"""Phase 4 — the bounded deviation-driven refinement cycle (docs/PHASE4_REFINEMENT_DESIGN.md).

After ``reconstruct()`` has built the plan and measured the FIRST deviation pass, this
module interprets the exact signed clusters (and the reconciliation), maps the single
most significant SUPPORTED discrepancy to the feature/parameter responsible, amends a
COPY of the plan, executes the amendment through the EXISTING builder, re-measures on
the same measured-only mask, and accepts or rejects under a strict monotonicity
contract:

* refinement must NEVER weaken an outcome classification — accept requires
  ``rank(new outcome) >= rank(old outcome)`` and no primary metric worsening;
* a state that is equal within the measurement noise floor everywhere is INCONCLUSIVE
  and therefore REJECTED ("did no verified good") — the loop is strictly non-harmful;
* a rejected amendment is restored through a 4-rung ladder (inverse sync → rebuild_from
  → apply_plan last resort → honest give-up), so the original result is preserved until
  the alternative PASSES;
* never rebuild the whole model when a focused fix suffices: param-only amendments go
  through ``builder.sync_parameters``, an appended dressing op through
  ``builder.apply_feature``, a moved hole through the builder's own focused
  ``rebuild_from``; ``apply_plan`` exists only as the reject ladder's last resort;
* ONE automatic iteration, ZERO LLM calls, ≤ 2-3 builder invocations, a phased wall
  budget with an injectable clock.

Everything outside the supported rule table (R1 hole/pocket width, R2 hole position,
R3 missing dressing, R4 first-band thickness; R5 reconciliation entries are
suggestion-only) is SURFACED, not executed: the report carries a ``suggestion`` for the
agent channel (``mode='plan'`` + ``plan_override``).

The driver ``run_refinement`` NEVER raises; every mutation goes through the builder's
own transactions, so an internal error leaves the original build intact.

Pure interpretation/mapping/acceptance (numpy + plain dicts) at module level; FreeCAD
is touched only inside execute / re-measure / restore.
"""

import copy
import math
import time

import numpy as np

from ..plan import builder
from . import intent as _intent

REFINE_BUDGET_S = 120.0
PHASE_BUDGET_S = {"interpret": 5.0, "amend": 2.0, "execute": 45.0, "remeasure": 60.0}
MAX_FACETS = 10_000                # the memory gate (case-07 class is agent territory)
MAX_CLUSTERS_EXAMINED = 3
MAX_BUILDER_CALLS = 3              # ≤2 on the amend path, ≤3 total incl. the last resort
OUTCOME_RANK = {"reconstruction_failed": 0, "reconstruction_unverified": 1,
                "partially_verified_parametric": 2, "verified_parametric": 3}
# the six outcome caps (reconstruct computes them from mesh/repair/units state, never
# from parameter values — identical before and after an amendment; the loop is FORBIDDEN
# from adding or removing them)
_OUTCOME_CAPS = ("extra_geometry", "synthetic_dimension_flags", "provenance_uncertain",
                 "units_unconfirmed", "scan_degraded", "partial_evidence_outcome_cap")

_COS10 = math.cos(math.radians(10.0))
_COS30 = math.cos(math.radians(30.0))
_COS60 = math.cos(math.radians(60.0))


def _eps(x) -> float:
    """The measurement noise floor: two states within it are equal."""
    return max(0.01 * abs(float(x)), 0.005)


def _clamp_delta(delta: float, old: float) -> float:
    """The amendment clamp (R1/R4): |Δ| <= max(0.5 mm, 25 % of the current value)."""
    limit = max(0.5, 0.25 * abs(old))
    return float(np.clip(delta, -limit, limit))


# ---------------------------------------------------------------------------
# interpretation


def interpret(result, scan_area=0.0) -> list:
    """Ranked discrepancy records: reconciliation entries first (suggestion-only
    priority, §2.2 R5), then decisive clusters by area (already sorted desc by
    ``_finish_clusters``). Filters E1 (decisive kind only — ``unsigned`` never drives)
    and E2 (>= 3 points AND area >= max(1 mm², 0.2 % of the scan area)). E0 (exact rows
    present) is a GATE checked by ``run_refinement``, not a filter here."""
    out = []
    unexplained = (((result.get("meta") or {}).get("reconciliation") or {})
                   .get("unexplained")) or []
    for entry in unexplained:
        out.append({"source": "reconciliation", "kind": "unexplained_holes",
                    "area_mm2": math.inf, "centroid": None, "cluster": None,
                    "text": str(entry)})
    area_bar = max(1.0, 0.002 * max(float(scan_area or 0.0), 0.0))
    for c in ((result.get("residual_analysis") or {}).get("clusters")) or []:
        if c.get("kind") not in ("missing", "excess"):
            continue                                   # E1: unsigned never drives
        if int(c.get("points") or 0) < 3:
            continue                                   # E2: the clustering floor
        if float(c.get("area_mm2") or 0.0) < area_bar:
            continue                                   # E2: per-point-share noise
        out.append({"source": "cluster", "kind": c["kind"],
                    "area_mm2": float(c["area_mm2"]),
                    "centroid": np.asarray(c["centroid_mm"], dtype=float),
                    "cluster": c})
    return out


def _resolve_ref(plan, ref):
    """A plan reference (a literal number or a parameter name) → float, or None."""
    if isinstance(ref, bool) or ref is None:
        return None
    if isinstance(ref, (int, float)):
        return float(ref)
    if isinstance(ref, str):
        spec = (plan.get("parameters") or {}).get(ref)
        if isinstance(spec, dict) and isinstance(spec.get("value"), (int, float)) \
                and not isinstance(spec.get("value"), bool):
            return float(spec["value"])
    return None


def _frame_of(plan):
    """The prismatic frame recovered from the plan's own feature records (§2.2): the
    first pad's sketch gives ``n`` / ``u`` (``v = n x u``) and the outline centre (the
    emission re-centres the base sketch on the section), and the pad-thickness
    parameters give ``total_h``. Returns None for plans without a sketch+pad pair
    (rotational / agent-amended routes never map automatically)."""
    feats = plan.get("features") or []
    pad = next((f for f in feats if isinstance(f, dict) and f.get("op") == "pad"), None)
    if pad is None:
        return None
    sk = next((f for f in feats if isinstance(f, dict) and f.get("id") == pad.get("sketch")),
              None)
    plane = (sk or {}).get("plane") if isinstance(sk, dict) else None
    if not isinstance(plane, dict):
        return None
    try:
        n = np.asarray([float(x) for x in plane["normal"]], dtype=float)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            return None
        n = n / nn
        u = np.asarray([float(x) for x in plane.get("x_dir") or [0.0, 0.0, 0.0]],
                       dtype=float)
        u = u - float(u @ n) * n
        un = float(np.linalg.norm(u))
        if un < 1e-9:
            ref = np.array([1.0, 0.0, 0.0]) if abs(float(n[0])) < 0.9 \
                else np.array([0.0, 1.0, 0.0])
            u = ref - float(ref @ n) * n
            un = float(np.linalg.norm(u))
        u = u / un
    except (KeyError, TypeError, ValueError):
        return None
    v = np.cross(n, u)
    params = plan.get("parameters") or {}
    thickness = 0.0
    for name, spec in params.items():
        if name == "plate_t" or (name.startswith("pad") and name.endswith("_t")):
            value = _resolve_ref(plan, name)
            thickness += value if value else 0.0
    if thickness <= 0.0:
        return None
    eu = _resolve_ref(plan, "plate_len")
    ev = _resolve_ref(plan, "plate_wid")
    if eu is None or ev is None:
        eu = ev = _resolve_ref(plan, "plate_d")   # a circular section outline
    return {"n": n, "u": u, "v": v,
            "base_origin": np.asarray([float(x) for x in plane.get("origin", [0, 0, 0])],
                                      dtype=float),
            "total_h": thickness, "extent_u": eu or 0.0, "extent_v": ev or 0.0,
            "first_pad": pad, "top_param": pad.get("length")}


def _hole_features(plan) -> list:
    """Hole / pocket features with their resolved axis frame: ``feat`` plus ``axis``
    (unit), ``axis_point`` (world), ``radius`` (planned), ``depth`` (mm), ``dia_param``."""
    feats = plan.get("features") or []
    by_id = {f.get("id"): f for f in feats if isinstance(f, dict)}
    out = []
    for feat in feats:
        if not isinstance(feat, dict) or feat.get("op") not in ("hole", "pocket"):
            continue
        if feat.get("counterbore") or feat.get("countersink"):
            continue            # coaxial sub-diameters: deferred to the agent (§2.3)
        plane = feat.get("plane") or {}
        try:
            axis = np.asarray([float(x) for x in plane["normal"]], dtype=float)
            an = float(np.linalg.norm(axis))
            if an < 1e-9:
                continue
            axis = axis / an
            x_dir = np.asarray([float(x) for x in plane.get("x_dir") or [0.0, 0.0, 0.0]],
                               dtype=float)
            x_dir = x_dir - float(x_dir @ axis) * axis
            xn = float(np.linalg.norm(x_dir))
            if xn < 1e-9:
                continue
            x_dir = x_dir / xn
            y_dir = np.cross(axis, x_dir)
            origin = np.asarray([float(x) for x in plane.get("origin", [0, 0, 0])],
                                dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        pos = feat.get("position") or {}
        px = _resolve_ref(plan, pos.get("x")) if feat["op"] == "hole" else 0.0
        py = _resolve_ref(plan, pos.get("y")) if feat["op"] == "hole" else 0.0
        axis_point = origin + (px or 0.0) * x_dir + (py or 0.0) * y_dir
        # the opening's size: a hole names ``diameter``; a pocket's sketch carries the
        # profile (circle diameter / rectangle width+height parameter names)
        dia_param = feat.get("diameter")
        radius = (_resolve_ref(plan, dia_param) or 0.0) / 2.0 if dia_param else None
        rect = None
        if feat["op"] == "pocket":
            psk = by_id.get(feat.get("sketch")) or {}
            profile = psk.get("profile") or {}
            if profile.get("shape") == "circle":
                dia_param = profile.get("diameter")
                radius = (_resolve_ref(plan, dia_param) or 0.0) / 2.0
            elif profile.get("shape") == "rectangle":
                rect = (profile.get("width"), profile.get("height"))
                radius = None
        depth = None if feat.get("through") else _resolve_ref(plan, feat.get("depth"))
        out.append({"feat": feat, "axis": axis, "axis_point": axis_point,
                    "x_dir": x_dir, "y_dir": y_dir, "radius": radius,
                    "depth": depth, "dia_param": dia_param, "rect": rect,
                    "pos_refs": (pos.get("x"), pos.get("y")),
                    "sketch": by_id.get(feat.get("sketch")) or None})
    return out


def _axial_ok(entry, rec, frame) -> bool:
    """The axial half of R1's geometry test: the centroid lies within the planned
    opening's axial span (± half the depth — or the frame height for a through
    opening — plus 1 mm)."""
    depth = rec["depth"] if rec["depth"] else frame["total_h"]
    rel = np.asarray(entry["centroid"], dtype=float) - rec["axis_point"]
    return abs(float(rel @ rec["axis"])) <= 0.5 * depth + 1.0


def _wall_geometry_ok(entry, rec, frame) -> bool:
    """R1's geometry signature for a round opening: axial span plus the radial wall
    band (0.5r .. 1.6r)."""
    if not _axial_ok(entry, rec, frame):
        return False
    if rec["radius"] is None or rec["radius"] <= 0.0:
        return False
    rel = np.asarray(entry["centroid"], dtype=float) - rec["axis_point"]
    radial = float(np.linalg.norm(rel - float(rel @ rec["axis"]) * rec["axis"]))
    return 0.5 * rec["radius"] <= radial <= 1.6 * rec["radius"]


def _mean_dist(entry) -> float:
    return float(entry["cluster"].get("mean_dist_mm") or 0.0)


# ---------------------------------------------------------------------------
# mapping rules (pure; each returns None when its evidence bar is not met)


def _map_hole_opening(entry, plan, frame, candidates):
    """R1 — hole / circle-pocket wall ⇒ diameter correction (scan-side magnitude:
    Δd = ±2·mean_dist_mm, clamped; nominals re-applied by ``amend``). A rectangular
    pocket maps the side whose outward normal best matches (centroid − rect centre),
    adjusting that side's length/width parameter (§2.2 R1)."""
    kind = entry["kind"]
    sign = 1.0 if kind == "excess" else -1.0
    for rec in candidates:
        if not _axial_ok(entry, rec, frame):
            continue
        if rec["radius"] is not None:
            if not _wall_geometry_ok(entry, rec, frame):
                continue
            ref = rec["dia_param"]
        elif rec["rect"]:
            ref = _rect_side_ref(entry, rec, plan)
            if ref is None:
                continue
        else:
            continue
        old = _resolve_ref(plan, ref)
        if old is None or old <= 0.0:
            continue                                     # E3: the param must be bound
        delta = 2.0 * _mean_dist(entry) * sign
        clamped = _clamp_delta(delta, old)
        if abs(clamped) < _eps(old):
            continue
        what = rec["feat"]["id"]
        return {"rule": "R1", "feature_id": what, "mode": "params",
                "params": {ref: old + clamped},
                "evidenced": {ref: old + delta},
                "op_append": None, "rebuild_id": None,
                "evidence": (f"{kind} cluster {entry['area_mm2']:.1f} mm2 at the wall "
                             f"of {what} (mean dist {_mean_dist(entry):.3f} mm)")}
    return None


def _rect_side_ref(entry, rec, plan):
    """The rectangle pocket's mapped side: the wall band the centroid sits in picks
    the parameter (width along the sketch x axis, height along y)."""
    w = _resolve_ref(plan, rec["rect"][0])
    h = _resolve_ref(plan, rec["rect"][1])
    if not w or not h:
        return None
    rel = np.asarray(entry["centroid"], dtype=float) - rec["axis_point"]
    du, dv = float(rel @ rec["x_dir"]), float(rel @ rec["y_dir"])
    band = max(1.0, 2.0 * _mean_dist(entry) + 0.5)
    mu, mv = abs(abs(du) - w / 2.0), abs(abs(dv) - h / 2.0)
    if min(mu, mv) > band:
        return None
    if mu <= mv and abs(dv) <= h / 2.0 + band:
        return rec["rect"][0]
    if mv < mu and abs(du) <= w / 2.0 + band:
        return rec["rect"][1]
    return None


def _map_position_pair(entry, entries, plan, frame, sigma_mm):
    """R2 — hole position drift: TWO opposite-sign decisive clusters on the SAME
    planned opening with centroid separation ≈ 2r ± 40 %.

    The crescent pair straddles the SCAN's opening (both crescents hug the scan wall
    at C_scan ± r), so the pair MIDPOINT estimates the scan hole centre and the
    amendment is the midpoint's offset from the modelled axis — the correction that
    leaves both crescents zero. (The design text spells the amendment
    "Δ = (missing − excess)/2"; under its own separation bar 2r ± 40 % that vector
    always has magnitude ≈ r, which is the WALL RADIUS, not the drift — the midpoint
    estimator is the arithmetic that satisfies the design's own F3 oracle and its
    "position corrected" pin.) Bars: |Δ| >= max(0.2 mm, 2σ) and |Δ| <= 0.5·r."""
    candidates = _hole_features(plan)
    for rec in candidates:
        if rec["radius"] is None or rec["radius"] <= 0.0:
            continue
        px_ref, py_ref = rec["pos_refs"]
        if any(isinstance(r, str) and r.startswith("pat") for r in (px_ref, py_ref)):
            continue         # pattern-owned geometry: re-anchoring is agent territory
        if not _wall_geometry_ok(entry, rec, frame):
            continue
        r = rec["radius"]
        for other in entries:
            if other is entry or other["source"] != "cluster":
                continue
            if other["kind"] not in ("missing", "excess") or other["kind"] == entry["kind"]:
                continue
            if not _wall_geometry_ok(other, rec, frame):
                continue
            sep = float(np.linalg.norm(other["centroid"] - entry["centroid"]))
            if not (0.6 * 2.0 * r <= sep <= 1.4 * 2.0 * r):
                continue
            midpoint = (entry["centroid"] + other["centroid"]) / 2.0
            shift = midpoint - rec["axis_point"]
            axis = rec["axis"]
            shift = shift - float(shift @ axis) * axis          # in-plane only
            mag = float(np.linalg.norm(shift))
            if mag < max(0.2, 2.0 * sigma_mm):
                continue               # below scan noise a move is not evidenced
            if mag > 0.5 * r:
                shift = shift * (0.5 * r / mag)
                mag = 0.5 * r
            return _position_amendment(rec, shift, entry, other, plan, frame)
    return None


def _position_amendment(rec, shift, entry, other, plan, frame):
    """Build the R2 mapping record from the world-space hole shift."""
    feat = rec["feat"]
    idx = feat["id"].rsplit("_", 1)[-1]
    params, evidenced = {}, {}
    px_ref = (feat.get("position") or {}).get("x")
    py_ref = (feat.get("position") or {}).get("y")
    bound = isinstance(px_ref, str) or isinstance(py_ref, str)
    if bound:
        # positions bound to parameters (sketch-frame components)
        su = float(shift @ rec["x_dir"])
        sv = float(shift @ rec["y_dir"])
        for ref, d in ((px_ref, su), (py_ref, sv)):
            if isinstance(ref, str):
                old = _resolve_ref(plan, ref)
                if old is None:
                    continue
                params[ref] = old + d
                evidenced[ref] = old + d
        mode, rebuild_id = "params", None
    else:
        # the engine's prismatic emission: the position is the literal sketch origin;
        # the provenance params hole_x{i}/hole_y{i} are FRAME coordinates
        mode, rebuild_id = "feature", feat["id"]
        du, dv = float(shift @ frame["u"]), float(shift @ frame["v"])
        for name, d in ((f"hole_x{idx}", du), (f"hole_y{idx}", dv)):
            old = _resolve_ref(plan, name)
            if old is not None:
                params[name] = round(old + d, 4)
                evidenced[name] = old + d
    if not params:
        params, evidenced = {f"hole_x{idx}": float(shift @ frame["u"])}, {}
    return {"rule": "R2", "feature_id": feat["id"], "mode": mode,
            "params": params, "evidenced": evidenced, "op_append": None,
            "rebuild_id": rebuild_id,
            "origin_shift": [float(x) for x in shift],
            "evidence": ("hole position drift (opposite-sign crescent pair at the "
                         f"wall of {feat['id']}; centroids "
                         f"{entry['area_mm2']:.1f}/{other['area_mm2']:.1f} mm2)")}


def _map_dressing(entry, plan, frame, survey, meta):
    """R3 — a measured dressing the plan never realized (or the recovery ladder
    dropped) ⇒ append the op through ``apply_feature``. Convex perimeter
    fillets/chamfers missing from the model read as ``excess`` at the edge; the
    concave case stays agent territory in v1."""
    if entry["kind"] != "excess":
        return None
    if abs(float(frame["n"][2])) <= _COS30:
        return None                       # the edge rules need a Z-aligned frame
    total_h = frame["total_h"]
    recs = (survey.get("dressings") or [])
    if not recs:
        return None
    from .reconstruct import _RULE_IDS    # lazy: reconstruct imports nothing from refine

    dropped = set((((meta.get("dressings_dropped") or {}).get("features")) or ()))
    existing = {f.get("edges") for f in plan.get("features") or []
                if isinstance(f, dict) and f.get("op") in ("fillet", "chamfer")}
    centroid = entry["centroid"]
    far_rule = "top_outer" if float(frame["n"][2]) >= 0.0 else "bottom_outer"
    base_rule = "bottom_outer" if far_rule == "top_outer" else "top_outer"
    base_origin, n = frame["base_origin"], frame["n"]
    for rec in recs:
        kind = rec.get("kind")
        if kind not in ("fillet", "chamfer"):
            continue
        try:
            point = np.asarray([float(x) for x in rec["point"]], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if kind == "fillet":
            if rec.get("axis") is None:
                continue
            axis = np.asarray([float(x) for x in rec["axis"]], dtype=float)
            if float(np.linalg.norm(axis)) < 1e-9 or abs(float(axis @ n)) > _COS60:
                continue                     # vertical-corner fillets stay informational
            radius = float(rec.get("radius_mm") or 0.0)
            if not (2.0 <= radius <= 0.6 * total_h):
                continue                     # the emission's own sane bounds
        else:
            if rec.get("normal") is None:
                continue
            normal = np.asarray([float(x) for x in rec["normal"]], dtype=float)
            if float(np.linalg.norm(normal)) < 1e-9 or abs(float(normal @ n)) <= _COS60:
                continue                     # perimeter chamfers tilt out of the plane
            radius = float(rec.get("width_mm") or 0.0)
            if not (0.5 <= radius <= 0.6 * total_h):
                continue
        if float(np.linalg.norm(centroid - point)) > max(1.0, 1.5 * radius):
            continue                         # the cluster must sit at THIS dressing
        height = float((point - base_origin) @ n)
        rule = far_rule if height > 0.5 * total_h else base_rule
        if rule in existing:
            continue                         # that face group already has its op
        pid, fid = (_RULE_IDS[rule][0], _RULE_IDS[rule][1]) if kind == "fillet" \
            else (_RULE_IDS[rule][2], _RULE_IDS[rule][3])
        if pid in (plan.get("parameters") or {}) and fid not in dropped:
            continue                         # param exists but the op does not: unknown state
        params, op = {}, {"id": fid, "edges": rule}
        if kind == "fillet":
            params[pid] = round(radius, 3)
            op.update({"op": "fillet", "size": pid})
        else:
            # the emission's width → cut-distance arithmetic, verbatim (P1.1b)
            a1 = math.radians(float((rec.get("angles_deg") or [45.0, 45.0])[0]))
            d1, d2 = radius * math.cos(a1), radius * math.sin(a1)
            dpid = pid.replace("chamfer_w", "chamfer_d")
            if abs(d1 - d2) > 0.15 * max(d1, d2, 1e-9):
                params[dpid] = round(d1, 3)
                params[f"{dpid}2"] = round(d2, 3)
                op.update({"op": "chamfer", "size": dpid, "size2": f"{dpid}2"})
            else:
                params[dpid] = round((d1 + d2) / 2.0, 3)
                op.update({"op": "chamfer", "size": dpid})
        return {"rule": "R3", "feature_id": fid, "mode": "feature_append",
                "params": params,
                "evidenced": {next(iter(params)): next(iter(params.values()))},
                "op_append": op, "rebuild_id": None, "size_param": next(iter(params)),
                "evidence": (f"missing {kind} (measured "
                             f"{rec.get('radius_mm', rec.get('width_mm')):.2f} mm) at the "
                             f"{rule} edge — excess cluster {entry['area_mm2']:.1f} mm2")}
    return None


def _map_face_offset(entry, plan, frame):
    """R4 — whole-face offset of the FIRST band's top face ⇒ plate_t correction."""
    pad = frame["first_pad"]
    param = pad.get("length") if isinstance(pad, dict) else None
    old = _resolve_ref(plan, param) if isinstance(param, str) else None
    if old is None or old <= 0.0:
        return None                                     # E3
    centroid = entry["centroid"]
    rel = centroid - frame["base_origin"]
    h = float(rel @ frame["n"])
    if abs(h - old) > 1.0:
        return None                    # only the top face level, mid-band never maps
    du, dv = float(rel @ frame["u"]), float(rel @ frame["v"])
    eu, ev = frame["extent_u"], frame["extent_v"]
    if eu <= 0.0 or ev <= 0.0:
        return None
    from_outline = min(eu / 2.0 - abs(du), ev / 2.0 - abs(dv))
    if from_outline < 0.25 * min(eu, ev):
        return None                    # mid-face, not an edge band
    if entry["area_mm2"] < 0.05 * eu * ev:
        return None                    # >= 5 % of that face's area
    delta = _mean_dist(entry) * (-1.0 if entry["kind"] == "excess" else 1.0)
    clamped = _clamp_delta(delta, old)
    if abs(clamped) < _eps(old):
        return None
    return {"rule": "R4", "feature_id": pad.get("id"), "mode": "params",
            "params": {param: old + clamped},
            "evidenced": {param: old + delta},
            "op_append": None, "rebuild_id": None,
            "evidence": (f"{entry['kind']} cluster {entry['area_mm2']:.1f} mm2 across "
                         f"the first band's top face (mean dist "
                         f"{_mean_dist(entry):.3f} mm)")}


def map_discrepancy(result, plan, frame, survey, meta, sigma_mm, scan_area):
    """Rank (``interpret``), then map the single most significant SUPPORTED
    discrepancy (E4: one amendment per cycle). Order per cluster: R3 (survey-
    corroborated dressing) → R2 (opposite-sign pair — more specific than a ring) →
    R1 → R4; the top-{MAX_CLUSTERS_EXAMINED} clusters are examined. Returns
    ``(mapping, None)`` or ``(None, {"reason", "suggestion"})`."""
    entries = interpret(result, scan_area)
    if not entries:
        return None, {"reason": "no beyond-warn clusters and nothing unexplained — "
                                "nothing to refine",
                      "suggestion": None}
    if entries[0]["source"] == "reconciliation":
        # R5: recognized-but-unrealized holes outrank everything as a PRIORITY, but
        # re-running the emission blind would not fix a synthesis decision — the
        # honest behaviour is the suggestion, never an auto-amendment
        text = entries[0]["text"]
        return None, {"reason": f"reconciliation unexplained ({text}) — not "
                                "auto-executed (R5)",
                      "suggestion": f"{text}; inspect the plan (mode='plan') and amend "
                                    "it via plan_override"}
    candidates = _hole_features(plan)
    for entry in [e for e in entries if e["source"] == "cluster"][:MAX_CLUSTERS_EXAMINED]:
        mapping = (_map_dressing(entry, plan, frame, survey, meta)
                   or _map_position_pair(entry, entries, plan, frame, sigma_mm)
                   or _map_hole_opening(entry, plan, frame, candidates)
                   or _map_face_offset(entry, plan, frame))
        if mapping is not None:
            return mapping, None
    top = entries[0]
    what = (f"{top['kind']} cluster {top['area_mm2']:.1f} mm2 at "
            f"{[round(float(x), 1) for x in top['centroid']]}") \
        if top["centroid"] is not None else top.get("text", "discrepancy")
    return None, {"reason": f"no supported mapping for the most significant "
                            f"discrepancy ({what}) — outside rules R1-R4",
                  "suggestion": f"top discrepancy {what}; take it through mode='plan' "
                                "+ plan_override (outline edits, rotational profiles, "
                                "counterbore/countersink and pattern members are agent "
                                "territory in v1)"}


# ---------------------------------------------------------------------------
# amend / execute / re-measure / accept / restore


def amend(plan, mapping, sigma_mm):
    """Deep-copy the plan, write the amended parameters with a ``scan_fit`` source
    (kind + 'refinement cycle 1' ref — the sheet's D/E columns), re-apply the G8
    nominal ladder to the CHANGED parameters only, append the R3 op, and shift the
    R2 feature's literal sketch origin. Returns ``(plan2, amendment)``; plan2 is
    None when a parameter vanished (E3 violated late)."""
    plan2 = copy.deepcopy(plan)
    changed = {}
    for name, new_value in mapping["params"].items():
        spec = (plan2.get("parameters") or {}).get(name)
        if not isinstance(spec, dict) or not isinstance(spec.get("value"), (int, float)):
            return None, f"parameter {name!r} is not a bound numeric parameter (E3)"
        changed[name] = (float(spec["value"]),
                         float(mapping.get("evidenced", {}).get(name, new_value)))
        spec["value"] = round(float(new_value), 4)
        spec["source"] = {"kind": "scan_fit",
                          "ref": f"refinement cycle 1: {mapping['evidence']}"[:120]}
    if mapping.get("origin_shift"):
        for feat in plan2["features"]:
            if feat.get("id") == mapping["feature_id"] and isinstance(feat.get("plane"),
                                                                      dict):
                origin = [float(x) + d for x, d in zip(feat["plane"].get("origin",
                                                                         [0, 0, 0]),
                                                       mapping["origin_shift"])]
                feat["plane"]["origin"] = [round(x, 6) for x in origin]
    if mapping.get("op_append"):
        plan2["features"].append(copy.deepcopy(mapping["op_append"]))
    snaps = []
    if changed:
        sub = {name: plan2["parameters"][name] for name in changed}
        snapped, snaps = _intent.apply_nominals(sub, sigma_mm)
        plan2["parameters"].update(snapped)
    amendment = {
        "rule": mapping["rule"], "feature_id": mapping["feature_id"],
        "mode": mapping["mode"], "evidence": mapping["evidence"],
        "changed": {name: {"old": old,
                           "new": plan2["parameters"][name]["value"],
                           "evidenced": round(ev, 4)}
                    for name, (old, ev) in changed.items()},
        "nominal_snaps": snaps}
    return plan2, amendment


class _Counter:
    """Builder-invocation budget: ≤ MAX_BUILDER_CALLS in the normal flow; the restore
    ladder's last resort (apply_plan) may exceed it — a half-restored document is the
    worst outcome, and the report records the overrun."""

    def __init__(self, limit=MAX_BUILDER_CALLS):
        self.count = 0
        self.limit = limit

    def call(self, fn, *args, **kwargs):
        self.count += 1
        return fn(*args, **kwargs)


def _body_valid(body) -> bool:
    try:
        if "Invalid" in list(getattr(body, "State", ())):
            return False
        shape = body.Shape
        return bool(shape is not None and shape.isValid()
                    and len(shape.Solids) == 1)
    except Exception:  # noqa: BLE001 - an undecidable body is not valid
        return False


def _body_volume(body):
    try:
        return float(body.Shape.Volume)
    except Exception:  # noqa: BLE001
        return None


def _execute(doc, plan2, mapping, build, counter):
    """One focused builder invocation (never a whole-plan rebuild): sync_parameters
    for a param amendment, apply_feature for an appended op, rebuild_from for a moved
    hole (the builder's own repair path — it re-applies only that feature and what
    follows it). Returns ``(ok, report, mode)``."""
    sheet_name = build.get("spreadsheet") or "Parameters"
    body_name = build.get("body") or "Body"
    mode = mapping.get("mode", "params")
    if counter.count >= counter.limit:
        return False, {"success": False, "error": "builder-call budget exhausted"}, mode
    if mode == "params":
        report = counter.call(builder.sync_parameters, doc, plan2,
                              sheet_name=sheet_name)
        ok = bool(report.get("success"))
    elif mode == "feature":
        report = counter.call(builder.rebuild_from, doc, plan2, mapping["rebuild_id"],
                              body_name=body_name, sheet_name=sheet_name)
        ok = bool(report.get("success"))
    else:
        report = counter.call(builder.apply_feature, doc, plan2, mapping["feature_id"],
                              body_name=body_name, sheet_name=sheet_name)
        ok = bool(report.get("success"))
    if ok:
        try:
            doc.recompute()
            builder.refresh_edge_rules(doc)   # a dressed body's edge links can go stale
            doc.recompute()
        except Exception:  # noqa: BLE001 - the settle must not fail the amendment
            pass
    return ok, report, mode


def remeasure(work_mesh, body, mask, warn_mm):
    """The SAME measured-only mask and engines as the first pass; ICP skipped
    (``register=False``) — RAW decides, and aligned rows are not needed to compare
    two candidate states of the same frame. Returns ``(rows, residual_analysis)``."""
    from . import deviation

    measured = deviation.registered_deviation(
        work_mesh, body.Shape, a="scan", b=body.Name, warn_mm=warn_mm, mask=mask,
        register=False)
    rows = measured["raw"]
    residual = deviation.exact_residuals(work_mesh, body.Shape, a="scan", b=body.Name,
                                         warn_mm=warn_mm, mask=mask)
    return rows + residual["rows"], residual


def _row_value(rows, rid):
    for r in rows:
        if r.get("id") == rid:
            return None if r.get("status") == "SKIP" else float(r.get("value") or 0.0)
    return None


def metrics_of(rows, volume_scan, body) -> dict:
    """The primary metric set M (§4.1) from deviation rows + the live body volume."""
    metrics = {"hausdorff": _row_value(rows, "hausdorff"),
               "scan_to_model.max": _row_value(rows, "scan_to_model.max"),
               "scan_to_model.rms": _row_value(rows, "scan_to_model.rms"),
               "model_to_scan.max": _row_value(rows, "model_to_scan.max"),
               "model_to_scan.rms": _row_value(rows, "model_to_scan.rms"),
               "scan_to_model.coverage": _row_value(rows, "scan_to_model.coverage"),
               "exact_scan_to_model.max": _row_value(rows, "exact_scan_to_model.max"),
               "exact_model_to_scan.max": _row_value(rows, "exact_model_to_scan.max")}
    volume = _body_volume(body) if body is not None else None
    metrics["volume_error"] = (abs(volume - volume_scan) / volume_scan
                               if volume is not None and volume_scan else None)
    return metrics


_WORSENS = ("hausdorff", "scan_to_model.max", "scan_to_model.rms",
            "model_to_scan.max", "model_to_scan.rms",
            "exact_scan_to_model.max", "exact_model_to_scan.max")


def accept(old_m, new_m, mapping, old_rank, new_rank, body):
    """The monotonicity contract (§4.2). REJECT on any primary metric worsening, an
    invalid body, or a lower outcome rank; ACCEPT only when the TARGETED evidence
    improved as well; equal-within-eps everywhere is inconclusive ⇒ REJECT."""
    improved = False
    if body is not None and not _body_valid(body):
        return False, "the amended body is invalid"
    for key in _WORSENS:
        old, new = old_m.get(key), new_m.get(key)
        if old is None or new is None:
            continue
        if new > old + _eps(old):
            return False, f"{key} worsened {old:.4f} -> {new:.4f} mm"
        if new < old - _eps(old):
            improved = True
    old_cov, new_cov = old_m.get("scan_to_model.coverage"), \
        new_m.get("scan_to_model.coverage")
    if old_cov is not None and new_cov is not None:
        if new_cov < old_cov - 1e-3:
            return False, f"coverage worsened {old_cov:.4f} -> {new_cov:.4f}"
        if new_cov > old_cov + 1e-3:
            improved = True
    old_ve, new_ve = old_m.get("volume_error"), new_m.get("volume_error")
    if old_ve is not None and new_ve is not None:
        if new_ve > old_ve + max(0.001, _eps(old_ve)):
            return False, f"volume error worsened {100 * old_ve:.2f}% -> " \
                          f"{100 * new_ve:.2f}%"
        if new_ve < old_ve - max(0.001, _eps(old_ve)):
            improved = True
    if new_rank < old_rank:
        return False, "outcome rank would drop — refinement never weakens " \
                      "classifications"
    # targeted evidence: the deciding cluster's own family must have moved the right way
    old_e1, new_e1 = old_m.get("exact_scan_to_model.max"), \
        new_m.get("exact_scan_to_model.max")
    old_e2, new_e2 = old_m.get("exact_model_to_scan.max"), \
        new_m.get("exact_model_to_scan.max")
    if mapping["rule"] == "R3":
        olds = [x for x in (old_e1, old_e2) if x is not None]
        news = [x for x in (new_e1, new_e2) if x is not None]
        if not news or len(news) < len(olds):
            return False, "re-measure exact rows SKIPped — inconclusive"
        if max(news) >= max(olds) - _eps(max(olds)):
            return False, (f"the appended dressing did not drop the exact max "
                           f"({max(olds):.4f} -> {max(news):.4f} mm)")
        improved = True
    else:
        for old, new in ((old_e1, new_e1), (old_e2, new_e2)):
            if old is None:
                continue
            if new is None:
                return False, "re-measure exact rows SKIPped — inconclusive"
            if new > old + _eps(old):
                return False, f"exact max worsened {old:.4f} -> {new:.4f} mm"
        changed = mapping.get("_changed") or {}
        for name, (old, new, evidenced) in changed.items():
            if abs(evidenced - old) > 1e-9:
                if not abs(new - evidenced) < abs(old - evidenced):
                    return False, f"parameter {name} did not move toward the evidence"
                improved = True
            elif abs(new - old) > _eps(old):
                return False, f"parameter {name} moved without evidence"
    if not improved:
        return False, "inconclusive — no metric moved beyond the noise floor"
    return True, "targeted evidence improved; no primary metric worsened"


def _outcome_of(hausdorff, warn_mm, meta) -> str:
    """The SAME expression reconstruct() uses, from the SAME meta flags (§5)."""
    ok = hausdorff is not None and hausdorff <= warn_mm
    if ok and not any(cap in meta for cap in _OUTCOME_CAPS):
        return "verified_parametric"
    return "partially_verified_parametric" if ok else "reconstruction_unverified"


def restore(doc, original_plan, build, volume_before, *, appended_id=None,
            first_feature=None, counter):
    """The §4.3 reject-restore ladder. Returns ``{"rung": 1..4, "ok": bool, ...}``;
    the original is preserved until an alternative state passes the volume + validity
    check of rung 1."""
    sheet_name = build.get("spreadsheet") or "Parameters"
    body_name = build.get("body") or "Body"

    def verified() -> bool:
        body = doc.getObject(body_name)
        if body is None or not _body_valid(body):
            return False
        volume = _body_volume(body)
        return volume is not None and volume_before is not None \
            and abs(volume - volume_before) <= 1e-6 * max(1.0, abs(volume_before))

    if appended_id is not None:
        report = counter.call(builder.drop_feature, doc, original_plan, appended_id,
                              body_name=body_name, sheet_name=sheet_name)
        if report.get("success") and verified():
            return {"rung": 1, "ok": True, "via": "drop_feature"}
    if counter.count < counter.limit:
        report = counter.call(builder.sync_parameters, doc, original_plan,
                              sheet_name=sheet_name)
        if report.get("success"):
            try:
                doc.recompute()
                builder.refresh_edge_rules(doc)
                doc.recompute()
            except Exception:  # noqa: BLE001
                pass
            if verified():
                return {"rung": 1, "ok": True, "via": "inverse sync"}
    if first_feature and counter.count < counter.limit:
        report = counter.call(builder.rebuild_from, doc, original_plan, first_feature,
                              body_name=body_name, sheet_name=sheet_name)
        if report.get("success") and verified():
            return {"rung": 2, "ok": True, "via": "rebuild_from"}
    # the sanctioned last resort (may exceed the builder-call budget — recorded)
    report = counter.call(builder.apply_plan, doc, original_plan)
    if report.get("success") and verified():
        return {"rung": 3, "ok": True, "via": "apply_plan",
                "budget_overrun": counter.count > counter.limit}
    return {"rung": 4, "ok": False,
            "note": "restore ladder exhausted — the document is left in the "
                    "restored-as-possible state; the caller forces "
                    "reconstruction_failed with meta.refinement_restore_failed"}


# ---------------------------------------------------------------------------
# the driver


def run_refinement(doc, work_mesh, result, *, warn_mm, budget_s=REFINE_BUDGET_S,
                   clock=None):
    """One bounded refinement cycle inside the reconstruct() call that already
    measured the first pass. Returns the report dict (``attempted/decision/reason/
    rule/metrics_before/metrics_after/amendment/restored/remaining_clusters/
    outcome_before/outcome_after/builder_calls``; an ACCEPTED report additionally
    carries ``plan``/``deviation``/``residual_analysis``/``nominal_snaps`` for the
    caller to swap in), or None when the cycle was NOT ATTEMPTED (§2.4 gating:
    non-prismatic or agent-amended route, failed build, SKIPped exact rows, or a
    scan beyond MAX_FACETS). NEVER raises."""
    counter = _Counter()
    try:
        return _run(doc, work_mesh, result, warn_mm=warn_mm, budget_s=budget_s,
                    clock=clock, counter=counter)
    except Exception as e:  # noqa: BLE001 - the cycle must never break the build
        return {"attempted": True, "decision": "skipped",
                "reason": f"internal error: {type(e).__name__}: {e}"[:200],
                "builder_calls": counter.count}


def _base_report(result, warn_mm, body, volume_scan, counter):
    rows = result.get("deviation") or []
    metrics = metrics_of(rows, volume_scan, body)
    outcome = _outcome_of(metrics.get("hausdorff"), warn_mm, result.get("meta") or {})
    return metrics, outcome, OUTCOME_RANK[outcome]


def _run(doc, work_mesh, result, *, warn_mm, budget_s, clock, counter):
    plan = result.get("plan") or {}
    meta = result.get("meta") or {}
    build = result.get("build") or {}
    # ---- §2.4 gating (not attempted ⇒ None; the render's suggestion is the seam's)
    if str(meta.get("route") or "") != "prismatic_extrusion":
        return None
    if not build.get("success"):
        return None
    if int(getattr(work_mesh, "CountFacets", 0) or 0) > MAX_FACETS:
        return None
    rows = result.get("deviation") or []
    if _row_value(rows, "exact_scan_to_model.max") is None \
            or _row_value(rows, "exact_model_to_scan.max") is None:
        return None                                     # E0: exact rows must be present
    body = doc.getObject(build.get("body") or "")
    if body is None:
        return None
    frame = _frame_of(plan)
    if frame is None:
        return None
    survey = result.get("survey") or {}
    sigma = float((survey.get("quality") or {}).get("sigma_mm") or 0.0)
    volume_scan = ((result.get("diagnostics") or {}).get("volume_mm3")) or None
    metrics_before, outcome_before, rank_before = _base_report(result, warn_mm, body,
                                                               volume_scan, counter)
    volume_before = _body_volume(body)
    # ---- phased wall budget (injectable clock)
    now = clock or time.monotonic
    t0 = now()
    cum, deadlines = 0.0, {}
    for phase, seconds in PHASE_BUDGET_S.items():
        cum += seconds
        deadlines[phase] = min(t0 + cum, t0 + max(1.0, float(budget_s)))

    def left(phase):
        return now() < deadlines[phase]

    def report(**extra):
        base = {"attempted": True, "builder_calls": counter.count,
                "metrics_before": metrics_before, "outcome_before": outcome_before,
                "rule": (extra.get("amendment") or {}).get("rule")}
        base.update(extra)
        return base

    # ---- 1 interpret + map
    if not left("interpret"):
        return report(decision="skipped",
                      reason="budget exhausted before interpretation — nothing written")
    mapping, why_not = map_discrepancy(result, plan, frame, survey, meta, sigma,
                                       float(getattr(work_mesh, "Area", 0.0) or 0.0))
    if mapping is None:
        return report(decision="skipped", reason=why_not["reason"],
                      suggestion=why_not.get("suggestion"))
    # ---- 2 amend
    if not left("amend"):
        return report(decision="skipped",
                      reason="budget exhausted before amend — nothing written")
    plan2, amendment = amend(plan, mapping, sigma)
    if plan2 is None:
        return report(decision="skipped", reason=amendment)
    # ---- 3 execute
    if not left("execute"):
        return report(decision="skipped",
                      reason="budget exhausted before execute — nothing written")
    ok, exec_report, mode = _execute(doc, plan2, mapping, build, counter)
    if not ok:
        # the builder rolled its own writes back (sync restores cells, _remove_created
        # removes objects) — verify, and only then touch the ladder
        body_now = doc.getObject(build.get("body") or "")
        vol_now = _body_volume(body_now) if body_now is not None else None
        intact = (body_now is not None and _body_valid(body_now)
                  and volume_before is not None and vol_now is not None
                  and abs(vol_now - volume_before)
                  <= 1e-6 * max(1.0, abs(volume_before)))
        restored = ({"rung": 0, "ok": True, "via": "builder rollback"} if intact
                    else restore(doc, plan, build, volume_before,
                                 appended_id=(mapping["feature_id"]
                                              if mode == "feature_append" else None),
                                 first_feature=(mapping["feature_id"]
                                                if mode == "feature" else None),
                                 counter=counter))
        return report(decision="rejected",
                      reason=f"amendment build failed: {exec_report.get('error')}",
                      amendment=amendment, restored=restored)
    # ---- 4 re-measure (same mask, no ICP)
    if not left("remeasure"):
        restored = restore(doc, plan, build, volume_before,
                           appended_id=(mapping["feature_id"]
                                        if mode == "feature_append" else None),
                           first_feature=(mapping["feature_id"] if mode == "feature"
                                          else None),
                           counter=counter)
        return report(decision="rejected",
                      reason="budget exhausted before re-measure — inconclusive, "
                             "original restored",
                      amendment=amendment, restored=restored)
    try:
        rows2, residual2 = remeasure(work_mesh, body, result.get("measured_mask"),
                                     warn_mm)
    except Exception as e:  # noqa: BLE001 - an unreadable state is a reject, not a crash
        restored = restore(doc, plan, build, volume_before,
                           appended_id=(mapping["feature_id"]
                                        if mode == "feature_append" else None),
                           first_feature=(mapping["feature_id"] if mode == "feature"
                                          else None),
                           counter=counter)
        return report(decision="rejected",
                      reason=f"re-measure failed ({type(e).__name__}: {e})"[:160],
                      amendment=amendment, restored=restored)
    # ---- 5 accept?
    metrics_after = metrics_of(rows2, volume_scan, body)
    outcome_after = _outcome_of(metrics_after.get("hausdorff"), warn_mm, meta)
    changed = {name: (vals["old"], plan2["parameters"][name]["value"],
                      vals["evidenced"])
               for name, vals in amendment["changed"].items()}
    mapping = {**mapping, "_changed": changed}
    ok, reason = accept(metrics_before, metrics_after, mapping, rank_before,
                        OUTCOME_RANK[outcome_after], body)
    if not ok:
        restored = restore(doc, plan, build, volume_before,
                           appended_id=(mapping["feature_id"]
                                        if mode == "feature_append" else None),
                           first_feature=(mapping["feature_id"] if mode == "feature"
                                          else None),
                           counter=counter)
        return report(decision="rejected", reason=reason, amendment=amendment,
                      metrics_after=metrics_after, outcome_after=outcome_after,
                      restored=restored,
                      remaining_clusters=(residual2.get("clusters") or []))
    # ---- accepted: the caller swaps plan/deviation/outcome (the report carries them)
    return report(decision="accepted", reason=reason, amendment=amendment,
                  metrics_after=metrics_after, outcome_after=outcome_after,
                  plan=plan2, deviation=rows2, residual_analysis=residual2,
                  nominal_snaps=amendment["nominal_snaps"],
                  remaining_clusters=(residual2.get("clusters") or []))
