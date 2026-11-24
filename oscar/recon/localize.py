# SPDX-License-Identifier: Apache-2.0
"""L-2: localized deviation regions with conservative feature attribution.

WHERE the generated solid differs from the source (spatial regions from the exact residual
clusters, both directions independently), and WHICH reconstructed feature is associated with
that difference (mesh region → recognized feature → FeaturePlan operation → CAD object).

Conservative by construction: a distance measurement alone never proves a cause — DIMENSION /
POSITION refinements require the full evidence chain (a recognized fit AND a plan operation
that disagree specifically); attribution is HIGH only when the chain is complete and
type-consistent, never on bounding-box overlap alone; insufficient evidence stays UNATTRIBUTED
or AMBIGUOUS. Pure functions on plain dicts — no FreeCAD, no document mutation."""

import math

TYPE_MISSING = "MISSING_MATERIAL"
TYPE_EXTRA = "EXTRA_MATERIAL"
TYPE_DIMENSION = "DIMENSION_ERROR"
TYPE_POSITION = "POSITION_ERROR"
TYPE_UNATTRIBUTED = "UNATTRIBUTED"

CONF_HIGH = "HIGH"
CONF_MEDIUM = "MEDIUM"
CONF_AMBIGUOUS = "AMBIGUOUS"
CONF_UNATTRIBUTED = "UNATTRIBUTED"


def _dist(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return [[round(min(xs), 3), round(min(ys), 3), round(min(zs), 3)],
            [round(max(xs), 3), round(max(ys), 3), round(max(zs), 3)]]


def _extent(box):
    return max(box[1][i] - box[0][i] for i in range(3))


def _world_position(feature):
    """The hole operation's world position. Two supported forms: a plain ``[x, y, z]``
    (list/tuple) and the plan's real sketch-frame ``{"x", "y"}`` resolved through the
    feature's ``plane`` (``origin + x*x_dir + y*(normal x x_dir)``). None when the
    feature carries neither."""
    pos = feature.get("position") or feature.get("center")
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    if isinstance(pos, dict) and "x" in pos and "y" in pos:
        plane = feature.get("plane") or {}
        origin = plane.get("origin")
        x_dir = plane.get("x_dir")
        normal = plane.get("normal")
        if origin and x_dir and normal:
            px, py = float(pos["x"]), float(pos["y"])
            y_dir = [normal[1] * x_dir[2] - normal[2] * x_dir[1],
                     normal[2] * x_dir[0] - normal[0] * x_dir[2],
                     normal[0] * x_dir[1] - normal[1] * x_dir[0]]
            return [float(origin[i]) + px * float(x_dir[i]) + py * y_dir[i] for i in range(3)]
    return None


def _hole_ops(plan):
    """The plan's hole operations with a usable position proxy: (op, world position,
    diameter in mm). The diameter may be a literal number or a PARAMETER NAME resolved
    through the plan's parameter table (the engine emits names for equal-diameter holes)."""
    parameters = (plan or {}).get("parameters") or {}
    out = []
    for f in (plan or {}).get("features") or []:
        if f.get("op") != "hole":
            continue
        world = _world_position(f)
        if not world:
            continue
        dia = f.get("diameter")
        if isinstance(dia, str):
            dia = (parameters.get(dia) or {}).get("value")
        out.append((f, world, float(dia) if dia is not None else 0.0))
    return out


def _dressing_ops(plan, parameters):
    """The plan's fillet/chamfer operations with their constructed size from the
    parameter table: (op, size_value_mm)."""
    out = []
    for f in (plan or {}).get("features") or []:
        if f.get("op") not in ("fillet", "chamfer"):
            continue
        size = (parameters or {}).get(f.get("size")) or {}
        value = size.get("value")
        if value is not None:
            out.append((f, float(value)))
    return out


def _nearest_dressing(point, survey):
    """The survey dressing strip nearest ``point``: (record, distance) — distance to the
    fitted cylinder's surface (axis-distance minus radius), so a cluster ON the strip
    reads ~0."""
    best, best_d = None, float("inf")
    for d in (survey or {}).get("dressings") or []:
        if d.get("kind") not in ("fillet", "chamfer") or d.get("point") is None:
            continue
        axis = d.get("axis") or (0, 0, 1)
        delta = [float(point[i]) - float(d["point"][i]) for i in range(3)]
        along = sum(delta[i] * float(axis[i]) for i in range(3))
        radial = math.sqrt(max(sum(delta[i] ** 2 for i in range(3)) - along * along, 0.0))
        dist = abs(radial - float(d.get("radius_mm") or d.get("width_mm") or 0.0))
        if dist < best_d:
            best, best_d = d, dist
    return best, best_d


def _object_map(build):
    """plan op id -> generated CAD object name (from the builder's report)."""
    return {f.get("id"): f.get("object") for f in (build or {}).get("features") or []}


def _nearest_hole(point, survey):
    """The recognized hole nearest ``point``: (hole, distance) or (None, inf). The
    distance is to the hole's fitted SURFACE (axis-line distance minus radius, the same
    convention the dressing chain uses) — a deviation at the hole's OPENING sits on the
    surface even though the fitted centre is half the plate away."""
    best, best_d = None, float("inf")
    for h in (survey or {}).get("holes") or []:
        centre = h.get("center") or (0, 0, 0)
        axis = h.get("axis") or (0, 0, 1)
        delta = [float(point[i]) - float(centre[i]) for i in range(3)]
        along = sum(delta[i] * float(axis[i]) for i in range(3))
        radial = math.sqrt(max(sum(delta[i] ** 2 for i in range(3)) - along * along, 0.0))
        radius = float(h.get("radius") or 0.0)
        d = 0.0 if radial <= radius else radial - radius   # inside the bore = on the feature
        if d < best_d:
            best, best_d = h, d
    return best, best_d


def localize_deviations(residual, *, survey=None, plan=None, build=None, warn_mm=0.3):
    """``{"regions": [...], "unattributed": n, "note": str}`` — localized deviation evidence
    from an ``exact_residuals`` result. Regions carry the §4 fields; unavailable statistics
    are None (never a fake zero). Attribution: HIGH requires the complete chain (recognized
    fit + plan op + CAD object, type-consistent); MEDIUM = op+object without the recognized
    link; AMBIGUOUS = several equally plausible operations; UNATTRIBUTED = none."""
    all_clusters = (residual or {}).get("clusters") or []
    clusters = [c for c in all_clusters if c.get("direction")]   # localization needs the L-1 feed
    regions = []
    skipped = len(all_clusters) - len(clusters)
    holes_ops = _hole_ops(plan)
    objects = _object_map(build)
    for cluster in clusters:
        points = cluster.get("member_points_mm") or []
        if len(points) < 2:   # matches _finish_clusters' thin-sample floor (2 agreeing points)
            skipped += 1
            continue
        dists = cluster.get("member_dists_mm") or []
        bbox = _bbox(points)
        centroid = cluster.get("centroid_mm") or [round(sum(p[i] for p in points) / len(points), 3)
                                                 for i in range(3)]
        dev_type = (TYPE_MISSING if cluster.get("kind") == "missing"
                    else TYPE_EXTRA if cluster.get("kind") == "excess" else TYPE_UNATTRIBUTED)
        region = {"direction": cluster["direction"],
                  "deviation_type": dev_type,
                  "centroid_mm": centroid,
                  "bounding_box_mm": bbox,
                  "sample_count": int(cluster.get("points") or len(points)),
                  "mean_deviation_mm": cluster.get("mean_dist_mm"),
                  "max_deviation_mm": cluster.get("max_dist_mm"),
                  "rms_deviation_mm": (round(math.sqrt(sum(d * d for d in dists) / len(dists)), 4)
                                       if dists else None),
                  "p95_deviation_mm": (round(sorted(dists)[min(len(dists) - 1,
                                                              math.ceil(0.95 * len(dists)) - 1)], 4)
                                       if dists else None),
                  "plan_operation_id": None,
                  "cad_object_name": None,
                  "recognized_feature": None,
                  "attribution_status": CONF_UNATTRIBUTED,
                  "attribution_confidence": CONF_UNATTRIBUTED,
                  "evidence": []}
        _attribute(region, cluster, points, bbox, centroid, survey, holes_ops, objects, warn_mm, plan=plan)
        regions.append(region)
    regions.sort(key=lambda r: -(r["max_deviation_mm"] or 0.0))
    for i, region in enumerate(regions, start=1):
        region["region_id"] = f"DEV-{i:03d}"
    out = {"regions": regions,
           "unattributed": sum(1 for r in regions
                               if r["attribution_status"] == CONF_UNATTRIBUTED)}
    if skipped:
        out["note"] = (f"{skipped} cluster(s) without the localization feed "
                       "(older residual data) were skipped")
    return out


def _attribute(region, cluster, points, bbox, centroid, survey, holes_ops, objects, warn_mm, plan=None):
    """Fill the attribution chain for one region. Evidence-based, conservative."""
    extent = max(_extent(bbox), 4.0 * float(warn_mm))
    # candidate plan operations: hole ops whose fitted BORE contains or borders the
    # region (bore-aware surface distance — a cluster hugging the rim reads ~0, an
    # unrelated hole reads its full distance)
    def _bore_distance(point, pos, dia):
        radius = dia / 2.0 if dia else 0.0
        radial = _dist(point, pos)
        return 0.0 if radial <= radius else radial - radius

    candidates = [(op, pos, dia) for op, pos, dia in holes_ops
                  if _bore_distance(centroid, pos, dia) <= extent]
    hole, hole_d = _nearest_hole(centroid, survey)
    near_hole = hole is not None and hole_d <= extent
    if len(candidates) > 1:
        region["attribution_status"] = CONF_AMBIGUOUS
        region["attribution_confidence"] = CONF_AMBIGUOUS
        region["evidence"].append(
            f"{len(candidates)} hole operations sit within the region's extent — "
            "the evidence does not separate them")
        return
    if not candidates:
        _attribute_dressing(region, centroid, survey, plan, objects, warn_mm, extent)
        if region["plan_operation_id"] is not None:
            return
        _attribute_slab(region, centroid, plan, objects)
        if near_hole:
            region["recognized_feature"] = (f"recognized hole r={hole.get('radius')} at "
                                            f"{[round(float(c), 2) for c in hole['center']]}")
            region["evidence"].append(
                "a recognized source hole lies in this region but no hole operation "
                "was constructed there — the deviation is unexplained by the tree")
        return
    op, pos, dia = candidates[0]
    region["plan_operation_id"] = op.get("id")
    region["cad_object_name"] = objects.get(op.get("id"))
    chain = [f"hole operation {op.get('id')} within {extent:.1f} mm of the region centroid"]
    if near_hole and hole.get("radius") is not None and dia:
        fit_d = 2.0 * float(hole["radius"])
        region["recognized_feature"] = (f"recognized hole r={hole.get('radius')} at "
                                        f"{[round(float(c), 2) for c in hole['center']]}")
        if region["cad_object_name"]:
            region["attribution_status"] = "attributed"
            region["attribution_confidence"] = CONF_HIGH
            chain.append("recognized fit, plan operation and generated object all present")
            # specific disagreement between the recognized fit and the constructed operation
            if abs(fit_d - dia) > max(0.1 * max(fit_d, dia), 2.0 * warn_mm):
                region["deviation_type"] = TYPE_DIMENSION
                chain.append(f"fitted diameter {fit_d:.2f} mm vs constructed {dia:.2f} mm")
            elif _dist(hole.get("center") or centroid, pos) > max(0.1 * extent, 2.0 * warn_mm):
                region["deviation_type"] = TYPE_POSITION
                chain.append("the constructed hole sits away from the fitted centre")
        else:
            region["attribution_confidence"] = CONF_MEDIUM
            chain.append("plan operation found but no generated object name in the report")
    else:
        if region["cad_object_name"]:
            region["attribution_status"] = "attributed"
            region["attribution_confidence"] = CONF_MEDIUM
            chain.append("spatial proximity to the operation only — no recognized-fit link")
        else:
            region["attribution_confidence"] = CONF_MEDIUM
            chain.append("plan operation found; object and recognized fit both unresolved")
    region["evidence"] = chain


def _attribute_dressing(region, centroid, survey, plan, objects, warn_mm, extent):
    """The fillet/chamfer chain: cluster on a recognized dressing strip → plan dressing
    op → CAD object. Same conservatism as the hole chain: HIGH needs fit+op+object;
    the DIMENSION refinement needs the fitted radius to disagree with the constructed
    size; a recognized strip without any dressing op stays UNATTRIBUTED, explained."""
    strip, strip_d = _nearest_dressing(centroid, survey)
    if strip is None or strip_d > extent:
        return
    kind = "fillet" if strip.get("kind") == "fillet" else "chamfer"
    region["recognized_feature"] = (f"recognized {kind} strip r={strip.get('radius_mm')} at "
                                    f"{[round(float(c), 2) for c in strip['point']]}")
    dress_ops = _dressing_ops(plan, (plan or {}).get("parameters"))
    matching = [(op, size) for op, size in dress_ops if op.get("op") == kind]
    if not matching:
        region["evidence"].append(
            f"a recognized {kind} strip lies in this region but no {kind} operation "
            "was constructed — the deviation is unexplained by the tree")
        return
    if len(matching) > 1:
        region["attribution_status"] = CONF_AMBIGUOUS
        region["attribution_confidence"] = CONF_AMBIGUOUS
        region["evidence"].append(
            f"{len(matching)} {kind} operations exist — the evidence does not separate them")
        return
    op, size = matching[0]
    region["plan_operation_id"] = op.get("id")
    region["cad_object_name"] = objects.get(op.get("id"))
    fitted = float(strip.get("radius_mm") or 0.0)
    chain = [f"{kind} operation {op.get('id')} matches the recognized strip"]
    if region["cad_object_name"]:
        region["attribution_status"] = "attributed"
        region["attribution_confidence"] = CONF_HIGH
        chain.append("recognized strip, plan operation and generated object all present")
        if fitted > 0 and abs(fitted - size) > max(0.1 * max(fitted, size), 2.0 * warn_mm):
            region["deviation_type"] = TYPE_DIMENSION
            chain.append(f"fitted radius {fitted:.2f} mm vs constructed size {size:.2f} mm")
    else:
        region["attribution_confidence"] = CONF_MEDIUM
        chain.append("plan operation found but no generated object name in the report")
    region["evidence"] = chain


def _attribute_slab(region, centroid, plan, objects):
    """Pad/pocket attribution, honestly weaker than the fit chains: a plan with EXACTLY ONE
    pad or pocket op whose sketch plane exists — an unexplained region anywhere on the plate
    is MEDIUM to that op (the slab owns the whole footprint). Never HIGH (no fitted surface
    ties a region to a slab), and several slab ops stay unattributed (which one is not
    separable without feature-contribution isolation)."""
    slab_ops = [f for f in (plan or {}).get("features") or []
                if f.get("op") in ("pad", "pocket")]
    if len(slab_ops) != 1:
        if len(slab_ops) > 1:
            region["evidence"].append(
                f"{len(slab_ops)} pad/pocket operations exist — which one owns the region "
                "is not separable without feature-contribution isolation")
        return
    op = slab_ops[0]
    if not (op.get("plane") or {}).get("origin"):
        return
    region["plan_operation_id"] = op.get("id")
    region["cad_object_name"] = objects.get(op.get("id"))
    region["attribution_status"] = "attributed" if region["cad_object_name"] else "attributed"
    region["attribution_confidence"] = CONF_MEDIUM
    region["evidence"].append(
        f"single {op.get('op')} operation owns the plate footprint; spatial attribution only — "
        "no fitted surface ties this region to the slab")


def localization_block(localized):
    """The compact report section (tools_recon renders it under the deviation rows)."""
    regions = (localized or {}).get("regions") or []
    if not regions:
        return []
    s2m = [r for r in regions if r["direction"] == "s2m"]
    m2s = [r for r in regions if r["direction"] == "m2s"]
    lines = ["LOCALIZED DEVIATIONS"]
    for r in regions[:8]:
        lines.append(f"  {r['region_id']}  {r['deviation_type']}  max "
                     f"{r['max_deviation_mm'] if r['max_deviation_mm'] is not None else '?'} mm"
                     f"  at {r['centroid_mm']}"
                     + (f"  op {r['plan_operation_id']} -> {r['cad_object_name']}"
                        if r["plan_operation_id"] else "")
                     + f"  [{r['attribution_confidence']}]")
    unattr = localized.get("unattributed") or 0
    if unattr:
        lines.append(f"UNATTRIBUTED REGIONS: {unattr}")
    if len(regions) > 8:
        lines.append(f"  … +{len(regions) - 8} more region(s)")
    lines.append(f"  directions: source->cad {len(s2m)} region(s), cad->source {len(m2s)} region(s)")
    return lines
