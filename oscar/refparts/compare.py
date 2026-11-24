"""Reconstruction deviation: rows that say how far a built model is from its source mesh.

The measurement the agent iterates on (issue: reconstruction accuracy): per-axis bounding-box
size deltas, the centre offset, the volume share, a budgeted WORST-COVERING sample of the
mesh's vertices against the built solid (``distToShape``), and the exact intersection-over-union
through ``common`` when the reference mesh sews into a solid.

Honesty rules (learned from the recon program's bounded-sample defect, where a uniform
250-point sample with a budget cut reported ``0.0 / PASS`` on a part that was 4.9 % wrong):

* every row is ``{"id", "kind", "a", "b", "metric", "value", "limit", "status", "detail"}``
  with ``kind="reference_deviation"`` and ``status`` in ``PASS | WARN | SKIP`` — NEVER ``FAIL``:
  the rows inform the correction loop, they do not gate the turn (the finish gate is
  ``run_validation``'s);
* a row that could not measure carries ``value=None`` and its reason in ``detail`` — a bounded
  sample reports the share it covered (``sampled n/total``), a volume share on an open mesh is
  withheld, and the exact IoU SKIPs with the sewing failure instead of pretending a number.

Asymmetric by design (documented): the sampled surface distance goes mesh -> solid only; a
mesh has no ``distToShape``. Points are taken in an even stride over the vertex list, which
covers a tessellation's geometry broadly; the stride and cap are in the ``detail`` text.

FreeCAD module (``Part`` is imported lazily); the row shape matches ``tools_verify_fit`` so the
existing fit-row renderers and persistence can adopt it unchanged.
"""

import math
import time

BBOX_TOL_MM = 0.2             # per-axis size delta tolerated as PASS
CENTER_TOL_MM = 0.2           # bbox centre offset tolerated as PASS
VOLUME_TOL_PCT = 1.0          # volume share within this is PASS
VOLUME_WARN_PCT = 5.0         # ... beyond this it would be, but WARN is the worst status
SURFACE_TOL_MM = 0.25         # mean sampled distance tolerated as PASS
IOU_TOL = 0.995               # exact IoU at or above this is PASS
DEFAULT_SAMPLE_CAP = 250      # mesh vertices sampled against the solid
MIN_SAMPLE = 60               # below this the sampled row SKIPs (value=None, never 0.0)
DEFAULT_BUDGET_S = 20.0
IOU_FACET_CAP = 8_000         # sewing beyond this holds FreeCAD for the better part of a minute

_ROW_KEYS = ("id", "kind", "a", "b", "metric", "value", "limit", "status", "detail")


def _row(rid: str, a: str, b: str, metric: str, value, limit: str, status: str, detail: str = "") -> dict:
    out = {"id": rid, "kind": "reference_deviation", "a": str(a or ""), "b": str(b or ""),
           "metric": metric, "value": value, "limit": limit, "status": status, "detail": detail}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            out["value"] = round(float(value), 4)
        except (TypeError, ValueError):
            out["value"] = None
    return out


def _skip(rid: str, a: str, b: str, metric: str, why: str) -> dict:
    return _row(rid, a, b, metric, None, "", "SKIP", why)


def _bbox(shape_or_mesh) -> tuple:
    bb = shape_or_mesh.BoundBox
    return (bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax)


def _sizes(box: tuple) -> list:
    return [box[i + 3] - box[i] for i in range(3)]


def _overlap(b1: tuple, b2: tuple, tol: float = 1e-6) -> bool:
    return all(b1[i] <= b2[i + 3] + tol and b2[i] <= b1[i + 3] + tol for i in range(3))


def sew_reference(mesh):
    """``(solid, "")`` of the reference mesh, or ``(None, why)`` when it cannot sew."""
    from ..cad.imported import solid_from_mesh

    return solid_from_mesh(mesh)


def deviation_rows(built, ref_mesh, *, sample_cap: int = DEFAULT_SAMPLE_CAP,
                   min_sample: int = MIN_SAMPLE, budget_s: float = DEFAULT_BUDGET_S,
                   clock=time.monotonic, a: str = "built", b: str = "reference",
                   ref_solid=None) -> list[dict]:
    """The deviation rows of ``built`` (a Part shape) against ``ref_mesh`` (a ``Mesh.Mesh``).

    Never raises on geometry: a measurement that cannot be made is a SKIP row with its
    reason. ``budget_s`` bounds the vertex sampling (checked continuously; a partial sample
    above ``min_sample`` is reported with its share, below it the row SKIPs with
    ``value=None``). ``ref_solid`` (a pre-sewn solid of the mesh) spares the exact-IoU row
    its sewing; the row also SKIPs above ``IOU_FACET_CAP`` facets, where sewing would hold
    the GUI thread for the better part of a minute."""
    rows: list[dict] = []
    deadline = clock() + float(budget_s)

    # --- bounding box: exact on both sides, no sampling involved --------------------
    try:
        bb_built, bb_ref = _bbox(built), _bbox(ref_mesh)
    except Exception as e:  # noqa: BLE001 — a broken shape is a SKIP, not a crash
        return [_skip("deviation.bbox_axis", a, b, "max_axis_size_delta_mm",
                      f"the bounding box could not be read ({type(e).__name__}: {str(e)[:80]})")]
    deltas = [abs(x - y) for x, y in zip(_sizes(bb_built), _sizes(bb_ref))]
    axis_names = ("x", "y", "z")
    worst = max(deltas)
    detail = ", ".join(f"{axis_names[i]} {deltas[i]:.3f}" for i in range(3))
    rows.append(_row("deviation.bbox_axis", a, b, "max_axis_size_delta_mm", worst,
                     f"<= {BBOX_TOL_MM:g} mm", "PASS" if worst <= BBOX_TOL_MM else "WARN",
                     f"per-axis size delta (mm): {detail}"))
    c_built = [(bb_built[i] + bb_built[i + 3]) / 2 for i in range(3)]
    c_ref = [(bb_ref[i] + bb_ref[i + 3]) / 2 for i in range(3)]
    offset = math.dist(c_built, c_ref)
    rows.append(_row("deviation.center_offset", a, b, "bbox_centre_offset_mm", offset,
                     f"<= {CENTER_TOL_MM:g} mm", "PASS" if offset <= CENTER_TOL_MM else "WARN",
                     f"bbox centre moved {tuple(round(c_built[i] - c_ref[i], 3) for i in range(3))} mm"))

    # --- volume: both sides must enclose one (an open mesh does not) ----------------
    closed = False
    try:
        closed = bool(ref_mesh.isSolid())
    except Exception:  # noqa: BLE001
        closed = False
    if not closed:
        rows.append(_skip("deviation.volume_share", a, b, "volume_delta_percent",
                          "the reference mesh is not closed, so it encloses no trustworthy volume"))
    else:
        try:
            ref_vol, built_vol = abs(float(ref_mesh.Volume)), abs(float(built.Volume))
        except Exception as e:  # noqa: BLE001
            rows.append(_skip("deviation.volume_share", a, b, "volume_delta_percent",
                              f"a volume could not be read ({type(e).__name__}: {str(e)[:60]})"))
        else:
            share = 0.0 if ref_vol <= 1e-9 else 100.0 * (built_vol - ref_vol) / ref_vol
            status = "PASS" if abs(share) <= VOLUME_TOL_PCT else "WARN"
            note = "extra material" if share > 0 else "missing material"
            rows.append(_row("deviation.volume_share", a, b, "volume_delta_percent", share,
                             f"within +-{VOLUME_TOL_PCT:g} %", status,
                             f"the model has {abs(share):.2f} % {note} against the mesh "
                             f"({built_vol:.1f} vs {ref_vol:.1f} mm3)"))

    # --- sampled surface distance: mesh vertices -> built solid --------------------
    try:
        points = list(ref_mesh.Points)
    except Exception as e:  # noqa: BLE001
        points = []
    if not points:
        rows.append(_skip("deviation.surface_sampled", a, b, "mean_sampled_distance_mm",
                          "the mesh has no points to sample"))
    else:
        import Part

        stride = max(1, math.ceil(len(points) / max(1, sample_cap)))
        distances: list[float] = []
        stop = False
        for i in range(0, len(points), stride):
            if clock() > deadline:
                stop = True
                break
            p = points[i]
            try:
                d = float(Part.Vertex(p.x, p.y, p.z).distToShape(built)[0])
            except Exception:  # noqa: BLE001 — one bad vertex does not kill the sample
                continue
            distances.append(d)
        if len(distances) < min_sample:
            rows.append(_skip("deviation.surface_sampled", a, b, "mean_sampled_distance_mm",
                              (f"the time budget ({budget_s:g} s) ran out after {len(distances)} of "
                               f"{len(points)} points" if stop else
                              f"only {len(distances)} points could be sampled of {len(points)}")))
        else:
            mean = sum(distances) / len(distances)
            worst_d = max(distances)
            share = 100.0 * len(distances) * stride / len(points)
            rows.append(_row("deviation.surface_sampled", a, b, "mean_sampled_distance_mm", mean,
                             f"<= {SURFACE_TOL_MM:g} mm",
                             "PASS" if mean <= SURFACE_TOL_MM else "WARN",
                             f"sampled {len(distances)}/{len(points)} mesh points ({share:.0f} %, "
                             f"stride {stride}), max {worst_d:.3f} mm"
                             + ("; the time budget stopped the sample" if stop else "")))

    # --- exact IoU through the kernel, only when the reference sews -----------------
    sewn, why = ((ref_solid, "") if ref_solid is not None
                 else (None, f"{int(getattr(ref_mesh, 'CountFacets', 0)):,} facets is beyond "
                             f"the {IOU_FACET_CAP:,} the exact IoU sews (it would hold "
                             f"FreeCAD for the better part of a minute)")
                 if int(getattr(ref_mesh, "CountFacets", 0)) > IOU_FACET_CAP
                 else sew_reference(ref_mesh))
    if sewn is None:
        rows.append(_skip("deviation.iou_exact", a, b, "iou",
                          f"the exact overlap needs the mesh as a solid: {why}"))
    else:
        try:
            if not _overlap(bb_built, _bbox(sewn)):
                iou = 0.0
            else:
                common = float(built.common(sewn).Volume)
                union = abs(float(built.Volume)) + abs(float(sewn.Volume)) - common
                iou = 0.0 if union <= 1e-9 else common / union
            rows.append(_row("deviation.iou_exact", a, b, "iou", iou, f">= {IOU_TOL:g}",
                             "PASS" if iou >= IOU_TOL else "WARN",
                             "exact solid intersection over union (kernel booleans)"))
        except Exception as e:  # noqa: BLE001
            rows.append(_skip("deviation.iou_exact", a, b, "iou",
                              f"the boolean failed ({type(e).__name__}: {str(e)[:80]})"))
    return rows


def render_rows(rows: list[dict], *, built: str, reference: str) -> str:
    """The model-facing block: one line per row, value + status + the detail that explains
    the next correction."""
    lines = [f"DEVIATION of '{built}' against the reference '{reference}':"]
    for r in rows:
        value = "n/a" if r.get("value") is None else f"{r['value']:.4f}"
        lines.append(f"- {r['id']}: {value} {r.get('metric', '')} [{r['status']}]"
                     + (f" — {r['detail']}" if r.get("detail") else ""))
    lines.append("Rows are PASS/WARN/SKIP (they inform the correction loop, they never gate the "
                 "turn): a WARN names the dimension to fix, a SKIP says what could not be measured.")
    return "\n".join(lines)
