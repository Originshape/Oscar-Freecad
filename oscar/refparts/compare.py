"""Comparing a reconstruction with its imported reference (Issue 003).

``compare_to_reference`` answers "how close is the built model to the reference?" with numbers instead of an eyeball:
per-axis size deviation of the bounding boxes, the offset between the box centres, the volume deviation and the
symmetric-difference volume of a boolean overlay. Every row is PASS / WARN / SKIP — NEVER FAIL: the check is
informational (a reconstruction that drifts must be reported, not blocked; the turn gate semantics stay untouched).

FreeCAD module (``Part`` booleans; GUI-thread only like ``measure``). The rows use the ``check_fit`` shape
``{"id", "kind", "a", "b", "metric", "value", "limit", "status", "detail"}`` with ``kind = "reference_deviation"``, so
they ride the ``fit_rows`` channel into ``validation.json`` and the Build Summary through the existing machinery
(catalog entry ``reference.deviation``).
"""

import math
import time

import Part

from . import measure

OVERLAP_WARN_PCT = 10.0     # symmetric difference above this share of the reference volume is a WARN
KIND = "reference_deviation"


def _row(rid: str, metric: str, value, limit, status: str, detail: str, a: str, b: str) -> dict:
    """One comparison row in the ``check_fit`` shape; ``value`` rounded to 3 decimals, never NaN."""
    try:
        value = round(float(value), 3)
        if not math.isfinite(value):
            value = 0.0
    except (TypeError, ValueError):
        value = 0.0
    return {"id": rid, "kind": KIND, "a": a, "b": b, "metric": metric, "value": value,
            "limit": limit, "status": status, "detail": detail[:220]}


def _symmetric_difference(built, ref) -> float:
    """The volume neither shape shares with the other: ``cut(built, ref) + cut(ref, built)`` in mm3 (raises on OCC failure)."""
    return float(built.cut(ref).Volume) + float(ref.cut(built).Volume)


def _center(box: tuple) -> tuple:
    return ((box[0] + box[3]) / 2.0, (box[1] + box[4]) / 2.0, (box[2] + box[5]) / 2.0)


def compare_shapes(built, ref, *, built_name: str = "", ref_name: str = "", warn_mm: float = 0.5,
                   warn_pct: float = 1.0, budget_s: float = 20.0, clock=None) -> list:
    """The deviation rows of ``built`` vs ``ref`` (two shapes in the same frame; GUI-thread only).

    ``warn_mm`` bounds the size deltas and the centre offset, ``warn_pct`` the volume deviation (percent of the
    reference volume); the symmetric-difference share is judged against the fixed ``OVERLAP_WARN_PCT``. A boolean that
    OCC cannot do (or that the time budget does not allow) degrades to one SKIP row — never an exception.
    """
    now = clock or time.monotonic
    deadline = now() + max(1.0, float(budget_s))
    bb, rb = measure.bbox_of(built), measure.bbox_of(ref)
    rows = []
    for i, ax in enumerate("xyz"):
        built_size, ref_size = bb[i + 3] - bb[i], rb[i + 3] - rb[i]
        delta = abs(built_size - ref_size)
        rows.append(_row(f"size.deviation_{ax}", "size_delta_mm", delta, round(float(warn_mm), 3),
                         "PASS" if delta <= warn_mm else "WARN",
                         f"built {measure._mm(built_size)} vs reference {measure._mm(ref_size)} mm along {ax}",
                         built_name, ref_name))
    dx, dy, dz = (bc - rc for bc, rc in zip(_center(bb), _center(rb)))
    offset = math.sqrt(dx * dx + dy * dy + dz * dz)
    rows.append(_row("offset.deviation", "center_offset_mm", offset, round(float(warn_mm), 3),
                     "PASS" if offset <= warn_mm else "WARN",
                     f"bounding-box centres moved by {measure._mm(dx)}, {measure._mm(dy)}, {measure._mm(dz)} mm",
                     built_name, ref_name))
    built_vol, ref_vol = abs(float(built.Volume)), abs(float(ref.Volume))
    if ref_vol > 0.0:
        pct = abs(built_vol - ref_vol) / ref_vol * 100.0
        rows.append(_row("volume.deviation", "volume_delta_pct", pct, round(float(warn_pct), 3),
                         "PASS" if pct <= warn_pct else "WARN",
                         f"built {measure._mm(built_vol)} vs reference {measure._mm(ref_vol)} mm3",
                         built_name, ref_name))
    else:
        rows.append(_row("volume.deviation", "volume_delta_pct", 0.0, round(float(warn_pct), 3), "SKIP",
                         "the reference has no volume to compare against", built_name, ref_name))
    if now() >= deadline:
        rows.append(_row("overlap.difference", "symmetric_difference_pct", 0.0, OVERLAP_WARN_PCT, "SKIP",
                         "the time budget ran out before the boolean overlay", built_name, ref_name))
        return rows
    try:
        difference = _symmetric_difference(built, ref)
        share = difference / ref_vol * 100.0 if ref_vol > 0.0 else 0.0
        rows.append(_row("overlap.difference", "symmetric_difference_pct", share, OVERLAP_WARN_PCT,
                         "PASS" if share <= OVERLAP_WARN_PCT else "WARN",
                         f"symmetric difference {measure._mm(difference)} mm3 ({measure._mm(share)} % of the reference)",
                         built_name, ref_name))
    except Exception as exc:  # noqa: BLE001 - a fragile boolean must not fail the check
        rows.append(_row("overlap.difference", "symmetric_difference_pct", 0.0, OVERLAP_WARN_PCT, "SKIP",
                         f"the boolean overlay failed ({type(exc).__name__}: {exc})", built_name, ref_name))
    return rows
