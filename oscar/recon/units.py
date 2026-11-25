"""Source-unit detection and mm/inch scale plausibility for imported meshes (K4).

Runtime facts this module is built on (verified on FreeCAD 1.1.4, 2026-11-14):
- Document/unit schemas do NOT affect imported mesh coordinates — a 30-unit STL is 30.0
  whatever the document says. Unit interpretation is entirely the application's job.
- The 3MF importer IGNORES the format's ``unit`` attribute (a unit="inch" 1×1×1 box imports
  as 1.0, not 25.4) — so real 3MF metadata is recovered by reading the SOURCE FILE, not the
  imported object. STL/OBJ/PLY/AST carry no unit metadata at all.
- ``Mesh.Mesh.transformGeometry(scale matrix)`` scales exactly, in place, on a copy — the
  safe conversion primitive (the original mesh is never touched).

Policy (K4-F): units are diagnosed, never assumed silently; no coordinate is ever scaled
without explicit user confirmation, and any approved conversion happens on a working copy,
recorded separately from geometry repair in the provenance ledger, with dimensions and
quality re-measured afterwards. Detection is transparent: the only positive mm/inch
mismatch evidence used is EXACT grid roundness of the three principal dimensions under one
interpretation and not the other (design intent, not object size); object size alone only
ever produces informational warnings.
"""

import math
import os
import zipfile

import numpy as np

INCH_MM = 25.4

# format → (explicit unit metadata?, note). The 3MF importer drops the attribute, so the
# SOURCE FILE is re-read here; STL/OBJ/PLY/AST are unitless by specification.
FORMAT_UNITS = {
    "stl": (False, "STL stores no unit metadata (unitless by specification)"),
    "obj": (False, "OBJ stores no unit metadata"),
    "ply": (False, "PLY stores no unit metadata"),
    "ast": (False, "AST stores no unit metadata"),
    "3mf": (True, "3MF declares a unit attribute; FreeCAD's importer IGNORES it, so the "
                  "source file is read directly"),
    "step": (True, "STEP carries units natively (handled by the Import module, not meshes)"),
}

MM_GRID = 1.0          # designed-in mm dims land on the 1 mm grid
INCH_GRID = 1.0 / 16.0  # designed-in inch dims land on the 1/16" fraction grid
EXACT_TOL = 0.002      # ≤0.2% off a grid = intentionally that dimension (sub-manufacturing
                       # roundness; a coincidence is far rougher — every flag carries the
                       # measured distances so the threshold is checkable, not believed)
UNIT_KNOWN = "UNIT_KNOWN"
UNIT_UNKNOWN = "UNIT_UNKNOWN"
SCALE_PLAUSIBLE = "SCALE_PLAUSIBLE"
POSSIBLE_MM_INCH_MISMATCH = "POSSIBLE_MM_INCH_MISMATCH"
SCALE_AMBIGUOUS = "SCALE_AMBIGUOUS"
NEEDS_USER_CONFIRMATION = "NEEDS_USER_CONFIRMATION"
GATE_STATUSES = (POSSIBLE_MM_INCH_MISMATCH, SCALE_AMBIGUOUS)   # these demote outcomes


def read_source_unit(path) -> dict:
    """Recover source-unit metadata from the ORIGINAL file. Only 3MF genuinely declares a
    unit; OBJ/PLY header mentions, if any, are reported as unverified hints, never as
    units. Never invents a unit."""
    ext = os.path.splitext(str(path))[1].lower().lstrip(".")
    out = {"format": ext or "unknown", "unit": "unknown", "explicit": False, "note": ""}
    if ext == "3mf":
        try:
            with zipfile.ZipFile(path) as z:
                name = next((n for n in z.namelist() if n.lower().endswith(".model")), None)
                if name:
                    head = z.read(name)[:4096].decode("utf-8", errors="replace")
                    low = head.lower()
                    if 'unit="' in low or "unit='" in low:
                        seg = low.split("unit=", 1)[1]
                        value = seg[1:seg.index(seg[0], 1)]
                        out.update(unit=value, explicit=True,
                                   note="declared by the 3MF source file (the FreeCAD "
                                        "importer ignores it — coordinates are unscaled)")
                        return out
        except Exception:  # noqa: BLE001 - an unreadable file reports unknown, never a guess
            pass
        out["note"] = "3MF without a readable unit attribute"
        return out
    if ext in ("obj", "ply"):
        try:
            with open(path, "rb") as fh:
                head = fh.read(2048).decode("utf-8", errors="replace").lower()
            if "unit" in head:
                out["note"] = "header mentions units — UNVERIFIED hint, not metadata"
                out["hint"] = True
        except OSError:
            pass
    if not out["note"] and ext in FORMAT_UNITS:
        out["note"] = FORMAT_UNITS[ext][1]
    return out


def _grid_roundness(dims, grid):
    """Max relative distance of the dimensions to ``grid`` (0 = exactly on grid)."""
    rel = [abs(d - round(d / grid) * grid) / max(d, 1e-12) for d in dims]
    return max(rel) if rel else 1.0


def _edge_stats(mesh):
    pts = np.array([[p.x, p.y, p.z] for p in mesh.Topology[0]], dtype=float)
    fac = np.array(mesh.Topology[1], dtype=int).reshape(-1, 3)
    if len(fac) == 0:
        return {"mean_edge_mm": 0.0, "median_edge_mm": 0.0}
    e = np.stack([pts[fac[:, 1]] - pts[fac[:, 0]],
                  pts[fac[:, 2]] - pts[fac[:, 1]],
                  pts[fac[:, 0]] - pts[fac[:, 2]]], axis=1)
    lengths = np.linalg.norm(e, axis=2).ravel()
    return {"mean_edge_mm": round(float(lengths.mean()), 5),
            "median_edge_mm": round(float(np.median(lengths)), 5)}


def analyse_units(mesh, source_path=None, user_unit=None) -> dict:
    """Transparent mm/inch plausibility report for an imported mesh. Read-only; no
    recognition runs; every status carries its evidence. ``user_unit`` ('mm'/'inch')
    outranks file metadata only when metadata is absent (a user statement about their own
    file is authoritative; both are recorded with their source)."""
    box = mesh.BoundBox
    dims = [box.XLength, box.YLength, box.ZLength]
    src = read_source_unit(source_path) if source_path else \
        {"format": "unknown", "unit": "unknown", "explicit": False,
         "note": "source file not available — no metadata could be read"}
    known = None
    if src["explicit"] and src["unit"] in ("millimeter", "millimetre", "mm"):
        known = "mm"
    elif src["explicit"] and src["unit"] in ("inch", "inches", "in"):
        known = "inch"
    elif user_unit in ("mm", "inch"):
        known = user_unit
    dims_mm = sorted(dims, reverse=True)
    dims_in = [round(d / INCH_MM, 5) for d in dims_mm]
    report = {
        "dims_source": [round(d, 4) for d in dims],
        "dims_mm": [round(d, 4) for d in dims_mm],
        "dims_inch": dims_in,
        "source_unit": known or "unknown",
        "source_unit_origin": ("file metadata" if known and src["explicit"] and
                               known != user_unit else
                               "user confirmation" if known else "none"),
        "source_unit_note": src["note"],
        "format": src["format"],
        "assumed_for_report": known or "mm",
        "evidence": [],
        "warnings": [],
        "confidence": None,
        "recommended_action": "",
    }
    report.update(_edge_stats(mesh))
    if mesh.isSolid():
        report["volume_mm3_if_source_is_mm"] = round(float(mesh.Volume), 3)

    # informational size observations — NEVER a unit verdict on their own
    max_dim = max(dims)
    if 0.0 < max_dim < 1.0:
        report["warnings"].append(
            f"very small under the mm reading (max dimension {max_dim:.3f}) — plausible for "
            "a miniature part; confirm units only if unexpected")
    if max_dim > 10_000.0:
        report["warnings"].append(
            f"very large under the mm reading (max dimension {max_dim:.0f} = "
            f"{max_dim / 1000.0:.1f} m) — plausible for a large part; confirm units only "
            "if unexpected")

    if known:
        report["status"] = UNIT_KNOWN
        report["confidence"] = "high" if src["explicit"] else "user-confirmed"
        report["evidence"].append(
            f"source unit {known!r} from {report['source_unit_origin']}; "
            f"dims in the other idiom would be "
            f"{[round(d / INCH_MM, 3) for d in dims_mm] if known == 'mm' else dims_mm}")
        report["recommended_action"] = "proceed in the confirmed units"
        return report

    # transparent design-intent evidence: exact grid roundness under each interpretation
    r_mm = _grid_roundness(dims_mm, MM_GRID)
    r_in = _grid_roundness(dims_mm, INCH_MM * INCH_GRID)
    report["evidence"].append(
        f"grid roundness of the three principal dimensions: mm-idiom {r_mm:.4%} off, "
        f"inch-idiom {r_in:.4%} off (≤{EXACT_TOL:.1%} = intentional dimension)")
    exact_mm = r_mm <= EXACT_TOL
    exact_in = r_in <= EXACT_TOL
    if exact_in and not exact_mm:
        report["status"] = POSSIBLE_MM_INCH_MISMATCH
        report["confidence"] = "medium"
        report["evidence"].append(
            f"dimensions {[round(d, 3) for d in dims_mm]} sit exactly on the inch fraction "
            f"grid as {[round(d / INCH_MM, 4) for d in dims_mm]} in "
            f"({[round(d / INCH_MM * 16) for d in dims_mm]} sixteenths) but not on the mm "
            "grid — consistent with a part designed in inches whose numbers were exported "
            "as-is (a ×25.4 interpretation)")
        report["recommended_action"] = ("confirm the intended units with the user before "
                                        "reconstruction; do not verify a reconstruction "
                                        "against an unconfirmed unit interpretation")
    elif exact_mm and exact_in:
        report["status"] = SCALE_AMBIGUOUS
        report["confidence"] = "low"
        report["evidence"].append("dimensions sit on BOTH idioms' grids — genuinely "
                                  "ambiguous from dimensions alone")
        report["recommended_action"] = "ask the user which unit the part was designed in"
    elif exact_mm:
        report["status"] = SCALE_PLAUSIBLE
        report["source_unit_status"] = UNIT_UNKNOWN
        report["confidence"] = "low"
        report["evidence"].append("dimensions sit on the mm grid and not the inch grid — "
                                  "consistent with mm; unit still formally unknown (no "
                                  "metadata), so mm is only ASSUMED for reporting")
        report["recommended_action"] = ("no action required; confirm units if the part was "
                                        "designed in inches")
    else:
        report["status"] = SCALE_PLAUSIBLE
        report["source_unit_status"] = UNIT_UNKNOWN
        report["confidence"] = "none"
        report["evidence"].append("dimensions sit on neither idiom's grid — no unit "
                                  "evidence either way (unusual sizes are NOT evidence)")
        report["recommended_action"] = ("no action required; confirm units only if the "
                                        "dimensions look wrong in hindsight")
    return report


def scaled_copy(mesh, factor: float):
    """A working copy of ``mesh`` scaled by ``factor`` (25.4 = inch numbers → mm). The
    original mesh is never touched (K4-F). ``transformGeometry`` scales exactly, in place,
    on the copy (verified)."""
    out = mesh.copy()
    t = _matrix_scale(factor)
    out.transformGeometry(t)
    return out


def _matrix_scale(factor):
    import FreeCAD as App

    t = App.Matrix()
    t.scale(factor, factor, factor)
    return t


def record_conversion(ledger: dict, factor: float, mesh_before, mesh_after) -> dict:
    """Record an APPROVED unit conversion in the ledger — separately from geometry repair.
    Facet CATEGORIES are unchanged (original stays original: conversion rescales the whole
    mesh uniformly); the key space moves, so the ledger's mesh digest is re-stamped for the
    converted state and per-key verification degrades honestly to the digest tier (documented)."""
    from . import provenance

    entry = {"operation": "unit_conversion", "factor": factor,
             "counts": {"facets_before": int(mesh_before.CountFacets),
                        "facets_after": int(mesh_after.CountFacets)},
             "facets_before": int(mesh_before.CountFacets),
             "facets_after": int(mesh_after.CountFacets),
             "bbox_before": [round(mesh_before.BoundBox.XLength, 4),
                             round(mesh_before.BoundBox.YLength, 4),
                             round(mesh_before.BoundBox.ZLength, 4)],
             "bbox_after": [round(mesh_after.BoundBox.XLength, 4),
                            round(mesh_after.BoundBox.YLength, 4),
                            round(mesh_after.BoundBox.ZLength, 4)],
             "keys": {}}
    ledger.setdefault("unit_operations", []).append(entry)
    ledger["mesh_digest"] = provenance.mesh_digest(mesh_after)
    return entry


def conversion_validation(mesh_before, mesh_after, factor) -> dict:
    """Check the converted geometry really is the intended interpretation: bbox ratio ==
    factor within 1e-6, facet count unchanged."""
    ratios = []
    for a, b in ((mesh_after.BoundBox.XLength, mesh_before.BoundBox.XLength),
                 (mesh_after.BoundBox.YLength, mesh_before.BoundBox.YLength),
                 (mesh_after.BoundBox.ZLength, mesh_before.BoundBox.ZLength)):
        if a > 0 and b > 0:
            ratios.append(a / b)
    ok = all(abs(r - factor) < 1e-6 * factor for r in ratios) and \
        int(mesh_after.CountFacets) == int(mesh_before.CountFacets)
    return {"valid": ok, "bbox_ratios": [round(r, 6) for r in ratios],
            "facet_count_preserved": int(mesh_after.CountFacets) == int(mesh_before.CountFacets)}
