#!/usr/bin/env python
"""P5.5: the FreeCAD-side CADBench adapter (mesh-to-CAD evaluation harness).

CADBench (anniedoris/CADBench, MIT) evaluates CadQuery programs on clean and
noisy meshes with IoU / SIoU / Chamfer / Valid-Shape-Rate metrics. Oscar is a
FreeCAD workbench, not a program generator, so this adapter runs the
RECONSTRUCTION ENGINE on each input STL headlessly and grades the OUTPUT
SOLID with the comparable metrics, reimplemented locally against the sewn
input solid:

- IoU: exact OCC boolean intersection-over-union (body vs sewn scan solid)
- Chamfer: bidirectional sampled NN distances (the deviation engine)
- VSR-equivalent: the pipeline ran and produced a valid solid Tip
- Parametric: feature/parameter counts and the recon outcome class

Usage (inside FreeCAD's python):
    python.exe -I cadbench_adapter.py --cases <dir-of-stls> --out <dir>

NOTE: the official CADBench dataset lives on HuggingFace (DeCoDELab/CADBench)
and is NOT bundled; point --cases at a downloaded copy (clean_mesh/ or
noisy_mesh/ split folders). Validation on the real dataset is BLOCKED until
one is available locally — the adapter is exercised on synthetic STLs by
tests/test_cadbench_adapter.py.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _grade_one(shape, mesh, recon):
    """Comparable metrics for one case: exact IoU + sampled Chamfer."""
    import numpy as np

    from oscar.cad.imported import solid_from_mesh
    from oscar.recon import deviation

    rows = {"outcome": recon.get("outcome"),
            "features": (recon.get("build") or {}).get("feature_count"),
            "parameters": (recon.get("build") or {}).get("parameter_count"),
            "iou": None, "chamfer_mm": None, "valid_solid": False}
    body_ok = bool((recon.get("build") or {}).get("success"))
    rows["valid_solid"] = body_ok and shape is not None and shape.isValid()
    try:
        scan_solid, _ = solid_from_mesh(mesh)
        if scan_solid is not None and scan_solid.Volume > 0 and rows["valid_solid"]:
            inter = shape.common(scan_solid).Volume
            union = shape.Volume + scan_solid.Volume - inter
            rows["iou"] = round(inter / union, 4) if union > 0 else None
            d1, _ = deviation._distances(deviation._mesh_points(mesh),
                                         deviation._shape_points(shape))
            d2, _ = deviation._distances(deviation._shape_points(shape),
                                         deviation._mesh_points(mesh))
            rows["chamfer_mm"] = round(
                float((np.mean(d1) + np.mean(d2)) / 2.0), 4)
    except Exception as exc:  # noqa: BLE001 - one case's metrics never stop a run
        rows["metric_error"] = f"{type(exc).__name__}: {exc}"[:160]
    return rows


def run(cases_dir, out_dir, warn_mm=0.5):
    import FreeCAD as App
    import Mesh

    sys.path.insert(0, ROOT)
    from oscar.recon import reconstruct

    stls = sorted(f for f in os.listdir(cases_dir) if f.lower().endswith(".stl"))
    results = []
    for name in stls:
        mesh = Mesh.Mesh(os.path.join(cases_dir, name))
        doc = App.newDocument("CB")
        t0 = time.monotonic()
        try:
            recon = reconstruct.reconstruct(doc, mesh, warn_mm=warn_mm)
            shape = None
            if (recon.get("build") or {}).get("success"):
                shape = doc.getObject(recon["build"]["body"]).Shape
            row = dict(_grade_one(shape, mesh, recon))
            row.update(case=name, facets=int(mesh.CountFacets),
                       seconds=round(time.monotonic() - t0, 2))
        except Exception as exc:  # noqa: BLE001 - a crashed case is a recorded result
            row = {"case": name, "facets": int(mesh.CountFacets), "error":
                   f"{type(exc).__name__}: {exc}"[:160]}
        finally:
            App.closeDocument(doc.Name)
        results.append(row)
        print("ran", name, "->", row.get("outcome") or row.get("error"))
    os.makedirs(out_dir, exist_ok=True)
    tag = time.strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(out_dir, f"cadbench-{tag}.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"cases": results}, fh, indent=1)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", required=True, help="directory of input STL files")
    ap.add_argument("--out", required=True, help="output directory for the JSON report")
    args = ap.parse_args(argv)
    rows = run(args.cases, args.out)
    graded = [r for r in rows if r.get("iou") is not None]
    if graded:
        mean_iou = sum(r["iou"] for r in graded) / len(graded)
        print(f"cases: {len(rows)} · graded: {len(graded)} · mean IoU {mean_iou:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
