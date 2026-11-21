#!/usr/bin/env python
"""B-1: the 15-case reconstruction benchmark — one command, deterministic, evidence.

Each case builds a SEEDED synthetic part (ground truth known), tessellates it into a
scan mesh, runs the full ``reconstruct.reconstruct`` pipeline in a throwaway document
and records: facets, wall time, peak memory (tracemalloc), the route, the outcome
class, feature/parameter counts, the sampled + exact deviation, the volume error
against the analytic truth, and whether the case's ORACLE passed (outcome class and
volume tolerance per case). Results land as Markdown + JSON under
``docs/research/benchmarks/``.

Run (FreeCAD 1.1.x console):
    "C:/Program Files/FreeCAD 1.1/bin/python.exe" -I scripts/benchmark_recon.py \
        --root D:/Oscar-Freecad --smoke
``--smoke`` runs a 3-case subset for quick verification; the full set runs in a few
minutes headless. Deterministic: every generator is seeded (``SEED``).
"""

import argparse
import json
import os
import sys
import time
import tracemalloc

SEED = 20261115

# Expected outcomes are minimum accepted classes. A stronger verified result satisfies a
# partially-verified target, while an unverified result never does. Geometry tolerances
# remain independent, unchanged gates below.
_OUTCOME_RANK = {"reconstruction_unverified": 0,
                 "partially_verified_parametric": 1,
                 "verified_parametric": 2}


def _outcome_meets(actual, required):
    """Whether the actual reconstruction class reaches the case's minimum target."""
    return (actual in _OUTCOME_RANK and required in _OUTCOME_RANK
            and _OUTCOME_RANK[actual] >= _OUTCOME_RANK[required])


def to_mesh(shape, dev):
    import FreeCAD as App
    import Mesh
    v, f = shape.tessellate(dev)
    return Mesh.Mesh(([App.Vector(*[p.x, p.y, p.z]) for p in v],
                      [tuple(int(i) for i in t) for t in f]))


def _cases():
    import FreeCAD as App
    import Part

    V = App.Vector

    def box(w, d, h, at=(0.0, 0.0, 0.0)):
        return Part.makeBox(w, d, h, V(*at))

    def bracket(w=60, d=40, h=10, r=3.3, holes=((15, 10), (45, 10), (15, 30), (45, 30))):
        shape = box(w, d, h)
        for x, y in holes:
            shape = shape.cut(Part.makeCylinder(r, h * 3, V(x, y, -h)))
        return shape

    def filleted_top(shape, h, r):
        return shape.makeFillet(r, [e for e in shape.Edges
                                    if abs(e.CenterOfMass.z - h) < 1e-6
                                    and e.Length > 1e-6])

    def chamfered_top(shape, h, c):
        return shape.makeChamfer(c, [e for e in shape.Edges
                                     if abs(e.CenterOfMass.z - h) < 1e-6])

    l_bracket = box(60, 40, 10).fuse(box(60, 10, 30, V(0, 0, 10)))
    rounded_base = box(50, 30, 8)
    rounded = rounded_base.makeFillet(6, [e for e in rounded_base.Edges
                                          if abs(e.CenterOfMass.z - 4) < 1e-6
                                          and e.Length > 1e-6])
    blind = box(40, 30, 10)
    blind = blind.cut(Part.makeCylinder(3.3, 6, V(15, 15, 10)))       # blind from top
    cbore = box(40, 30, 10)
    cbore = cbore.cut(Part.makeCylinder(3.3, 30, V(20, 15, -10)))
    cbore = cbore.cut(Part.makeCylinder(6, 4, V(20, 15, 6)))
    csink = box(40, 30, 10)
    csink = csink.cut(Part.makeCylinder(3.3, 30, V(20, 15, -10)))
    csink = csink.cut(Part.makeCone(3.3, 7, 3, V(20, 15, 7)))
    shaft = Part.makeCylinder(8, 20).fuse(Part.makeCylinder(5, 15, V(0, 0, 20)))
    dome = Part.makeCylinder(10, 5).fuse(
        Part.makeSphere(10, V(0, 0, 5)).common(Part.makeBox(40, 40, 20, V(-20, -20, 5))))

    return [
        ("01-clean-box", (box(30, 20, 8), 0.5), 0.5, "verified_parametric", 0.02),
        ("02-noisy-plate", (box(40, 30, 8), 0.4), 0.5, "verified_parametric", 0.03),
        ("03-bracket-4-holes", (bracket(), 0.3), 0.5, "verified_parametric", 0.02),
        # P1.2 level decomposition: the L builds as stacked pads from its two
        # measured section bands (verified, exact geometry)
        ("04-l-bracket", (l_bracket, 0.3), 0.5, "verified_parametric", 0.02),
        ("05-rounded-plate", (rounded, 0.3), 0.5, "verified_parametric", 0.03),
        # P1.1b: the chamfer binds its CUT DISTANCE (derived from the measured
        # strip width and junction angles), not the width itself — verified
        ("06-chamfered-plate", (chamfered_top(box(40, 30, 10), 10, 2.0), 0.4),
         0.5, "verified_parametric", 0.01),
        # Target: partially verified with a measured fillet. Keep that target so a
        # current UNVERIFIED result remains a visible benchmark failure. The build
        # succeeds, but sampled/exact distances exceed warn_mm; the 3% volume bound
        # independently catches volume regressions.
        ("07-filleted-bracket", (filleted_top(bracket(), 10, 2.5), 0.3),
         0.5, "partially_verified_parametric", 0.03),
        # P1.2 exact deviation: the blind floor measures 0.0 (the old sampled
        # cloud could not see it and honestly stayed unverified)
        ("08-blind-hole", (blind, 0.3), 0.5, "verified_parametric", 0.02),
        ("09-counterbore", (cbore, 0.3), 0.5, "verified_parametric", 0.02),
        ("10-countersink", (csink, 0.3), 0.5, "verified_parametric", 0.03),
        ("11-stepped-shaft", (shaft, 0.4), 0.5, "verified_parametric", 0.02),
        # H2 now fits the spherical meridian: the full 1.1.4 run measured -0.36%
        # volume error, while Hausdorff remains above warn_mm and stays UNVERIFIED.
        ("12-dome-cap", (dome, 0.4), 0.5, "reconstruction_unverified", 0.01),
        # 0.05 mm noise on a coarse planar tessellation degrades recognition — the
        # volume still lands within 15 % and the outcome stays honest
        ("13-noisy-bracket", (bracket(), 0.3), 0.5, "reconstruction_unverified", 0.15),
        ("14-off-nominal-plate", (box(40.04, 30.06, 8.02), 0.4),
         0.5, "verified_parametric", 0.02),
        ("15-fine-mesh-perf", (bracket(), 0.1), 1.0, "verified_parametric", 0.02),
    ]


def run_case(root, name, spec, warn_mm, want_outcome, vol_tol, noise=None):
    import FreeCAD as App
    sys.path.insert(0, root)
    from oscar.recon import reconstruct

    shape, dev = spec
    mesh = noise(shape, 0.05, SEED + 1) if noise else to_mesh(shape, dev)
    doc = App.newDocument("Bench")
    tracemalloc.start()
    t0 = time.monotonic()
    try:
        result = reconstruct.reconstruct(doc, mesh, warn_mm=warn_mm)
        error = None
    except Exception as exc:  # noqa: BLE001 - a crashed case is a recorded result
        result, error = {}, f"{type(exc).__name__}: {exc}"[:160]
    elapsed = time.monotonic() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    truth = shape.Volume
    volume_error = None
    if result.get("build", {}).get("success"):
        body = doc.getObject(result["build"]["body"])
        volume_error = (body.Shape.Volume - truth) / truth
    row = {
        "case": name, "facets": int(mesh.CountFacets),
        "seconds": round(elapsed, 2), "peak_mb": round(peak / 1e6, 1),
        "route": (result.get("meta") or {}).get("route"),
        "outcome": result.get("outcome"),
        "features": (result.get("build") or {}).get("feature_count"),
        "parameters": (result.get("build") or {}).get("parameter_count"),
        "hausdorff_mm": next((r["value"] for r in result.get("deviation") or []
                              if r["id"] == "hausdorff"), None),
        "exact_max_mm": next((r["value"] for r in result.get("deviation") or []
                              if r["id"] == "exact_scan_to_model.max"), None),
        "volume_error": None if volume_error is None else round(volume_error, 4),
        "error": error,
    }
    ok = (error is None and _outcome_meets(result.get("outcome"), want_outcome)
          and volume_error is not None and abs(volume_error) <= vol_tol)
    row["oracle"] = "PASS" if ok else "FAIL"
    App.closeDocument(doc.Name)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cases = _cases()
    if args.smoke:
        cases = [c for c in cases if c[0] in ("01-clean-box", "03-bracket-4-holes",
                                              "11-stepped-shaft")]
    def _noisy_variant(shape, dev, sigma, seed):
        import numpy as np
        import Mesh
        import FreeCAD as App
        m = to_mesh(shape, dev)
        pts, fac = m.Topology
        rng = np.random.default_rng(seed)
        P = np.array([[p.x, p.y, p.z] for p in pts], dtype=float)
        P = P + sigma * rng.standard_normal(P.shape)
        return Mesh.Mesh(([App.Vector(*p) for p in P],
                          [tuple(int(i) for i in f) for f in fac]))

    rows = []
    for name, spec, warn, want, tol in cases:
        noise = None
        if name in ("02-noisy-plate", "13-noisy-bracket"):
            noise = lambda shape, sigma, seed: _noisy_variant(shape, spec[1], sigma, seed)  # noqa: E731
        print(f"running {name} ...", flush=True)
        rows.append(run_case(args.root, name, spec, warn, want, tol, noise=noise))
    out_dir = args.out or os.path.join(args.root, "docs", "research", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    tag = time.strftime("%Y%m%d-%H%M%S")
    header = ("| case | facets | s | peak MB | route | outcome | feats | params | "
              "hausdorff | exact max | vol err | oracle |")
    sep = "|" + "---|" * 12
    lines = [f"# Reconstruction benchmark — {tag} (seed {SEED})",
             "", header, sep]
    for r in rows:
        lines.append(
            "| {case} | {facets} | {seconds} | {peak_mb} | {route} | {outcome} | "
            "{features} | {parameters} | {hausdorff_mm} | {exact_max_mm} | "
            "{volume_error} | {oracle} |".format(**{**r, "route": r["route"] or "-",
                                                    "outcome": r["outcome"] or "-"}))
    passed = sum(1 for r in rows if r["oracle"] == "PASS")
    lines += ["", f"**{passed}/{len(rows)} cases pass their oracle.**",
              "_Editability: every PASS row is a bound PartDesign/Feature chain "
              "(feature/parameter counts above); faceted solids are never counted._"]
    md = os.path.join(out_dir, f"benchmark-{tag}.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, f"benchmark-{tag}.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"seed": SEED, "rows": rows}, fh, indent=1)
    print("\n".join(lines))
    print(f"\nwritten: {md}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
