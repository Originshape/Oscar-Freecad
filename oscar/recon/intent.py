"""Nominal dimension snapping with evidence-band protection (G8).

Manufactured parts carry DESIGN INTENT: plate thicknesses land on whole/half/tenth
millimetres, angles on 15°/1°/0.5° grids, hole diameters on the common drill series.
A measured 10.02 at sigma 0.05 is almost certainly a designed 10.0. This module proposes
and (for lengths and angles) applies such nominals to a plan's measured parameters —
under the two-stage honesty rule:

1. the BAND: a candidate only qualifies within ``SIGMA_BAND`` sigma of the measurement,
   and the COARSEST qualifying grid wins (10.02 → 10.0, never 10.1); an exact value is
   left alone; out-of-band measurements keep their measured value;
2. the RE-VALIDATION: the plan is built from the snapped values and the deviation rows
   are computed on that model (downstream in ``reconstruct``), so a snap that fought the
   evidence shows up in the deviation instead of hiding in a round number.

Diameters are NEVER forced: a measured 6.62 mm may be a worn 6.6 or a true 6.62, so the
nearest common drill size is RECORDED as a suggestion only. Positions are untouched (no
design convention). Every decision — applied or suggested — is logged with its Δ and
σ-ratio and lands in the provenance ledger (``nominal_snaps``) and the tool report.

Pure functions on plain dicts; no FreeCAD, no document access, any thread.
"""

SIGMA_BAND = 2.0              # a candidate must sit within 2σ of the measurement
LENGTH_STEPS = (1.0, 0.5, 0.1)          # coarsest first: integers, halves, tenths
ANGLE_STEPS = (15.0, 1.0, 0.5)          # coarsest first
# common metric jobber drills (mm) — a suggestion table, never a forced value
DRILLS_MM = (1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.2,
             4.5, 4.8, 5.0, 5.2, 5.5, 5.8, 6.0, 6.2, 6.6, 7.0, 7.5, 8.0, 8.5, 9.0,
             9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 15.0, 16.0,
             17.0, 18.0, 19.0, 20.0)

# plan-parameter roles by name (the prismatic route's measured parameters):
# plate_* and blind/counterbore depths are LENGTHS (snapped), hole/cbore/csink
# diameters are DIAMETERS (drill suggestion only), countersink angles are ANGLES



def _round_step(value, step):
    return round(round(value / step) * step, 4)


def _ladder_candidate(value, sigma, steps):
    """(nominal, delta, sigma_ratio) of the COARSEST grid step whose nearest value sits
    within the band; None when the value is already on-grid or nothing qualifies."""
    for step in steps:
        nominal = _round_step(value, step)
        delta = abs(value - nominal)
        if delta <= 1e-9:
            return None                                   # already nominal on this grid
        if delta <= SIGMA_BAND * max(sigma, 1e-12):
            return nominal, delta, delta / max(sigma, 1e-12)
    return None


def length_candidate(value, sigma):
    """The coarsest length nominal within 2σ (integer → 0.5 → 0.1 mm)."""
    return _ladder_candidate(value, sigma, LENGTH_STEPS)


def angle_candidate(value, sigma):
    """The coarsest angle nominal within 2σ (15° → 1° → 0.5°)."""
    return _ladder_candidate(value, sigma, ANGLE_STEPS)


def drill_candidate(diameter):
    """(nominal, delta) — the nearest common drill size. A SUGGESTION: hole diameters
    are measured from wall fits and a worn drill or a non-stock size is real; callers
    record this but never change the value."""
    best = min(DRILLS_MM, key=lambda d: abs(diameter - d))
    return best, abs(diameter - best)


def _role(name):
    def numbered(prefix):
        return name == prefix or (name.startswith(prefix) and name[len(prefix):].isdigit())

    if numbered("cs_a"):
        return "angle"
    if numbered("hole_d") or numbered("cb_d") or numbered("cs_d"):
        return "diameter"
    if name in ("plate_len", "plate_wid", "plate_t") or numbered("hole_dp") \
            or numbered("cb_dp"):
        return "length"
    return None


def apply_nominals(params, sigma_mm):
    """Post-measurement design-intent refinement of a plan parameter dict (G8).

    ``params`` is the plan's ``{name: {"value", "source"}}`` mapping. Lengths and angles
    snap to the coarsest in-band nominal — the VALUE changes and the source records
    ``measured→nominal (Δ…, n.nnσ)``; diameters get their nearest drill size logged as a
    SUGGESTION only; everything else is untouched. Returns ``(params, log)`` with the
    log carrying ``applied`` True/False, the measured value, the nominal, Δ and the
    σ-ratio for every decision — the provenance and report feed."""
    sigma = max(float(sigma_mm or 0.0), 1e-12)
    log = []
    out = dict(params)
    for name, spec in params.items():
        value = spec.get("value") if isinstance(spec, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        value = float(value)
        role = _role(name)
        if role == "length":
            got = length_candidate(value, sigma)
            if got:
                nominal, delta, ratio = got
                out[name] = {**spec, "value": nominal,
                             "source": {**_src(spec),
                                        "ref": _src(spec).get("ref", "") +
                                               f" [nominal snap {value:g}→{nominal:g} "
                                               f"(Δ{delta:.3g}, {ratio:.2g}σ)]"}}
                log.append({"param": name, "role": "length", "measured": value,
                            "nominal": nominal, "delta": round(delta, 5),
                            "sigma_ratio": round(ratio, 3), "applied": True})
        elif role == "angle":
            got = angle_candidate(value, sigma)
            if got:
                nominal, delta, ratio = got
                out[name] = {**spec, "value": nominal,
                             "source": {**_src(spec),
                                        "ref": _src(spec).get("ref", "") +
                                               f" [nominal snap {value:g}°→{nominal:g}° "
                                               f"(Δ{delta:.3g}, {ratio:.2g}σ)]"}}
                log.append({"param": name, "role": "angle", "measured": value,
                            "nominal": nominal, "delta": round(delta, 5),
                            "sigma_ratio": round(ratio, 3), "applied": True})
        elif role == "diameter":
            nominal, delta = drill_candidate(value)
            if abs(delta) > 1e-9:
                log.append({"param": name, "role": "diameter", "measured": value,
                            "nominal": nominal, "delta": round(delta, 5),
                            "sigma_ratio": round(delta / sigma, 3), "applied": False,
                            "note": "nearest common drill size — suggestion only, the "
                                    "measured diameter keeps driving the model"})
    return out, log


def _src(spec):
    source = spec.get("source") if isinstance(spec, dict) else None
    return dict(source) if isinstance(source, dict) else {}
