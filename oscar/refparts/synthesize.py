"""Plan fragments synthesized from a measurement (boss emission).

Closes the generation gap the capability audit called N2: the ``boss`` op was
registered (``plan/_schema_v2``) with a builder (``plan/_dressbuild.build_boss``)
but nothing ever emitted it — measured raised cylinders had no path into a
FeaturePlan. ``measured_plan`` / ``boss_plan_features`` turn the bosses a
measurement recognized into schema-v2 plan pieces (parameters with ``source``
strings + ``boss`` features) that build through the EXISTING builder as native,
spreadsheet-editable features; no separate generation architecture.

Coordinates are the measurement's BOARD frame (origin at the PCB lower-left
corner, +Z up): ``measured_plan`` therefore emits the measured plate itself,
corner-anchored (rectangle centred on derived ``plate_l / 2`` parameters), so
the plan is self-consistent and builds standalone. Bosses standing off the SAME
face at the SAME height share ONE feature with several ``profiles`` (one
attached sketch, one pad — the builder resolves the face rule against the tip,
so a second same-face boss FEATURE could only attach to the first boss's cap).

This module is pure (no FreeCAD import): it reads a ``measure.Measurement`` or
the JSON-safe dict ``Measurement.to_dict()`` produces and returns plain dicts.
"""

import json

_MAX_SHOWN = 3                     # boss features the emission line spells out


def _bosses_of(m) -> list:
    """The bosses of a ``Measurement`` or its ``to_dict()`` form (never None)."""
    if isinstance(m, dict):
        return m.get("bosses") or []
    return getattr(m, "bosses", None) or []


def _pcb_of(m) -> "dict | None":
    if isinstance(m, dict):
        return m.get("pcb")
    return getattr(m, "pcb", None)


def _object_of(m) -> str:
    if isinstance(m, dict):
        return str(m.get("object") or "the reference")
    return str(getattr(m, "object", "") or "the reference")


def boss_plan_features(m, *, param_prefix: str = "boss") -> tuple:
    """``(parameters, features)`` plan fragments for the bosses of a measurement.

    Every (face, height) group is one ``boss`` feature whose ``profiles`` are the
    measured circles (centres as measured literals, in the BOARD frame);
    diameters and heights bind parameters carrying
    ``source {kind: step_measurement, ref: ...}`` so the plan stays
    contract-compliant and editable. Parameters are shared when all bosses agree
    on the value (``<prefix>_d`` / ``<prefix>_h``), else numbered per group or
    per distinct diameter in first-appearance order (``<prefix>1_d`` ...).
    A measurement without bosses yields ``({}, [])``.
    """
    bosses = _bosses_of(m)
    where = _object_of(m)
    groups: list = []               # [(face, h, [boss, ...])] in measurement order
    for b in bosses:
        for face, h, members in groups:
            if face == b["face"] and abs(h - b["h"]) <= 1e-3:
                members.append(b)
                break
        else:
            groups.append((b["face"], b["h"], [b]))
    diameters: list = []            # distinct diameters, first-appearance order
    for b in bosses:
        if not any(abs(b["d"] - d) <= 1e-3 for d in diameters):
            diameters.append(b["d"])
    one_d = len(diameters) <= 1
    params: dict = {}
    features: list = []
    for i, (face, h, members) in enumerate(groups, 1):
        h_name = f"{param_prefix}_h" if len(groups) == 1 else f"{param_prefix}{i}_h"

        def sourced(name: str, value: float, b: dict) -> None:
            if name not in params:
                params[name] = {"value": float(value), "source": {
                    "kind": "step_measurement",
                    "ref": f"measured on {where}: raised cylinder d {b['d']:g} h {b['h']:g} mm "
                           f"on the {b['face']} face at ({b['x']:g}, {b['y']:g})"}}

        profiles = []
        for b in members:
            j = next(j for j, d in enumerate(diameters, 1) if abs(b["d"] - d) <= 1e-3)
            d_name = f"{param_prefix}_d" if one_d else f"{param_prefix}{j}_d"
            sourced(d_name, b["d"], b)
            profiles.append({"shape": "circle", "diameter": d_name, "center": [b["x"], b["y"]]})
        sourced(h_name, h, members[0])
        places = ", ".join(f"({b['x']:g}, {b['y']:g})" for b in members[:4]) \
            + (f" and {len(members) - 4} more" if len(members) > 4 else "")
        features.append({"id": f"{param_prefix}{i}", "op": "boss", "face": face,
                         "length": h_name,
                         "note": f"measured boss{'es' if len(members) != 1 else ''} on the {face} face at {places}",
                         "profiles": profiles})
    return params, features


def measured_plan(m, *, plan_id: str = "measured_board", param_prefix: str = "boss") -> "dict | None":
    """A complete buildable schema-2 plan of the measured board with its bosses, in
    the measurement's BOARD frame: a corner-anchored plate (the rectangle centred
    on derived ``plate_l / 2`` / ``plate_w / 2`` parameters, so editing a size
    grows the board away from the corner and the bosses stay put) plus the boss
    features of ``boss_plan_features``. ``None`` when the measurement found no
    PCB or no bosses (there is nothing to place)."""
    bosses = _bosses_of(m)
    pcb = _pcb_of(m)
    if not bosses or not isinstance(pcb, dict):
        return None
    where = _object_of(m)
    params, features = boss_plan_features(m, param_prefix=param_prefix)

    def measured(name: str, value: float, what: str) -> dict:
        return {"value": float(value), "source": {
            "kind": "step_measurement", "ref": f"measured on {where}: PCB {what} (exact B-rep)"}}

    plate = {
        "plate_l": measured("plate_l", pcb["length"], "length"),
        "plate_w": measured("plate_w", pcb["width"], "width"),
        "plate_t": measured("plate_t", pcb["thickness"], "thickness"),
        "plate_cx": {"expr": "plate_l / 2"},
        "plate_cy": {"expr": "plate_w / 2"},
    }
    base = [
        {"id": "base_sk", "op": "sketch",
         "plane": {"origin": [0, 0, 0], "normal": [0, 0, 1], "x_dir": [1, 0, 0]},
         "profile": {"shape": "rectangle", "width": "plate_l", "height": "plate_w",
                     "center": ["plate_cx", "plate_cy"]}},
        {"id": "base_pad", "op": "pad", "sketch": "base_sk", "length": "plate_t"},
    ]
    return {"schema_version": 2, "id": plan_id, "part_class": "plate", "units": "mm",
            "target": "freecad",
            "parameters": {**plate, **params},
            "features": base + features}


def emission_line(m, *, cap: int = _MAX_SHOWN) -> str:
    """The model-facing guidance line for a measurement that found bosses ("" without
    them): names the count and embeds a compact, buildable plan of the measured
    board (at most ``cap`` boss features, the rest counted) so the plan the model
    writes carries the measured numbers and their ``source`` strings. Positions
    are in the board frame — the line says so."""
    bosses = _bosses_of(m)
    if not bosses:
        return ""
    plan = measured_plan(m)
    base, feats = plan["features"][:2], plan["features"][2:]
    fragment = json.dumps({"parameters": plan["parameters"],
                           "features": base + feats[:max(1, cap)]},
                          separators=(",", ":"))
    more = f" (+{len(feats) - cap} more not shown)" if len(feats) > cap else ""
    return (f"Bosses measured ({len(bosses)}): plan them as native editable `boss` ops — a buildable plan of "
            f"the measured board follows (positions in the board frame, origin at the PCB lower-left corner; "
            f"adapt them when your plan uses another origin): {fragment}{more}")
