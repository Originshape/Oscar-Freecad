"""Parts, placements and references of a schema-v2 build (private to oscar.plan).

A v2 plan with ``parts`` builds one PartDesign Body per part, never fused. A
part's placement is three offsets (``x`` / ``y`` / ``z``, mm) and optional
rotations (``rx`` / ``ry`` / ``rz``, degrees about the world axes, applied as
yaw-pitch-roll: ``rz`` about Z last), each a parameter name bound to the
Body's ``Placement`` by expression - change the parameter, recompute, the part
moves. The ``assembly`` block holds the same placements by name for parts
that carry none of their own; ``references`` (mating faces) are measured on
the finished Bodies and reported, never enforced.
"""

from . import schema
from ._buildcore import resolve

PART_PROP = "OscarPlanPart"        # plan part id stamped on the part's Body

# placement key -> (Placement property path, parameter unit suffix)
_BASE = {"x": "Placement.Base.x", "y": "Placement.Base.y", "z": "Placement.Base.z"}
_ROT = {"rx": "Placement.Rotation.Roll", "ry": "Placement.Rotation.Pitch",
        "rz": "Placement.Rotation.Yaw"}

_OPPOSED = {"top": "bottom", "bottom": "top", "front": "back", "back": "front",
            "left": "right", "right": "left"}
_SIDE = {"top": ("z", True), "bottom": ("z", False), "front": ("y", False),
         "back": ("y", True), "left": ("x", False), "right": ("x", True)}


def units(plan: dict) -> list[dict]:
    """``schema.plan_units`` with the assembly placements merged in: a part
    without a ``placement`` of its own takes the assembly placement that names
    it (``placement_name`` says which). Each unit also carries ``index``."""
    out = schema.plan_units(plan)
    by_part = {}
    asm = plan.get("assembly") if isinstance(plan.get("assembly"), dict) else {}
    for pl in asm.get("placements") or []:
        if isinstance(pl, dict) and isinstance(pl.get("part"), str):
            by_part[pl["part"]] = pl
    for i, unit in enumerate(out):
        unit["index"] = i
        unit["placement_name"] = None
        if unit["placement"] is None and unit["part"] in by_part:
            pl = by_part[unit["part"]]
            unit["placement"] = {k: v for k, v in pl.items()
                                 if k in ("x", "y", "z", "rx", "ry", "rz")}
            unit["placement_name"] = pl.get("name")
    return out


def place_body(body, spec: dict | None, plan: dict, sheet_name: str,
               part_id: str) -> None:
    """Set ``body.Placement`` from the placement ``spec`` and bind every
    parameter-named offset / rotation by expression to ``<sheet>.<alias>``."""
    if not spec:
        return
    import FreeCAD as App

    def num(key):
        v = spec.get(key, 0)
        return resolve(plan, v, f"part {part_id} placement.{key}", "")

    body.Placement = App.Placement(
        App.Vector(num("x"), num("y"), num("z")),
        App.Rotation(num("rz"), num("ry"), num("rx")))   # yaw, pitch, roll
    for key, path in _BASE.items():
        if isinstance(spec.get(key), str):
            body.setExpression(path, f"{sheet_name}.{spec[key]}")
    for key, path in _ROT.items():
        if isinstance(spec.get(key), str):
            body.setExpression(path, f"{sheet_name}.{spec[key]} * 1 deg")


def stamp_part(body, part_id: str) -> None:
    """Record which plan part the Body realises."""
    if PART_PROP not in body.PropertiesList:
        body.addProperty("App::PropertyString", PART_PROP, "Oscar",
                         "Plan part this Body realises")
    setattr(body, PART_PROP, part_id or "")


def part_of(obj) -> str | None:
    """The plan part id stamped on ``obj`` (None when it carries none)."""
    if obj is None or PART_PROP not in getattr(obj, "PropertiesList", []):
        return None
    return getattr(obj, PART_PROP) or None


def _extent(body, face: str) -> float | None:
    shape = getattr(body, "Shape", None)
    if shape is None or shape.isNull():
        return None
    axis, positive = _SIDE[face]
    bb = shape.BoundBox
    return getattr(bb, axis.upper() + ("Max" if positive else "Min"))


def reference_report(plan: dict, bodies: dict) -> tuple[list[dict], list[str]]:
    """Measure every ``references`` entry on the built Bodies (``bodies``:
    part id -> Body object): the signed gap between the two faces' planes
    along their axis (positive = apart, negative = overlapping; for two faces
    facing the same way the offset between the planes) against the expected
    ``gap`` parameter (0 when none). Informational: returns ``(rows,
    warnings)`` and never raises."""
    rows: list[dict] = []
    warnings: list[str] = []
    values = schema.parameter_values(plan)
    for ref in plan.get("references") or []:
        try:
            a, b = ref["a"], ref["b"]
            ea, eb = _extent(bodies[a["part"]], a["face"]), _extent(bodies[b["part"]], b["face"])
            if ea is None or eb is None or _SIDE[a["face"]][0] != _SIDE[b["face"]][0]:
                rows.append({"id": ref.get("id"), "ok": None,
                             "note": "faces on different axes or a Body without geometry: not measured"})
                continue
            if _OPPOSED[a["face"]] == b["face"]:
                # a's outward side meets b's inward side: b lies beyond a
                gap = (eb - ea) if _SIDE[a["face"]][1] else (ea - eb)
            else:
                gap = abs(ea - eb)
            want = float(values.get(ref.get("gap"), 0.0)) if ref.get("gap") else 0.0
            ok = abs(gap - want) <= 0.01 + 1e-6 * abs(want)
            rows.append({"id": ref.get("id"), "kind": ref.get("kind"),
                         "gap_mm": round(gap, 4), "expected_mm": round(want, 4),
                         "ok": ok})
            if not ok:
                warnings.append(f"reference {ref.get('id')}: {a['part']}.{a['face']} to "
                                f"{b['part']}.{b['face']} measure {gap:g} mm apart, "
                                f"expected {want:g} (informational: nothing was changed)")
        except Exception as e:  # noqa: BLE001 - a report row must never fail a build
            rows.append({"id": ref.get("id") if isinstance(ref, dict) else None,
                         "ok": None, "note": f"not measured: {type(e).__name__}: {e}"})
    return rows, warnings


__all__ = ["PART_PROP", "units", "place_body", "stamp_part", "part_of",
           "reference_report"]
