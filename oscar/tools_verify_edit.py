"""Deep validation: the +-10% editability rebuild.

It works on ANY document with a ``Params`` /
``Parameters`` sheet: it takes the (up to 8) parameters that drive the most object properties, pushes each one
+-10%, recomputes, asks ``model_health`` (STRICT) whether the edit created an error that the unedited model did not
have, and puts every cell back exactly as it was. A model whose dimensions are "parameters" only on paper (a
parameter edit fragments the shell, or changes nothing) is therefore caught before the user finds out.

GUI-THREAD ONLY (it edits and recomputes a document): call it from a ``dispatch`` task. The time cap (default 20 s)
is checked before every edit, so a slow model costs at most one edit more; the parameters that were not reached are
reported as ``not_checked``.
"""

import math
import time
from typing import Any, Callable

DELTA = 0.10
MAX_PARAMS = 8
BUDGET_S = 20.0
_ANGLE_UNITS = ("deg", "°", "degree", "degrees")


def _cell_value(sheet: Any, name: str) -> float | None:
    """The numeric value of the aliased cell (``getattr(v, 'Value', v)``: a unitless cell is a plain number)."""
    try:
        raw = sheet.get(name)
    except Exception:  # noqa: BLE001
        return None
    value = getattr(raw, "Value", raw)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _signature(snapshot: Any) -> dict[str, tuple]:
    """``{object: (volume, bbox, placement)}`` of the geometry entries of a ``StateSnapshot``: what an edit must
    change for a parameter to count as driving the model (the hash key alone changes on every recompute)."""
    out: dict[str, tuple] = {}
    for name, entry in (getattr(snapshot, "objects", None) or {}).items():
        if entry.get("volume") is None and entry.get("bbox") is None:
            continue
        bbox = entry.get("bbox")
        out[name] = (entry.get("volume"), tuple(bbox) if bbox else None, str(entry.get("placement")))
    return out


def pick_parameters(table: dict, limit: int = MAX_PARAMS) -> tuple[list[dict], list[dict]]:
    """``(chosen rows, skipped)`` from a ``read_param_table`` dict.

    A parameter is perturbed when it is a plain non-zero number that at least one object property is bound to;
    the ``limit`` highest-``bound`` ones win (ties by name). Skipped, with the reason: formulas (they follow their
    sources), count / stage parameters (rebuilt through ``set_parameter``, a percentage of 4 hooks is not a model),
    unbound parameters (nothing to drive), zero and non-numeric values."""
    chosen: list[dict] = []
    skipped: list[dict] = []
    for r in (table or {}).get("rows") or []:
        name = str(r.get("name", ""))
        value = r.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            skipped.append({"parameter": name, "reason": "not a number"})
        elif r.get("expr"):
            skipped.append({"parameter": name, "reason": "formula: follows its sources"})
        elif r.get("needs_rebuild") or name.endswith("_count"):
            skipped.append({"parameter": name, "reason": "count / stage parameter: rebuilt by set_parameter"})
        elif int(r.get("bound") or 0) <= 0:
            skipped.append({"parameter": name, "reason": "no object property is bound to it"})
        elif float(value) == 0.0:
            skipped.append({"parameter": name, "reason": "value is 0: a percentage of 0 changes nothing"})
        else:
            chosen.append(r)
    chosen.sort(key=lambda r: (-int(r.get("bound") or 0), str(r.get("name"))))
    for r in chosen[limit:]:
        skipped.append({"parameter": str(r.get("name")), "reason": f"beyond the {limit} highest-bound parameters"})
    return chosen[:limit], skipped


def _perturbed(value: float, sign: int, delta: float, unit: str) -> float | None:
    """``value * (1 +- delta)``; an angle is never pushed past its geometric limit (360, or just below 180 when it
    starts under 180): ``None`` when the parameter already sits on that limit, so the edit is left out."""
    new = value * (1 + sign * delta)
    if unit.strip().lower() in _ANGLE_UNITS:
        cap = 360.0 if value > 180.0 else 179.99
        if new > cap:
            new = cap
            if new <= value + 1e-9:
                return None
    return new


def _content(value: float, unit: str) -> str:
    """The cell text of a numeric edit: a FORMULA (``=2.2 mm``); a text cell ``-1.5 mm`` would be stored as a string."""
    text = f"{value:.10g}"
    return f"={text} {unit.strip()}".rstrip() if unit.strip() else f"={text}"


def editability_test(doc: Any, table: dict, *, delta: float = DELTA, max_params: int = MAX_PARAMS,
                     budget_s: float = BUDGET_S, clock: Callable[[], float] = time.monotonic,
                     should_stop: Callable[[], bool] | None = None) -> dict:
    """Perturb the driver parameters of ``doc`` by +-``delta`` and prove the model survives. GUI-thread only.

    ``table`` is ``cad.health.read_param_table(doc)``. For every chosen parameter and sign the cell is rewritten,
    the document recomputed and ``model_health(strict=True)`` consulted: the edit is ``FAIL`` when it produces an
    error finding the unedited model did not have (the problems are ``CODE object`` texts). A parameter whose two
    edits changed no object's volume / bounding box / placement is ``inert`` (it is bound but drives nothing).
    Every cell is restored VERBATIM in a ``finally`` (also on Stop and on an exception).

    Returns ``{"ok", "checked", "parameters": [{"parameter", "value", "unit", "bound", "plus", "minus", "inert"}],
    "broken": [{"parameter", "delta", "value", "problems"}], "inert": [...], "skipped": [...], "not_checked":
    [...], "restored": bool, "elapsed_s", "budget_s"}`` where ``plus`` / ``minus`` are ``{"value", "ok",
    "problems"}`` or ``None`` (left out: not reached or past an angle limit). ``{"ok": False, "error": ...}`` when
    there is nothing to edit."""
    from .cad import health_eval

    started = clock()
    sheet = doc.getObject(str((table or {}).get("sheet") or "")) if (table or {}).get("sheet") else None
    if sheet is None:
        return {"ok": False, "error": "the document has no Params / Parameters spreadsheet", "parameters": [],
                "broken": [], "inert": [], "skipped": [], "not_checked": [], "restored": True, "checked": 0,
                "elapsed_s": 0.0, "budget_s": budget_s}
    chosen, skipped = pick_parameters(table, max_params)
    result: dict = {"ok": True, "checked": 0, "parameters": [], "broken": [], "inert": [], "skipped": skipped,
                    "not_checked": [], "restored": True, "elapsed_s": 0.0, "budget_s": budget_s}
    if not chosen:
        result["error"] = "no bound numeric parameter to edit (formulas, counts and unbound rows are skipped)"
        result["ok"] = False
        return result
    originals: dict[str, tuple[str, str]] = {}     # name -> (cell, original content)
    for r in chosen:
        cell = r.get("cell") or sheet.getCellFromAlias(r["name"])
        originals[r["name"]] = (cell, sheet.getContents(cell))
    base_keys = {f.key() for f in health_eval.model_health(doc, strict=True).findings}
    base_sig = _signature(health_eval.snapshot_state(doc, prev_findings=[]))

    def stopped() -> bool:
        try:
            return bool(should_stop and should_stop())
        except Exception:  # noqa: BLE001
            return False

    def run_edit(name: str, cell: str, value: float, unit: str) -> tuple[dict, dict[str, tuple]]:
        sheet.set(cell, _content(value, unit))
        doc.recompute()
        report = health_eval.model_health(doc, strict=True)
        problems = [f"{f.code} {f.object}".strip() for f in report.errors if f.key() not in base_keys]
        sig = _signature(health_eval.snapshot_state(doc, prev_findings=[]))
        return {"value": round(value, 4), "ok": not problems, "problems": problems}, sig

    try:
        for index, r in enumerate(chosen):
            name = str(r["name"])
            cell, original = originals[name]
            unit = str(r.get("unit") or "")
            base_value = _cell_value(sheet, name)
            if base_value is None:
                result["skipped"].append({"parameter": name, "reason": "its cell is not a plain number"})
                continue
            entry = {"parameter": name, "value": round(base_value, 6), "unit": unit, "bound": int(r.get("bound") or 0),
                     "plus": None, "minus": None, "inert": False}
            moved = False
            out_of_time = False
            for sign, key in ((+1, "plus"), (-1, "minus")):
                if clock() - started > budget_s or stopped():
                    out_of_time = True
                    break
                value = _perturbed(base_value, sign, delta, unit)
                if value is None:
                    continue
                try:
                    side, sig = run_edit(name, cell, value, unit)
                except Exception as e:  # noqa: BLE001 - a crashing recompute is the finding
                    side, sig = {"value": round(value, 4), "ok": False,
                                 "problems": [f"{type(e).__name__}: {str(e)[:80]}"]}, base_sig
                entry[key] = side
                result["checked"] += 1
                if sig != base_sig:
                    moved = True
                if not side["ok"]:
                    result["broken"].append({"parameter": name, "delta": sign * delta, "value": side["value"],
                                             "problems": side["problems"]})
            sheet.set(cell, original)
            doc.recompute()
            if out_of_time and entry["plus"] is None and entry["minus"] is None:
                result["not_checked"].extend(str(x["name"]) for x in chosen[index:])
                break
            entry["inert"] = bool(not moved and (entry["plus"] or entry["minus"]))
            if entry["inert"]:
                result["inert"].append(name)
            result["parameters"].append(entry)
            if out_of_time:
                result["not_checked"].extend(str(x["name"]) for x in chosen[index + 1:])
                break
    finally:
        for name, (cell, original) in originals.items():
            try:
                if sheet.getContents(cell) != original:
                    sheet.set(cell, original)
            except Exception:  # noqa: BLE001
                result["restored"] = False
        try:
            doc.recompute()
        except Exception:  # noqa: BLE001
            pass
        for name, (cell, original) in originals.items():
            try:
                if sheet.getContents(cell) != original:
                    result["restored"] = False
            except Exception:  # noqa: BLE001
                result["restored"] = False
    result["ok"] = not result["broken"]
    result["elapsed_s"] = round(clock() - started, 2)
    return result
