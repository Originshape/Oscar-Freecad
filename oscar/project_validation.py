"""Validation records of :mod:`oscar.project` (private, PURE: no FreeCAD, no Qt, no disk).

``tools_verify.run_validation`` hands the Project a ``ValidationResult`` (a dataclass,
or its dict) and ``validate_model`` / ``check_fit`` may hand over a PARTIAL one (only the health part, only
the fit rows). This module turns either into the ``validation.json`` shape (``normalize``) and merges a
partial record into the record of the SAME model state (``merge``), so a later narrow run never wipes the
evidence of an earlier wide one, while a record of another model state is never carried over.

The record keys are exactly ``schema, ts, after_step, doc, ok, health, gate_errors, fit_rows, dfm, coverage,
ran, not_validated``; ``ValidationResult.text`` / ``fit_ran`` / ``fail_rows`` are redundant renderings and are
dropped. ``FIT_ENTRY`` (which ``check_fit`` row belongs to which catalog entry) lives here once: the Build
Summary classification (``project_docs.classify_validation``) and the merge both use it.
"""

from typing import Any

from .project_util import _json_safe

# The ordered catalog of "things the app can say it validated".
CATALOG = ("health", "fit.interference", "fit.clearance", "fit.ports", "fit.retention", "dfm",
           "editability", "rebuild", "step_collision", "physical")
FIT_IDS = ("fit.interference", "fit.clearance", "fit.ports", "fit.retention")

# Which check_fit rows belong to which catalog entry: by row kind / id prefix (the alignment rows of the kit are
# filed under the entry they are about).
FIT_ENTRY = {
    "fit.interference": lambda r: r.get("kind") == "interference",
    "fit.clearance": lambda r: r.get("kind") == "clearance",
    "fit.ports": lambda r: (r.get("kind") in ("containment", "alignment")
                            and not str(r.get("id", "")).startswith("hooks.")),
    "fit.retention": lambda r: r.get("kind") == "retention" or str(r.get("id", "")).startswith("hooks."),
}

HEALTH_ROWS_CAP = 400       # object rows kept in validation.json (a 5000-object document must not make every write megabytes)
HEALTH_FINDINGS_CAP = 200


def _dicts(value: Any) -> list[dict]:
    """The dict entries of ``value`` when it is a list (anything else gives ``[]``)."""
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    return [x for x in value if isinstance(x, str)] if isinstance(value, list) else []


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def group_of(row: dict) -> str | None:
    """The fit catalog entry a ``check_fit`` row belongs to, or None (a ``coverage`` row such as
    ``reference.status`` belongs to none of them)."""
    for entry, belongs in FIT_ENTRY.items():
        if belongs(row):
            return entry
    return None


def has_fail(rows: list[dict]) -> bool:
    return any(str(r.get("status", "")).upper() == "FAIL" for r in rows)


def normalize(validation: dict, *, now: str, default_after_step: int) -> dict:
    """``validation`` (a ``ValidationResult`` as a dict, or any dict of that shape) as a ``validation.json`` record.

    Missing fields get defaults (``schema`` 1, ``ts`` = ``now``, ``after_step`` = ``default_after_step``, empty
    lists, ``ok`` computed as ``not gate_errors and no FAIL fit row``); wrongly typed fields are replaced by their
    defaults instead of raising; the input is not modified. ``health.rows`` is capped at ``HEALTH_ROWS_CAP``
    (``rows_omitted`` says how many were dropped) and ``health.findings`` at ``HEALTH_FINDINGS_CAP``."""
    data = _json_safe(validation) if isinstance(validation, dict) else {}
    health = data.get("health") if isinstance(data.get("health"), dict) else {}
    if len(_dicts(health.get("rows"))) > HEALTH_ROWS_CAP:
        rows = _dicts(health.get("rows"))
        health = {**health, "rows": rows[:HEALTH_ROWS_CAP], "rows_omitted": len(rows) - HEALTH_ROWS_CAP}
    if len(_dicts(health.get("findings"))) > HEALTH_FINDINGS_CAP:
        health = {**health, "findings": _dicts(health.get("findings"))[:HEALTH_FINDINGS_CAP]}
    gate_errors = _dicts(data.get("gate_errors"))
    fit_rows = _dicts(data.get("fit_rows"))
    ok = data.get("ok")
    after_step = data.get("after_step")
    return {
        "schema": 1,
        "ts": now,
        "after_step": default_after_step if after_step is None else _int(after_step, default_after_step),
        "doc": str(data.get("doc") or ""),
        "ok": bool(ok) if ok is not None else (not gate_errors and not has_fail(fit_rows)),
        "health": health,
        "gate_errors": gate_errors,
        "fit_rows": fit_rows,
        "dfm": _dicts(data.get("dfm")),
        "coverage": _dicts(data.get("coverage")),
        "ran": _strings(data.get("ran")),
        "not_validated": _strings(data.get("not_validated")),
    }


def same_state(old: dict | None, new: dict) -> bool:
    """True when ``old`` describes the SAME model state as ``new``: the same document and the same ``after_step``
    (the newest geometry step when the validation ran). Anything else (a later step, a rewind, another
    document) means ``old`` says nothing about the model ``new`` looked at."""
    return (isinstance(old, dict) and str(old.get("doc") or "") == new["doc"]
            and old.get("after_step") == new["after_step"])


def merge(old: dict | None, new: dict) -> dict:
    """``new`` (a normalized record) plus whatever ``old`` validated that ``new`` did not run, when both describe
    the same model state (``same_state``); otherwise ``new`` unchanged.

    Per section: ``health`` and ``gate_errors`` come from ``old`` unless ``new.ran`` has ``health``; the fit
    rows of every ``fit.*`` entry that ``new`` did not run are carried over (a row that belongs to no entry
    follows the fit rows as a whole); ``dfm`` unless ``new.ran`` has ``dfm``; ``coverage`` when ``new`` has
    none; ``editability`` is carried by id. ``ran`` is the union in catalog order, ``ok`` stays False when a
    carried section holds a FAIL row or gate error, ``not_validated`` loses what is validated now."""
    if not same_state(old, new):
        return new
    old = normalize(old, now=str(old.get("ts") or new["ts"]), default_after_step=new["after_step"])
    ran_new, ran_old = set(new["ran"]), set(old["ran"])
    carried: set[str] = set()
    out = dict(new)
    if "health" not in ran_new and "health" in ran_old:
        out["health"], out["gate_errors"] = old["health"], old["gate_errors"]
        carried.add("health")
    stale_fit = [e for e in FIT_IDS if e in ran_old and e not in ran_new]
    if stale_fit:
        fresh_fit = any(e in ran_new for e in FIT_IDS)
        rows = list(new["fit_rows"])
        for row in old["fit_rows"]:
            group = group_of(row)
            if group in stale_fit or (group is None and not fresh_fit):
                rows.append(row)
        out["fit_rows"] = rows
        carried.update(stale_fit)
    if "dfm" not in ran_new and "dfm" in ran_old:
        out["dfm"] = old["dfm"]
        carried.add("dfm")
    if not new["coverage"] and old["coverage"]:
        out["coverage"] = old["coverage"]
    if "editability" not in ran_new and "editability" in ran_old:
        carried.add("editability")
    merged_ran = [c for c in CATALOG if c in ran_new or c in carried]
    out["ran"] = merged_ran + [r for r in new["ran"] if r not in CATALOG]
    carried_bad = (("health" in carried and bool(old["gate_errors"]))
                   or (any(e in carried for e in FIT_IDS) and has_fail(out["fit_rows"])))
    out["ok"] = bool(new["ok"]) and not carried_bad
    out["not_validated"] = [x for x in dict.fromkeys([*new["not_validated"], *old["not_validated"]])
                            if x not in out["ran"]]
    return out


def stale_reason(validation: dict | None, steps: list) -> str:
    """Why ``validation`` does not describe the model as it is now, or ``""`` when it does.

    ``steps`` are the step records INCLUDING superseded ones (``Project.read_steps(include_superseded=True)``). A
    validation is STALE when a geometry step came after the one it ran at (``after_step`` below the newest
    non-superseded successful geometry step), or when the model was rewound below it (``after_step`` above that
    newest step and a superseded geometry step sits at or below ``after_step``): the model is in a state the
    validation never looked at. A missing ``after_step`` counts as 0."""
    if not validation:
        return ""
    after = _int(validation.get("after_step"), 0)
    live = [s.index for s in steps if s.ok and s.geometry and not s.superseded]
    newest = max(live, default=0)
    if after < newest:
        return f"the validation ran after step {after} but the model changed at step {newest}; validate again"
    if after > newest and any(s.superseded and s.ok and s.geometry and s.index <= after for s in steps):
        return f"the validation ran after step {after} but the model was rewound to step {newest}; validate again"
    return ""


def tree_from_health(validation: dict | None) -> list[dict]:
    """Model-tree rows (``{"name", "type", "solids", "volume", "bbox", "tip"}``) from the ``health.rows`` of a
    validation record, for a Build Summary written when the live document cannot be read (headless / closed).

    Same rule as the live tree: only objects with solids (or a mesh) are listed; a TIP is a solid that no other
    object consumes, that is not hidden and that is not part of the reference (``role == "ref"`` / ``Ref_*``);
    tips first, then by descending volume."""
    rows = []
    for r in _dicts(((validation or {}).get("health") or {}).get("rows")):
        solids = _int(r.get("solids"), 0)
        is_mesh = r.get("shape_type") == "Mesh"
        if solids <= 0 and not is_mesh:
            continue
        name = str(r.get("name") or "")
        is_ref = r.get("role") == "ref" or name.startswith("Ref_")
        rows.append({"name": name, "type": r.get("type"), "solids": solids, "volume": r.get("volume"),
                     "bbox": r.get("bbox"),
                     "tip": bool(solids > 0 and not r.get("consumed") and r.get("visible") is not False and not is_ref)})
    rows.sort(key=lambda row: (not row["tip"], -(row["volume"] or 0.0)))
    return rows
