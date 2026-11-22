"""Text generators of :mod:`oscar.project`: NOTES.md, the Build Summary and rebuild.py (private, PURE).

Nothing here touches FreeCAD, the disk or the clock: every function turns plain data (the manifest,
ledger / params / provenance / notes / validation / spec dicts, step records) into text, so the output is
deterministic for the same inputs and cheap to regenerate after every step (under 50 ms for 200 steps).

The validated / NOT validated rule lives here once (``classify_validation``): the Build Summary and the
NOTES "Validation" section both use it, so what the app claims never depends on model text.
"""

import pprint
from typing import Any, Iterable

from .project_validation import CATALOG as VALIDATION_CATALOG
from .project_validation import FIT_ENTRY as _FIT_ENTRY

# ---------------------------------------------------------------------------
# validated / NOT validated catalog (the ordered ids and the fit-row grouping live in
# project_validation so the Project's validation merge uses the same definition)

VALIDATION_LABELS = {
    "health": "Model health (valid solids, no recompute errors)",
    "fit.interference": "Part-to-part interference",
    "fit.clearance": "Clearances",
    "fit.ports": "Port openings contain every connector",
    "fit.retention": "Snap-fit retention (lid lift test)",
    "dfm": "Printability rules (minimum wall, vent web, hook strain)",
    "editability": "Parameter editability (+-10% rebuild)",
    "rebuild": "rebuild.py reproduces the model headless",
    "step_collision": "STEP-based collision check",
    "physical": "Physical fit, printing, thermal behaviour and <MATERIAL> shrinkage on a real printer",
}

STEP_REFERENCE_STATUSES = ("step-imported", "step-measured", "user-attached")


def _count_line(rows: Iterable[dict], order: tuple[str, ...] = ("PASS", "WARN", "INFO")) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("status", "")).upper()] = counts.get(str(row.get("status", "")).upper(), 0) + 1
    parts = [f"{counts[s]} {s}" for s in order if counts.get(s)]
    return ", ".join(parts)


def _ids(rows: Iterable[dict], status: str) -> str:
    return ", ".join(str(r.get("id", "?")) for r in rows if str(r.get("status", "")).upper() == status)


STALE_REASON = "stale: the model changed after the last validation"


def classify_validation(validation: dict | None, *, reference_status: str = "none", material: str = "",
                        rebuild_ok: bool = False, stale: str = "") -> tuple[list[dict], list[dict]]:
    """Split the catalog into what was validated and what was not.

    A catalog entry is VALIDATED iff ``validation["ran"]`` contains it and its rows have no FAIL (and, for
    the fit entries, not every row was skipped). ``physical`` is ALWAYS not validated. ``step_collision``
    is validated only when the reference status is a STEP one AND the reference interference rows were
    really computed (neither SKIP nor missing). ``rebuild`` is validated when ``rebuild_ok`` (the headless
    replay check passed after the last geometry step). Every other entry that did not run is NOT validated
    with the reason ``not run``.

    ``stale`` (a non-empty reason from ``project_validation.stale_reason``) means the validation does not describe
    the model as it is now: NOTHING it ran counts, and the entries that would have said ``not run`` say
    ``STALE_REASON`` instead.

    Returns ``(validated, not_validated)``: ``[{"id", "label", "result"}]`` and ``[{"id", "label", "reason"}]``
    in catalog order. The ``physical`` label carries the material (``ABS shrinkage``) when it is known.
    """
    validation = {} if stale else (validation or {})
    ran = set(validation.get("ran") or [])
    fit_rows = [r for r in (validation.get("fit_rows") or []) if isinstance(r, dict)]
    dfm_rows = [r for r in (validation.get("dfm") or []) if isinstance(r, dict)]
    gate_errors = validation.get("gate_errors") or []
    health = validation.get("health") or {}
    validated: list[dict] = []
    not_validated: list[dict] = []

    def good(entry: str, result: str) -> None:
        validated.append({"id": entry, "label": VALIDATION_LABELS[entry], "result": result})

    def bad(entry: str, reason: str) -> None:
        label = VALIDATION_LABELS[entry]
        if entry == "physical":
            label = label.replace("<MATERIAL>", material or "material")
        not_validated.append({"id": entry, "label": label, "reason": reason})

    for entry in VALIDATION_CATALOG:
        if entry == "physical":
            bad(entry, "the app cannot check a physical print; test it on your printer")
        elif entry == "health":
            if "health" not in ran:
                bad(entry, "not run")
            elif gate_errors:
                bad(entry, f"{len(gate_errors)} error(s): " + "; ".join(
                    f"{e.get('code', '')} {e.get('object', '')}".strip() for e in gate_errors[:3]))
            else:
                counts = health.get("counts") or {}
                partial = ("; some objects were not checked (time budget)" if health.get("budget_exceeded") else "")
                if health.get("valid", True):
                    good(entry, f"valid ({counts.get('objects', 0)} objects, {counts.get('solids', 0)} solids{partial})")
                else:
                    good(entry, f"no new errors ({counts.get('errors', 0)} pre-existing{partial})")
        elif entry in _FIT_ENTRY:
            rows = [r for r in fit_rows if _FIT_ENTRY[entry](r)]
            if entry not in ran:
                bad(entry, "not run")
            elif any(str(r.get("status", "")).upper() == "FAIL" for r in rows):
                bad(entry, "FAIL: " + _ids(rows, "FAIL"))
            elif rows and all(str(r.get("status", "")).upper() == "SKIP" for r in rows):
                skipped = next((r.get("detail") for r in rows if r.get("detail")), "")
                bad(entry, "skipped" + (f": {skipped}" if skipped else ""))
            else:
                good(entry, _count_line(rows) or "no applicable checks")
        elif entry == "dfm":
            if "dfm" not in ran:
                bad(entry, "not run")
            elif any(str(r.get("status", "")).upper() == "FAIL" for r in dfm_rows):
                bad(entry, "FAIL: " + _ids(dfm_rows, "FAIL"))
            else:
                good(entry, _count_line(dfm_rows) or "no rules triggered")
        elif entry == "editability":
            if "editability" in ran:
                good(entry, "driver parameters rebuilt at +-10%")
            else:
                bad(entry, "not run")
        elif entry == "rebuild":
            if rebuild_ok:
                good(entry, "the headless replay matched the live model")
            else:
                bad(entry, "not run")
        elif entry == "step_collision":
            ref_rows = [r for r in fit_rows if str(r.get("id", "")).startswith("interference.reference")]
            computed = [r for r in ref_rows if str(r.get("status", "")).upper() not in ("SKIP", "")]
            if reference_status not in STEP_REFERENCE_STATUSES:
                if reference_status == "none":
                    bad(entry, "no reference part was sourced")
                else:
                    bad(entry, "the reference is nominal board data; no STEP file was imported")
            elif "fit.interference" not in ran or not computed:
                bad(entry, "interference against the imported reference was not run")
            elif any(str(r.get("status", "")).upper() == "FAIL" for r in computed):
                bad(entry, "FAIL: " + _ids(computed, "FAIL"))
            else:
                good(entry, "interference against the imported reference solids: "
                     + (_count_line(computed) or "checked"))
    if stale:
        for item in not_validated:
            if item["reason"] in ("not run", "interference against the imported reference was not run"):
                item["reason"] = STALE_REASON
    return validated, not_validated


# ---------------------------------------------------------------------------
# small formatting helpers

def clip(text: Any, limit: int = 120) -> str:
    """One-line, table-safe text clipped to ``limit`` characters (newlines and ``|`` neutralised)."""
    s = " ".join(str(text if text is not None else "").split()).replace("|", "\\|")
    return s if len(s) <= limit else s[: max(1, limit - 1)].rstrip() + "…"


def fmt_number(value: Any, digits: int = 1) -> str:
    """``18432.1`` style number (``digits`` decimals for floats), ``-`` for None, integers as they are, other
    values as clipped text."""
    if value is None or value == "":
        return "-"
    if isinstance(value, (bool, int)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return clip(value, 40)


def fmt_bbox(bbox: Any) -> str:
    """``94.0 x 65.0 x 22.0`` from ``[xmin, ymin, zmin, xmax, ymax, zmax]``; ``-`` when unavailable."""
    try:
        x0, y0, z0, x1, y1, z1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return "-"
    return f"{x1 - x0:.1f} x {y1 - y0:.1f} x {z1 - z0:.1f}"


def _table(header: list[str], rows: list[list[Any]], cap: int | None = None, limit: int = 100) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    shown = rows if cap is None else rows[:cap]
    for row in shown:
        lines.append("| " + " | ".join(clip(c, limit) for c in row) + " |")
    if cap is not None and len(rows) > cap:
        lines.append(f"+{len(rows) - cap} more")
    return lines


def _param_value(row: dict) -> str:
    value = row.get("value")
    if isinstance(value, float):
        return f"{value:g}"
    return "" if value is None else str(value)


def _evidence_steps(requirement: dict, steps: list[dict]) -> str:
    """Step-record indexes that evidence the requirement: the evidence of every step that covers it."""
    req_id = requirement.get("id")
    covering = [s for s in steps if req_id in (s.get("covers") or [])
                or s.get("id") in (requirement.get("covered_by") or [])]
    indexes = sorted({int(e.get("step_index")) for s in covering for e in (s.get("evidence") or [])
                      if isinstance(e.get("step_index"), int) and e.get("step_index") > 0
                      and e.get("kind") in ("auto", "confirmed")})
    return ", ".join(f"#{i}" for i in indexes) if indexes else "-"


def _coverage_of(coverage: list[dict] | None, req_id: str) -> dict | None:
    for row in coverage or []:
        if isinstance(row, dict) and row.get("requirement") == req_id:
            return row
    return None


def _material_line(material: str, shrink: float | None, shrink_comp: str) -> str:
    if not material:
        return ""
    pct = f" {shrink * 100:g}%" if isinstance(shrink, (int, float)) else ""
    if shrink_comp == "none":
        return f"{material} shrinkage{pct} is NOT compensated; not validated on a printer."
    return f"{material} shrinkage{pct} is compensated by clearance only; not validated on a printer."


_MATERIAL_SHRINK = {"ABS": 0.008, "ASA": 0.007, "PLA": 0.003, "PETG": 0.004, "PC": 0.007, "NYLON": 0.015}


def material_shrink(material: str) -> float | None:
    """Shrink fraction of a known material (``oscar.kit.materials`` has the same table; this pure copy keeps the
    notes working while the kit is absent)."""
    return _MATERIAL_SHRINK.get(str(material or "").upper())


# ---------------------------------------------------------------------------
# Build Summary

_SUMMARY_CAPS = (
    # (tree rows, requirement rows, parameter rows, validated, not validated, open items, provenance extras)
    (12, 10, 14, 10, 8, 8, 3),
    (8, 8, 10, 8, 6, 6, 2),
    (5, 5, 6, 6, 5, 4, 1),
    (3, 3, 3, 4, 4, 3, 0),
)
_SUMMARY_MAX_LINES = 70
_SUMMARY_MAX_CHARS = 7000


def _bullets(items: list[str], cap: int, empty: str) -> list[str]:
    if not items:
        return [f"- {empty}"]
    lines = [f"- {clip(i, 200)}" for i in items[:cap]]
    if len(items) > cap:
        lines.append(f"+{len(items) - cap} more")
    return lines


def render_summary(*, outcome: str, reference_status: str, project_dir: str, tree: list[dict],
                   requirements: list[dict], steps: list[dict], coverage: list[dict] | None,
                   params: list[dict], provenance: dict, validated: list[dict], not_validated: list[dict],
                   open_items: list[str], files: list[str]) -> str:
    """The ``## Build Summary`` block, at most 70 lines and 7000 characters (``+N more`` for truncated tables).

    The Requirements section is omitted when the ledger has no requirements; every other heading is always
    present, so a test and the panel can rely on them. ``requirements`` / ``steps`` are the ledger's dicts,
    ``coverage`` is ``validation["coverage"]`` (the kit's spec_coverage rows), ``tree`` the Model tree rows
    ``{"name","type","solids","volume","bbox"}``.
    """
    reference = (provenance or {}).get("reference") or {}
    entries = (provenance or {}).get("entries") or []
    entry = next((e for e in entries if e.get("id") == reference.get("entry")), None)
    for caps in _SUMMARY_CAPS:
        tree_cap, req_cap, par_cap, val_cap, nval_cap, open_cap, prov_cap = caps
        lines = ["## Build Summary",
                 f"**Outcome:** {outcome} · reference status: {reference_status} · project: {project_dir}",
                 "### Model tree"]
        lines += _table(["Name", "Type", "Solids", "Volume mm3", "BBox mm"],
                        [[r.get("name"), r.get("type"), fmt_number(r.get("solids")), fmt_number(r.get("volume")),
                          fmt_bbox(r.get("bbox"))] for r in tree], tree_cap, 60)
        if requirements:
            lines.append("### Requirements")
            rows = []
            for req in requirements:
                cov = _coverage_of(coverage, req.get("id", ""))
                ids = ", ".join(req.get("covered_by") or []) or "-"
                if cov and cov.get("detail"):
                    ids += f" ({cov['detail']})"
                status = str(req.get("status", "pending"))
                if cov and str(cov.get("status", "")) == "missing":
                    status += " - geometry missing"
                rows.append([req.get("id"), req.get("text"), status, ids])
            lines += _table(["id", "requirement", "status", "steps"], rows, req_cap, 70)
        lines.append("### Parameters")
        if params:
            lines += _table(["name", "value", "unit", "source"],
                            [[p.get("name"), _param_value(p), p.get("unit", ""), p.get("source", "")] for p in params],
                            par_cap, 40)
        else:
            lines.append("- no parameters recorded")
        lines.append("### Provenance")
        source = (entry or {}).get("source") or "n/a"
        sha = (entry or {}).get("sha256") or ""
        shown = reference.get("name") or reference.get("id") or ("none" if reference_status == "none" else "(unnamed)")
        lines.append(f"- Reference: {shown} - status "
                     f"**{reference_status}** - source: {clip(source, 90)} - sha256: "
                     f"{sha if len(sha) == 64 else 'n/a (no file)'}")
        extras = [e for e in entries if e is not entry][:prov_cap]
        for e in extras:
            lines.append(f"- {e.get('kind', 'source')} {e.get('id', '')}: {clip(e.get('claim') or e.get('source'), 90)}"
                         f" - {e.get('status', '')}")
        lines.append("### Validated")
        lines += _bullets([f"{v['label']}: {v['result']}" for v in validated], val_cap, "nothing validated")
        lines.append("### NOT validated")
        lines += _bullets([f"{n['label']} ({n['reason']})" for n in not_validated], nval_cap, "nothing")
        lines.append("### Open items")
        lines += _bullets(open_items, open_cap, "none")
        lines.append("**Files:** " + (", ".join(files) if files else "none"))
        text = "\n".join(lines)
        if len(lines) <= _SUMMARY_MAX_LINES and len(text) <= _SUMMARY_MAX_CHARS:
            return text
    return text[: _SUMMARY_MAX_CHARS - 3] + "..."   # unreachable in practice: the last cap level is small


# ---------------------------------------------------------------------------
# NOTES.md

def render_notes(*, title: str, ts: str, request: str, ledger: dict | None, steps: list[Any], params: list[dict],
                 provenance: dict, notes: list[dict], validation: dict | None, spec: dict | None,
                 reference_status: str, rebuild_ok: bool, files: list[str],
                 unresolved_extra: list[str], stale: str = "") -> str:
    """The generated NOTES.md: ten sections in a fixed order, deterministic for the same inputs.

    ``steps`` are StepRecord-like objects (``index, tool, title, ok, snapshot, snapshot_note,
    checkpoint_note, superseded, replay``), ``ledger`` the C5 dict (or None), ``validation`` the latest
    validation.json dict, ``spec`` the EnclosureSpec dict (or None), ``files`` the relative artifact list,
    ``stale`` the reason the validation does not describe the model as it is now (``""`` = it does; the
    Validation section then claims nothing, exactly like the Build Summary).
    """
    ledger = ledger or {}
    plan_steps = ledger.get("steps") or []
    requirements = ledger.get("requirements") or []
    material = str((spec or {}).get("material") or "")
    out = [f"# {clip(title or 'Oscar project', 120)} - project notes",
           f"_Generated by Oscar {ts}. Do not edit: it is rewritten after every step._", ""]

    out += ["## 1. Request", (request or ledger.get("request") or "(no request recorded)").strip(), ""]

    out.append("## 2. Requirements and status")
    if requirements:
        out += _table(["id", "requirement", "status", "evidence steps"],
                      [[r.get("id"), r.get("text"), r.get("status", "pending"), _evidence_steps(r, plan_steps)]
                       for r in requirements], None, 200)
    else:
        out.append("(none)")
    out.append("")

    out.append("## 3. Plan")
    if plan_steps:
        out += _table(["id", "phase", "step", "status", "note"],
                      [[s.get("id"), s.get("phase"), s.get("title"), s.get("status", "pending"), s.get("note", "")]
                       for s in plan_steps], None, 200)
    else:
        out.append("(none)")
    out.append("")

    out.append("## 4. Parameters")
    if params:
        out += _table(["name", "value", "unit", "source", "note", "bound"],
                      [[p.get("name"), _param_value(p) or p.get("expr", ""), p.get("unit", ""), p.get("source", ""),
                        p.get("note", ""),
                        "needs rebuild (use set_parameter)" if p.get("needs_rebuild") else p.get("bound", 0)]
                       for p in params], None, 80)
    else:
        out.append("(none)")
    out.append("")

    out.append("## 5. Provenance")
    reference = (provenance or {}).get("reference") or {}
    out.append(f"reference status: {reference_status}"
               + (f" ({reference.get('name') or reference.get('id')})" if reference.get("id") or reference.get("name") else ""))
    entries = (provenance or {}).get("entries") or []
    if entries:
        out += _table(["id", "kind", "claim", "source", "status", "confidence", "sha256"],
                      [[e.get("id"), e.get("kind"), e.get("claim"), e.get("source"), e.get("status"),
                        e.get("confidence"), e.get("sha256") or "-"] for e in entries], None, 120)
    else:
        out.append("(no sources recorded)")
    out.append("")

    out.append("## 6. Assumptions")
    assumptions: list[str] = []
    for n in notes:
        if n.get("kind") == "assumption":
            refs = f" ({', '.join(n.get('refs') or [])})" if n.get("refs") else ""
            assumptions.append(f"{n.get('id')}{refs}: {clip(n.get('text'), 400)}")
    for p in params:
        if p.get("source") == "assumption":
            unit = f" {p['unit']}" if p.get("unit") else ""
            note = f": {clip(p.get('note'), 200)}" if p.get("note") else ""
            assumptions.append(f"parameter {p.get('name')} = {_param_value(p) or p.get('expr', '')}{unit} (assumed){note}")
    if spec and material:
        line = _material_line(material, material_shrink(material), str(spec.get("shrink_comp") or "clearance"))
        if line:
            assumptions.append(line)
    out += [f"- {a}" for a in assumptions] or ["(none)"]
    out.append("")

    out.append("## 7. Validation")
    validated, not_validated = classify_validation(validation, reference_status=reference_status,
                                                   material=material, rebuild_ok=rebuild_ok, stale=stale)
    out.append("Validated:")
    out += [f"- {v['label']}: {v['result']}" for v in validated] or ["- (nothing validated yet)"]
    out.append("NOT validated:")
    out += [f"- {n['label']} ({n['reason']})" for n in not_validated]
    out.append("")

    out.append("## 8. Unresolved")
    unresolved: list[str] = []
    for s in plan_steps:
        if s.get("status") in ("blocked", "skipped", "changed") and s.get("note"):
            unresolved.append(f"{s.get('id')} {s.get('status')}: {clip(s.get('note'), 300)} ({clip(s.get('title'), 80)})")
    for f in ((validation or {}).get("health") or {}).get("findings") or []:
        if f.get("severity") in ("error", "warning"):
            unresolved.append(f"health {f.get('severity')} {f.get('code', '')} {f.get('object', '')}: "
                              f"{clip(f.get('message'), 200)}".replace("  ", " "))
    for row in (validation or {}).get("fit_rows") or []:
        if str(row.get("status", "")).upper() == "FAIL":
            unresolved.append(f"fit FAIL {row.get('id')}: {clip(row.get('detail') or row.get('metric'), 200)}")
    for n in notes:
        if n.get("kind") == "unresolved":
            unresolved.append(f"{n.get('id')}: {clip(n.get('text'), 400)}")
    unresolved += unresolved_extra
    out += [f"- {u}" for u in unresolved] or ["(none)"]
    out.append("")

    out.append("## 9. Steps log")
    if steps:
        rows = []
        for s in steps:
            note = s.snapshot_note or s.checkpoint_note
            if not s.ok:
                note = s.result_head or "failed"
            cells = [str(s.index), s.tool, s.title, "yes" if s.ok else "no", s.snapshot or "-", note]
            if s.superseded:
                cells = [f"~~{c}~~" if c and c != "-" else c for c in cells]
            rows.append(cells)
        out += _table(["#", "tool", "title", "ok", "snapshot", "note"], rows, None, 120)
    else:
        out.append("(no steps recorded)")
    out.append("")

    out.append("## 10. Artifacts")
    out += [f"- {f}" for f in files] or ["(none)"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# rebuild.py

REBUILD_TEMPLATE = '''#!/usr/bin/env python
# Generated by Oscar. Rebuilds {title} by replaying the recorded steps through the same code path the app used.
# Run headless:  freecadcmd -c "exec(open({script_path}, encoding='utf-8').read())"   (the app refuses to replay into a document that is open in it)
# Environment:   OSCAR_REBUILD_SET="wall_t=2.4,clearance=0.35"   re-runs with changed parameters
# Needs the Oscar package (OSCAR_MOD_DIR below); standalone use without it is not supported.
import sys
import FreeCAD as App
OSCAR_MOD_DIR = {mod_dir}
PROJECT_DIR = {project_dir}
DOC_NAME = {doc_name}
PARAMS = {params}
STEPS = {steps}

def main():
    import json, os
    if OSCAR_MOD_DIR not in sys.path:
        sys.path.insert(0, OSCAR_MOD_DIR)
    if getattr(App, "GuiUp", False) and DOC_NAME in App.listDocuments():
        print("REBUILD_REFUSED: document %r is open in this session; run headless instead (see line 3)" % DOC_NAME)
        return 2
    from oscar import project
    overrides = dict(kv.split("=", 1) for kv in os.environ.get("OSCAR_REBUILD_SET", "").split(",") if "=" in kv)
    result = project.replay_steps(project.steps_from_literal(STEPS), project_root=PROJECT_DIR,
                                  doc_name=DOC_NAME, overrides=overrides)
    for r in result.results:
        print(("PASS " if r["ok"] else "FAIL ") + "step %03d %s" % (r["index"], r["tool"]))
    print("REBUILD_RESULT " + json.dumps(project.replay_summary(result)))   # {{"docs": {{name: {{object: volume|null}}}}}}, is_geometry objects only
    print("REBUILD_OK" if result.ok else "REBUILD_FAILED")
    return 0 if result.ok else 1

_RC = main()
if not getattr(App, "GuiUp", False):
    sys.exit(_RC)    # headless: the exit status is the signal; a bare `main()` exits 0 even on failure (verified). Never sys.exit inside the GUI console.
'''


def render_rebuild(*, title: str, script_path: str, mod_dir: str, project_dir: str, doc_name: str,
                   params: dict[str, str], steps: list[dict]) -> str:
    """Text of rebuild.py: paths and names as ``repr`` literals (never ``r"..."``: apostrophes in
    user names, Windows paths), ``STEPS`` as a PYTHON literal (``pprint``: True/False/None, never JSON)."""
    safe_title = '"' + " ".join(str(title or "project").split()).replace('"', "'") + '"'
    return REBUILD_TEMPLATE.format(
        title=safe_title, script_path=repr(script_path), mod_dir=repr(mod_dir), project_dir=repr(project_dir),
        doc_name=repr(doc_name), params=pprint.pformat(params, width=110, sort_dicts=False),
        steps=pprint.pformat(steps, width=110, sort_dicts=False))
