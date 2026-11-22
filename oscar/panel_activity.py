# SPDX-License-Identifier: Apache-2.0
"""Pure model of the LOGS tab's visual activity timeline.

Input: ``steps.jsonl`` step records — plain dicts with the ``StepRecord`` schema
(``index / ts / turn_id / round / call_id / tool / title / args / ok / result_head /
result_chars / dur_s / doc / delta / health / snapshot / snapshot_note / checkpoint /
checkpoint_note / script / replay / geometry``). Output: one ``Entry`` dict per visible
timeline row, in chronological order, grouped into ``analysis / plan / build / verify /
export`` sections.

PURE PYTHON: no Qt, no FreeCAD, no oscar imports — fully testable from plain dicts.
Nothing is invented: every field of an entry is derived from fields the records
actually carry (the tool mapping table only re-words the tool name; a missing
snapshot stays missing; an unknown tool falls back to a generic sentence).

Derivations:

* status — ``ok`` missing/``None`` -> ``running`` (a live step); ``ok`` False ->
  ``failed``; health invalid or warnings -> ``warning``; a geometry step whose own
  notes say ``unchanged`` and whose delta is empty -> ``skipped``; else ``completed``.
* recovery — a failed record whose NEXT record retries the same tool gets a note in
  its explanation saying whether the retry recovered or failed again.
* collapse — runs of >= 3 consecutive records of the same tool with the same status
  (repeated retries / repeated checks) become ONE entry whose ``detail`` carries the
  raw records under ``{"collapsed": [...]}``.
* snapshots — kept only where the record carries a path; CONSECUTIVE entries with the
  same snapshot keep only the first (byte-identical neighbours count as the same when
  ``snapshot_dir`` resolves both files).
"""

import dataclasses
import hashlib
import os

# The entry fields, in display order.
ENTRY_FIELDS = ("timestamp", "op", "title", "explanation", "status", "result_text",
                "error_text", "snapshot_path", "detail", "group")

STATUSES = ("running", "completed", "failed", "warning", "skipped")
GROUPS = ("analysis", "plan", "build", "verify", "export")

GROUP_LABELS = {
    "analysis": "Analysis",
    "plan": "Plan & parameters",
    "build": "Build",
    "verify": "Verify & save",
    "export": "Export",
}

# Which timeline group a tool belongs to (unknown tools fall back to "build", the
# default of a mutating call).
TOOL_GROUPS = {
    # analysis: bring reference data in and measure / recognise it
    "import_file": "analysis",
    "import_reference_part": "analysis",
    "source_reference_part": "analysis",
    "measure_reference_part": "analysis",
    "reconstruct_mesh": "analysis",
    # plan: documents, parameter sheets and notes that set up the build
    "create_document": "plan",
    "reload_document": "plan",
    "set_parameters": "plan",
    "set_parameter": "plan",
    "bind_expression": "plan",
    "enclosure_spec": "plan",
    "record_note": "plan",
    # build: geometry
    "create_object": "build",
    "edit_object": "build",
    "delete_object": "build",
    "insert_part_from_library": "build",
    "apply_feature_plan": "build",
    "reapply_plan_feature": "build",
    "build_enclosure": "build",
    "add_cutout": "build",
    "add_vents": "build",
    "execute_code": "build",
    "execute_code_headless": "build",
    "execute_code_async": "build",
    # verify: checks, validation, saves
    "validate_model": "verify",
    "run_fem_analysis": "verify",
    "save_document": "verify",
    # export
    "export_model": "export",
}

# One human sentence per recorded tool (what the step DID, not its outcome — the
# outcome is the status / result / error line). Generic fallback for unknown tools.
TOOL_EXPLANATIONS = {
    "create_document": "Created a new document",
    "reload_document": "Reloaded the document",
    "import_file": "Imported a file",
    "import_reference_part": "Imported a reference part",
    "source_reference_part": "Fetched a reference part",
    "measure_reference_part": "Measured the reference part",
    "reconstruct_mesh": "Analysed the mesh for reconstruction",
    "create_object": "Created an object",
    "edit_object": "Edited an object",
    "delete_object": "Deleted an object",
    "insert_part_from_library": "Inserted a part from the library",
    "apply_feature_plan": "Applied a feature plan",
    "reapply_plan_feature": "Re-applied a plan feature",
    "build_enclosure": "Built the enclosure",
    "add_cutout": "Added a cutout",
    "add_vents": "Added vents",
    "set_parameters": "Set parameters on the sheet",
    "set_parameter": "Set a parameter on the sheet",
    "bind_expression": "Bound an expression to a parameter",
    "enclosure_spec": "Reviewed the enclosure specification",
    "record_note": "Recorded a project note",
    "execute_code": "Ran custom Python code",
    "execute_code_headless": "Ran custom Python code (headless)",
    "execute_code_async": "Started background Python code",
    "validate_model": "Validated the model",
    "run_fem_analysis": "Ran a FEM analysis",
    "save_document": "Saved the document",
    "export_model": "Exported the model",
}

_RESULT_CLIP = 400          # characters of result_head kept per entry
_COLLAPSE_MIN = 3           # shortest run of identical (tool, status) that collapses


def _record_dict(record) -> dict:
    """``record`` as a plain dict: a dict as is (copy), ``StepRecord`` via ``to_line()``,
    a dataclass via ``asdict``; anything else becomes ``{}`` (never raises)."""
    if isinstance(record, dict):
        return dict(record)
    to_line = getattr(record, "to_line", None)
    if callable(to_line):
        try:
            data = to_line()
            return dict(data) if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - a broken record must not kill the timeline
            return {}
    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        return dataclasses.asdict(record)
    return {}


def _first_line(text: str, clip: int) -> str:
    """The first non-empty line of ``text``, clipped to ``clip`` characters."""
    for line in str(text or "").splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= clip else line[:clip] + "…"
    return ""


def _status_of(rec: dict) -> str:
    ok = rec.get("ok", None)
    if ok is None:
        return "running"
    if not ok:
        return "failed"
    health = rec.get("health")
    if isinstance(health, dict):
        if health.get("valid") is False:
            return "warning"
        try:
            if int(health.get("warnings") or 0) > 0:
                return "warning"
        except (TypeError, ValueError):
            pass
    delta = rec.get("delta") if isinstance(rec.get("delta"), dict) else {}
    empty_delta = not (delta.get("added") or delta.get("removed") or delta.get("changed"))
    if rec.get("geometry") and empty_delta and str(rec.get("snapshot_note") or "") == "unchanged":
        return "skipped"
    return "completed"


def _explanation_of(rec: dict, status: str) -> str:
    tool = str(rec.get("tool") or "")
    base = TOOL_EXPLANATIONS.get(tool) or f"Ran {tool.replace('_', ' ') or 'a tool'}"
    if status == "warning":
        health = rec.get("health") if isinstance(rec.get("health"), dict) else {}
        try:
            errors = int(health.get("errors") or 0)
        except (TypeError, ValueError):
            errors = 0
        if health.get("valid") is False or errors:
            base += f" — health check reported {errors or ''} error{'s' if errors != 1 else ''}".replace("  ", " ")
        else:
            base += " — health check reported warnings"
    return base


def _same_snapshot(rel_a: str, rel_b: str, snapshot_dir: str | None) -> bool:
    """True when two consecutive snapshot paths point at the same picture: the same
    relative path, or byte-identical files under ``snapshot_dir`` (missing files are
    never hashed — a path that cannot be checked counts as different)."""
    if not rel_a or not rel_b:
        return False
    if rel_a == rel_b:
        return True
    if not snapshot_dir:
        return False
    path_a = os.path.join(snapshot_dir, *rel_a.split("/"))
    path_b = os.path.join(snapshot_dir, *rel_b.split("/"))
    try:
        if not (os.path.isfile(path_a) and os.path.isfile(path_b)):
            return False
        if os.path.getsize(path_a) != os.path.getsize(path_b):
            return False
        digests = set()
        for path in (path_a, path_b):
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
            digests.add(digest.hexdigest())
        return len(digests) == 1
    except OSError:
        return False


def _recovery_notes(records: "list[dict]") -> "dict[int, str]":
    """Per failed record: what the NEXT record's retry of the same tool did."""
    notes: dict[int, str] = {}
    for i, rec in enumerate(records):
        if rec.get("ok") is not False:
            continue
        nxt = records[i + 1] if i + 1 < len(records) else None
        if not isinstance(nxt, dict) or nxt.get("tool") != rec.get("tool"):
            continue
        try:
            step = int(nxt.get("index") or 0)
        except (TypeError, ValueError):
            step = 0
        where = f" (step {step})" if step else ""
        if nxt.get("ok") is True:
            notes[i] = f"retried and recovered{where}"
        elif nxt.get("ok") is False:
            notes[i] = f"retried and failed again{where}"
    return notes


def build_timeline(step_records, snapshot_dir=None) -> "list[dict]":
    """The timeline entries of ``step_records`` (plain dicts, ``StepRecord`` objects or
    a mix; chronological input order is kept). ``snapshot_dir`` — the project root —
    resolves each record's relative ``snapshot`` path to an absolute one and enables
    byte-comparison of consecutive snapshots; without it entries keep the path exactly
    as the record carries it."""
    records: list[dict] = []
    for record in step_records or []:
        rec = _record_dict(record)
        if rec:
            records.append(rec)
    # live append safety: a started record and its finished successor share call_id —
    # keep the LAST (the finished one) of each duplicate call
    by_call: dict[str, dict] = {}
    order: list[str] = []
    for rec in records:
        call_id = str(rec.get("call_id") or "")
        key = call_id or f"#{len(order)}"
        if key not in by_call:
            order.append(key)
        by_call[key] = rec
    records = [by_call[key] for key in order]

    notes = _recovery_notes(records)

    entries: list[dict] = []
    run: list[tuple[int, dict, str]] = []   # (record position, record, status)

    def flush_run() -> None:
        if not run:
            return
        if len(run) < _COLLAPSE_MIN:
            for pos, rec, status in run:
                entries.append(_entry_of(rec, status, notes.get(pos, ""), snapshot_dir))
            run.clear()
            return
        first_pos, first, status = run[0]
        last = run[-1][1]
        count = len(run)
        title = str(first.get("title") or first.get("tool") or "")
        explanation = (_explanation_of(first, status)
                       + f" — {count} consecutive attempt{'s' if count != 1 else ''}"
                       + f" ({_short_time(str(first.get('ts') or ''))}–{_short_time(str(last.get('ts') or ''))})")
        snapshot = str(last.get("snapshot") or "")
        entries.append({
            "timestamp": str(first.get("ts") or ""),
            "op": str(first.get("tool") or ""),
            "title": title,
            "explanation": explanation,
            "status": status,
            "result_text": _first_line(str(last.get("result_head") or ""), _RESULT_CLIP),
            "error_text": _first_line(str(last.get("result_head") or ""), _RESULT_CLIP) if status == "failed" else "",
            "snapshot_path": _snapshot_path(snapshot, snapshot_dir),
            "detail": {"collapsed": [rec for _, rec, _ in run]},
            "group": TOOL_GROUPS.get(str(first.get("tool") or ""), "build"),
        })
        run.clear()

    statuses = [_status_of(rec) for rec in records]
    for pos, (rec, status) in enumerate(zip(records, statuses)):
        tool = str(rec.get("tool") or "")
        if run and (str(run[-1][1].get("tool") or "") != tool or run[-1][2] != status):
            flush_run()
        run.append((pos, rec, status))
    flush_run()

    # dedupe identical consecutive snapshots: keep the first, blank the rest
    previous_rel = ""
    for entry in entries:
        rel = _rel_of_entry(entry, snapshot_dir)
        if rel and _same_snapshot(previous_rel, rel, snapshot_dir):
            entry["snapshot_path"] = ""
            # keep previous_rel: a run of 3 identical pictures blanks entries 2..n
        elif rel:
            previous_rel = rel
    return entries


def _rel_of_entry(entry: dict, snapshot_dir: str | None) -> str:
    """The record-relative snapshot path of ``entry`` ("" when it has none)."""
    path = str(entry.get("snapshot_path") or "")
    if not path:
        return ""
    if snapshot_dir and path.startswith(os.path.abspath(snapshot_dir) + os.sep):
        return os.path.relpath(path, snapshot_dir).replace(os.sep, "/")
    return path.replace(os.sep, "/")


def _snapshot_path(rel: str, snapshot_dir: str | None) -> str:
    rel = str(rel or "").strip()
    if not rel:
        return ""
    if not snapshot_dir:
        return rel
    return os.path.join(snapshot_dir, *rel.split("/"))


def _entry_of(rec: dict, status: str, recovery: str, snapshot_dir: str | None) -> dict:
    tool = str(rec.get("tool") or "")
    explanation = _explanation_of(rec, status)
    if recovery:
        explanation += f" — {recovery}"
    result_head = str(rec.get("result_head") or "")
    return {
        "timestamp": str(rec.get("ts") or ""),
        "op": tool,
        "title": str(rec.get("title") or tool),
        "explanation": explanation,
        "status": status,
        "result_text": _first_line(result_head, _RESULT_CLIP),
        "error_text": _first_line(result_head, _RESULT_CLIP) if status == "failed" else "",
        "snapshot_path": _snapshot_path(str(rec.get("snapshot") or ""), snapshot_dir),
        "detail": rec,
        "group": TOOL_GROUPS.get(tool, "build"),
    }


def entry_key(entry: dict):
    """A stable identity for one entry (the timeline tab reuses the row widget whose
    key did not change): the record index, or the collapsed run's first..last index."""
    detail = entry.get("detail") if isinstance(entry.get("detail"), dict) else {}
    collapsed = detail.get("collapsed")
    if isinstance(collapsed, list) and collapsed:
        first = _index_of(collapsed[0])
        last = _index_of(collapsed[-1])
        return ("run", first, last)
    return ("step", _index_of(detail))


def _index_of(rec) -> int:
    try:
        return int((rec or {}).get("index") or 0)
    except (TypeError, ValueError):
        return 0


def _short_time(ts: str) -> str:
    """``2026-11-15T19:50:37`` -> ``19:50:37`` (the raw value when it does not parse)."""
    text = str(ts or "")
    return text.split("T", 1)[1] if "T" in text and len(text.split("T", 1)[1]) >= 5 else text
