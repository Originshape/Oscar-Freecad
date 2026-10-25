"""Text renderings of the verification tools: pure Python, no FreeCAD.

``validate_model`` prints the health report, the parameter table and the editability line; ``check_fit`` prints one
line per row and a count line. Everything here takes plain dicts (the shapes ``read_param_table``, ``check_fit`` and
``editability_test`` produce) so the formats are testable without a document.
"""

from typing import Any, Iterable

MAX_PARAM_ROWS = 40          # "(max 40 rows, then +N more)"
FIT_STATUSES = ("PASS", "WARN", "FAIL", "SKIP")


def fmt_num(value: Any) -> str:
    """``2`` / ``2.4`` / ``0.0125`` / ``1.2346e+06``: a compact number; text as it is; ``?`` for None / non-finite."""
    if value is None:
        return "?"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:        # NaN
            return "?"
        return f"{value:.6g}"
    return str(value)


def fmt_val(value: Any) -> str:
    """A row value: a float always shows a decimal point (``0.0``, ``2.0``, ``0.64``)."""
    if isinstance(value, float) and value == value and abs(value) < 1e15:
        text = f"{value:.6g}"
        return text if any(c in text for c in ".eE") else text + ".0"
    return fmt_num(value)


def _clip(text: Any, limit: int) -> str:
    s = " ".join(str(text if text is not None else "").split())
    return s if len(s) <= limit else s[:max(1, limit - 1)] + "~"


def params_block(table: dict | None, max_rows: int = MAX_PARAM_ROWS) -> str:
    """The ``PARAMETERS`` block of ``validate_model`` for a ``read_param_table`` dict.

    Header ``PARAMETERS (sheet 'Params', 31 rows, 24 bound to objects)`` (a row is bound when at least one object
    property is expression-bound to it), then ``name value unit source bound`` columns, at most ``max_rows`` rows and
    ``(+N more)``. A kit stage / count parameter that is not expression-bound shows ``0 (rebuild: set_parameter)``:
    editing its cell alone changes nothing. No sheet: one line saying so."""
    rows = list((table or {}).get("rows") or [])
    sheet = str((table or {}).get("sheet") or "")
    if not sheet:
        return "PARAMETERS: no 'Params' or 'Parameters' spreadsheet in this document (no driving parameters to check)"
    bound = sum(1 for r in rows if int(r.get("bound") or 0) > 0)
    out = [f"PARAMETERS (sheet '{sheet}', {len(rows)} rows, {bound} bound to objects)"]
    if not rows:
        return out[0]
    shown = rows[:max(1, int(max_rows))]
    cells = []
    for r in shown:
        flag = str(int(r.get("bound") or 0))
        if r.get("needs_rebuild"):
            flag += " (rebuild: set_parameter)"
        cells.append((_clip(r.get("name", ""), 40), fmt_num(r.get("value")), _clip(r.get("unit", ""), 8),
                      _clip(r.get("source", ""), 18), flag))
    heads = ("name", "value", "unit", "source", "bound")
    widths = [max(len(heads[i]), max(len(c[i]) for c in cells)) for i in range(4)]
    out.append("  ".join(heads[i].ljust(widths[i]) for i in range(4)) + "  bound")
    for c in cells:
        out.append("  ".join(c[i].ljust(widths[i]) for i in range(4)) + "  " + c[4])
    if len(rows) > len(shown):
        out.append(f"(+{len(rows) - len(shown)} more)")
    return "\n".join(out)


def fit_line(row: dict) -> str:
    """``PASS interference.shell_lid  Shell vs Lid: common_volume_mm3 = 0.0 (<= 0.05)``; a SKIP row carries its
    reason (``SKIP interference.reference_shell  no Ref_* object in the document``); a WARN / FAIL appends its detail."""
    status = str(row.get("status", "")).upper()
    rid = str(row.get("id", "?"))
    a, b = str(row.get("a", "") or ""), str(row.get("b", "") or "")
    detail = _clip(row.get("detail", ""), 220)
    if status == "SKIP":
        return f"SKIP {rid}  {detail or 'not evaluated'}"
    who = f"{a} vs {b}" if a and b else (a or b)
    metric = str(row.get("metric", "") or "")
    if metric and str(row.get("kind", "")) != "coverage":
        body = f"{metric} = {fmt_val(row.get('value'))}"
        limit = str(row.get("limit", "") or "")
        if limit:
            body += f" ({limit})"
    else:
        body = ""
    head = f"{who}: {body}" if who and body else (who or body)
    if not head:                                   # a coverage row: the detail IS the content
        return f"{status:<4} {rid}  {detail}".rstrip()
    line = f"{status:<4} {rid}  {head}".rstrip()
    if detail and status in ("WARN", "FAIL"):
        line += f" - {detail}"
    return line


def fit_counts(rows: Iterable[dict]) -> dict[str, int]:
    """``{"PASS": n, "WARN": n, "FAIL": n, "SKIP": n}`` for the rows (an unknown status counts as SKIP)."""
    counts = {s: 0 for s in FIT_STATUSES}
    for r in rows:
        status = str(r.get("status", "")).upper()
        counts[status if status in counts else "SKIP"] += 1
    return counts


def fit_summary(rows: Iterable[dict]) -> str:
    """``FIT: 9 PASS, 1 WARN, 0 FAIL, 1 SKIP``."""
    c = fit_counts(rows)
    return "FIT: " + ", ".join(f"{c[s]} {s}" for s in FIT_STATUSES)


def fit_block(rows: list[dict]) -> str:
    """One line per row, then the count line."""
    return "\n".join([fit_line(r) for r in rows] + [fit_summary(rows)])


def dfm_block(rows: list[dict]) -> str:
    """``DFM: 8 PASS, 1 WARN, 2 INFO`` plus one line per WARN / FAIL row (PASS and INFO rows are counted only)."""
    if not rows:
        return ""
    counts: dict[str, int] = {}
    for r in rows:
        counts[str(r.get("status", "")).upper()] = counts.get(str(r.get("status", "")).upper(), 0) + 1
    out = ["DFM: " + ", ".join(f"{counts[s]} {s}" for s in ("PASS", "WARN", "FAIL", "INFO") if counts.get(s))]
    for r in rows:
        if str(r.get("status", "")).upper() in ("WARN", "FAIL"):
            out.append(f"{str(r.get('status')).upper():<4} {r.get('id', '?')}  {_clip(r.get('detail', ''), 200)}")
    return "\n".join(out)


def _side(entry: dict | None) -> str:
    if not entry:
        return "not checked"
    if entry.get("ok"):
        return "ok"
    problems = entry.get("problems") or []
    return f"FAIL({_clip('; '.join(str(p) for p in problems[:2]), 70)})" if problems else "FAIL"


def edit_block(result: dict | None, delta: float = 0.10) -> str:
    """The ``EDITABILITY`` block of ``validate_model(deep=True)`` for an ``editability_test`` dict:
    ``wall_t +10% ok, -10% ok | clearance +10% ok, -10% FAIL(FRAGMENTS Shell)`` plus the inert / skipped /
    not-checked lists (each only when non-empty)."""
    if not result:
        return ""
    pct = f"{delta * 100:g}%"
    params = list(result.get("parameters") or [])
    if result.get("error"):
        return f"EDITABILITY not run: {result['error']}"
    head = (f"EDITABILITY ({len(params)} parameter{'s' if len(params) != 1 else ''}, +-{pct}, "
            f"{result.get('elapsed_s', 0):.1f} s)")
    out = [head]
    if params:
        out.append(" | ".join(f"{p['parameter']} +{pct} {_side(p.get('plus'))}, -{pct} {_side(p.get('minus'))}"
                              for p in params))
    inert = [str(p) for p in (result.get("inert") or [])]
    if inert:
        out.append("inert (changing them moved no object): " + ", ".join(inert[:12]))
    late = [str(p) for p in (result.get("not_checked") or [])]
    if late:
        out.append(f"not checked ({result.get('budget_s', 20):g} s cap): " + ", ".join(late[:12]))
    skipped = result.get("skipped") or []
    if skipped:
        out.append("skipped: " + ", ".join(f"{s['parameter']} ({_clip(s.get('reason', ''), 40)})" for s in skipped[:8])
                   + (f" (+{len(skipped) - 8} more)" if len(skipped) > 8 else ""))
    if result.get("restored") is False:
        out.append("WARNING: a parameter cell could not be restored; check the Params sheet")
    return "\n".join(out)
