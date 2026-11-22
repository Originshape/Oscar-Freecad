"""Verification tools: ``validate_model``, ``check_fit`` and ``run_validation``.

``validate_model`` answers "is the model sound?" for any document (every object type, status strings, solids, sketch
DoF, expression errors, the expression-bound parameter table; ``deep=True`` also rebuilds with each driver parameter
moved +-10%). ``check_fit`` runs the geometric fit checks between parts (generic rows from ``tools_verify_fit``, the
kit's rows through the guarded ``kit.checks.fit_rows``). ``run_validation`` is the single entry the FinishGate and the
Project call at the end of a work turn; ``feature_tree_status`` (``tools.py``) delegates to ``validate_model``.

Who judges what (3.6, binding): ``ValidationResult.ok`` and the gate look ONLY at ``gate_errors``: health errors that
are NEW since the turn started (key not in ``ctx.state["baseline"][doc]``) or sit on an object this turn created or
changed (``ctx.state["touched"][doc]``). A user's own broken sketch or an invalid vendor solid in a ``Ref_*`` part
that existed before the turn never fails the gate and is shown as ``pre-existing``; with no baseline known every
error counts (an explicit "check my model" request judges everything).

Threading: tool handlers and ``run_validation`` run on the agent thread; every document access goes through
``cad.dispatch`` (the ``health`` wrappers or one ``_gui`` task), the context is captured BEFORE dispatching, and no
lock is held across a dispatch. Both tools are read-only (not mutating, not geometry): no step record, no checkpoint.
"""

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

from . import tools_verify_edit as _edit
from . import tools_verify_fit as _fit
from . import tools_verify_text as _text
from .cad import dispatch as _dispatch
from .cad import health
from .cad.health_types import Finding, HealthReport
from .toolctx import ToolContext, current_context, use_context
from .tools import ToolError, _require_document, tool
from .turnlog import log_event

_PHASES = ("BUILD", "VERIFY", "DELIVER")
FIT_INTENTS = ("enclosure", "fixture")
_HEALTH_TIMEOUT_S = 90.0
_FIT_TIMEOUT_S = 120.0
_FIT_BUDGET_S = 100.0
_EDIT_TIMEOUT_S = 120.0
_ROWS_FULL_LIMIT = 60
_PRE_PREFIX = "pre-existing (not caused by this task, not counted): "


# ---------------------------------------------------------------------------------------------- the result


@dataclass
class ValidationResult:
    """What ``run_validation`` returns. The fields of ``tests/support/fakes.FakeValidation`` are the same;
    ``editability`` is an addition (empty unless ``deep``)."""

    ok: bool                       # not gate_errors and not fail_rows
    doc: str
    health: dict                   # HealthReport.to_dict() of the WHOLE document, strict mode (display and NOTES)
    gate_errors: list = field(default_factory=list)   # the health errors that count (new / touched) + editability breaks
    fit_rows: list = field(default_factory=list)
    fit_ran: bool = False
    fail_rows: list = field(default_factory=list)
    dfm: list = field(default_factory=list)
    coverage: list = field(default_factory=list)
    ran: list = field(default_factory=list)            # VALIDATION_CATALOG ids that actually ran
    not_validated: list = field(default_factory=list)  # catalog ids that did not run or are always unverifiable
    text: str = ""                 # model-facing rendering (health + FIT lines + DFM lines)
    after_step: int = 0            # project.last_geometry_step() at run time
    editability: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """A JSON-safe plain dict (``validation.json`` shape plus ``text``)."""
        return asdict(self)


def _catalog() -> tuple[str, ...]:
    from .project_docs import VALIDATION_CATALOG

    return tuple(VALIDATION_CATALOG)


def fit_entry(row: dict) -> str:
    """The ``VALIDATION_CATALOG`` entry a fit row belongs to (same split as the Build Summary):
    interference / clearance / retention by kind, the kit's port and alignment rows under ``fit.ports``, the
    ``hooks.*`` rows under ``fit.retention``; ``""`` for coverage rows."""
    kind, rid = str(row.get("kind", "")), str(row.get("id", ""))
    if kind == "interference":
        return "fit.interference"
    if kind == "clearance":
        return "fit.clearance"
    if kind == "retention" or rid.startswith("hooks."):
        return "fit.retention"
    if kind in ("containment", "alignment"):
        return "fit.ports"
    return ""


# ---------------------------------------------------------------------------------------------- context readers


def _state(ctx: ToolContext) -> dict:
    return ctx.state if isinstance(ctx.state, dict) else {}


def _spec_of(ctx: ToolContext) -> dict | None:
    """The EnclosureSpec dict of the bound project (``spec.json``), else None."""
    project = ctx.project
    if project is None:
        return None
    try:
        spec = project.read_spec()
    except Exception:  # noqa: BLE001
        return None
    return spec if isinstance(spec, dict) and spec else None


def _reference_status(ctx: ToolContext) -> str:
    for owner in (ctx.ledger, ctx.project):
        value = getattr(owner, "reference_status", "") if owner is not None else ""
        if isinstance(value, str) and value:
            return value
    return "none"


def _after_step(ctx: ToolContext) -> int:
    try:
        if ctx.project is not None:
            return int(ctx.project.last_geometry_step())
        if ctx.ledger is not None:
            return int(getattr(ctx.ledger, "last_geometry_step", 0) or 0)
    except Exception:  # noqa: BLE001
        pass
    return 0


def _baseline_keys(ctx: ToolContext, doc: str) -> set[tuple[str, str]] | None:
    """The finding keys known when the turn started for ``doc``; None when the agent recorded no baseline for it."""
    base = _state(ctx).get("baseline")
    if not isinstance(base, dict) or doc not in base:
        return None
    return {(str(k[0]), str(k[1])) for k in (base[doc] or ()) if isinstance(k, (list, tuple)) and len(k) >= 2}


def _touched_names(ctx: ToolContext, doc: str) -> set[str]:
    touched = _state(ctx).get("touched")
    names = touched.get(doc) if isinstance(touched, dict) else None
    return {str(n) for n in (names or ())}


def split_findings(report: HealthReport, baseline: set | None, touched: set[str]) -> tuple[list[Finding], list[Finding]]:
    """``(counted, pre_existing)`` error findings. An error counts when it is new (key not in ``baseline``), sits on an
    object the turn touched, or when no baseline is known; otherwise it is pre-existing. Pure."""
    counted, pre = [], []
    for f in report.errors:
        if baseline is None or f.key() not in baseline or (f.object and f.object in touched):
            counted.append(f)
        else:
            pre.append(f)
    return counted, pre


# ---------------------------------------------------------------------------------------------- GUI plumbing


def _gui(ctx: ToolContext, task: Callable[[], Any], timeout: float, op: str) -> tuple[Any, str]:
    """Run ``task`` on the GUI thread with the turn's Stop signal; ``(value, "")`` or ``(None, reason)``."""
    try:
        result, failed = _dispatch.dispatch(task, timeout=timeout, op=op, should_stop=ctx.should_stop)
    except Exception as e:  # noqa: BLE001 - a verification tool never crashes the turn
        return None, f"{type(e).__name__}: {e}"
    if failed:
        return None, _dispatch.failure_text(result)
    return result, ""


def _open_doc(name: str):
    import FreeCAD as App

    docs = App.listDocuments()
    return docs.get(name)


# ---------------------------------------------------------------------------------------------- checking


class _ArgError(Exception):
    """Argument problems of a fit run (the tool turns them into one ToolError)."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def _health_report(ctx: ToolContext, doc_name: str | None, deep: bool) -> HealthReport | None:
    """The STRICT health report of ``doc_name`` (None = the turn's last document, then the active one)."""
    name = doc_name or _state(ctx).get("last_doc") or None
    report = health.health_via_dispatch(name, strict=True, deep=deep, timeout=_HEALTH_TIMEOUT_S)
    if report is None and not doc_name and name and not ctx.should_stop():
        report = health.health_via_dispatch(None, strict=True, deep=deep, timeout=_HEALTH_TIMEOUT_S)
    return report


def _gate_entry(f: Finding) -> dict:
    return f.to_dict()


def _dfm(spec: dict | None) -> list[dict]:
    """The kit's design-for-manufacture rows for ``spec`` (pure); ``[]`` without a spec, a kit or a known board."""
    if spec is None:
        return []
    try:
        from .kit import checks as kit_checks
        from .kit import spec as kit_spec

        board = kit_spec.load_board(str(spec.get("board") or "rpi4b"))
        return [dict(r) for r in (kit_checks.dfm_rows(spec, board) or []) if isinstance(r, dict)]
    except ModuleNotFoundError as e:
        if e.name not in ("oscar.kit", "oscar.kit.checks", "oscar.kit.spec"):
            raise
        return []
    except Exception as e:  # noqa: BLE001 - the DFM rows are advice, never a reason to fail the validation
        log_event("health_error", op="dfm_rows", doc="", error=f"{type(e).__name__}: {e}"[:200])
        return []


def _fit_part(ctx: ToolContext, doc: str, *, spec: dict | None, parts: list[str] | None = None,
              reference: str | None = None, ids: set[str] | None = None, lift_mm: float = 0.5,
              coverage: bool = False) -> dict:
    """The fit rows of ``doc`` (generic + kit) in ONE GUI task; ``{"rows", "coverage"}``.

    Raises ``_ArgError`` for unresolvable ``parts`` / ``reference``. A GUI timeout or Stop gives
    ``{"rows": [], "coverage": [], "error": reason}``."""
    status = _reference_status(ctx)

    def task() -> dict:
        d = _open_doc(doc)
        if d is None:
            return {"problems": [f"document '{doc}' is not open"]}
        rows, problems = _fit.run_generic(d, parts=parts, reference=reference, spec=spec, reference_status=status,
                                          lift_mm=lift_mm, ids=ids, budget_s=_FIT_BUDGET_S)
        if problems:
            return {"problems": problems}
        kit, cover = _fit.kit_rows(d, spec, status, ids, coverage=coverage)
        return {"rows": rows + kit, "coverage": cover}

    result, error = _gui(ctx, task, _FIT_TIMEOUT_S, "check_fit")
    if error or not isinstance(result, dict):
        log_event("health_error", op="check_fit", doc=doc, error=str(error)[:200])
        return {"rows": [], "coverage": [], "error": error or "no answer from the GUI thread"}
    if result.get("problems"):
        raise _ArgError(list(result["problems"]))
    return {"rows": list(result.get("rows") or []), "coverage": list(result.get("coverage") or [])}


def _editability(ctx: ToolContext, doc: str, table: dict) -> dict:
    def task() -> dict:
        d = _open_doc(doc)
        if d is None:
            return {"ok": False, "error": f"document '{doc}' is not open"}
        return _edit.editability_test(d, table, should_stop=ctx.should_stop)

    result, error = _gui(ctx, task, _EDIT_TIMEOUT_S, "editability")
    if error or not isinstance(result, dict):
        return {"ok": False, "error": error or "no answer from the GUI thread"}
    return result


def _edit_gate_errors(edit: dict) -> list[dict]:
    """An editability break is a gate error: the parameter exists but editing it breaks the model."""
    out = []
    for b in edit.get("broken") or []:
        sign = "+" if float(b.get("delta") or 0) > 0 else "-"
        problems = ", ".join(b.get("problems") or ["the model breaks"])
        out.append({"severity": "error", "code": "EDITABILITY", "object": str(b.get("parameter", "")),
                    "message": f"changing '{b.get('parameter')}' by {sign}10% breaks the model ({problems}); "
                               "make the geometry follow the parameter robustly, or bind it to fewer features"})
    return out


def _rows_text(report: HealthReport, pre_keys: set) -> str:
    """The report as the model reads it: pre-existing errors are shown as warnings so they never invite the model to
    'fix' (or delete) what the user or an earlier turn left; the stored report keeps the real severities."""
    findings = [replace(f, severity="warning", message=_PRE_PREFIX + f.message)
                if (f.severity == "error" and f.key() in pre_keys) else f for f in report.findings]
    shown = replace(report, findings=findings)
    if len(shown.rows) > _ROWS_FULL_LIMIT:
        return shown.render("compact", max_rows=40)
    return shown.render("full")


def _compose_text(report: HealthReport, pre_keys: set, rows: list[dict], fit_ran: bool, dfm: list[dict],
                  edit: dict) -> str:
    parts = [_rows_text(report, pre_keys)]
    if edit:
        parts.append(_text.edit_block(edit))
    if fit_ran:
        parts.append(_text.fit_block(rows))
    block = _text.dfm_block(dfm)
    if block:
        parts.append(block)
    return "\n".join(p for p in parts if p)


def _ran_ids(health_ran: bool, rows: list[dict], fit_ran: bool, dfm: list[dict], edit: dict) -> list[str]:
    ran: set[str] = set()
    if health_ran:
        ran.add("health")
    if fit_ran:
        ran.update(e for e in (fit_entry(r) for r in rows) if e)
    if dfm:
        ran.add("dfm")
    if edit and edit.get("checked"):
        ran.add("editability")
    return [c for c in _catalog() if c in ran]


def run_validation(doc_name: str | None, *, ctx: "ToolContext | None" = None, deep: bool = False,
                   fit: bool | None = None) -> "ValidationResult | None":
    """The single entry the FinishGate and the Project call. ANY thread.

    ``doc_name=None`` means the document of the turn's last geometry step (``ctx.state["last_doc"]``), else the
    active document. ``fit=None`` runs the fit checks iff the intent needs them (``enclosure`` / ``fixture``) or a
    ``spec.json`` exists; ``deep=True`` adds the +-10% editability rebuild (a break becomes a gate error with the code
    ``EDITABILITY``) and the exhaustive ``isValid`` pass. The sections that ran are persisted through
    ``ctx.project.record_validation`` when a project is bound (the Project merges them with the stored record of the
    same document and geometry step, so a ``check_fit`` the model ran earlier is not lost).

    Returns ``None`` when there is nothing to judge: the document does not exist, the GUI thread timed out or Stop was
    pressed. The agent treats that as "no validation claim" (the VERIFY step is skipped, never passed): a missing
    document must not look like a clean one."""
    ctx = ctx or current_context()
    with use_context(ctx):
        return _run_validation(ctx, doc_name, deep, fit)


def _run_validation(ctx: ToolContext, doc_name: str | None, deep: bool, fit: bool | None) -> "ValidationResult | None":
    started = time.monotonic()
    report = _health_report(ctx, doc_name, deep)
    if report is None:
        log_event("validation_run", ok=None, error="no health report (no document, timeout or Stop)")
        return None
    doc = report.doc
    counted, pre = split_findings(report, _baseline_keys(ctx, doc), _touched_names(ctx, doc))
    gate = [_gate_entry(f) for f in counted]
    spec = _spec_of(ctx)
    run_fit = bool(fit) if fit is not None else (ctx.intent in FIT_INTENTS or spec is not None)
    rows: list[dict] = []
    coverage: list[dict] = []
    fit_error = ""
    if run_fit:
        try:
            part = _fit_part(ctx, doc, spec=spec, coverage=True)
            rows, coverage, fit_error = part["rows"], part["coverage"], part.get("error", "")
        except _ArgError:   # cannot happen without parts / reference; a vanished document is handled as an error
            fit_error = "the fit checks could not start"
    fit_ran = run_fit and not fit_error
    dfm = _dfm(spec)
    edit: dict = {}
    if deep:
        edit = _editability(ctx, doc, health.param_table_via_dispatch(doc))
        gate.extend(_edit_gate_errors(edit))
    fail_rows = [r for r in rows if r.get("status") == "FAIL"]
    ran = _ran_ids(True, rows, fit_ran, dfm, edit)
    text = _compose_text(report, {f.key() for f in pre}, rows, fit_ran, dfm, edit)
    if fit_error:
        text += f"\nFIT checks did not run: {fit_error}"
    result = ValidationResult(
        ok=not gate and not fail_rows, doc=doc, health=report.to_dict(), gate_errors=gate, fit_rows=rows,
        fit_ran=fit_ran, fail_rows=fail_rows, dfm=dfm, coverage=coverage, ran=ran,
        not_validated=[c for c in _catalog() if c not in ran], text=text, after_step=_after_step(ctx), editability=edit)
    computed = {"doc", "health", "gate_errors", "ran", "not_validated", "after_step"}
    if fit_ran:
        computed |= {"fit_rows", "fit_ran", "fail_rows", "coverage"}
    if dfm:
        computed.add("dfm")
    if edit:
        computed.add("editability")
    _persist(ctx, {k: v for k, v in result.to_dict().items() if k in computed}, doc)
    log_event("validation_run", ok=result.ok, fail_rows=len(fail_rows), gate_errors=len(gate),
              seconds=round(time.monotonic() - started, 2))
    return result


# ---------------------------------------------------------------------------------------------- persistence


def _persist(ctx: ToolContext, part: dict, doc: str) -> None:
    """Hand the Project the part of a validation that ran (never raises: a disk problem is logged).

    ``part`` carries only the sections that were computed plus ``ran`` (the catalog ids they cover): the Project
    merges a partial record into the stored record of the SAME model state (same document and ``after_step``), so a
    ``check_fit`` the model ran earlier is not wiped by a later health-only run, and a record of another model state is
    replaced. ``ok`` is left to the Project (``not gate_errors and no FAIL row`` of the merged record)."""
    project = ctx.project
    if project is None:
        return
    try:
        record = {k: v for k, v in part.items() if k not in ("text", "ok", "fail_rows", "fit_ran")}
        record["doc"] = doc
        record.setdefault("after_step", _after_step(ctx))
        project.record_validation(record)
    except Exception as e:  # noqa: BLE001 - the dossier must not break a verification
        log_event("project_error", where="record_validation", error=f"{type(e).__name__}: {e}"[:200])


# ---------------------------------------------------------------------------------------------- validate_model


def _fit_needed(ctx: ToolContext) -> bool:
    return ctx.intent in FIT_INTENTS or _spec_of(ctx) is not None


def _apply_to_ledger(ctx: ToolContext, result: ValidationResult, complete: bool) -> None:
    """Fold an explicit ``validate_model`` into the ledger: a FAILING result always (the VERIFY step reopens and a fix
    step appears), a passing one only when it is a COMPLETE validation for this turn (no fit checks are owed), so a
    health-only pass can never stand in for the fit checks the finish gate still has to run."""
    ledger = ctx.ledger
    if ledger is None or not hasattr(ledger, "apply_validation"):
        return
    if result.ok and not complete:
        return
    try:
        ledger.apply_validation(result)
    except Exception as e:  # noqa: BLE001
        log_event("ledger_error", where="apply_validation", error=f"{type(e).__name__}: {e}"[:200])


def _validate(tool_name: str, doc_name: str, deep: bool, include_params: bool) -> tuple[str, None]:
    ctx = current_context()
    actual = _require_document(doc_name)
    report = _health_report(ctx, actual, deep)
    if report is None:
        raise ToolError(f"{tool_name}: document '{actual}' could not be checked (the GUI thread was busy or Stop was "
                        "pressed); call it again")
    counted, pre = split_findings(report, _baseline_keys(ctx, actual), _touched_names(ctx, actual))
    gate = [_gate_entry(f) for f in counted]
    table: dict = {}
    if include_params or deep:
        table = health.param_table_via_dispatch(actual)
    edit: dict = {}
    if deep:
        edit = _editability(ctx, actual, table)
        gate.extend(_edit_gate_errors(edit))
    lines = [f"VALIDATE doc '{actual}' (deep={'True' if deep else 'False'})",
             _rows_text(report, {f.key() for f in pre})]
    if include_params:
        lines.append(_text.params_block(table))
    if edit:
        lines.append(_text.edit_block(edit))
    n = len(gate)
    if n:
        lines.append(f"RESULT: INVALID ({n} error{'s' if n != 1 else ''})")
    elif pre:
        lines.append(f"RESULT: VALID ({len(pre)} pre-existing error{'s' if len(pre) != 1 else ''} not counted)")
    else:
        lines.append("RESULT: VALID")
    text = "\n".join(lines)
    ran = _ran_ids(True, [], False, [], edit)
    result = ValidationResult(ok=not gate, doc=actual, health=report.to_dict(), gate_errors=gate, ran=ran,
                              not_validated=[c for c in _catalog() if c not in ran], text=text,
                              after_step=_after_step(ctx), editability=edit)
    _persist(ctx, {k: v for k, v in result.to_dict().items()
                   if k in ("doc", "health", "gate_errors", "ran", "editability", "after_step")}, actual)
    _apply_to_ledger(ctx, result, complete=not _fit_needed(ctx))
    if gate:
        raise ToolError(text)
    return text, None


_DOC = {"type": "string", "description": "Name of the Oscar document"}


@tool(
    "validate_model",
    "Check the model's health: recompute errors, invalid or open solids, fragments, conflicting sketches, "
    "expression errors, and the parameter table. Returns the findings and VALID or INVALID. deep=true also "
    "rebuilds with each key parameter changed by +-10% to prove the model is editable. Run it after your last "
    "change and after every fix; an INVALID result lists real defects to repair, not to explain away.",
    {"type": "object", "properties": {
        "doc_name": _DOC,
        "deep": {"type": "boolean", "default": False,
                 "description": "Also rebuild with the key parameters moved +-10% (slower: up to about 20 s)."},
        "include_params": {"type": "boolean", "default": True,
                           "description": "Append the parameter table (value, unit, source, objects bound)."}},
     "required": ["doc_name"]},
    category="verify", phases=_PHASES,
    title=lambda a: "Validate model (deep)" if a.get("deep") else "Validate model",
)
def _validate_model(doc_name: str, deep: bool = False, include_params: bool = True):
    return _validate("validate_model", doc_name, bool(deep), bool(include_params))


def feature_tree_status(doc_name: str):
    """``tools.feature_tree_status`` delegates here: the same report as ``validate_model`` (health of every object
    and the parameter table), without the deep rebuild."""
    return _validate("feature_tree_status", doc_name, False, True)


# ---------------------------------------------------------------------------------------------- check_fit


def _check_fit_args(parts: Any, reference: Any, checks: Any, spec: Any, lift_mm: Any) -> tuple:
    """Validate the arguments before any document access; every problem is listed in one error."""
    problems: list[str] = []
    names = None
    if parts is not None:
        if not isinstance(parts, list) or not all(isinstance(p, str) and p.strip() for p in parts):
            problems.append("'parts' must be a list of object names")
        elif len(parts) < 2:
            problems.append("'parts' needs at least two object names (omit it to check the tagged shell, lid and reference)")
        elif len(parts) > _fit.MAX_PARTS:
            problems.append(f"'parts' has {len(parts)} names; at most {_fit.MAX_PARTS} are checked pairwise")
        else:
            names = [p.strip() for p in parts]
    ids, id_problems = _fit.select_ids(checks if isinstance(checks, list) else None)
    if checks is not None and not isinstance(checks, list):
        problems.append("'checks' must be a list of check ids")
    problems.extend(id_problems)
    if spec is not None and not isinstance(spec, dict):
        problems.append("'spec' must be an object (a partial or full EnclosureSpec)")
    try:
        lift = float(lift_mm)
        if not 0.05 <= lift <= 5.0:
            problems.append(f"'lift_mm' is {lift:g}; use a value from 0.05 to 5 mm (default 0.5)")
    except (TypeError, ValueError):
        problems.append("'lift_mm' must be a number")
        lift = 0.5
    ref = str(reference).strip() if isinstance(reference, str) and reference.strip() else None
    if problems:
        raise ToolError(f"check_fit: {'; '.join(problems)}.")
    return names, ref, ids, (spec if isinstance(spec, dict) else None), lift


@tool(
    "check_fit",
    "Run geometric fit checks between parts: interference (shell vs lid vs reference), clearances, port openings "
    "versus connectors, standoff alignment, and a snap-fit retention test that lifts the lid. Returns PASS / WARN / "
    "FAIL rows. A FAIL is a real defect to fix, not to explain away; change the geometry or the parameter "
    "(set_parameter) and run it again.",
    {"type": "object", "properties": {
        "doc_name": _DOC,
        "parts": {"type": "array", "items": {"type": "string"},
                  "description": "Object names to check; default = the tagged shell, lid and reference."},
        "reference": {"type": "string",
                      "description": "Name of the reference part (Ref_...). Default: the tagged reference."},
        "checks": {"type": "array", "items": {"type": "string"},
                   "description": "Subset of check ids, e.g. ['interference.shell_lid']. Default: all that apply."},
        "spec": {"type": "object", "description": "EnclosureSpec override; default = the project's spec.json."},
        "lift_mm": {"type": "number", "default": 0.5, "minimum": 0.05, "maximum": 5}},
     "required": ["doc_name"]},
    category="verify", phases=_PHASES,
    title=lambda a: "Check fit",
)
def _check_fit(doc_name: str, parts: list | None = None, reference: str | None = None, checks: list | None = None,
               spec: dict | None = None, lift_mm: float = 0.5):
    names, ref, ids, spec_override, lift = _check_fit_args(parts, reference, checks, spec, lift_mm)
    ctx = current_context()
    actual = _require_document(doc_name)
    use_spec = spec_override if spec_override is not None else _spec_of(ctx)
    try:
        part = _fit_part(ctx, actual, spec=use_spec, parts=names, reference=ref, ids=ids, lift_mm=lift)
    except _ArgError as e:
        raise ToolError(f"check_fit: {'; '.join(e.problems)}.") from None
    if part.get("error"):
        raise ToolError(f"check_fit: the checks did not run ({part['error']}); call it again")
    rows = part["rows"]
    if not rows:
        raise ToolError("check_fit: no check applies to the selected ids; omit 'checks' to run every check that applies")
    fails = [r for r in rows if r.get("status") == "FAIL"]
    text = "\n".join([f"CHECK_FIT doc '{actual}' (lift_mm={lift:g})", _text.fit_block(rows)])
    ran = _ran_ids(False, rows, True, [], {})
    _persist(ctx, {"doc": actual, "fit_rows": rows, "fit_ran": True, "fail_rows": fails, "ran": ran,
                   "after_step": _after_step(ctx)}, actual)
    if fails:
        ids_text = ", ".join(str(r.get("id")) for r in fails[:6])
        raise ToolError(text + f"\nFAIL rows ({ids_text}) are defects: fix the geometry or the parameter "
                               "(set_parameter), then run check_fit again.")
    return text, None
