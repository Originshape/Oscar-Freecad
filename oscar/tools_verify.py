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

import os
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
_EDIT_TIMEOUT_S = 300.0     # wraps the adaptive editability pass (tools_verify_edit.MAX_BUDGET_S is 240 s)
_PARAMETRIC_TIMEOUT_S = 30.0
_ROWS_FULL_LIMIT = 60
_PRE_PREFIX = "pre-existing (not caused by this task, not counted): "
# Warning-severity findings a parametric turn cannot wave through (same new-or-touched rule as the errors).
_PARAMETRIC_PROMOTED = ("NO_GEOMETRIC_CHANGE", "FRAGMENTS")


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
    reopen: dict = field(default_factory=dict)

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
    if kind == "reference_deviation":
        return "reference.deviation"
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


# ---------------------------------------------------------------------------------------------- parametric contract


def _parametric_wanted(ctx: ToolContext) -> bool:
    """The bound ledger says the request asked for a parametric, spreadsheet-driven model."""
    return bool(getattr(ctx.ledger, "parametric", False))


def _parametric_state(ctx: ToolContext, doc: str, touched: set[str]) -> "dict | None":
    """``{"sheet": <name or "">, "bindings": <n>}`` of ``doc`` read on the GUI thread; ``None`` when it could not be
    read (the health gate still judges; the parametric claim is simply not widened by a guess).

    With ``touched`` (the objects THIS task built or changed, from the Project's step records) bindings count only
    on those objects — a binding the user brought with the document never satisfies the contract. Without records
    (``touched`` empty) the whole document is judged: there is no evidence to discriminate with."""
    from .plan.builder import PLAN_MARK

    def task() -> "dict | None":
        d = _open_doc(doc)
        if d is None:
            return None
        sheet = next((o for o in d.Objects if o.isDerivedFrom("Spreadsheet::Sheet")
                      and (o.Name in ("Params", "Parameters") or o.Label in ("Params", "Parameters")
                           or getattr(o, "Label2", "") == PLAN_MARK)), None)
        eligible = set(touched) or None
        bindings = 0
        if sheet is not None:
            prefixes = (f"{sheet.Name}.", f"{sheet.Label}.")
            for o in d.Objects:
                if o is sheet or (eligible is not None and o.Name not in eligible):
                    continue
                engine = getattr(o, "ExpressionEngine", None) or ()
                if any(str(expr).startswith(prefixes) for _prop, expr in engine):
                    bindings += 1
        return {"sheet": sheet.Name if sheet is not None else "", "bindings": bindings}

    result, error = _gui(ctx, task, _PARAMETRIC_TIMEOUT_S, "parametric")
    if error or not isinstance(result, dict):
        log_event("health_error", op="parametric", doc=doc, error=str(error)[:200])
        return None
    return result


def _parametric_gate_errors(ctx: ToolContext, doc: str, touched: set[str]) -> list[dict]:
    """``PARAMETRIC_UNMET`` gate errors: the request asked for a parametric model and the document is not one
    (no spreadsheet, or a spreadsheet nothing this task built or changed is bound to)."""
    state = _parametric_state(ctx, doc, touched)
    if state is None:
        return []
    how = ("build it with `apply_feature_plan`, or write the sheet with `set_parameters` and bind the dimensions "
           "by expression; if this model profile does not offer the plan tool, say plainly in the report that a "
           "fully parametric model was not achieved")
    if not state["sheet"]:
        return [{"severity": "error", "code": "PARAMETRIC_UNMET", "object": "",
                 "message": f"the request asks for a parametric model but document '{doc}' has no Params / "
                            f"Parameters spreadsheet; {how}"}]
    if not state["bindings"]:
        return [{"severity": "error", "code": "PARAMETRIC_UNMET", "object": state["sheet"],
                 "message": f"the Params spreadsheet '{state['sheet']}' exists but nothing this task built or "
                            f"changed is bound to it: no feature follows any parameter; {how}"}]
    return []


def _promoted_findings(report: HealthReport, baseline: set | None, touched: set[str]) -> list[Finding]:
    """The warning findings a parametric turn cannot wave through (a cut that removed nothing, loose fragments),
    promoted to errors under the same new-or-touched rule as ``split_findings``. Pure."""
    return [replace(f, severity="error",
                    message=f"{f.message} (counted: every operation of a parametric build must change the model)")
            for f in report.findings
            if f.severity != "error" and f.code in _PARAMETRIC_PROMOTED
            and (baseline is None or f.key() not in baseline or (f.object and f.object in touched))]


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


def _reopen_probe(ctx: ToolContext, doc: str) -> dict:
    """D-2 item 8, OPT-IN (OSCAR_VALIDATE_REOPEN=1): save a COPY of the document, reopen it and
    verify the feature tree survives — the Tip volume and the parameter expressions are the
    same. Default-off: the row then reports `not checked` (an opt-in check is never claimed).
    Evidence only for the text; a failure is a gate error (a model that cannot reopen is not
    verified)."""
    if os.environ.get("OSCAR_VALIDATE_REOPEN", "") != "1":
        return {"checked": False, "note": "not checked (opt-in: OSCAR_VALIDATE_REOPEN=1)"}
    import FreeCAD as App

    def task() -> dict:
        import tempfile

        d = App.getDocument(doc) if doc in App.listDocuments() else None
        if d is None:
            return {"checked": False, "note": f"document '{doc}' is not open"}
        fd, path = tempfile.mkstemp(suffix=".FCStd", prefix="oscar_reopen_")
        os.close(fd)
        try:
            def _solids(doc_o):
                return [o.Name for o in doc_o.Objects
                        if getattr(o, "Shape", None) is not None and o.Shape.Solids]

            def _volume(doc_o):
                return sum(abs(float(o.Shape.Volume)) for o in doc_o.Objects
                           if getattr(o, "Shape", None) is not None and o.Shape.Solids)

            bodies = _solids(d)
            vol_before = _volume(d)
            sheets = [o for o in d.Objects if getattr(o, "TypeId", "") == "Spreadsheet::Sheet"]
            expr_before = sum(len(o.ExpressionEngine) for o in d.Objects)
            d.saveCopy(path)
            reopened = App.openDocument(path)
            try:
                r_bodies = _solids(reopened)
                vol_after = _volume(reopened)
                r_expr = sum(len(o.ExpressionEngine) for o in reopened.Objects)
                ok = (vol_before > 0 and vol_after > 0
                      and abs(vol_after - vol_before) <= 1e-6 * max(vol_before, 1.0)
                      and r_expr >= expr_before and len(r_bodies) == len(bodies))
                return {"checked": True, "ok": bool(ok),
                        "volume_preserved": vol_after, "expressions": r_expr,
                        "note": "" if ok else "the reopened copy differs (tip volume or expressions)"}
            finally:
                App.closeDocument(reopened.Name)
        except Exception as exc:  # noqa: BLE001 - a probe failure is reported, never raised
            return {"checked": True, "ok": False, "note": f"{type(exc).__name__}: {exc}"[:160]}
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    result, error = _gui(ctx, task, 60.0, "reopen probe")
    if error or not isinstance(result, dict):
        return {"checked": False, "note": f"probe failed: {error or 'no result'}"}
    return result


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
    ``EDITABILITY``) and the exhaustive ``isValid`` pass. When the bound ledger says the request asked for a
    parametric, spreadsheet-driven model (``Ledger.parametric``), ``deep`` is forced on and the document must hold a
    ``Params`` / ``Parameters`` (or plan-stamped) spreadsheet with at least one expression-bound feature — otherwise
    a ``PARAMETRIC_UNMET`` gate error says the contract is unmet. The sections that ran are persisted through
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
    parametric = _parametric_wanted(ctx)
    if parametric and not deep:
        deep = True                    # a parametric request owes the +-10% editability proof at finish
    report = _health_report(ctx, doc_name, deep)
    if report is None:
        log_event("validation_run", ok=None, error="no health report (no document, timeout or Stop)")
        return None
    doc = report.doc
    baseline, touched = _baseline_keys(ctx, doc), _touched_names(ctx, doc)
    counted, pre = split_findings(report, baseline, touched)
    gate = [_gate_entry(f) for f in counted]
    if parametric:
        gate.extend(_parametric_gate_errors(ctx, doc, touched))
        gate.extend(_gate_entry(f) for f in _promoted_findings(report, baseline, touched))
    gate.extend(_cutter_gate_errors(ctx, doc))
    gate.extend(_params_unbound_gate_errors(ctx, doc))
    gate.extend(_non_parametric_gate_errors(ctx, doc))
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
    reopen: dict = {"checked": False, "note": "not checked"}
    if deep:
        edit = _editability(ctx, doc, health.param_table_via_dispatch(doc))
        gate.extend(_edit_gate_errors(edit))
        reopen = _reopen_probe(ctx, doc)
        if reopen.get("checked") and not reopen.get("ok", True):
            gate.append(f"REOPEN {reopen.get('note') or 'failed'}")
    fail_rows = [r for r in rows if r.get("status") == "FAIL"]
    ran = _ran_ids(True, rows, fit_ran, dfm, edit)
    text = _compose_text(report, {f.key() for f in pre}, rows, fit_ran, dfm, edit)
    if fit_error:
        text += f"\nFIT checks did not run: {fit_error}"
    if deep:
        text += ("\nREOPEN: ok" if reopen.get("checked") and reopen.get("ok")
                 else f"\nREOPEN: {reopen.get('note') or 'failed'}")
    result = ValidationResult(
        ok=not gate and not fail_rows, doc=doc, health=report.to_dict(), gate_errors=gate, fit_rows=rows,
        fit_ran=fit_ran, fail_rows=fail_rows, dfm=dfm, coverage=coverage, ran=ran,
        not_validated=[c for c in _catalog() if c not in ran], text=text, after_step=_after_step(ctx), editability=edit, reopen=reopen)
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
    gate.extend(_cutter_gate_errors(ctx, actual))
    gate.extend(_params_unbound_gate_errors(ctx, actual))
    gate.extend(_non_parametric_gate_errors(ctx, actual))
    table: dict = {}
    if include_params or deep:
        table = health.param_table_via_dispatch(actual)
    edit: dict = {}
    if deep:
        edit = _editability(ctx, actual, table)
        gate.extend(_edit_gate_errors(edit))
    lines = [f"VALIDATE doc '{actual}' (deep={'True' if deep else 'False'})",
             _rows_text(report, {f.key() for f in pre})]
    lines.extend(f"ERROR {e['code']}: {e['message']}" for e in gate
                 if e.get("code") in ("LEFTOVER_CUTTERS", "PARAMS_UNBOUND", "NON_PARAMETRIC_DELIVERABLE"))
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


# ---------------------------------------------------------------------------------------------- compare_to_reference


def _shape_volume(obj) -> float:
    try:
        return abs(float(obj.Shape.Volume))
    except Exception:  # noqa: BLE001 - an exotic shape simply loses the largest-volume contest
        return 0.0


# loose Part primitives are helper cutters when a distinct final solid exists (Issue 003 Area 5)
_HELPER_TYPES = frozenset({"Part::Box", "Part::Cylinder", "Part::Sphere", "Part::Cone", "Part::Torus"})
_BOOLEAN_TYPES = frozenset({"Part::Cut", "Part::Fuse", "Part::MultiFuse", "Part::Common", "Part::Section"})


def _boxes_overlap(a, b) -> bool:
    """True when two bounding boxes intersect (strictly — touching faces do not count)."""
    return (a.XMin < b.XMax and a.XMax > b.XMin
            and a.YMin < b.YMax and a.YMax > b.YMin
            and a.ZMin < b.ZMax and a.ZMax > b.ZMin)


def _leftover_cutters(doc, touched: set) -> list:
    """Names of THIS TURN'S visible, unconsumed helper cutters: loose Part primitives that OVERLAP
    a distinct final solid, in a document that shows boolean intent (a Part boolean exists — the
    model was cutting/fusing and left a piece unused). False-positive-safe (Issue 003 Area 5): the
    user's own pre-existing primitives, consumed construction objects, hidden cutters, the
    reference, primitive-only models, sibling deliverable parts and documents with no booleans are
    NOT leftovers. GUI-thread only."""
    from .cad import health_eval

    geo = _fit._geometry(doc, health_eval)
    if not any(o.TypeId in _BOOLEAN_TYPES for o in geo):
        return []
    consumed = {o.Name for g in geo for o in (getattr(g, "OutList", None) or []) if getattr(o, "Name", "")}
    primitives = [o for o in geo
                  if o.TypeId in _HELPER_TYPES and o.Name in touched and not _fit.is_reference(o)
                  and o.Name not in consumed and _fit._visible(o)]
    finals = [o for o in geo if o.TypeId not in _HELPER_TYPES and o.Name not in consumed]
    if not finals:
        return []
    flagged = []
    for obj in primitives:
        try:
            if any(_boxes_overlap(obj.Shape.BoundBox, other.Shape.BoundBox) for other in finals):
                flagged.append(obj.Name)
        except Exception:  # noqa: BLE001 - a shapeless object simply cannot overlap
            continue
    return flagged


def _params_unbound_gate_errors(ctx: ToolContext, doc_name: str) -> list:
    """M1 F4 (RC5): a deliverable parameter sheet whose parameters bind NOTHING is decoration, not a
    parametric model. When the document holds non-reference geometry but the reported parameter sheet
    has zero expression bindings, validation fails. Only a sheet THIS TURN wrote counts (its name is
    in the touched set): a user's or an earlier turn's sheet is never penalized here. Silent when the
    scan cannot run, when there is no sheet or no geometry, and whenever anything at all is bound."""
    touched = _touched_names(ctx, doc_name)
    if not touched:
        return []

    def task() -> dict:
        doc = _open_doc(doc_name)
        if doc is None:
            return {}
        from .cad import health_eval
        table = health_eval.read_param_table(doc)
        if not table.get("sheet") or not table.get("rows"):
            return {}
        if not [o for o in doc.Objects if health_eval.is_geometry(o) and not _fit.is_reference(o)]:
            return {}
        return {"sheet": table["sheet"], "params": len(table["rows"]),
                "bound": sum(int(r.get("bound", 0)) for r in table["rows"])}

    info, failure = _gui(ctx, task, 15.0, "params_unbound")
    if failure or not info or info.get("bound") or info.get("sheet") not in touched:
        return []
    return [{"code": "PARAMS_UNBOUND", "object": info["sheet"],
             "message": f"the parameter sheet {info['sheet']!r} holds {info['params']} parameter(s) but nothing "
                        "is expression-bound to them: a populated spreadsheet is not a parametric model — build "
                        "through apply_feature_plan (or bind_expression) so every driving dimension moves the "
                        "geometry"}]


def _non_parametric_gate_errors(ctx: ToolContext, doc_name: str) -> list:
    """M1 F5 (RC4, the 16:54 session): loose solids and booleans delivered as a RECONSTRUCTION.
    When the turn's request is a reconstruction (``flow_intent.is_reconstruction_request``) and the
    touched geometry holds no native sketches / PartDesign features and nothing is expression-bound,
    the deliverable is not a parametric reconstruction and cannot validate as one. Ordinary boolean
    workflows (no reconstruction wording), feature-built and bound deliverables never trip it."""
    from .flow_intent import is_reconstruction_request

    if not is_reconstruction_request(str(getattr(ctx.ledger, "request", "") or "")):
        return []
    touched = _touched_names(ctx, doc_name)
    if not touched:
        return []

    def task() -> dict:
        doc = _open_doc(doc_name)
        if doc is None:
            return {}
        from .cad import health_eval
        names: list[str] = []
        boolean = False
        for name in touched:
            obj = doc.getObject(name)
            if obj is None:
                continue
            type_id = str(obj.TypeId)
            if type_id.startswith(("PartDesign::", "Sketcher::")):
                return {}                     # native parametric features exist: not this gate's case
            boolean = boolean or type_id in ("Part::Cut", "Part::Fuse", "Part::Common", "Part::MultiFuse")
            if health_eval.is_geometry(obj) and not _fit.is_reference(obj):
                names.append(obj.Name)
        if not names or not (boolean or len(names) > 1):
            return {}                         # a lone imported solid is a reference, not a deliverable claim
        table = health_eval.read_param_table(doc)
        return {"names": names,
                "bound": sum(int(r.get("bound", 0)) for r in table.get("rows", []))}

    info, failure = _gui(ctx, task, 15.0, "non_parametric")
    if failure or not info or info.get("bound"):
        return []
    listing = ", ".join(info["names"][:6])
    return [{"code": "NON_PARAMETRIC_DELIVERABLE", "object": listing,
             "message": f"the reconstruction is loose solids and booleans ({listing}): no native sketches or "
                        "PartDesign features and nothing bound to a parameter — a parametric reconstruction needs "
                        "editable features (build it with apply_feature_plan, or say plainly that only a faceted / "
                        "boolean approximation was made)"}]


def _cutter_gate_errors(ctx: ToolContext, doc_name: str) -> list:
    """Gate errors for leftover helper cutters (Issue 003 Area 5); silent when the scan cannot run."""
    touched = _touched_names(ctx, doc_name)

    def task() -> list:
        doc = _open_doc(doc_name)
        return _leftover_cutters(doc, touched) if doc is not None else []

    names, failure = _gui(ctx, task, 30.0, "leftover_cutters")
    if failure or not names:
        return []
    listing = ", ".join(names[:8])
    return [{"code": "LEFTOVER_CUTTERS", "object": listing,
             "message": f"leftover helper cutters stay in the model ({listing}): delete or hide them; "
                        "the deliverable is one final solid"}]


def _settle_reference_view(built, refs) -> list:
    """M1 F3: a successful comparison hands the viewport to the reconstruction — the reference
    objects are hidden (they stay in the document; toggling Visibility shows them again) and the
    built body is shown and framed. Best-effort: never fails the comparison."""
    hidden = []
    try:
        for ref in refs:
            if getattr(ref, "Visibility", False):
                ref.Visibility = False
                hidden.append(ref.Name)
        if hasattr(built, "Visibility") and not built.Visibility:
            built.Visibility = True
        from .cad import views
        views.fit_view()                     # cosmetic; a no-op in console sessions
    except Exception:  # noqa: BLE001 - visibility must never break the comparison
        pass
    return hidden


def _compare_task(doc_name: str, wanted: str, ref_name: str, warn_mm: float, warn_pct: float) -> dict:
    """GUI thread: resolve the built object and the reference, compare their GLOBAL shapes (Issue 003)."""
    import Part

    from .cad import health_eval
    from .refparts import compare, measure

    doc = _open_doc(doc_name)
    if doc is None:
        return {"error": f"compare_to_reference: document '{doc_name}' is not open"}
    if wanted:
        built = _fit._find_object(doc, wanted)
        if built is None:
            return {"error": f"compare_to_reference: '{wanted}' is not an object of document '{doc.Name}'"}
        if not health_eval.is_geometry(built) or not hasattr(built, "Shape"):
            return {"error": f"compare_to_reference: '{wanted}' is not solid geometry (a sketch or mesh cannot be compared)"}
    else:
        candidates = [o for o in _fit._geometry(doc, health_eval) if not _fit.is_reference(o)]
        if not candidates:
            return {"error": f"compare_to_reference: document '{doc.Name}' holds no solid geometry to compare"}
        visible = [o for o in candidates if _fit._visible(o)] or candidates
        built = max(visible, key=_shape_volume)
    refs, ref_error = _fit.reference_objects(doc, health_eval, ref_name or None)
    if ref_error or not refs:
        problem = ref_error or f"no reference object (Ref_* / OscarRole ref) in document '{doc.Name}'"
        return {"error": f"compare_to_reference: no reference to compare against ({problem}); import it first "
                         "(import_reference_part, or import_file for an attached file)"}
    ref_display = ref_name or (refs[0].Name if len(refs) == 1 else "reference")
    rows = compare.compare_shapes(measure.global_shape(built),
                                  Part.makeCompound([measure.global_shape(o) for o in refs]),
                                  built_name=built.Name, ref_name=ref_display,
                                  warn_mm=warn_mm, warn_pct=warn_pct)
    hidden = _settle_reference_view(built, refs)
    return {"rows": rows, "built": built.Name, "reference": ref_display, "hidden": hidden}


@tool(
    "compare_to_reference",
    "Compare the built model with the imported reference geometry: per-axis size deviation of the bounding boxes, the "
    "box-centre offset, the volume deviation and a boolean symmetric-difference share, each PASS / WARN / SKIP with the "
    "real numbers. Informational: a WARN says how far the reconstruction drifted (report the numbers); it never fails "
    "the turn.",
    {"type": "object", "properties": {
        "doc_name": _DOC,
        "object": {"type": "string",
                   "description": "The built object; default = the largest visible solid that is not the reference."},
        "reference": {"type": "string",
                      "description": "The reference object or container (Ref_... or any name); default = the tagged reference."},
        "warn_mm": {"type": "number", "default": 0.5, "minimum": 0.01, "maximum": 10,
                    "description": "WARN threshold for the size and offset deviations, mm."},
        "warn_pct": {"type": "number", "default": 1.0, "minimum": 0.01, "maximum": 50,
                     "description": "WARN threshold for the volume deviation, percent of the reference volume."}},
     "required": ["doc_name"]},
    category="verify", phases=_PHASES,
    title=lambda a: "Compare with reference",
)
def _compare_to_reference(doc_name: str, object: str | None = None, reference: str | None = None,  # noqa: A002
                          warn_mm: float = 0.5, warn_pct: float = 1.0):
    ctx = current_context()
    actual = _require_document(doc_name)

    def task() -> dict:
        return _compare_task(actual, str(object or ""), str(reference or ""), float(warn_mm), float(warn_pct))

    result, failure = _gui(ctx, task, _FIT_TIMEOUT_S, "compare_to_reference")
    if failure or not isinstance(result, dict) or result.get("error"):
        message = failure or (result or {}).get("error") or "the comparison returned nothing"
        raise ToolError(f"compare_to_reference: {message}")
    rows = result["rows"]
    lines = [f"COMPARE_TO_REFERENCE doc '{actual}' (built '{result['built']}' vs reference "
             f"'{result['reference']}')", _text.fit_block(rows)]
    if result.get("hidden"):
        lines.append(f"Reference hidden after the comparison ({', '.join(result['hidden'])}); "
                     "toggle its Visibility in the model tree to see it again.")
    text = "\n".join(lines)
    _persist(ctx, {"doc": actual, "fit_rows": rows, "fit_ran": True, "fail_rows": [],
                   "ran": _ran_ids(False, rows, True, [], {}), "after_step": _after_step(ctx)}, actual)
    return text, None
