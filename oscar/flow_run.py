"""Turn outcome texts, the round governor line and the agent configuration (private module of ``oscar.flow``).

PURE PYTHON (no FreeCAD, no Qt). Everything here is plain text composition and data: the ledger is only read through
its public methods (``counts``, ``current_step``, ``pending_required``), so this module never imports ``oscar.flow``.
``oscar.flow`` re-exports the public names, so callers import everything from there.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from .flow_guard import WatchdogConfig

if TYPE_CHECKING:   # annotation only: importing oscar.flow here at run time would be circular (flow re-exports this module)
    from .flow import Ledger

__all__ = ["NOT_VERIFIED_HEADER", "NOT_VERIFIED_FOOTER", "NO_CHANGES_TEXT", "compose_unverified", "compose_capped",
           "compose_needs_help", "compose_failed", "phase_guidance", "health_label", "format_governor", "AgentConfig"]

NOT_VERIFIED_HEADER = "NOT VERIFIED: this turn ended with items the app could not close:"
NOT_VERIFIED_FOOTER = "The summary below is the model's own account; treat any claim of success in it as unverified."
NO_CHANGES_TEXT = ("Oscar made no changes in the model for this request (the model only described a plan twice). "
                   "Try again, or say what blocked it.")


# ---------------------------------------------------------------------------------------------------
# Outcome texts


def compose_unverified(reasons: Sequence[str], model_text: str, summary: str = "") -> str:
    """The text of an ``unverified`` turn: the NOT VERIFIED banner, the model's own text, the Build Summary (if any)."""
    parts = [NOT_VERIFIED_HEADER, *[f"- {r}" for r in reasons], NOT_VERIFIED_FOOTER, ""]
    text = "\n".join(parts)
    body = (model_text or "").strip()
    if body:
        text += "\n" + body
    if summary:
        text += "\n\n" + summary
    return text.rstrip() if not summary else text


def compose_capped(model_text: str, tool_rounds: int, pending_ids: Sequence[str], summary: str = "") -> str:
    """The text of a ``capped`` turn: the wrap-up, the Continue footer, the Build Summary (if any)."""
    footer = (f"(Stopped after {tool_rounds} tool rounds with {len(pending_ids)} required steps open: "
              f"{', '.join(pending_ids)}. Press Continue to keep going.)")
    body = (model_text or "").strip()
    text = (body + "\n\n" if body else "") + footer
    return text + ("\n\n" + summary if summary else "")


def compose_needs_help(explanation: str) -> str:
    """``Oscar is stuck: <explanation>. What I need from you: ...`` (breaker or watchdog budget exhausted)."""
    explanation = (explanation or "the model kept failing").strip().rstrip(".")
    return (f'Oscar is stuck: {explanation}. What I need from you: attach the missing file, relax a requirement, '
            'or say "skip it".')


def compose_failed(reasons: Sequence[str], empty_reply_message: str = "") -> str:
    """The text of a work turn that made no change after the nudge budget (zero geometry)."""
    head = empty_reply_message.strip() if empty_reply_message else NO_CHANGES_TEXT
    line = next((r for r in reasons if r.startswith("the model made no change")), reasons[0] if reasons else "")
    return head + (f"\n- {line}" if line else "")


# ---------------------------------------------------------------------------------------------------
# Governor line and configuration

_FALLBACK_GUIDANCE = {
    "INTAKE": "Capture the request: confirm the requirements and the plan, then start with the first open step.",
    "SOURCE": ("Source the reference part first (find_reference_part, source_reference_part). If no file can be obtained "
               "the app falls back to nominal board data, so do not stall."),
    "PLAN": "Refine the plan and the spec briefly, then build; do not describe the whole model in your thinking.",
    "PARAMS": ("Put every driving dimension in the Params sheet (set_parameters or build_enclosure stage params), each with a "
               "source, then build from it."),
    "BUILD": "Build one named step per call, read the result and the health line, and fix errors before the next step.",
    "VERIFY": "Run validate_model / check_fit, fix every FAIL or health error, then give a short report.",
    "DELIVER": ("Everything is built: write a short final answer (what was built, key dimensions, assumptions); the app writes "
                "the notes and exports."),
}


def phase_guidance(phase: str) -> str:
    """One short guidance sentence for a phase: ``prompt.PHASE_GUIDANCE`` when that module exists, else a built-in fallback."""
    try:
        import importlib

        text = importlib.import_module(f"{__package__}.prompt").PHASE_GUIDANCE.get(phase)
        if text:
            return str(text)
    except ModuleNotFoundError as e:
        if e.name != f"{__package__}.prompt":
            raise
    except (AttributeError, KeyError, TypeError):
        pass
    return _FALLBACK_GUIDANCE.get(phase, "")


def _short(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` characters with an ellipsis (no elision marker: used inside the one-line governor)."""
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def health_label(last_health: dict | None) -> str:
    """``OK`` / ``<n> ERR`` / ``?`` (no health report yet) from the per-turn ``ctx.state["last_health"]`` mapping."""
    if not last_health:
        return "?"
    errors, invalid = 0, False
    for report in last_health.values():
        if not isinstance(report, dict):
            continue
        counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
        errors += int(counts.get("errors", report.get("errors", 0)) or 0)
        invalid = invalid or report.get("valid") is False
    if errors:
        return f"{errors} ERR"
    return "1 ERR" if invalid else "OK"


def format_governor(ledger: "Ledger", *, round_no: int, cap: int, health: str = "?", phase_changed: bool = False,
                    warn_fraction: float = 0.8) -> str:
    """The ``[governor]`` line appended to a results message, plus ``[phase] ...`` when the phase changed.

    Grammar: ``[governor] round n/cap · PHASE · steps d/t done · now: S7 "title" · pending: S8, S9 · health OK``;
    from ``warn_fraction`` of the budget a reminder is appended. At most 400 characters on the first line.
    """
    ledger._refresh()
    c = ledger.counts()
    cur = ledger.current_step()
    parts = [f"[governor] round {round_no}/{cap}", ledger.phase, f"steps {c['done']}/{c['total']} done"]
    if cur is not None:
        parts.append(f'now: {cur.id} "{_short(cur.title, 60)}"')
    pend = [s.id for s in ledger.pending_required() if cur is None or s.id != cur.id]
    if pend:
        shown = ", ".join(pend[:6]) + (f" +{len(pend) - 6}" if len(pend) > 6 else "")
        parts.append(f"pending: {shown}")
    parts.append(f"health {health}")
    if cap > 0 and round_no >= warn_fraction * cap:
        parts.append(f"{int(round(warn_fraction * 100))}% of the round budget used: finish the required steps, then stop and report.")
    line = " · ".join(parts)
    if len(line) > 400:
        line = line[:399] + "…"
    if phase_changed:
        line += f"\n[phase] {ledger.phase}: {phase_guidance(ledger.phase)}"
    return line


@dataclass
class AgentConfig:
    """Limits and switches of one agent turn."""

    max_tool_rounds: int = 30
    max_rounds_hard: int = 60
    raise_step: int = 10
    warn_fraction: float = 0.8
    cutoff_retries: int = 2
    gate_nudges_max: int = 3
    breaker_trips_max: int = 3
    stall_rounds: int = 6                        # 5.8 stall rule
    recovery_notes_max: int = 12                 # turn-wide ceiling on injected synthetic notes of ALL mechanisms
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    reasoning_plan: str = "medium"               # first provider round of a turn when Profile.reasoning == "auto"
    reasoning_exec: str = "low"                  # every later round
    exec_max_tokens: int = 12000
    auto_validate: bool = True
    steer: bool = True
    flow_enforcement: bool = True                # False (env OSCAR_FLOW=0): ledger, gate, governor, watchdog, breaker, hooks are off
