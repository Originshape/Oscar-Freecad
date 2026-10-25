"""The plan tools of the model: ``plan_set`` and ``plan_update``.

The app seeds a plan (a ``flow.Ledger``) before the first model call; these two tools are how the model
refines it. Both operate on ``current_context().ledger`` and run on the agent thread (no GUI dispatch).
They are read-only for the document: not mutating, never geometry, so no step record, health check or
checkpoint is made for them; the agent emits ``plan_changed`` when the ledger revision moves.

Importing this module registers the tools (``oscar.tools`` imports it at the end of its own import).
"""

from .flow import LedgerError, PHASES
from .tools import ToolError, current_context, tool

__all__ = ["plan_set", "plan_update"]

_STATUSES = ["pending", "running", "done", "blocked", "skipped", "changed"]


def _ledger(tool_name: str):
    """The bound ledger of this turn; raises a model-facing ``ToolError`` when no plan is bound."""
    ledger = current_context().ledger
    if ledger is None:
        raise ToolError(f"{tool_name}: no plan is bound to this turn (the app keeps a plan only for build tasks).")
    return ledger


@tool(
    "plan_set",
    "Edit the build plan the app seeded for this task: add, rename or re-scope requirements and plan steps. "
    "By default your edits are MERGED into the app's plan (keep the ids you were given); pass replace=true only "
    "to replace it entirely. Returns the current plan.",
    {"type": "object", "properties": {
        "requirements": {
            "type": "array",
            "description": "[{id?, text, covers?}]. Keep given ids (R1..). Omit id to add a new requirement.",
            "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "text": {"type": "string"},
                "covers": {"type": "array", "items": {"type": "string"}}}, "required": ["text"]}},
        "steps": {
            "type": "array",
            "description": "[{id?, title, phase, acceptance?, covers?}]. Omit id to append a new step.",
            "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "title": {"type": "string"},
                "phase": {"type": "string", "enum": list(PHASES)},
                "acceptance": {"type": "string"},
                "covers": {"type": "array", "items": {"type": "string"}}}, "required": ["title"]}},
        "replace": {"type": "boolean", "default": False}}},
    category="plan",
)
def plan_set(requirements=None, steps=None, replace=False):
    """Apply a plan edit to the bound ledger and return its compact rendering (plus any warnings).

    Raises ``ToolError`` with the ledger's model-facing message when the edit is invalid (nothing is changed
    then) or when no ledger is bound. A call with neither ``requirements`` nor ``steps`` just shows the plan.
    """
    ledger = _ledger("plan_set")
    try:
        warnings = ledger.plan_set(requirements or None, steps or None, replace=bool(replace))
    except LedgerError as e:
        raise ToolError(str(e)) from e
    text = ledger.render_compact()
    if warnings:
        text += "\nNOTES: " + " ".join(warnings)
    return text, None


@tool(
    "plan_update",
    "Mark a plan step (S..) or requirement (R..) as running, done, blocked, skipped or changed. 'blocked', "
    "'skipped' and 'changed' need a note with the reason. The app records tool results as evidence automatically; "
    "call this when a step is finished or cannot be done.",
    {"type": "object", "properties": {
        "id": {"type": "string", "description": "S5 or R2"},
        "status": {"type": "string", "enum": _STATUSES},
        "note": {"type": "string", "description": "Reason (required for blocked/skipped/changed) or where the result is."},
        "evidence": {"type": "string", "description": "Optional: object names or facts that show the step is done."}},
     "required": ["id", "status"]},
    category="plan",
)
def plan_update(id, status, note="", evidence=""):  # noqa: A002 - the schema names the argument `id`
    """Set one step or requirement status on the bound ledger; returns the confirmation plus the compact plan.

    Raises ``ToolError`` with model-facing messages for an unknown id, a bad status, a missing reason,
    an app-owned step or a derived requirement status. ``done`` without recorded work is stored as a claim
    (the finish gate does not accept a claim alone).
    """
    ledger = _ledger("plan_update")
    if not ledger.steps and not ledger.requirements:
        raise ToolError("plan_update: this turn has no plan (it is a question or an inspection); there is nothing to update.")
    try:
        confirmation = ledger.plan_update(id, status, note, evidence)
    except LedgerError as e:
        raise ToolError(str(e)) from e
    return f"{confirmation}\n{ledger.render_compact()}", None
