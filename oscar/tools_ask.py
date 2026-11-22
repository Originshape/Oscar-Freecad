# SPDX-License-Identifier: Apache-2.0
"""The ``ask_user`` tool: the model's one way to request an explicit user decision.

Importing this module registers the tool. The handler runs on the agent worker thread: it validates
the request with ``oscar.askuser`` (malformed questions are rejected, never repaired), opens a
``PendingRequest`` in ``askuser.REGISTRY`` bound to this conversation and turn, renders the card on
the GUI thread through the panel-set ``surface`` hook (no surface ⇒ headless: the registry still
works) and then blocks COOPERATIVELY in ``REGISTRY.wait`` — 0.25 s slices, Stop-honouring, bounded
by the request timeout — until the panel resolves the matching id. The structured answer protocol
(``answer``/``index``/``value``, wizard ``answers`` in order, plus request/conversation/run ids and
timestamps) is returned to the model verbatim; a request that ends without an answer is a ToolError,
never an invented one.
"""

import json

from . import askuser
from .tools import ToolError, current_context, tool
from .turnlog import log_event

__all__ = ["ask_user"]

# Set by the panel (panel.py) at mount: ``surface(request)`` renders the decision card on the GUI
# thread and returns at once — the worker waits on the registry Event, not on this call.
surface = None

_QUESTION_SCHEMA = {"type": "object", "properties": {
    "title": {"type": "string", "description": "The question headline; required"},
    "subtitle": {"type": "string", "description": "One line of context or evidence for the question"},
    "mono_subtitle": {"type": "boolean", "default": False, "description": "Render the subtitle in monospace (paths, numbers)"},
    "options": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "object", "properties": {
        "label": {"type": "string", "description": "The option's short answer text"},
        "description": {"type": "string", "description": "What choosing it means"},
        "likeliest": {"type": "boolean", "default": False,
                      "description": "Mark EXACTLY ONE option — the evidence-supported default — and list it first"}},
        "required": ["label"]}},
    "custom_value": {"type": "object", "properties": {"label": {"type": "string"},
                                                      "unit": {"type": "string"}},
                     "description": "Offer a custom value entry with this label (and optional unit)"},
    "allow_free_text": {"type": "boolean", "default": False, "description": "Allow a typed free-text answer"},
}}

_SCHEMA = dict(_QUESTION_SCHEMA)
_SCHEMA["properties"] = dict(_QUESTION_SCHEMA["properties"])
_SCHEMA["properties"]["wizard"] = {
    "type": "array", "minItems": 2, "maxItems": 6,
    "description": "2 to 6 related questions of ONE decision stage, asked together on one card; "
                   "each item uses the single-question fields. Do not batch unrelated questions.",
    "items": _QUESTION_SCHEMA}


@tool(
    "ask_user",
    "Ask the user ONE decision you cannot resolve from evidence: a question with 2-4 mutually "
    "exclusive options (mark exactly one evidence-supported likeliest, listed first), or a wizard "
    "of 2-6 related questions of the same decision stage. Use file and conversation evidence "
    "first; never ask for what you can measure or infer; never re-ask an answered question unless "
    "the evidence changed. The result is the user's structured answer.",
    _SCHEMA,
    category="session",
)
def ask_user(title: str = "", subtitle: str = "", mono_subtitle: bool = False, options=None,
             custom_value=None, allow_free_text: bool = False, wizard=None):
    """Request the decision, wait for it on this worker thread, return the structured answer.

    Raises ``ToolError`` for a malformed question, a turn without a bound conversation, a card that
    could not be displayed, or a request that ended cancelled/timed out — no answer is invented."""
    ctx = current_context()
    payload = ({"wizard": wizard} if wizard is not None else
               {"title": title, "subtitle": subtitle, "mono_subtitle": mono_subtitle,
                "options": options or [], "custom_value": custom_value,
                "allow_free_text": allow_free_text})
    normalized, err = (askuser.validate_wizard(payload) if wizard is not None
                       else askuser.validate_question(payload))
    if normalized is None:
        raise ToolError(f"ask_user: {err}")
    if not ctx.conversation_id or not ctx.turn_id:
        raise ToolError("ask_user: no conversation is bound to this turn (the question cannot be routed back)")

    request = askuser.REGISTRY.open(normalized, conversation_id=ctx.conversation_id, run_id=ctx.turn_id)
    _display(request)
    settled = askuser.REGISTRY.wait(request, should_stop=ctx.should_stop)
    log_event("ask_return", request_id=request.id, state=settled.state)
    if settled.state != "answered":
        raise ToolError(f"ask_user: the request was {settled.state} before an answer arrived — "
                        f"nothing was recorded; re-ask only if the evidence changed")
    return _answer_text(normalized, settled.answer)


def _display(request: "askuser.PendingRequest") -> None:
    """Render the card on the GUI thread via the panel hook; headless (no hook) is not an error."""
    hook = surface
    if hook is None:
        return
    from .cad import dispatch   # imported here: the module itself stays importable without Qt
    try:
        dispatch.dispatch(lambda: hook(request), timeout=60.0, op="ask_user_card", reentrant=True)
    except Exception as e:  # noqa: BLE001 - an unrenderable question must not hang the turn
        askuser.REGISTRY.cancel_run(request.run_id)
        raise ToolError(f"ask_user: the decision card could not be displayed ({e})") from e


def _answer_text(question: dict, answer: dict) -> str:
    """The model-facing result: a readable summary line, then the structured answer as JSON."""
    if "wizard" in question:
        lines = [f"Q: {q['title']}" for q in question["wizard"]]
        chosen = [f"A: {a['answer']} (option {a['index']})" for a in answer["answers"]]
        summary = "\n".join(f"{q}\n{a}" for q, a in zip(lines, chosen))
    else:
        summary = f"{question['title']}\nA: {answer['answer']} (option {answer['index']})"
    return f"USER ANSWERED:\n{summary}\nstructured answer: {json.dumps(answer, ensure_ascii=False)}"
