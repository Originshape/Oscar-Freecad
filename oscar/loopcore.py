"""Shared agent-loop helpers (pure Python: no FreeCAD, no Qt, no ``oscar.tools`` / ``oscar.project``).

Everything here is used by ``oscar.agent``:

* the three fixed notes the app injects into a conversation (cut-off, tool budget, watchdog),
* the stop-reason helpers (one mapping, defined in ``oscar.stopreason``),
* ``call_chat``: calls ``provider.chat`` with only the options that provider accepts,
* ``real_user_request``: reduces the stored user message to what the user actually typed,
* small utilities (``make_note``, ``args_hash``, ``clip``).

The module imports ``oscar.providers`` lazily (``make_note`` only) so it stays importable in any interpreter.
"""

import hashlib
import inspect
import json
import re
import uuid

from .stopreason import is_cut_off, normalize_stop_reason  # noqa: F401  (re-exported on purpose)

__all__ = [
    "CUTOFF_NOTE", "BUDGET_NOTE", "WATCHDOG_NOTE", "NO_TEXT_PLACEHOLDER", "TOKEN_CUT_NOTE", "CREDITS_CUT_NOTE",
    "CREDITS_EMPTY_MESSAGE", "shortened_for_credits",
    "normalize_stop_reason", "is_cut_off", "args_problem", "bad_arguments_message", "call_chat",
    "is_synthetic", "real_user_request", "make_note", "args_hash", "clip", "first_line", "unique_call_ids",
]

# Appended to a reply cut at its output cap (agent._cut_note): the model's own cap, or the lower one OriginShape set
# because the account's credits are low (``x-oscar-normalized`` says ``credits``; audit 2026-10-28: that was blamed on
# the model's output budget).
TOKEN_CUT_NOTE = "\n\n(The reply was cut off at the output-token limit.)"
CREDITS_CUT_NOTE = ("\n\n(The reply was cut off: your OriginShape credits are low, so the reply was shortened to what "
                    "they can pay for.)")
CREDITS_EMPTY_MESSAGE = ("Your OriginShape credits are low, so the reply was shortened to what they can pay for, and "
                         "the model spent all of it on thinking. Add credits or choose a plan, or start a new chat: a "
                         "shorter chat needs fewer credits.")


def shortened_for_credits(provider: object) -> bool:
    """True when OriginShape lowered the last reply's output budget because the account's credits are low."""
    return "credits" in str(getattr(provider, "last_normalized", "") or "").split(",")


# What the app says when a reasoning model spent the whole output budget on thinking (agent._CUTOFF_NOTE).
CUTOFF_NOTE = (
    "[Your previous reply hit the output-token limit while you were still "
    "thinking, so it contained no answer and no tool call. Do not repeat that "
    "deliberation. Decide now and act: issue the tool calls for the next step "
    "(several in one response is fine), or reply briefly.]"
)

# Appended to the results message of the last allowed tool round (agent._BUDGET_NOTE; the words "Tool budget" are pinned by a test).
BUDGET_NOTE = (
    "[Tool budget for this turn is used up. Do not call any more tools. Reply "
    "now with a short summary: what was done, what is still missing, and how "
    "the user can continue.]"
)

# Template: the app closed a reply that thought too long. Fields: {chars} {secs} {now}.
WATCHDOG_NOTE = (
    "[The app stopped your previous reply after {chars} characters of thinking / {secs} seconds without a tool call. "
    "Do not deliberate again. Act now: make the tool call for the next step ({now}). "
    "Keep thinking under 300 words per round.]"
)

# Assistant content the app puts in front of a synthetic user note when the model's own reply had no text,
# so roles keep alternating.
NO_TEXT_PLACEHOLDER = "(the model replied without text)"

_TRUNCATED_GUIDANCE = (
    "Your output was cut off at the output-token limit. Re-issue the call with complete arguments and split "
    "long scripts into several smaller execute_code calls (under 40 lines each)."
)
_MALFORMED_GUIDANCE = (
    "Fix the JSON (double quotes, no trailing commas, newlines escaped as \\n) and re-issue the call."
)


def args_problem(call) -> str:
    """Why a tool call's arguments could not be used: ``""`` when they decoded fine.

    Works for old call objects that only carry the legacy ``{"_raw": ...}`` marker (no ``args_error`` field).
    """
    error = getattr(call, "args_error", "") or ""
    if error:
        return error
    arguments = getattr(call, "arguments", None)
    if isinstance(arguments, dict) and "_raw" in arguments:
        return "arguments could not be decoded"
    return ""


def bad_arguments_message(call, stop_reason: str = "") -> str:
    """The model-facing result for a call whose arguments were not decodable (always contains ``not valid JSON``).

    The guidance depends on the cause: output cut off at the token limit (``args_error`` starts with
    ``truncated`` or the reply's stop reason is a cut-off) versus plain malformed JSON.
    """
    problem = args_problem(call) or "arguments could not be decoded"
    truncated = problem.startswith("truncated") or is_cut_off(stop_reason)
    guidance = _TRUNCATED_GUIDANCE if truncated else _MALFORMED_GUIDANCE
    return f"Not executed: the arguments for this call were not valid JSON ({problem}). {guidance}"


def call_chat(provider, system: str, messages: list, specs: list, should_stop, **opts):
    """Generator over ``provider.chat(...)`` that passes only the options the provider's signature accepts.

    ``opts`` with a ``None`` value are dropped (None means "provider default"). Old scripted providers have
    exactly the four legacy parameters, so ``tool_choice`` / ``reasoning`` / ``max_tokens`` /
    ``parallel_tool_calls`` silently do not reach them. Closing this generator closes the provider's.
    """
    wanted = {k: v for k, v in opts.items() if v is not None}
    if wanted:
        try:
            params = inspect.signature(provider.chat).parameters
        except (TypeError, ValueError):
            params = {}
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            wanted = {k: v for k, v in wanted.items() if k in params}
    return provider.chat(system, messages, specs, should_stop, **wanted)


def unique_call_ids(calls, used: set, blocks=None) -> int:
    """Make the ids of one reply's tool calls unique, IN PLACE; returns how many were renamed.

    A tool result is paired with its call by id, so two calls with one id (a local server that numbers its calls
    ``0`` in every reply, a model that copies an id, an empty id) leave one result unmatched and every later request is
    rejected by strict hosts (Anthropic: ``tool_use ids must be unique``; OpenAI: a ``tool_call_id`` answered twice).
    ``used`` is the set of ids the conversation already holds: an id in it, or an empty id, gets a fresh one
    (``<id>_2``, ``<id>_3``, ... / ``call_<hex>``). ``used`` is extended with every id of ``calls``. ``blocks`` (the
    reply's provider blocks) are patched too, in order, so a replayed ``tool_use`` block keeps naming the id its
    result carries.
    """
    by_old: dict[str, list[str]] = {}     # the id a call arrived with -> the ids its occurrences end up with, in order
    count = 0
    for tool_call in calls or ():
        old = str(getattr(tool_call, "id", "") or "")
        new = old
        if not new or new in used:
            if not old:
                new = f"call_{uuid.uuid4().hex[:12]}"
            else:
                number = 2
                new = f"{old}_{number}"
                while new in used:
                    number += 1
                    new = f"{old}_{number}"
            tool_call.id = new
            count += 1
        used.add(new)
        by_old.setdefault(old, []).append(new)
    if count and blocks:
        for index, block in enumerate(blocks):
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            queue = by_old.get(str(block.get("id") or ""))
            if queue:
                chosen = queue.pop(0)
                if chosen != block.get("id"):
                    blocks[index] = dict(block, id=chosen)
    return count


def is_synthetic(message) -> bool:
    """True for a message the app injected (cut-off / gate / watchdog / breaker / steer / wrap-up / resume note)."""
    return bool(getattr(message, "synthetic", False))


_ATTACH_MARK = "\n\n--- attached files ---"
_STATE_BLOCK = re.compile(r"<workspace_state>.*?(?:</workspace_state>|\Z)", re.S)
_APP_LINE = re.compile(r"^\s*\[(?:governor|phase)\].*$", re.M)


def _is_note_paragraph(paragraph: str) -> bool:
    text = paragraph.strip()
    return text.startswith("[") and text.endswith("]")


def _typed_text(content: str) -> str:
    """The part of one stored user message the user typed (5.2 input rule)."""
    text = content or ""
    cut = text.find(_ATTACH_MARK)
    if cut >= 0:
        text = text[:cut]
    text = _STATE_BLOCK.sub("", text)
    text = _APP_LINE.sub("", text)
    paragraphs = re.split(r"\n[ \t]*\n", text.strip())
    # Leading "[...]" notes (revert / rewind notes the panel prepends); a lone bracketed paragraph is the user's own text.
    while len(paragraphs) > 1 and _is_note_paragraph(paragraphs[0]):
        paragraphs.pop(0)
    return "\n\n".join(paragraphs).strip()


def real_user_request(history: list) -> str:
    """What the user typed in the latest real user turn ("" when there is none).

    The newest user message that is not synthetic and carries no tool results, reduced to the typed text:
    cut at ``--- attached files ---``, leading ``[...]`` note paragraphs dropped, ``<workspace_state>``
    blocks and ``[governor]`` / ``[phase]`` lines removed. The classifier and the requirement extraction
    must never see notes or attachment text (a revert note contains the word "change").
    """
    for message in reversed(history or []):
        if getattr(message, "role", "") != "user" or getattr(message, "tool_results", None) or is_synthetic(message):
            continue
        typed = _typed_text(getattr(message, "content", "") or "")
        if typed:
            return typed
    return ""


def make_note(text: str, tag: str):
    """A synthetic user message: the app's own note to the model (``synthetic=True``, ``tag`` names the mechanism)."""
    from .providers import Message

    return Message(role="user", content=text, synthetic=True, tag=tag)


def args_hash(name: str, args: dict) -> str:
    """Stable 12-hex identity of a call (tool name + arguments), used to detect an identical repeated call."""
    try:
        blob = json.dumps(args, sort_keys=True, default=str)[:20000]
    except (TypeError, ValueError):
        blob = repr(args)[:20000]
    return hashlib.sha1(f"{name}\x00{blob}".encode("utf-8", "replace")).hexdigest()[:12]


def clip(text: str, limit: int = 12000, head: float = 0.67) -> str:
    """Head + elision marker + tail, at most about ``limit`` characters of the original kept."""
    text = text if isinstance(text, str) else str(text)
    if limit <= 0 or len(text) <= limit:
        return text
    head_n = max(1, int(limit * head))
    tail_n = max(0, limit - head_n)
    omitted = len(text) - head_n - tail_n
    tail = text[-tail_n:] if tail_n else ""
    return f"{text[:head_n]}\n...({omitted} chars omitted)...\n{tail}"


def first_line(text: str, limit: int = 160) -> str:
    """The first non-empty line of ``text``, clipped to ``limit`` characters."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= limit else line[: limit - 1] + "…"
    return ""
