"""Legacy pieces of the agent loop: the prompt aliases and the history-size helpers (pure Python).

Split out of ``oscar.agent`` (which re-exports every name here, so ``agent.PERSONA``, ``agent.compact_history``,
``agent.needs_execution_nudge`` ... keep working) to keep the loop module small. Nothing here touches FreeCAD or
Qt. ``needs_execution_nudge`` and ``compact_history`` serve the flow-off path (``OSCAR_FLOW=0``); ``PERSONA`` /
``STRATEGY`` are aliases of ``oscar.prompt``.
"""

import json
import re

from . import loopcore
from . import prompt as _prompt
from .providers import Message, ToolResult


# Safety valves: a runaway tool loop must not hammer an API forever, and one
# giant session must not eat the whole context window.
MAX_TOOL_ROUNDS = 30
MAX_HISTORY_MESSAGES = 80   # compaction trigger (checked at turn start)
KEEP_HISTORY_MESSAGES = 40  # what compaction trims down to (at most)
KEEP_SCREENSHOTS = 4        # screenshots that survive compaction
# Mid-turn size guard: one long build turn can outgrow the model's context
# window long before 80 messages. When the estimated transcript exceeds
# this many characters, older tool results are stubbed (structure intact —
# every tool call keeps a matching result, just short).
CONTEXT_CHAR_BUDGET = 200_000        # ~50k tokens across providers
TOOL_RESULT_KEEP_CHARS = 400         # what an old result keeps when stubbed
# An image costs about the same tokens (~1-1.5k for a viewport screenshot or
# a 1280 px photo) however many base64 characters carry it, so the estimate
# charges this fixed amount per image: counting len(base64) made one photo or
# two screenshots exceed the whole budget and wiped vision for the turn.
IMAGE_COST_CHARS = 6_000

# The prompt text lives in ``oscar.prompt``; these names stay for the tests and callers that read them.
# ``STRATEGY`` is the FULL core prompt (every conditional fragment, so the pinned literals ``EXECUTION CONTRACT``,
# ``never end the turn at``, ``HOUSING/ENCLOSURE`` and ``download_file`` are always present); ``PERSONA`` is its role
# paragraph. The loop itself sends ``prompt.build_system_prompt(...)`` (see ``oscar.agent_ctx``), never the two joined.
PERSONA = _prompt.CORE.split("\n\n", 1)[0]
STRATEGY = _prompt.CORE

_BUDGET_NOTE = loopcore.BUDGET_NOTE

_BAD_ARGUMENTS = (
    "Not executed: the arguments for this call were not valid JSON — the "
    "output was probably cut off by the output-token limit. Re-issue the call "
    "with complete arguments; split very long scripts into several smaller "
    "execute_code calls."
)

_TRIMMED_NOTE = "[Earlier conversation was trimmed to save context.]\n\n"

# A reasoning model's thinking counts against max_tokens. When it uses the
# whole budget before producing a tool call or an answer, the reply is
# empty with stop reason length/max_tokens. The turn is then nudged (at

MAX_CUTOFF_RETRIES = 2

_CUTOFF_PLACEHOLDER = (
    "(this reply was cut off at the output-token limit before it contained "
    "any answer or tool call)"
)

_CUTOFF_NOTE = loopcore.CUTOFF_NOTE


def _cut_off(stop_reason: str) -> bool:
    """Generation stopped at the output-token limit (either wire name)."""
    return loopcore.is_cut_off(stop_reason)


def _message_size(m: Message) -> int:
    total = len(m.content or "")
    for call in m.tool_calls:
        total += len(call.name or "") + len(json.dumps(call.arguments,
                                                       default=str))
    for result in m.tool_results:
        total += len(result.content or "")
        if result.image_b64:
            total += IMAGE_COST_CHARS
    total += IMAGE_COST_CHARS * sum(1 for b in m.images if b)
    return total


def stub_oversized_history(history: list[Message],
                           budget: int = CONTEXT_CHAR_BUDGET) -> bool:
    """Break up an oversized transcript MID-TURN without breaking it.

    The provider contract requires every tool call to keep its matching
    result, so nothing is dropped here: old tool results are truncated to
    TOOL_RESULT_KEEP_CHARS (with an elision marker), old screenshots are
    dropped (keeping the newest KEEP_SCREENSHOTS), and old plain-text
    messages are capped. The newest messages pass through untouched. Runs
    every round once the estimate crosses the budget, so a 30-round build
    turn cannot outgrow the window. Returns True when it stubbed anything.
    """
    if sum(_message_size(m) for m in history) <= budget:
        return False
    import copy as _copy

    boundary = len(history)
    acc = 0
    for i in range(len(history) - 1, -1, -1):
        acc += _message_size(history[i])
        if acc > budget:
            boundary = i + 1
            break
    changed = False
    # Image keep-set walks NEWEST-first (matching compact_history), so the
    # survivors are the freshest screenshots — counting oldest-first kept
    # stale images and dropped the ones the model just received.
    keep_images: set[int] = set()
    shots = 0
    for m in reversed(history):
        for result in m.tool_results:
            if result.image_b64:
                shots += 1
                if shots <= KEEP_SCREENSHOTS:
                    keep_images.add(id(result))
    for i, m in enumerate(history):
        if i >= boundary:
            continue  # newest region stays verbatim
        if m.tool_results:
            for result in m.tool_results:
                if len(result.content or "") > TOOL_RESULT_KEEP_CHARS:
                    result.content = (result.content[:TOOL_RESULT_KEEP_CHARS]
                                      + " …(elided to save context)")
                    changed = True
                if result.image_b64 and id(result) not in keep_images:
                    result.image_b64 = ""
                    changed = True
        elif len(m.content or "") > 2000 and m.role == "assistant":
            m.content = m.content[:2000] + " …(elided to save context)"
            changed = True

    # Meet the budget: if the policy-kept screenshots still push the
    # transcript over, shed them oldest-first (a fresh screenshot is
    # worth more than an old one, but none is worth blowing the window).
    # The newest one stays: text the stubber cannot shrink (the user's own
    # message) must not cost the model the screenshot it just asked for.
    def _total() -> int:
        return sum(_message_size(m) for m in history)

    if _total() > budget:
        kept_order = [r for m in reversed(history)
                      for r in m.tool_results if r.image_b64]
        for result in reversed(kept_order[1:]):  # oldest kept first out
            if _total() <= budget:
                break
            result.image_b64 = ""
            changed = True
            if id(result) in keep_images:
                keep_images.discard(id(result))
    return changed


# Words that make a request CAD WORK: if the turn produced no tool calls
# and the user's prompt contains one of these, the answer is nudged to
# actually execute instead of stopping at a plan.
_EXECUTION_WORDS = (
    "build", "create", "make", "model", "design", "draw", "sketch",
    "housing", "enclosure", "case", "bracket", "plate", "shaft", "part",
    "box", "shape", "geometry", "3d", "print", "cad", "dimension",
    "assembly", "fillet", "chamfer", "cut", "hole", "vent",
    "convert", "extrude", "pad", "pocket", "cutout", "cut-out", "redo",
)
# Short, ambiguous stems must be whole words (plain inflections allowed:
# "holes", "parts", "cutting") so "whole", "apart", "keypad" or "inventory"
# do not match; the longer stems match as word prefixes ("designing").
# Either kind may carry a "re"/"re-" prefix ("rebuild", "re-create",
# "redesign", "remodel"): the substring match this replaced caught those.
_SHORT_WORDS = {"3d", "cad", "cut", "pad", "box", "hole", "vent", "case",
                "part", "make", "plate", "shape", "print", "cutout", "cut-out",
                "redo"}
_WORD_RE = re.compile(
    r"\b(?:re-?)?(?:"
    + "|".join(re.escape(w) for w in _EXECUTION_WORDS if w not in _SHORT_WORDS)
    + r")|\b(?:re-?)?(?:"
    + "|".join(re.escape(w) for w in sorted(_SHORT_WORDS))
    + r")(?:s|es|ed|d|ing|ting)?\b")

_ATTACHMENTS_MARK = "\n\n--- attached files ---"  # attachments.build_user_message
# Notes the application puts in front of the user's text: the panel's pending
# revert notes (full and partial, panel._revert_turn) and compact_history's
# trimmed-conversation note.
_APP_NOTE_PREFIXES = (
    "[I reverted every CAD change",
    "[I tried to revert one of your earlier turns",
    _TRIMMED_NOTE.strip(),
)
# Every app note ends "]" + newline (or the end of the text); the partial
# revert note embeds the engine's free-form error text, which may hold ']'.
_APP_NOTE_END_RE = re.compile(r"\]\s*(?:\n|$)")


def _users_own_words(message: str) -> str:
    """What the user typed: the message minus the workspace block, inlined
    attachment text and the application's leading notes."""
    text = (message or "").split("<workspace_state>")[0]
    text = text.split(_ATTACHMENTS_MARK)[0]
    while True:
        stripped = text.lstrip()
        for prefix in _APP_NOTE_PREFIXES:
            if stripped.startswith(prefix):
                closing = _APP_NOTE_END_RE.search(stripped)
                end = closing.start() if closing else stripped.find("]")
                stripped = stripped[end + 1:] if end >= 0 else ""  # through the bracket
                break
        else:
            return stripped
        text = stripped


def needs_execution_nudge(user_text: str, tools_used_this_turn: int) -> bool:
    """True when a no-tool reply should be pushed to actually build.

    Questions ("how do I…?", short factual prompts) stay unanswered-by-
    tools on purpose; work requests must end in geometry. Only the user's
    own words count: a file they attached or a note the application added
    says nothing about what they asked for.

    Deprecated, kept for compatibility: superseded by the flow (``flow.py`` intent classifier + FinishGate), which
    decides from the typed request and the ledger, in any language. Reachable only with ``OSCAR_FLOW=0``.
    """
    if tools_used_this_turn > 0:
        return False
    lowered = _users_own_words(user_text).strip().lower()
    if not lowered:
        return False
    if lowered.endswith("?") and len(lowered) < 120:
        return False
    return _WORD_RE.search(lowered) is not None


_EXECUTION_NOTE = (
    "You described an approach but did not run a single tool this turn — "
    "no CAD work exists yet. Execute your plan NOW, step by step, in this "
    "same turn: one numbered step at a time, each carried out with the "
    "actual tools (create_document / create_object / edit_object / "
    "execute_code / get_view …), verifying with get_objects after each "
    "milestone. Do not stop at a plan, do not wait for confirmation, do "
    "not hand me instructions I could run myself. If a step is genuinely "
    "impossible, complete the steps that are possible and say exactly "
    "which step blocked and why."
)


def compact_history(
    history: list[Message],
    limit: int = MAX_HISTORY_MESSAGES,
    keep: int = KEEP_HISTORY_MESSAGES,
) -> bool:
    """Drop the oldest turns once the history outgrows ``limit``.

    Runs only at the start of a turn and cuts only at a real user message
    (never between a tool call and its result, never at an app-injected
    synthetic note such as a cut-off / gate / steer note), so the window
    never opens with an orphaned tool result and always starts with a user
    turn. The retained messages lose their thinking blocks (they were bound
    to the prefix that was just removed) and all but the newest screenshots.
    Mutates ``history`` in place; returns True when it trimmed.

    Deprecated, kept for compatibility: superseded by ``context.py`` (the pinned task memory and the derived,
    token-budgeted request view, which never rewrite the stored history). Still the trim of last resort for chats
    whose history outgrew ``limit``.
    """
    if len(history) <= limit:
        return False
    boundaries = [
        i for i, m in enumerate(history)
        if m.role == "user" and not m.tool_results and not getattr(m, "synthetic", False)
    ]
    cut = next((i for i in boundaries if len(history) - i <= keep), None)
    if cut is None:
        cut = boundaries[-1] if boundaries else 0
    if cut <= 0:
        return False
    del history[:cut]
    for message in history:
        if message.provider_blocks:
            message.provider_blocks = [
                b for b in message.provider_blocks
                if b.get("type") not in ("thinking", "redacted_thinking")
            ]
    shots = 0
    for message in reversed(history):
        for result in reversed(message.tool_results):
            if result.image_b64:
                shots += 1
                if shots > KEEP_SCREENSHOTS:
                    result.image_b64 = ""
    history[0].content = _TRIMMED_NOTE + (history[0].content or "")
    return True


def _keep_last_screenshot(results: list[ToolResult]) -> None:
    """Keep one screenshot per round: the last one shows the model after
    every call in the round, so earlier ones are redundant tokens (and the
    number of images per request is capped by the providers)."""
    with_images = [r for r in results if r.image_b64]
    for result in with_images[:-1]:
        result.image_b64 = ""
        result.content += "\n(screenshot omitted — the last call of this step shows the model)"


def _summarize_arguments(name: str, arguments: dict) -> str:
    """One-line argument summary for the activity log."""
    keys = {
        "create_document": ["name"],
        "create_object": ["doc_name", "obj_type", "obj_name"],
        "edit_object": ["doc_name", "obj_name"],
        "delete_object": ["doc_name", "obj_name"],
        "get_objects": ["doc_name"],
        "get_object": ["doc_name", "obj_name"],
        "get_view": ["view_name", "focus_object"],
        "insert_part_from_library": ["relative_path"],
        "import_file": ["path"],
        "reload_document": ["doc_name", "discard_unsaved"],
        "run_fem_analysis": ["doc_name", "analysis_name"],
    }
    arguments = arguments if isinstance(arguments, dict) else {}
    parts = []
    for key in keys.get(name, [k for k in arguments if k != "_raw"][:3]):
        value = arguments.get(key)
        if value is not None:
            text = json.dumps(value) if not isinstance(value, str) else value
            parts.append(f"{key}={text[:60]}")
    return ", ".join(parts) if parts else ""
