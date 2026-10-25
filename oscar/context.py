"""Context management for the agent loop: token estimates, the pinned task memory and the derived request view.

Pure Python: no FreeCAD and no Qt at import time (``workspace_snapshot`` imports FreeCAD lazily, inside the
function, and must be called on the GUI thread).

The stored conversation (``history``, a list of ``providers.Message``) is APPEND-ONLY from the point of view of this
module: nothing here ever mutates it. Each round the agent calls ``build_request_view`` and sends the returned copy.
When the estimated prompt exceeds the budget the view is shrunk in a fixed order (old script arguments, thinking of
earlier turns, old tool results, old screenshots, then whole old rounds), while the pinned task memory
(``pinned_memory``: the request, the plan, the parameters, the objects, the sourcing outcome) stands in for what was
dropped. The memory text changes only when the plan, the parameters or the objects really change, never per round, so
the prefix that OpenAI-compatible hosts cache from the start of the prompt stays valid.

Public names: ``MemoryInputs``, ``pinned_memory``, ``estimate_tokens_text``, ``estimate_tokens``,
``round_boundaries``, ``elide_script_args``, ``script_paths_from_steps``, ``alternation_problems``,
``assert_alternating``, ``build_request_view``, ``too_small_notice``, ``workspace_snapshot`` and the constants
``ULTRA_LITE_TOOLS`` / ``ULTRA_LITE_MEMORY_CHARS`` the agent uses for the floor-rule degradation.
"""

import copy
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import context_guard as _context_guard
from .providers import MAX_WIRE_IMAGES, Message, ToolCall, ToolResult

__all__ = ["MemoryInputs", "pinned_memory", "estimate_tokens_text", "estimate_tokens", "round_boundaries",
           "elide_script_args", "script_paths_from_steps", "alternation_problems", "assert_alternating",
           "build_request_view", "too_small_notice", "workspace_snapshot", "ULTRA_LITE_TOOLS",
           "ULTRA_LITE_MEMORY_CHARS"]

# -- constants -------------------------------------------------------------
CHARS_PER_TOKEN = 3.5
ARGS_CHARS_PER_TOKEN = 3.0
IMAGE_TOKENS = 1500
MESSAGE_OVERHEAD_TOKENS = 8
DEFAULT_CONTEXT_WINDOW = 32768
BUDGET_FRACTION = 0.70
OUTPUT_RESERVE_CAP = 8192
FLOOR_FRACTION = 0.9
CALIBRATION_RANGE = (0.5, 3.0)
NEWEST_ROUNDS_PROTECTED = 2
ELIDE_MIN_CHARS = 600
RESULT_KEEP_HEAD = 300
RESULT_KEEP_TAIL = 100
RESULT_ELISION_MARK = " ...(elided)... "
MEMORY_MIN_CHARS = 1200
ULTRA_LITE_MEMORY_CHARS = 1200
DEFAULT_MEMORY_CHARS = 6000
REQUEST_HARD_CAP = 16000
SCRIPT_TOOLS = ("execute_code", "execute_code_headless", "execute_code_async")
# The tools an ultra-lite request offers; ``build_enclosure`` / ``enclosure_spec`` only when offered.
ULTRA_LITE_TOOLS = ("get_objects", "get_view", "create_object", "edit_object", "execute_code", "plan_update",
                    "build_enclosure", "enclosure_spec", "import_file")
_THINKING_TYPES = ("thinking", "redacted_thinking")
_OMITTED_NOTE = "[Earlier conversation was omitted to save context.]"
_APP_LINE_PREFIXES = ("[governor]", "[phase]")


# -- token estimates -----------------------------------------------------------------------------------------------

def estimate_tokens_text(text: str) -> int:
    """Estimated tokens of ``text``: ``ceil(len(text) / 3.5)`` (0 for empty or non-text input)."""
    if not text or not isinstance(text, str):
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _text_raw(text: Any) -> float:
    return len(text) / CHARS_PER_TOKEN if isinstance(text, str) and text else 0.0


def _args_raw(arguments: Any) -> float:
    try:
        return len(json.dumps(arguments, default=str, ensure_ascii=False)) / ARGS_CHARS_PER_TOKEN
    except (TypeError, ValueError):
        return len(str(arguments)) / ARGS_CHARS_PER_TOKEN


def _spec_raw(spec: Any) -> float:
    """Raw tokens of one tool spec: ``len(json.dumps(parameters)) / 3.5 + len(description) / 3.5``."""
    if isinstance(spec, Mapping):
        desc, params = spec.get("description"), spec.get("parameters", spec.get("input_schema"))
    else:
        desc, params = getattr(spec, "description", ""), getattr(spec, "parameters", None)
    try:
        schema = len(json.dumps(params, default=str, ensure_ascii=False)) if params is not None else 0
    except (TypeError, ValueError):
        schema = len(str(params))
    return (schema + len(desc or "")) / CHARS_PER_TOKEN


def _tools_raw(tools: Iterable[Any]) -> float:
    return sum(_spec_raw(t) for t in (tools or ()))


def _block_raw(block: Any) -> float:
    """Raw tokens of a provider block that is NOT already counted through ``content`` / ``tool_calls`` (thinking)."""
    if not isinstance(block, Mapping):
        return 0.0
    kind = block.get("type")
    if kind == "thinking":
        return _text_raw(block.get("thinking"))
    if kind == "redacted_thinking":
        return _text_raw(block.get("data"))
    return 0.0


def _message_raw(message: Any) -> float:
    """Uncalibrated estimate of one message: 8 overhead, text / 3.5, call arguments / 3.0, 1500 per image."""
    total = float(MESSAGE_OVERHEAD_TOKENS) + _text_raw(getattr(message, "content", ""))
    for call in getattr(message, "tool_calls", None) or ():
        total += _args_raw(call.arguments)
    for result in getattr(message, "tool_results", None) or ():
        total += _text_raw(result.content)
        if result.image_b64:
            total += IMAGE_TOKENS
    total += IMAGE_TOKENS * len(getattr(message, "images", None) or ())
    for block in getattr(message, "provider_blocks", None) or ():
        total += _block_raw(block)
    return total


def _raw_total(messages: Sequence[Any], system: str, tools: Iterable[Any]) -> float:
    return _text_raw(system) + _tools_raw(tools) + sum(_message_raw(m) for m in messages)


def _clip_k(value: float) -> float:
    low, high = CALIBRATION_RANGE
    return min(max(value, low), high)


def estimate_tokens(messages: list, system: str = "", tools: Sequence = (), *,
                    usage_prompt_tokens: int | None = None, since_index: int = 0) -> int:
    """Estimated prompt tokens of ``system`` + ``tools`` + ``messages``.

    Raw rule: text ``len / 3.5``, tool-call arguments ``len(json.dumps(args)) / 3.0``, every image a flat 1500,
    8 per message, thinking blocks by their text, tool specs ``(len(json.dumps(parameters)) + len(description)) / 3.5``.
    Calibration is MULTIPLICATIVE: when ``usage_prompt_tokens`` (the measured prompt tokens of an earlier request)
    is given, ``k = clip(usage / raw(system + tools + messages[:since_index]), 0.5, 3.0)`` and the result is
    ``k * raw(everything)``. ``since_index`` is the number of leading messages the usage figure covers; ``0`` (the
    default) means all of ``messages``, so the result is the measured figure itself (clipped to the 0.5 - 3.0 band).
    """
    raw = _raw_total(messages, system, tools)
    if usage_prompt_tokens and usage_prompt_tokens > 0:
        covered = messages if not since_index or since_index >= len(messages) else messages[:max(0, since_index)]
        base = _raw_total(covered, system, tools)
        if base > 0:
            raw *= _clip_k(usage_prompt_tokens / base)
    return math.ceil(raw)


# -- message helpers -----------------------------------------------------------------------------------------------

def _is_synthetic(message: Any) -> bool:
    return bool(getattr(message, "synthetic", False))


def _is_real_user(message: Any) -> bool:
    """A user message the user wrote (not a tool-results message, not an app note)."""
    return message.role == "user" and not message.tool_results and not _is_synthetic(message)


def _copy_call(call: Any) -> Any:
    clone = copy.copy(call)
    if isinstance(call.arguments, dict):
        clone.arguments = dict(call.arguments)
    return clone


def _copy_message(message: Any) -> Any:
    """A copy whose lists, calls and results are new objects (strings are shared), so neither this module nor a provider
    adapting the request in place can ever touch the stored history."""
    clone = copy.copy(message)
    clone.tool_calls = [_copy_call(c) for c in message.tool_calls or ()]
    clone.tool_results = [copy.copy(r) for r in message.tool_results or ()]
    clone.images = list(message.images or [])
    clone.provider_blocks = list(message.provider_blocks or [])
    return clone


def round_boundaries(history: list) -> list[int]:
    """The indices where whole rounds can be cut: ``i`` with ``history[i]`` an assistant message and ``history[i-1]`` a user message.

    A cut before ``history[i]`` keeps every assistant tool-call message together with its results message and never
    leaves a synthetic note without the assistant message it pairs with (a pair is always assistant first, note
    second, so the prefix cut removes both or neither). A synthetic message is never a cut point.
    """
    return [i for i in range(1, len(history))
            if history[i].role == "assistant" and history[i - 1].role == "user" and not _is_synthetic(history[i])]


def _line_count(code: str) -> int:
    return max(1, len(code.splitlines()))


def _script_marker(code: str, saved_path: str | None) -> str:
    lines = _line_count(code)
    if saved_path:
        return f"[script elided: {lines} lines, saved as {saved_path}]"
    return f"[script elided: {lines} lines]"


def _elided_code(call: Any, saved_path: str | None) -> str | None:
    """The marker that replaces ``call``'s ``code`` argument, or None when the call is not an elidable script."""
    if getattr(call, "name", "") not in SCRIPT_TOOLS or not isinstance(call.arguments, dict):
        return None
    code = call.arguments.get("code")
    if not isinstance(code, str) or len(code) <= ELIDE_MIN_CHARS:
        return None
    return _script_marker(code, saved_path)


def elide_script_args(call, saved_path: str | None) -> "ToolCall":
    """A COPY of ``call`` whose ``code`` argument (``execute_code`` / ``execute_code_headless`` / ``execute_code_async``, longer than 600 characters) is replaced by ``[script elided: NN lines, saved as <saved_path>]``.

    ``saved_path`` is the project-relative path of the saved script (``scripts/007_shell.py``); without it the marker is
    ``[script elided: NN lines]``. Any other call, a short script or call with undecodable arguments is returned as an
    unchanged copy. The original is never modified.
    """
    clone = copy.copy(call)
    if isinstance(call.arguments, dict):
        clone.arguments = dict(call.arguments)
        marker = _elided_code(call, saved_path)
        if marker is not None:
            clone.arguments["code"] = marker
    return clone


def _elide_in_blocks(blocks: list, call_id: str, marker: str) -> list:
    """A copy of ``blocks`` where the ``tool_use`` of ``call_id`` carries ``marker`` as its code."""
    out = []
    for block in blocks:
        if not isinstance(block, dict):
            out.append(block)
            continue
        kind = block.get("type")
        if kind == "tool_use" and block.get("id") == call_id and isinstance(block.get("input"), dict) \
                and isinstance(block["input"].get("code"), str):
            block = dict(block)
            block["input"] = {**block["input"], "code": marker}
        out.append(block)
    return out


def script_paths_from_steps(steps: Iterable[Any]) -> dict[str, str]:
    """``{tool call id: project-relative script path}`` from Project step records (``call_id`` and ``script`` attributes).

    Superseded records and records without a script are skipped. The agent passes the result as ``script_paths`` to
    ``build_request_view`` so an elided script points at the file the app saved.
    """
    paths: dict[str, str] = {}
    for step in steps or ():
        call_id = getattr(step, "call_id", "")
        script = getattr(step, "script", "")
        if call_id and script and not getattr(step, "superseded", False):
            paths[str(call_id)] = str(script)
    return paths


# -- alternation ---------------------------------------------------------------------------------------------------

def alternation_problems(view: Sequence[Any]) -> list[str]:
    """What is wrong with ``view`` as a request: empty, not starting with a plain user message, two consecutive assistant messages, an unanswered or orphaned tool call, a trailing assistant message.

    Allowed: consecutive user messages (the tool-results message, a screenshot message and a content message follow each
    other exactly as the providers already emit them).
    """
    if not view:
        return ["the request view is empty"]
    problems: list[str] = []
    first = view[0]
    if first.role != "user" or first.tool_results:
        problems.append("the view must start with a user message that carries no tool results")
    for i in range(1, len(view)):
        if view[i].role == "assistant" and view[i - 1].role == "assistant":
            problems.append(f"messages {i - 1} and {i} are both assistant messages")
    for i, message in enumerate(view):
        if message.role == "assistant" and message.tool_calls:
            nxt = view[i + 1] if i + 1 < len(view) else None
            if nxt is None or nxt.role != "user":
                problems.append(f"assistant message {i} has tool calls but no results message follows")
            else:
                answered = {r.tool_call_id for r in nxt.tool_results}
                missing = [c.id for c in message.tool_calls if c.id not in answered]
                if missing:
                    problems.append(f"assistant message {i} has unanswered tool calls {missing}")
        if message.role == "user" and message.tool_results:
            prev = view[i - 1] if i else None
            known = {c.id for c in prev.tool_calls} if prev is not None and prev.role == "assistant" else set()
            orphan = [r.tool_call_id for r in message.tool_results if r.tool_call_id not in known]
            if orphan:
                problems.append(f"user message {i} carries results {orphan} without their tool calls")
    if view[-1].role == "assistant":
        problems.append("the request must end with a user message")
    return problems


def assert_alternating(view: Sequence[Any]) -> None:
    """Raise ``AssertionError`` naming every problem of ``alternation_problems(view)``; return None for a valid view."""
    problems = alternation_problems(view)
    if problems:
        raise AssertionError("; ".join(problems))


# -- pinned task memory --------------------------------------------------------------------------------------------

@dataclass
class MemoryInputs:
    """What ``pinned_memory`` renders; built by the agent on the worker thread from state it already holds (no GUI access)."""

    request: str                      # the original request, verbatim (turn 1 of the task)
    ledger: dict | None               # C5 dict
    params: list[dict]                # read_param_table rows
    objects: list[dict]               # [{"name","type","solids","volume","doc"}]
    reference: dict | None            # provenance["reference"]
    files: list[str]                  # attachment / download / project paths worth remembering
    decisions: list[str]              # record_note(kind="decision") texts and gate waivers
    project_dir: str = ""
    turn_index: int = 1


_PINNED_NAMES = ("Params", "Parameters", "NOTES", "PLAN")
_ICONS = {"done": "[x]", "running": "[~]", "pending": "[ ]", "blocked": "[!]", "skipped": "[-]", "changed": "[=]"}
_TERMINAL = ("done", "blocked", "skipped", "changed")
_CONTAINER_TYPES = ("App::Part", "PartDesign::Body", "App::DocumentObjectGroup")


def _clip_text(text: Any, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: max(0, limit - 1)] + "…"


def _fmt_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value == int(value) and abs(value) < 1e9:
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def _param_item(row: Mapping) -> str:
    name = str(row.get("name") or "").strip()
    if not name:
        return ""
    value = _fmt_number(row.get("value", ""))
    unit = str(row.get("unit") or "").strip()
    if unit and unit not in value:
        value = f"{value} {unit}"
    source = str(row.get("source") or "").strip()
    return f"{name}={value}" + (f" ({source})" if source else "")


def _object_desc(row: Mapping) -> str:
    parts: list[str] = []
    solids = row.get("solids")
    volume = row.get("volume")
    if isinstance(solids, (int, float)) and not isinstance(solids, bool) and solids > 0:
        parts.append("solid" if solids == 1 else f"{int(solids)} solids")
        if isinstance(volume, (int, float)) and not isinstance(volume, bool) and volume:
            parts.append(f"{_fmt_number(round(volume))} mm3")
    elif row.get("type"):
        parts.append(str(row["type"]))
    if row.get("hidden") or row.get("visible") is False:
        parts.append("hidden")
    if row.get("valid") is False:
        parts.append("INVALID")
    return ", ".join(parts)


def _object_item(row: Mapping) -> str:
    name = str(row.get("name") or "").strip()
    if not name:
        return ""
    if name in _PINNED_NAMES:
        return name
    desc = _object_desc(row)
    return f"{name} [{desc}]" if desc else name


def _object_sections(objects: Sequence[Mapping]) -> list[tuple[str, list[str]]]:
    """``[(doc name, items)]``: pinned objects (Params / NOTES / PLAN) first within each document, order otherwise kept."""
    docs: dict[str, list[Mapping]] = {}
    for row in objects or ():
        if isinstance(row, Mapping) and row.get("name"):
            docs.setdefault(str(row.get("doc") or ""), []).append(row)
    sections = []
    for doc, rows in docs.items():
        ordered = [r for r in rows if r["name"] in _PINNED_NAMES] + [r for r in rows if r["name"] not in _PINNED_NAMES]
        sections.append((doc, [i for i in (_object_item(r) for r in ordered) if i]))
    return sections


class _Items:
    """One memory line made of ``header`` and ``items`` that can be shortened to ``n`` items plus a ``+N more`` tail."""

    def __init__(self, header: str, items: list[str], hint: str = "", cap: int | None = None):
        self.header, self.items, self.hint = header, items, hint
        self.n = len(items) if cap is None else min(cap, len(items))

    def render(self, n: int | None = None) -> str:
        count = self.n if n is None else n
        shown = self.items[:count]
        line = self.header + " | ".join(shown)
        if count < len(self.items):
            tail = f"... +{len(self.items) - count} more" + (f": {self.hint}" if self.hint else "")
            line += (" | " if shown else "") + tail
        return line


def _current_step(steps: Sequence[Mapping]) -> Mapping | None:
    """The running step, else the first non-terminal required step (the ledger's ``current_step`` rule)."""
    return next((s for s in steps if s.get("status") == "running"), None) or next(
        (s for s in steps if s.get("status") not in _TERMINAL and s.get("required", True)), None)


def _plan_fallback(ledger: Mapping) -> list[str]:
    """Plan and requirement lines rendered from the ledger dict (used when ``flow`` cannot render it)."""
    steps = [s for s in ledger.get("steps") or () if isinstance(s, Mapping)]
    current = _current_step(steps)
    lines = []
    for s in steps:
        line = f"{_ICONS.get(s.get('status'), '[ ]')} {s.get('id', '?')} {s.get('title', '')}"
        note = s.get("note") or ""
        if s.get("status") in ("done", "skipped") and note:
            line += f" - {note}"
        elif s.get("status") in ("blocked", "changed"):
            line += f" - {s.get('status')}: {note}" if note else f" - {s.get('status')}"
        if current is s:
            line += "   <- now"
        lines.append(line)
    reqs = []
    for r in ledger.get("requirements") or ():
        if isinstance(r, Mapping):
            covered = ",".join(r.get("covered_by") or ()) or "no step"
            reqs.append(f"{r.get('id', '?')} {r.get('status', 'pending')} ({covered})")
    if reqs:
        lines.append("REQUIREMENTS: " + " | ".join(reqs))
    return lines


def _plan_lines(ledger: Mapping, max_lines: int = 16) -> list[str]:
    """The plan lines of ``Ledger.render_compact`` without its header (``flow`` renders; a local fallback covers failures)."""
    try:
        from . import flow

        text = flow.Ledger.from_dict(copy.deepcopy(dict(ledger))).render_compact(max_lines)
        lines = text.split("\n")[1:]
        if lines or not ledger.get("steps"):
            return lines
    except Exception:  # noqa: BLE001 - a malformed ledger must never break the memory
        pass
    return _plan_fallback(ledger)


def _compact_plan(ledger: Mapping) -> str:
    """``PLAN: S1 [x] | S5 [~] <- now | ...``: ids and states only (the tight-budget form)."""
    steps = [s for s in ledger.get("steps") or () if isinstance(s, Mapping)]
    current = _current_step(steps)
    items = []
    for s in steps:
        item = f"{s.get('id', '?')} {_ICONS.get(s.get('status'), '[ ]')}"
        items.append(item + (" <- now" if s is current else ""))
    return "PLAN: " + " | ".join(items) if items else ""


def _reference_line(reference: Mapping | None) -> str:
    if not isinstance(reference, Mapping):
        return ""
    entry = reference.get("entry") if isinstance(reference.get("entry"), Mapping) else {}
    status = str(reference.get("status") or "").strip()
    ident = str(reference.get("id") or reference.get("name") or "").strip()
    if not ident and status in ("", "none"):
        return ""
    source = str(reference.get("source") or entry.get("source") or "n/a")
    sha = str(reference.get("sha256") or entry.get("sha256") or "")
    return (f"REFERENCE: {ident or 'none'} · status {status or 'none'} · source {_clip_text(source, 120)} · "
            f"sha256 {sha[:12] if sha else 'n/a'}")


def pinned_memory(inp: MemoryInputs, *, max_chars: int = DEFAULT_MEMORY_CHARS, tool_names: Iterable[str] | None = None) -> str:
    """The pinned task memory: the request, plan, requirements, parameters, objects, reference, files and decisions.

    Exact skeleton (sections whose data is empty are omitted)::

        <task_memory>
        REQUEST (verbatim, turn 1):  ...
        INTENT: enclosure · PHASE: BUILD / PLAN: ... / REQUIREMENTS: ... / PARAMETERS (sheet 'Params'): ...
        OBJECTS (doc 'X'): ... / REFERENCE: ... / FILES: ... / DECISIONS: ...
        </task_memory>

    It carries NO round counter and NO ledger revision (those ride in the trailing ``[governor]`` line), so the text
    changes only when the plan, the parameters or the objects really change. When the text exceeds ``max_chars`` the
    OBJECTS line is shortened first, then PARAMETERS, DECISIONS, FILES, then the plan collapses to ids and states; the
    REQUEST is never shortened (beyond a 16000 character hard cap that only a pasted document can reach). The
    parameter / object ``+N more`` hints name ``get_parameters`` only when ``tool_names`` is None or contains it.
    The result is deterministic for equal inputs. Pass ``max_chars=1200`` for the ultra-lite degradation.
    """
    names = None if tool_names is None else frozenset(tool_names)
    request = (inp.request or "").strip()
    if len(request) > REQUEST_HARD_CAP:
        request = request[:REQUEST_HARD_CAP] + f"\n[request clipped: {len(inp.request.strip()) - REQUEST_HARD_CAP} more characters]"
    head = [f"REQUEST (verbatim, turn {int(inp.turn_index or 1)}):", request or "(none)"]
    ledger = inp.ledger if isinstance(inp.ledger, Mapping) else None
    plan_lines: list[str] = []
    req_line = ""
    if ledger is not None:
        head.append(f"INTENT: {ledger.get('intent') or 'other'} · PHASE: {ledger.get('phase') or 'INTAKE'}")
        for line in _plan_lines(ledger):
            if line.startswith("REQUIREMENTS: "):
                req_line = line
            else:
                plan_lines.append(line)
    params_hint = "get_parameters" if names is None or "get_parameters" in names else "see the Params sheet"
    params = _Items("PARAMETERS (sheet 'Params'): ",
                    [i for i in (_param_item(r) for r in inp.params or () if isinstance(r, Mapping)) if i],
                    params_hint, cap=24)
    objects: list[_Items] = []
    for doc, items in _object_sections(inp.objects):
        header = f"OBJECTS (doc '{doc}'): " if doc else "OBJECTS: "
        objects.append(_Items(header, items, "get_objects", cap=40))
    reference = _reference_line(inp.reference)
    file_items = ([f"project {inp.project_dir}"] if inp.project_dir else []) + \
        [f"file {f}" for f in dict.fromkeys(str(x) for x in inp.files or () if x)]
    files = _Items("FILES: ", file_items)
    decisions = _Items("DECISIONS: ", [_clip_text(d, 160) for d in (inp.decisions or ()) if str(d or "").strip()],
                       cap=12)

    def assemble(compact: bool = False) -> str:
        lines = list(head)
        if ledger is not None and compact:
            plan = _compact_plan(ledger)
            lines.extend([plan] if plan else [])
        elif ledger is not None:
            if plan_lines:
                lines.append("PLAN:")
                lines.extend(plan_lines)
            if req_line:
                lines.append(req_line)
        if params.items:
            lines.append(params.render())
        for sec in objects:
            if sec.items:
                lines.append(sec.render())
        if reference:
            lines.append(reference)
        if files.items:
            lines.append(files.render())
        if decisions.items:
            lines.append(decisions.render())
        return "<task_memory>\n" + "\n".join(lines) + "\n</task_memory>"

    text = assemble()
    for sec in [*objects, params, decisions, files]:
        if len(text) <= max_chars:
            break
        low, high = 0, sec.n
        while low < high:                      # the largest item count that fits (monotonic in n)
            mid = (low + high + 1) // 2
            sec.n = mid
            if len(assemble()) <= max_chars:
                low = mid
            else:
                high = mid - 1
        sec.n = low
        text = assemble()
    if len(text) > max_chars and ledger is not None:
        text = assemble(compact=True)
    return text


# -- derived request view ------------------------------------------------------------------------------------------

def _window(caps: Any) -> int:
    value = getattr(caps, "context_window", None) if caps is not None else None
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else DEFAULT_CONTEXT_WINDOW


def _max_output(caps: Any) -> int:
    value = getattr(caps, "max_output", None) if caps is not None else None
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else 4096


def too_small_notice(window: int) -> str:
    """The visible notice (and the ``failed`` turn text) for a context window too small for the task."""
    return (f"The context window of this model ({int(window)} tokens) is too small for this task; raise num_ctx / "
            "the context window to at least 16384 in the profile's Advanced settings.")


def _strip_app_lines(messages: list) -> None:
    """Remove ``[governor]`` / ``[phase]`` lines from every message except the newest tool-results message."""
    newest = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].tool_results), -1)
    for i, message in enumerate(messages):
        content = message.content or ""
        if i == newest or "[governor]" not in content and "[phase]" not in content:
            continue
        lines = content.split("\n")
        kept = [ln for ln in lines if not ln.lstrip().startswith(_APP_LINE_PREFIXES)]
        if len(kept) != len(lines):
            message.content = "\n".join(kept).rstrip()


def _request_from_memory(memory: str) -> str:
    match = re.search(r"REQUEST \(verbatim, turn \d+\):\n(.*?)\n(?:INTENT: |PLAN: |PLAN:\n|REQUIREMENTS: |PARAMETERS |OBJECTS|"
                      r"REFERENCE: |FILES: |DECISIONS: |</task_memory>)", memory or "", re.S)
    return match.group(1).strip() if match else ""


def _request_in_view(messages: Sequence[Any], memory: str) -> bool:
    """False only when the memory names a request and no retained real user message contains its first 200 characters."""
    probe = _request_from_memory(memory)[:200].strip()
    if not probe:
        return True
    return any(m.role == "user" and not m.tool_results and probe in (m.content or "") for m in messages)


def _typed_text(content: str) -> str:
    """What the user typed in a stored user message: the text before the attachment list and the workspace block."""
    text = content or ""
    for marker in ("\n\n--- attached files ---", "<workspace_state>"):
        at = text.find(marker)
        if at >= 0:
            text = text[:at]
    return text.strip()


def _with_current_request(memory: str, history: Sequence[Any], cut: int) -> str:
    """``memory`` plus the newest real user message when the cut dropped it and the memory does not already hold its text.

    The pinned memory carries the request the agent handed over; in a multi-turn chat that may be the first turn's
    brief, and a cut in the middle of a long turn would otherwise leave the model without what the user just asked.
    """
    base = memory or _OMITTED_NOTE
    last = next((i for i in range(len(history) - 1, -1, -1) if _is_real_user(history[i])), -1)
    if last < 0 or last >= cut:
        return base
    typed = _typed_text(history[last].content)
    if not typed or typed[:120] in base:
        return base
    clipped = typed if len(typed) <= 1500 else typed[:1500] + " [...]"
    line = f"CURRENT REQUEST (the user's latest message, verbatim):\n{clipped}"
    if base.rstrip().endswith("</task_memory>"):
        return base.rstrip()[: -len("</task_memory>")].rstrip("\n") + "\n" + line + "\n</task_memory>"
    return f"{base}\n{line}"


def _clip_result(result: ToolResult) -> ToolResult | None:
    text = result.content or ""
    if len(text) <= RESULT_KEEP_HEAD + RESULT_KEEP_TAIL + len(RESULT_ELISION_MARK):
        return None
    clone = copy.copy(result)
    clone.content = text[:RESULT_KEEP_HEAD] + RESULT_ELISION_MARK + text[-RESULT_KEEP_TAIL:]
    return clone


def _build_view(history: list, *, memory: str, caps: Any, system: str, tools: Sequence, usage_prompt_tokens: int | None,
                keep_screenshots: int, script_paths: Mapping[str, str] | None, sent_estimate: int | None,
                allow_cut: bool) -> tuple[list, dict]:
    window = _window(caps)
    budget = max(0, int(BUDGET_FRACTION * window) - min(_max_output(caps), OUTPUT_RESERVE_CAP))
    stats: dict[str, Any] = {"before": 0, "after": 0, "dropped_rounds": 0, "elided_scripts": 0, "clipped_results": 0,
                             "dropped_images": 0, "memory": False, "too_small": False, "dropped_thinking": 0,
                             "budget": budget, "window": window, "k": 1.0}
    msgs = [_copy_message(m) for m in history]                                    # step 1
    _strip_app_lines(msgs)                                                        # step 2
    sizes = [_message_raw(m) for m in msgs]
    fixed = _text_raw(system) + _tools_raw(tools)
    k = 1.0                                                                       # step 3
    if usage_prompt_tokens and usage_prompt_tokens > 0:
        if sent_estimate and sent_estimate > 0:
            k = _clip_k(usage_prompt_tokens / sent_estimate)
        elif len(msgs) > 2:
            previous = fixed + sum(sizes[:-2])
            k = max(1.0, _clip_k(usage_prompt_tokens / previous)) if previous > 0 else 1.0
    stats["k"] = round(k, 4)
    memory_raw = MESSAGE_OVERHEAD_TOKENS + _text_raw(memory or _OMITTED_NOTE)   # the message that stands in for the dropped rounds
    stats["before"] = math.ceil(k * (fixed + sum(sizes)))
    bounds = round_boundaries(msgs)
    protected = bounds[-2] if len(bounds) >= 2 else 0           # start of the newest two rounds
    mem_min = _text_raw((memory or "")[:MEMORY_MIN_CHARS])
    stats["too_small"] = k * (fixed + mem_min + sum(sizes[protected:])) > FLOOR_FRACTION * window

    def refresh(i: int) -> None:
        sizes[i] = _message_raw(msgs[i])

    def over(extra: float = 0.0) -> bool:
        return k * (fixed + sum(sizes) + extra) > budget

    cut = 0
    if over():
        # (a) elide the code of old script calls (arguments AND the copy of provider_blocks)
        for i in range(protected):
            m = msgs[i]
            if m.role != "assistant" or not m.tool_calls:
                continue
            changed = False
            for j, call in enumerate(m.tool_calls):
                path = (script_paths or {}).get(call.id)
                marker = _elided_code(call, path)
                if marker is None:
                    continue
                m.tool_calls[j] = elide_script_args(call, path)
                if m.provider_blocks:
                    m.provider_blocks = _elide_in_blocks(m.provider_blocks, call.id, marker)
                stats["elided_scripts"] += 1
                changed = True
            if changed:
                refresh(i)
        # (a2) thinking blocks of rounds from EARLIER user turns (exactly what the turn-start compaction drops)
        if over():
            last_user = max((i for i, m in enumerate(msgs) if _is_real_user(m)), default=-1)
            for i in range(min(protected, last_user)):
                m = msgs[i]
                if m.role == "assistant" and m.provider_blocks:
                    kept = [b for b in m.provider_blocks if not (isinstance(b, dict) and b.get("type") in _THINKING_TYPES)]
                    if len(kept) != len(m.provider_blocks):
                        stats["dropped_thinking"] += len(m.provider_blocks) - len(kept)
                        m.provider_blocks = kept
                        refresh(i)
        # (b) clip old tool results
        if over():
            for i in range(protected):
                m = msgs[i]
                if not m.tool_results:
                    continue
                for j, result in enumerate(m.tool_results):
                    clipped = _clip_result(result)
                    if clipped is not None:
                        m.tool_results[j] = clipped
                        stats["clipped_results"] += 1
                refresh(i)
        # (c) drop images from all but the newest ``keep_screenshots``
        if over():
            keep = max(0, min(int(keep_screenshots), MAX_WIRE_IMAGES))
            seen = 0
            for i in range(len(msgs) - 1, -1, -1):
                m = msgs[i]
                touched = False
                for j in range(len(m.tool_results) - 1, -1, -1):
                    if m.tool_results[j].image_b64:
                        seen += 1
                        if seen > keep and i < protected:
                            clone = copy.copy(m.tool_results[j])
                            clone.image_b64 = ""
                            m.tool_results[j] = clone
                            stats["dropped_images"] += 1
                            touched = True
                for j in range(len(m.images) - 1, -1, -1):
                    seen += 1
                    if seen > keep and i < protected:
                        del m.images[j]
                        stats["dropped_images"] += 1
                        touched = True
                if touched:
                    if m.role == "user" and not (m.content or m.tool_results or m.images):
                        m.content = "[image omitted to save context]"
                    refresh(i)
        # (d) drop the oldest whole rounds; the pinned memory stands in for them
        if allow_cut and over():
            candidates = [b for b in bounds if 0 < b <= protected]
            if candidates:
                suffix = [0.0] * (len(sizes) + 1)
                for i in range(len(sizes) - 1, -1, -1):
                    suffix[i] = suffix[i + 1] + sizes[i]
                chosen = candidates[-1]
                for b in candidates:
                    if k * (fixed + memory_raw + suffix[b]) <= budget:
                        chosen = b
                        break
                cut = chosen
    if cut:
        stats["dropped_rounds"] = sum(1 for m in msgs[:cut] if m.role == "assistant")
        msgs = msgs[cut:]
        sizes = sizes[cut:]
    # steps 6 and 7: the memory is materialized only when rounds were dropped or the request left the window
    if cut or not _request_in_view(msgs, memory):
        note = _with_current_request(memory, history, cut) if cut else (memory or _OMITTED_NOTE)
        first = msgs[0] if msgs else None
        if first is not None and first.role == "user" and not first.tool_results and not cut:
            first.content = f"{note}\n\n{first.content}" if first.content else note
        else:
            msgs.insert(0, Message(role="user", content=note))
        stats["memory"] = True
    stats["raw_after"] = math.ceil(fixed + sum(_message_raw(m) for m in msgs))
    stats["after"] = math.ceil(k * stats["raw_after"])
    return msgs, stats


def build_request_view(history: list, *, memory: str, caps: "Any | None", system: str = "", tools: Sequence = (),
                       usage_prompt_tokens: int | None = None, keep_screenshots: int = 3,
                       script_paths: Mapping[str, str] | None = None, sent_estimate: int | None = None) -> tuple[list, dict]:
    """``(view, stats)``: the messages to send this round, derived from the stored ``history`` (never mutated).

    1. messages are copied; 2. ``[governor]`` / ``[phase]`` lines survive only in the newest tool-results message;
    3. the prompt is estimated (``estimate_tokens`` rule) and, when it exceeds ``0.70 * caps.context_window -
    min(caps.max_output, 8192)`` (window 32768 when unknown), the view shrinks in this order, each step only on messages
    OLDER than the newest two rounds: (a) code of old ``execute_code`` calls is elided in ``ToolCall.arguments`` and in
    the copy of ``provider_blocks``; (a2) thinking blocks of earlier user turns are dropped; (b) old tool results keep
    their first 300 and last 100 characters; (c) screenshots beyond the newest ``keep_screenshots`` (and the provider
    cap of 6) are dropped; (d) the oldest WHOLE rounds are dropped at ``round_boundaries``. The newest two rounds and
    the thinking blocks of the open loop are never touched; if the request is still over budget it is accepted (a real
    overflow error teaches the provider layer the true window). 4. The view always starts with a user message: the
    pinned ``memory`` is prepended (or merged into a qualifying first user message) ONLY when rounds were dropped or
    the original request is no longer in the view, so two consecutive views are byte-identical except for their last
    messages. 5. ``stats`` carries ``before``, ``after``, ``dropped_rounds``, ``elided_scripts``, ``clipped_results``,
    ``dropped_images``, ``memory``, ``too_small`` (the floor rule: system + tools + minimal memory +
    newest two rounds exceed 90% of the window; the agent then degrades to ultra-lite or refuses), ``raw_after`` (the
    UNCALIBRATED estimate of the returned view: pass it back as ``sent_estimate`` next round).

    Calibration is multiplicative: with ``usage_prompt_tokens`` (last round's measured prompt tokens) and
    ``sent_estimate`` (last round's ``stats["raw_after"]``) ``k = clip(usage / sent_estimate, 0.5, 3.0)``; without
    ``sent_estimate`` the previous request is approximated by the history minus its last two messages and ``k`` is only
    allowed to RAISE the estimate (``k >= 1``), so a missing figure can never make the view too big. ``script_paths``
    maps tool call id -> saved script path (``script_paths_from_steps``). The view ends with the alternation check; a
    cut that would break alternation is retried without cutting (``stats["alternation_fallback"]``).
    """
    if not history:
        return [], {"before": 0, "after": 0, "dropped_rounds": 0, "elided_scripts": 0, "clipped_results": 0,
                    "dropped_images": 0, "memory": False, "too_small": False, "dropped_thinking": 0, "raw_after": 0}
    opts: dict[str, Any] = dict(memory=memory or "", caps=caps, system=system or "", tools=list(tools or ()),
                                usage_prompt_tokens=usage_prompt_tokens, keep_screenshots=keep_screenshots,
                                script_paths=script_paths, sent_estimate=sent_estimate)
    view, stats = _build_view(history, allow_cut=True, **opts)
    baseline = len(alternation_problems(history))
    problems = alternation_problems(view)
    if len(problems) > baseline:
        view, stats = _build_view(history, allow_cut=False, **opts)
        stats["alternation_fallback"] = True
        problems = alternation_problems(view)
        if len(problems) > baseline:
            stats["alternation_problems"] = problems
    return view, stats


# -- workspace snapshot --------------------------------------------------------------------------------------------

_ORIGIN_NAMES = re.compile(r"^(?:Origin|(?:X|Y|Z)_Axis|(?:XY|XZ|YZ)_Plane|Origin_Point)\d*$")


def _is_origin_feature(obj: Any) -> bool:
    type_id = getattr(obj, "TypeId", "")
    if type_id == "App::Origin":
        return True
    if type_id in ("App::Line", "App::Plane", "App::Point"):
        return bool(getattr(obj, "Role", "") or _ORIGIN_NAMES.match(getattr(obj, "Name", "")))
    return False


def _children_of(obj: Any) -> list:
    try:
        return [c for c in (getattr(obj, "Group", None) or []) if not _is_origin_feature(c)]
    except Exception:  # noqa: BLE001
        return []


def _snapshot_lines(doc: Any, max_objects: int) -> tuple[int, list[str]]:
    """``(object count, object lines)`` of one document: origin features hidden, containers collapsed, pinned names first."""
    objects = [o for o in list(getattr(doc, "Objects", []) or []) if not _is_origin_feature(o)]
    hidden_names: set[str] = set()
    for obj in objects:
        if getattr(obj, "TypeId", "") == "App::Part":
            stack = _children_of(obj)
            while stack:
                child = stack.pop()
                if child.Name not in hidden_names:
                    hidden_names.add(child.Name)
                    if getattr(child, "TypeId", "") == "App::Part":
                        stack.extend(_children_of(child))
    top = [o for o in objects if o.Name not in hidden_names]
    pinned = [o for o in top if o.Name in _PINNED_NAMES or getattr(o, "Label", "") in _PINNED_NAMES]
    top = pinned + [o for o in top if o not in pinned]
    lines: list[str] = []
    for obj in top[:max_objects]:
        try:
            flags = [obj.TypeId]
            if not getattr(obj, "Visibility", True):
                flags.append("hidden")
            # A label is free text from whatever file was opened: one line, no tags (context_guard.safe_name).
            label = f" '{_context_guard.safe_name(obj.Label)}'" if obj.Label != obj.Name else ""
            line = f"    {obj.Name}{label} [{', '.join(flags)}]"
            if getattr(obj, "TypeId", "") == "App::Part":
                count = len(_children_of(obj))
                line += f" ({count} children)" if count != 1 else " (1 child)"
            lines.append(line)
        except Exception:  # noqa: BLE001 - one broken object must not blank the snapshot
            continue
    if len(top) > max_objects:
        lines.append(f"    ... {len(top) - max_objects} more (use get_objects)")
    return len(objects), lines


def workspace_snapshot(max_docs: int = 8, max_objects: int = 40) -> str:
    """Open documents and their objects as a compact text block (the ``<workspace_state>`` body); never raises.

    Per document: ``Params`` / ``Parameters`` / ``NOTES`` / ``PLAN`` objects first, every other object once, an
    ``App::Part`` collapsed with its children into one line (``Part_Shell [App::Part] (6 children)``, the children are
    not listed again), origin features hidden, at most ``max_objects`` object lines then ``... N more (use get_objects)``.
    When the GUI is up the current selection and the active workbench follow the documents. FreeCAD is imported lazily
    here; call on the GUI thread (the panel constructs the worker there). Returns ``""`` without FreeCAD.
    """
    try:
        import FreeCAD
    except ImportError:
        return ""
    try:
        docs = FreeCAD.listDocuments()
    except Exception:  # noqa: BLE001
        return ""
    if not docs:
        return "No documents are open (create one with create_document)."
    try:
        active = getattr(FreeCAD.ActiveDocument, "Name", "") if FreeCAD.ActiveDocument else ""
    except Exception:  # noqa: BLE001
        active = ""
    lines = ["Open documents:"]
    for name, doc in list(docs.items())[:max_docs]:
        try:
            count, object_lines = _snapshot_lines(doc, max_objects)
        except Exception:  # noqa: BLE001
            count, object_lines = 0, []
        lines.append(f"- {name}{' (active)' if name == active else ''}: {count} object(s)")
        lines.extend(object_lines)
    if len(docs) > max_docs:
        lines.append(f"- ... {len(docs) - max_docs} more documents (use list_documents)")
    lines.extend(_gui_lines(FreeCAD))
    return "\n".join(lines)


def _gui_lines(FreeCAD: Any) -> list[str]:
    """``Selection: ...`` and ``Active workbench: ...`` lines when the GUI is up (empty list otherwise)."""
    try:
        if not getattr(FreeCAD, "GuiUp", False):
            return []
        import FreeCADGui as Gui
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    try:
        picked: list[str] = []
        for sel in Gui.Selection.getSelectionEx() or []:
            subs = list(getattr(sel, "SubElementNames", []) or [])
            if subs:
                picked.extend(f"{sel.ObjectName}.{s}" for s in subs)
            else:
                picked.append(sel.ObjectName)
        if picked:
            out.append("Selection: " + ", ".join(picked[:8]) + (f" (+{len(picked) - 8} more)" if len(picked) > 8 else ""))
    except Exception:  # noqa: BLE001
        pass
    try:
        bench = Gui.activeWorkbench()
        out.append(f"Active workbench: {bench.name() if bench is not None else 'unknown'}")
    except Exception:  # noqa: BLE001
        pass
    return out
