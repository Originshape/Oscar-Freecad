"""Text tool mode and inline reasoning splitting. No FreeCAD, no Qt, no network.

A model without native tool calling (``ModelCaps.tools == "text"``) is driven through plain text:

* ``text_protocol`` / ``system_with_protocol`` - the compact protocol appended to the END of the system prompt, with
  one line per tool of the CURRENT request (the native ``tools`` array is omitted);
* ``format_call_block`` / ``format_result_block`` / ``assistant_text`` / ``results_text`` - the history wire format:
  an assistant message with calls becomes ``text + one <tool_call>{json}</tool_call> block per call`` and the results
  go back as ONE user message of ``<tool_result id= name= error=>`` blocks, so a server that rejects OpenAI
  ``tool_calls`` / ``role: "tool"`` history still sees the whole loop;
* ``extract_text_tool_calls`` - the tolerant parser (``<tool_call>`` blocks, fenced JSON, a bare JSON object or
  array, ``[TOOL_CALLS]``), also used on NATIVE-mode replies that carry no ``tool_calls`` (block-only replies);
* ``InlineSplitter`` - streaming-safe split of inline ``<think>`` / ``<|channel|>analysis<|message|>`` spans into
  reasoning, so a tag split across chunks never leaks;
* ``TextStreamFilter`` - the stream side of all of it: splits reasoning, WITHHOLDS text from the first tool-call
  marker until the end of the stream (the user never sees raw JSON) and extracts the calls at the end.

``oscar.providers`` re-exports ``extract_text_tool_calls``. ``ToolCall`` is imported lazily (this module is imported
by ``oscar.providers`` itself).
"""

import json
import re
import uuid

from .provider_args import decode_tool_arguments
from .provider_stream import ToolProgress

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
MISTRAL_MARK = "[TOOL_CALLS]"
_ARGS_CLIP = 20_000
_LINE_DESC_MAX = 160
_ENUM_SHOWN_MAX = 5

_NAME_KEYS = ("name", "tool", "function", "tool_name", "function_name")
_ARGS_KEYS = ("arguments", "parameters", "args", "input", "params")

# -- the prompt side ---------------------------------------------------------------------------------------

_HEAD = (
    "TOOL CALLING PROTOCOL. This model has no native tool calling. To call tools reply with one or more blocks, "
    "each exactly:\n"
    "<tool_call>\n"
    '{"name": "<tool name>", "arguments": {<json object>}}\n'
    "</tool_call>\n"
    "Rules: one block per call; only JSON inside a block; no prose inside a block; after the blocks stop and "
    "wait for the results.\n")
_PLAIN_ANSWER = "If you do not need a tool, answer in plain text with no block.\n"
_MUST_CALL = "You MUST reply with a tool_call block.\n"
_AT_MOST_ONE = "Call at most one tool per reply.\n"
_TOOLS_HEAD = "TOOLS (name(param: type, required ones end with *): description):\n"


def _first_sentence(text: str, limit: int = _LINE_DESC_MAX) -> str:
    """The first sentence of ``text`` on one line, at most ``limit`` characters (an ellipsis marks a clip)."""
    flat = " ".join((text or "").split())
    match = re.search(r"(?<=[.!?])\s", flat)
    if match:
        flat = flat[:match.start()]
    if len(flat) > limit:
        flat = flat[:limit - 3].rstrip() + "..."
    return flat


def _type_names(schema) -> str:
    """The ``type`` names of a JSON-schema node (a union is ``a|b``; enums with at most 5 values are listed)."""
    if not isinstance(schema, dict):
        return "any"
    values = schema.get("enum")
    if isinstance(values, list) and 0 < len(values) <= _ENUM_SHOWN_MAX:
        return "|".join(str(v) for v in values)
    kind = schema.get("type")
    if isinstance(kind, str):
        return kind
    if isinstance(kind, list) and kind:
        return "|".join(str(k) for k in kind)
    for union in ("anyOf", "oneOf"):
        members = schema.get(union)
        if isinstance(members, list) and members:
            names: list[str] = []
            for member in members:
                text = _type_names(member)
                if text not in names:
                    names.append(text)
            return "|".join(names)
    return "any"


def tool_line(spec) -> str:
    """``- name(param: type*, other: type): first sentence of the description`` for one tool."""
    parameters = getattr(spec, "parameters", None)
    parameters = parameters if isinstance(parameters, dict) else {}
    properties = parameters.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = parameters.get("required")
    required = set(required) if isinstance(required, list) else set()
    params = ", ".join(f"{name}: {_type_names(schema)}{'*' if name in required else ''}"
                       for name, schema in properties.items())
    line = f"- {spec.name}({params})"
    description = _first_sentence(getattr(spec, "description", ""))
    return f"{line}: {description}" if description else line


def text_protocol(tools, tool_choice: str | None = None, parallel_tool_calls: bool | None = None) -> str:
    """The protocol text for this request, or ``""`` when no protocol should be sent.

    ``tool_choice``: ``None`` / ``"auto"`` the protocol as is; ``"required"`` adds "You MUST reply with a
    tool_call block." (replacing the "answer in plain text" line); ``"none"`` omits the protocol entirely; a tool
    name adds "Call ONLY <name>.". ``parallel_tool_calls=False`` adds "Call at most one tool per reply.".
    """
    if not tools or tool_choice == "none":
        return ""
    names = {t.name for t in tools}
    out = _HEAD
    if tool_choice == "required":
        out += _MUST_CALL
    elif isinstance(tool_choice, str) and tool_choice not in ("", "auto") and tool_choice in names:
        out += f"Call ONLY {tool_choice}.\n"
    else:
        out += _PLAIN_ANSWER
    if parallel_tool_calls is False:
        out += _AT_MOST_ONE
    return out + _TOOLS_HEAD + "\n".join(tool_line(t) for t in tools)


def system_with_protocol(system: str, tools, tool_choice: str | None = None,
                         parallel_tool_calls: bool | None = None) -> str:
    """``system`` with the protocol appended at the END (the static prefix stays the same inside a phase)."""
    protocol = text_protocol(tools, tool_choice, parallel_tool_calls)
    if not protocol:
        return system
    return f"{system}\n\n{protocol}" if system else protocol


# -- the history wire format -----------------------------------------------------------------------------

def format_call_block(name: str, arguments) -> str:
    """One ``<tool_call>`` block; arguments that could not be decoded are replayed as ``{}`` like the native wire."""
    bad = not isinstance(arguments, dict) or "_raw" in arguments
    payload = {"name": name, "arguments": {} if bad else arguments}
    return f"{TOOL_OPEN}\n{json.dumps(payload, ensure_ascii=False)}\n{TOOL_CLOSE}"


def _escape_result(text: str) -> str:
    """A result must never be able to close its own block (or open a call block the loop could mistake)."""
    return text.replace("</tool_result>", "<\\/tool_result>").replace(TOOL_OPEN, "<\\tool_call>")


def format_result_block(result, note: str = "") -> str:
    """``<tool_result id="..." name="..." error="true|false">text</tool_result>`` for one ``ToolResult``."""
    text = result.content or "(no output)"
    if note:
        text = f"{text}\n{note}"
    return (f'<tool_result id="{result.tool_call_id}" name="{result.name}" '
            f'error="{"true" if result.is_error else "false"}">\n{_escape_result(text)}\n</tool_result>')


def assistant_text(message) -> str:
    """An assistant message as text-mode content: its text plus one call block per tool call."""
    parts = [message.content.strip()] if message.content and message.content.strip() else []
    parts.extend(format_call_block(c.name, c.arguments if not c.args_error else {}) for c in message.tool_calls)
    return "\n".join(parts)


def results_text(message, screenshot_note: str = "", *, vision_ok: bool = True) -> str:
    """The text of the ONE user message that carries every tool result (plus the user's own text, if any)."""
    blocks = [format_result_block(r, screenshot_note if (r.image_b64 and not vision_ok) else "")
              for r in message.tool_results]
    if message.content and message.content.strip():
        blocks.append(message.content)
    return "\n".join(blocks)


def openai_text_wire(m, *, vision_ok: bool, images_note: str, screenshot_note: str) -> list[dict]:
    """One ``Message`` as OpenAI-compatible chat messages in text tool mode (a list of exactly one message).

    Assistant: ``content`` = text + one ``<tool_call>`` block per call, no ``tool_calls`` key. User: ONE message
    holding every ``<tool_result>`` block (and the user's own text); screenshots follow as image parts of that same
    message, only when vision is on (otherwise a one-line note says what was left out).
    """
    if m.role == "assistant":
        return [{"role": "assistant", "content": assistant_text(m) or "(no reply)"}]
    text = results_text(m, screenshot_note, vision_ok=vision_ok)
    images: list[tuple[str, str]] = []
    if vision_ok:
        images = [(f"(screenshot after {r.name})", r.image_b64) for r in m.tool_results if r.image_b64]
        images += [("", b64) for b64 in m.images]
    elif m.images:
        text = f"{text}\n{images_note}".strip()
    if not images:
        return [{"role": "user", "content": text or "(empty message)"}]
    parts: list[dict] = [{"type": "text", "text": text or "(see the images)"}]
    for label, b64 in images:
        if label:
            parts.append({"type": "text", "text": label})
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return [{"role": "user", "content": parts}]


# -- the parser ---------------------------------------------------------------------------------------------

def _new_call(name: str, raw_args):
    """A ``ToolCall`` for ``name`` from decoded arguments (a dict) or a raw argument string (tolerant decode)."""
    from .providers import ToolCall

    call_id = f"call_{uuid.uuid4().hex[:12]}"
    if isinstance(raw_args, dict):
        return ToolCall(id=call_id, name=name, arguments=raw_args)
    if raw_args is None:
        return ToolCall(id=call_id, name=name, arguments={})
    if isinstance(raw_args, str):
        arguments, error = decode_tool_arguments(raw_args)
        return ToolCall(id=call_id, name=name, arguments=arguments, args_error=error,
                        raw_args=raw_args[:_ARGS_CLIP] if error else "")
    return ToolCall(id=call_id, name=name, arguments={"_raw": str(raw_args)},
                    args_error="arguments must be a JSON object", raw_args=str(raw_args)[:_ARGS_CLIP])


def _balanced_end(text: str, start: int) -> int:
    """Index just past the JSON object / array that opens at ``text[start]``, or -1 when it never closes (string-aware)."""
    if start >= len(text) or text[start] not in "{[":
        return -1
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _canonical(name, names) -> str | None:
    """The tool name ``name`` stands for (``functions.x`` / ``default_api.x`` prefixes are dropped), else None."""
    if not isinstance(name, str):
        return None
    cleaned = name.strip().strip("`'\"")
    if cleaned in names:
        return cleaned
    tail = cleaned.rsplit(".", 1)[-1]
    return tail if tail in names else None


def _call_shape(obj, names, *, strict: bool):
    """``(name, raw_arguments)`` when ``obj`` is a call-shaped dict naming a known tool, else None.

    ``strict`` (fenced / bare JSON) additionally requires an arguments key: ``{"name": "x"}`` quoted in prose is
    data, while inside an explicit ``<tool_call>`` block a bare name is a call without arguments.
    """
    if not isinstance(obj, dict):
        return None
    function = obj.get("function")
    if isinstance(function, dict):                      # OpenAI shape: {"type": "function", "function": {...}}
        inner = _call_shape(function, names, strict=strict)
        if inner is not None:
            return inner
    for key in _NAME_KEYS:
        name = obj.get(key)
        if isinstance(name, str):
            canonical = _canonical(name, names)
            if canonical is None:
                continue
            for akey in _ARGS_KEYS:
                if akey in obj:
                    return canonical, obj[akey]
            return None if strict else (canonical, None)
    return None


def _payload_calls(text: str, names, *, strict: bool) -> list:
    """The calls of one JSON payload (an object, or an array of objects), tolerant decoding; ``[]`` when none."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text, strict=False)
    except ValueError:
        arguments, error = decode_tool_arguments(text)        # fences, trailing commas, python dicts
        value = arguments if not error else None
        if value is None:
            return []
    items = value if isinstance(value, list) else [value]
    calls = []
    for item in items:
        shape = _call_shape(item, names, strict=strict)
        if shape is not None:
            calls.append(_new_call(*shape))
    return calls


_TOOL_OPEN_RE = re.compile(r"<tool_call\s*>", re.IGNORECASE)
_TOOL_CLOSE_RE = re.compile(r"</tool_call\s*>", re.IGNORECASE)
_CLOSE_AFTER_RE = re.compile(r"\s*</tool_call\s*>", re.IGNORECASE)
_FENCE_RE = re.compile(r"```[ \t]*(?:json|tool_call|tool|JSON)?[ \t]*\r?\n?(.*?)```", re.DOTALL)
_MISTRAL_V3_RE = re.compile(r"\s*([A-Za-z_][\w.\-]*)\s*\[ARGS\]\s*", re.DOTALL)
_BARE_SCAN_MAX_CHARS = 200_000   # beyond this a reply is a document, not a call: skip the quadratic bare-JSON scan


def _scan(text: str, names, *, allow_bare: bool = True) -> tuple[list, list[tuple[int, int]]]:
    """``(calls, spans)``: every recognized call in ``text`` and the (start, end) span of the block that held each.

    Spans never overlap and are sorted. Only calls naming a tool in ``names`` count; a block that yields no known
    call is not a span (its text stays).
    """
    names = frozenset(names or ())
    if not names or not text:
        return [], []
    found: list[tuple[int, int, list]] = []

    def taken(start: int, end: int) -> bool:
        return any(start < b and a < end for a, b, _ in found)

    # 1. <tool_call> ... </tool_call>  (an unclosed block counts when its JSON is complete: the stop token may
    #    have eaten the closing tag)
    for match in _TOOL_OPEN_RE.finditer(text):
        start = match.start()
        if taken(start, match.end()):
            continue
        close = _TOOL_CLOSE_RE.search(text, match.end())
        if close is not None:
            inner, end = text[match.end():close.start()], close.end()
        else:
            first = re.search(r"[{\[]", text[match.end():])
            if first is None:
                continue
            json_start = match.end() + first.start()
            json_end = _balanced_end(text, json_start)
            if json_end < 0:
                continue
            inner, end = text[match.end():json_end], json_end
        calls = _payload_calls(inner, names, strict=False)
        if not calls and close is not None:
            # the closing tag found first may sit INSIDE a JSON string (a script that quotes the tag): take the
            # balanced object instead, and the real closing tag right behind it when there is one
            first = re.search(r"[{\[]", text[match.end():])
            json_end = _balanced_end(text, match.end() + first.start()) if first is not None else -1
            if json_end > 0:
                calls = _payload_calls(text[match.end():json_end], names, strict=False)
                if calls:
                    tail = _CLOSE_AFTER_RE.match(text, json_end)
                    end = tail.end() if tail else json_end
        if calls:
            found.append((start, end, calls))
    # 2. [TOOL_CALLS] [...]   and the older  [TOOL_CALLS]name[ARGS]{...}
    for match in re.finditer(re.escape(MISTRAL_MARK), text):
        if taken(match.start(), match.end()):
            continue
        pos = match.end()
        v3 = _MISTRAL_V3_RE.match(text, pos)
        if v3 is not None and v3.end() < len(text) and text[v3.end()] in "{[":
            end = _balanced_end(text, v3.end())
            name = _canonical(v3.group(1), names)
            if end > 0 and name:
                found.append((match.start(), end, [_new_call(name, text[v3.end():end])]))
            continue
        while pos < len(text) and text[pos] in " \t\r\n":
            pos += 1
        end = _balanced_end(text, pos)
        if end < 0:
            continue
        calls = _payload_calls(text[pos:end], names, strict=False)
        if calls:
            found.append((match.start(), end, calls))
    # 3. fenced blocks
    for match in _FENCE_RE.finditer(text):
        if taken(match.start(), match.end()):
            continue
        calls = _payload_calls(match.group(1), names, strict=True)
        if calls:
            found.append((match.start(), match.end(), calls))
    # 4. a bare JSON object / array that starts a line
    if allow_bare and len(text) <= _BARE_SCAN_MAX_CHARS:
        for match in re.finditer(r"(?m)^[ \t]*([{\[])", text):
            start = match.start(1)
            if taken(start, start + 1):
                continue
            end = _balanced_end(text, start)
            if end < 0:
                continue
            calls = _payload_calls(text[start:end], names, strict=True)
            if calls:
                found.append((start, end, calls))
    found.sort(key=lambda item: item[0])
    calls = [c for _, _, group in found for c in group]
    return calls, [(a, b) for a, b, _ in found]


def _join_pieces(text: str, spans: list[tuple[int, int]]) -> str:
    """``text`` with ``spans`` removed: the remaining pieces are stripped and joined by a blank line."""
    pieces: list[str] = []
    cursor = 0
    for start, end in [*spans, (len(text), len(text))]:
        piece = text[cursor:start].strip()
        cursor = end
        if piece:
            pieces.append(piece)
    return "\n\n".join(pieces)


def extract_text_tool_calls(text: str, tool_names, *, block_only: bool = False,
                            max_calls: int = 0) -> tuple[list, str]:
    """``(calls, text_without_the_blocks)`` for tool calls written as text.

    Accepts ``<tool_call>...</tool_call>``, a fenced ```json block holding ``{"name", "arguments"}`` (also the
    ``tool`` / ``function`` / ``tool_name`` and ``parameters`` / ``args`` / ``input`` aliases and the OpenAI
    ``{"type": "function", "function": {...}}`` shape), a bare JSON object or array of them that starts a line, and
    ``[TOOL_CALLS][...]``. Arguments may be an object or a JSON string (decoded tolerantly: a bad one yields a call
    with ``args_error``). Names that are not in ``tool_names`` are ignored; ids are minted ``call_<12 hex>``.

    ``block_only=True`` (native-mode replies that carried no ``tool_calls``): calls are returned ONLY when the whole
    text consists of the block(s) (nothing but whitespace around them), so a final answer that quotes an example
    call is never executed. ``max_calls`` > 0 keeps only the first calls (``parallel_tool_calls=False``). When no
    call is recognized the text is returned unchanged.
    """
    original = text or ""
    calls, spans = _scan(original, tool_names)
    if not calls:
        return [], original
    remaining = _join_pieces(original, spans)
    if block_only and remaining.strip():
        return [], original
    if max_calls and len(calls) > max_calls:
        calls = calls[:max_calls]
    return calls, remaining.strip()




# -- inline reasoning splitter -----------------------------------------------------------------------------

_THINK_PAIRS = {"<think>": "</think>", "<thinking>": "</thinking>", "<reasoning>": "</reasoning>"}
_CHANNEL_RE = re.compile(r"<\|channel\|>\s*([A-Za-z_]+)[^<]*?<\|message\|>")
_ANALYSIS_CLOSER = "<|end|>"
_DROP_TOKENS = ("<|start|>assistant", "<|return|>", "<|end|>", "<|call|>")
_TEXT_TAGS = (*_THINK_PAIRS, *_DROP_TOKENS)
_CHANNEL_HEADER_MAX = 120


class InlineSplitter:
    """Splits ``<think>...</think>`` and ``<|channel|>analysis<|message|>...<|end|>`` spans out of a text stream.

    ``feed(chunk)`` returns ``[(kind, text)]`` with ``kind`` ``"text"`` or ``"reasoning"``. A tag split across
    chunks is held back until it can be classified (``"<th"`` waits for the next chunk); ``finish()`` flushes what is
    pending (an unclosed reasoning span stays reasoning). Text without a ``<`` passes through chunk for chunk,
    unchanged. Other Harmony channels (``final``, ``commentary``) lose only their header; ``<|start|>assistant`` /
    ``<|end|>`` / ``<|return|>`` / ``<|call|>`` tokens in the text state are dropped.
    """

    def __init__(self):
        self._buf = ""
        self._closer: str | None = None     # set while inside a reasoning span: the tag that ends it

    @property
    def in_reasoning(self) -> bool:
        """True while inside a reasoning span."""
        return self._closer is not None

    @staticmethod
    def _could_become_tag(tail: str) -> bool:
        """True when ``tail`` (starting at ``<``, no complete tag at its start) may still become a tag."""
        if any(tag.startswith(tail) for tag in _TEXT_TAGS):
            return True
        if "<|channel|>".startswith(tail):
            return True
        return tail.startswith("<|channel|>") and "<|message|>" not in tail and len(tail) <= _CHANNEL_HEADER_MAX

    @staticmethod
    def _partial_suffix(buf: str, closer: str) -> int:
        """Length of the longest suffix of ``buf`` that is a proper prefix of ``closer``."""
        for size in range(min(len(closer) - 1, len(buf)), 0, -1):
            if closer.startswith(buf[-size:]):
                return size
        return 0

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        """Consume ``chunk``; the events that can be released now."""
        self._buf += chunk or ""
        out: list[tuple[str, str]] = []

        def emit(kind: str, text: str) -> None:
            if text:
                out.append((kind, text))

        while self._buf:
            if self._closer is not None:
                at = self._buf.find(self._closer)
                if at >= 0:
                    emit("reasoning", self._buf[:at])
                    self._buf = self._buf[at + len(self._closer):]
                    self._closer = None
                    continue
                keep = self._partial_suffix(self._buf, self._closer)
                emit("reasoning", self._buf[:len(self._buf) - keep])
                self._buf = self._buf[len(self._buf) - keep:]
                break
            lt = self._buf.find("<")
            if lt < 0:
                emit("text", self._buf)
                self._buf = ""
                break
            emit("text", self._buf[:lt])
            self._buf = self._buf[lt:]
            tag = self._match_tag()
            if tag is not None:
                length, closer = tag
                self._buf = self._buf[length:]
                self._closer = closer
                continue
            if self._could_become_tag(self._buf):
                break                                   # wait for more data
            emit("text", "<")
            self._buf = self._buf[1:]
        return out

    def _match_tag(self) -> tuple[int, str | None] | None:
        """``(length, closer)`` of the complete tag at the start of the buffer (closer None = a dropped token)."""
        for opener, closer in _THINK_PAIRS.items():
            if self._buf.startswith(opener):
                return len(opener), closer
        for token in _DROP_TOKENS:
            if self._buf.startswith(token):
                return len(token), None
        header = _CHANNEL_RE.match(self._buf)
        if header is not None:
            return header.end(), _ANALYSIS_CLOSER if header.group(1).lower() == "analysis" else None
        return None

    def finish(self) -> list[tuple[str, str]]:
        """Flush the pending buffer (an unfinished tag is released as plain content of the current state)."""
        kind = "reasoning" if self._closer is not None else "text"
        out = [(kind, self._buf)] if self._buf else []
        self._buf = ""
        self._closer = None
        return out


# -- the stream filter ---------------------------------------------------------------------------------------

_PARTIAL_MARKERS = (TOOL_OPEN, MISTRAL_MARK)
_NAME_PEEK_RE = re.compile(r'"(?:name|tool|function|tool_name)"\s*:\s*"([^"\\]{1,80})"')
_FENCE_HEAD_RE = re.compile(r"```[ \t]*([A-Za-z_]*)[ \t]*\r?\n?[ \t\r\n]*")
_BARE_CALL_RE = re.compile(
    r'\[?\s*\{\s*"(?:name|tool|function|tool_name|function_name|type|arguments|parameters|args|input|params)"')
_UNDECIDED_CHARS = 24


class TextStreamFilter:
    """The text side of a provider stream: reasoning split, tool-call hold-back, extraction at the end.

    ``feed(chunk)`` -> ``[(kind, payload)]`` events to deliver now (``text`` / ``reasoning`` / ``tool_progress``);
    ``finish(...)`` -> ``(events, calls)``. In TEXT mode (``text_mode=True``) text is withheld from the first
    ``<tool_call>`` / ``[TOOL_CALLS]`` / JSON-fence marker (anywhere in the reply) or a reply that starts with a
    bare JSON call, until the end of the stream: the user never sees raw tool markup, and whatever the end of the
    stream shows not to be a call is released as text. In NATIVE mode text is withheld only when the reply STARTS
    with such a marker (a block-only reply cannot start with prose) and calls are extracted only when the reply is
    block-only. Text with no marker characters passes through chunk for chunk.
    """

    def __init__(self, tool_names=(), *, text_mode: bool = False, split_think: bool = True, max_calls: int = 0):
        self.tool_names = frozenset(tool_names or ())
        self.text_mode = bool(text_mode)
        self.max_calls = int(max_calls or 0)
        self._splitter = InlineSplitter() if split_think else None
        self._visible = ""
        self._emitted = 0                 # characters of ``_visible`` already delivered
        self._held = False                # a confirmed marker: everything from it is withheld until the end
        self._progress = ToolProgress()
        self._reasoning: list[str] = []
        self._peeked_name = ""

    @property
    def reasoning_text(self) -> str:
        """Every reasoning chunk this filter produced (for replay as ``reasoning_content``)."""
        return "".join(self._reasoning)

    # -- feeding

    def feed(self, chunk: str) -> list[tuple[str, object]]:
        """Consume one text chunk; the events to deliver now."""
        pieces = self._splitter.feed(chunk) if self._splitter is not None else [("text", chunk or "")]
        return self._absorb(pieces)

    def _absorb(self, pieces) -> list[tuple[str, object]]:
        out: list[tuple[str, object]] = []
        for kind, text in pieces:
            if kind == "reasoning":
                self._reasoning.append(text)
                out.append(("reasoning", text))
            elif text:
                self._visible += text
        self._release(out)
        return out

    # -- hold-back

    def _marker(self) -> tuple[int, bool] | None:
        """``(index, confirmed)`` of the earliest withheld position in the unemitted text, or None.

        ``confirmed`` is True for a real marker (held to the end of the stream); False for something that may
        still turn into one (a partial tag, a code fence whose content has not arrived): re-evaluated every chunk.
        """
        text, start = self._visible, self._emitted
        at_start = start == 0
        found: list[int] = []
        match = _TOOL_OPEN_RE.search(text, start)
        if match:
            found.append(match.start())
        mark = text.find(MISTRAL_MARK, start)
        if mark >= 0:
            found.append(mark)
        confirmed_idx = min(found) if found else None
        fence = self._fence_start(text, start)
        undecided_idx: int | None = None
        if fence is not None:
            if fence[1]:
                confirmed_idx = fence[0] if confirmed_idx is None else min(confirmed_idx, fence[0])
            else:
                undecided_idx = fence[0]
        lead = len(text) - len(text.lstrip())
        if at_start and text[lead:lead + 1] in ("{", "["):
            frag = text[lead:]
            if _BARE_CALL_RE.match(frag):
                confirmed_idx = lead if confirmed_idx is None else min(confirmed_idx, lead)
            elif len(frag.strip()) < _UNDECIDED_CHARS and (undecided_idx is None or lead < undecided_idx):
                undecided_idx = lead
        if confirmed_idx is None and undecided_idx is None:
            partial = self._partial_marker(text, start)
            if partial is None:
                return None
            undecided_idx = partial
        index = confirmed_idx if confirmed_idx is not None else undecided_idx
        if undecided_idx is not None and (confirmed_idx is None or undecided_idx < confirmed_idx):
            index, confirmed = undecided_idx, False
        else:
            confirmed = True
        if not self.text_mode and (not at_start or text[:index].strip()):
            return None                       # native mode: prose before a marker cannot be a block-only reply
        return index, confirmed

    @staticmethod
    def _fence_start(text: str, start: int) -> tuple[int, bool] | None:
        """``(index, confirmed)`` of a code fence that may hold a call, else None.

        Confirmed: JSON-ish content after an empty / ``json`` / ``tool_call`` / ``tool`` fence. Unconfirmed: a fence
        whose language or first content character has not arrived yet.
        """
        pos = start
        while True:
            at = text.find("```", pos)
            if at < 0:
                return None
            head = _FENCE_HEAD_RE.match(text, at)
            language = head.group(1).lower() if head else ""
            end = head.end() if head else at + 3
            rest = text[end:]
            if language in ("", "json", "tool_call", "tool"):
                if not rest:
                    return at, False
                if rest[0] in "{[":
                    return at, True
            elif end >= len(text):
                return at, False             # "```py" may still become "```python" or "```json"
            pos = at + 3

    @staticmethod
    def _partial_marker(text: str, start: int) -> int | None:
        """Index of a trailing partial marker (``<tool_c``, ``[TOOL_``, a lone backtick pair) that may complete."""
        tail = text[start:]
        for size in range(min(len(tail), max(len(m) for m in _PARTIAL_MARKERS)), 0, -1):
            suffix = tail[-size:]
            if suffix in ("`", "``") or any(m.startswith(suffix) and m != suffix for m in _PARTIAL_MARKERS):
                return len(text) - size
        return None

    def _release(self, out: list) -> None:
        """Deliver what may be shown (everything before the first withheld position) and report tool progress."""
        if self._held:
            self._peek_progress(out)
            return
        marker = self._marker()
        if marker is None:
            self._deliver(out, len(self._visible))
            return
        index, confirmed = marker
        hold = index
        while hold > self._emitted and self._visible[hold - 1] in " \t\r\n":
            hold -= 1
        self._deliver(out, hold)
        if confirmed:
            self._held = True
            self._peek_progress(out)

    def _deliver(self, out: list, upto: int) -> None:
        if upto > self._emitted:
            out.append(("text", self._visible[self._emitted:upto]))
            self._emitted = upto

    def _peek_progress(self, out: list) -> None:
        """A throttled ``tool_progress`` while a (text-mode) call streams in, so the watchdog and the UI see activity."""
        if not self._peeked_name:
            match = _NAME_PEEK_RE.search(self._visible, self._emitted)
            if match and match.group(1) in self.tool_names:
                self._peeked_name = match.group(1)
        event = self._progress.update(0, self._peeked_name, len(self._visible) - self._emitted)
        if event is not None:
            out.append(event)

    # -- end of stream

    def finish(self, *, stop_reason: str = "", complete: bool = True,
               extract: bool = True) -> tuple[list[tuple[str, object]], list]:
        """``(events, calls)`` at the end of the stream.

        ``complete=False`` (the stream ended early): everything pending is released as text, no calls are
        extracted. ``extract=False`` (the server already delivered native calls): same, the text is not scanned.
        A reply cut at the token limit still yields the blocks that completed, in text mode only.
        """
        out: list[tuple[str, object]] = []
        if self._splitter is not None:
            out.extend(self._absorb(self._splitter.finish()))
        pending = self._visible[self._emitted:]
        if not pending:
            return out, []
        calls: list = []
        tail = pending
        if complete and extract:
            calls, tail = self._extract(pending, stop_reason)
        if calls:
            final = self._progress.final(0, calls[0].name, len(pending))
            if final is not None:
                out.append(final)
        if tail:
            out.append(("text", tail))
        self._emitted = len(self._visible)
        return out, calls

    def _truncated_call(self, pending: str, stop_reason: str):
        """A ``ToolCall`` with ``args_error`` for a text-mode call block cut off at the output-token limit, else None.

        A weak model that writes a long script into a ``<tool_call>`` block runs out of tokens in the middle of it:
        nothing in the text parses, and without this the raw markup would be released as the reply (shown to the user,
        replayed as the model's own words, judged by the finish gate as "stopped with unfinished work"). The call that
        is returned carries the same ``truncated (...)`` reason a native call gets, so the loop answers it with the
        "split the script" guidance instead. Needs a CONFIRMED marker (the text was withheld), a stop reason that is a
        cut-off and a tool name that is one of this request's tools.
        """
        from .stopreason import is_cut_off

        if not (self.text_mode and self._held and is_cut_off(stop_reason)):
            return None
        block = pending.lstrip()
        peek = _NAME_PEEK_RE.search(block)
        if peek is None or peek.group(1) not in self.tool_names:
            return None
        opener = _TOOL_OPEN_RE.match(block)
        inner = block[opener.end():] if opener else block
        arguments, error = decode_tool_arguments(inner, stop_reason=stop_reason)
        if not error:
            return None                                   # it decodes after all: nothing to report
        from .providers import ToolCall

        return ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=peek.group(1), arguments=arguments, args_error=error,
                        raw_args=inner[:_ARGS_CLIP])

    def _extract(self, pending: str, stop_reason: str) -> tuple[list, str]:
        """``(calls, text_to_release)`` for the withheld text; ``([], pending)`` when nothing qualifies."""
        from .stopreason import normalize_stop_reason

        calls, spans = _scan(pending, self.tool_names)
        if not calls:
            truncated = self._truncated_call(pending, stop_reason)
            if truncated is not None:
                return [truncated], ""
            return [], pending
        remaining = _join_pieces(pending, spans)
        if not self.text_mode:
            block_only = (not remaining and not self._visible[:self._emitted].strip()
                          and normalize_stop_reason(stop_reason) != "max_tokens")
            if not block_only:
                return [], pending
        if self.max_calls and len(calls) > self.max_calls:
            calls = calls[:self.max_calls]
        if remaining and self._visible[:self._emitted].strip():
            remaining = "\n\n" + remaining
        return calls, remaining
