"""Parsing helpers shared by the provider stream parsers. No FreeCAD, no Qt.

* ``sse_lines`` - ``data:`` payloads of an SSE body (UTF-8, comment lines counted as liveness).
* ``ndjson_lines`` - parsed JSON objects of an NDJSON body (the native Ollama stream).
* ``ToolProgress`` - throttled ``tool_progress`` events.
* ``usage_numbers`` / ``stop_info`` - the normalized ``stop_info`` event.
* ``looks_like_image_rejection`` / ``looks_like_tools_unsupported`` / ``context_window_from_error`` /
  ``stream_error`` / ``error_text`` - reading what a server said when it refused or failed a request.

``oscar.providers`` re-exports these under their historic underscore names.
"""

import json
import re
import threading
import time
from typing import Any, Generator

from .models import unambiguous_image_rejection
from .provider_reasoning import mentions_field
from .provider_transport import ProviderError, retriable_message
from .stopreason import normalize_stop_reason

StreamEvent = tuple[str, Any]


def looks_like_image_rejection(detail: str) -> bool:
    """True when a 400 body is (probably) about image content: the request is retried without images.

    Text-only models (e.g. DeepSeek's chat endpoint) reject multimodal content blocks outright; the
    detected phrases cover both wire dialects. This is the BROAD test: a size / base64 / media-type complaint
    matches too, which degrades the current session. Whether the model itself lacks vision (and so whether that may
    be remembered across sessions) is ``models.unambiguous_image_rejection``.
    """
    lower = (detail or "").lower()
    if unambiguous_image_rejection(lower):
        return True
    if "content.type is invalid" in lower or "content type is invalid" in lower:
        return True
    if "image" in lower and any(
        word in lower for word in ("invalid", "not support", "unsupported", "allowed values")
    ):
        return True
    return False


_TOOLS_UNSUPPORTED_PHRASES = (
    "does not support tools", "doesn't support tools", "do not support tools", "does not support tool",
    "doesn't support tool", "tools is not supported", "tools are not supported", "tools not supported",
    "tool use is not supported", "tool use not supported", "tool calling is not supported",
    "tool calling not supported", "tool_use is not supported", "function calling is not supported",
    "does not support function calling", "doesn't support function calling", "no endpoints found that support tool",
    "tool choice requires --enable-auto-tool-choice", "requires --enable-auto-tool-choice", "requires --jinja",
    "tools param requires", "does not support the tools", "tools are not available",
)


def looks_like_tools_unsupported(detail: str) -> bool:
    """True when a 400 / 404 body says this model or endpoint cannot do native tool calling.

    The answer is to switch the model to text tool mode (``learn(tools="text")``) and retry. ``tool_choice``
    complaints are NOT this (they have their own rule): ``tools`` must appear as a whole word.
    """
    lower = " ".join((detail or "").lower().split())
    if any(phrase in lower for phrase in _TOOLS_UNSUPPORTED_PHRASES):
        return True
    if mentions_field(lower, "tools") and any(
            word in lower for word in ("unrecognized request argument", "unknown parameter", "unknown field",
                                       "extra inputs are not permitted", "unsupported parameter", "not permitted")):
        return True
    return False


# The reader thread publishes SSE comment lines (": keepalive") as liveness through this thread-local.
LIVE = threading.local()


_SSE_FIELDS = frozenset({"data", "event", "id", "retry"})
_STRAY_CLIP = 200


def _line_iterator(response):
    """The lines of a streaming response, split on LF (and CR) ONLY.

    ``requests`` splits on ``str.splitlines()`` when it decodes, which also breaks a line at U+2028, U+2029,
    U+0085, VT, FF and the FS / GS / RS separators. JSON serializers leave those characters unescaped inside
    strings, so a ``data:`` line whose text or tool arguments hold one was cut in two and the whole chunk was
    silently dropped. ``delimiter="\\n"`` keeps every line whole; a response object that does not accept it (test
    fakes with the plain ``iter_lines(decode_unicode=True)`` shape) is iterated the old way. A raw CR can only be a
    line terminator (JSON escapes it), so lone-CR streams are still split.
    """
    try:
        lines = response.iter_lines(decode_unicode=True, delimiter="\n")
    except TypeError:
        lines = response.iter_lines(decode_unicode=True)
    for raw in lines:
        if isinstance(raw, str) and "\r" in raw:
            yield from raw.split("\r")
        else:
            yield raw


def sse_lines(response) -> Generator[str, None, None]:
    """Yield ``data:`` payload strings from an SSE response body.

    Sets ``response.encoding = "utf-8"`` first (a ``text/event-stream`` without a charset would otherwise
    be decoded as ISO-8859-1 and non-ASCII text would turn into mojibake). Comment lines (starting with
    ``:``) are counted as liveness, not data. A body that ends without a single ``data:`` line but with
    other text (an HTML login page, a JSON error answered with status 200) raises a non-retriable
    ``ProviderError`` quoting its first 200 characters instead of looking like an empty reply.
    """
    try:
        response.encoding = "utf-8"
    except (AttributeError, TypeError):
        pass
    alive = getattr(LIVE, "callback", None)
    seen_data = False
    stray = ""
    for raw in _line_iterator(response):
        if not raw:
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        if raw.startswith(":"):
            if alive is not None:
                alive()
            continue
        if raw.startswith("data:"):
            seen_data = True
            yield raw[5:].strip()
        elif not seen_data and len(stray) < _STRAY_CLIP and raw.split(":", 1)[0].strip() not in _SSE_FIELDS:
            stray = f"{stray} {raw.strip()}".strip()
    if stray and not seen_data:
        raise ProviderError(
            f"the server answered with something other than an event stream: {stray[:_STRAY_CLIP]}", kind="stream")


def ndjson_lines(response) -> Generator[dict, None, None]:
    """Yield the JSON objects of an NDJSON response body (one object per line), UTF-8.

    Blank lines are skipped. A body that ends without a single JSON object but with other text (an HTML error page
    answered with status 200, a proxy banner) raises a non-retriable ``ProviderError`` quoting its first 200
    characters instead of looking like an empty reply. Lines that are not JSON objects after a valid one are ignored.
    """
    try:
        response.encoding = "utf-8"
    except (AttributeError, TypeError):
        pass
    alive = getattr(LIVE, "callback", None)
    seen = False
    stray = ""
    for raw in _line_iterator(response):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        text = (raw or "").strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except ValueError:
            if not seen and len(stray) < _STRAY_CLIP:
                stray = f"{stray} {text}".strip()
            continue
        if isinstance(value, dict):
            seen = True
            if alive is not None:
                alive()
            yield value
    if stray and not seen:
        raise ProviderError(
            f"the server answered with something other than an NDJSON stream: {stray[:_STRAY_CLIP]}", kind="stream")


class ToolProgress:
    """Throttles ``tool_progress`` events: at most one per interval per tool index, plus a final one."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._last: dict[Any, float] = {}
        self._final: set = set()

    def update(self, index, name: str, chars: int) -> StreamEvent | None:
        """The event to emit now, or None while the interval since the last one has not passed."""
        now = time.monotonic()
        last = self._last.get(index)
        if last is not None and now - last < self.interval:
            return None
        self._last[index] = now
        return ("tool_progress", {"name": name, "chars": chars, "index": index if isinstance(index, int) else 0})

    def final(self, index, name: str, chars: int) -> StreamEvent | None:
        """The closing event of a tool call (once per index)."""
        if index in self._final:
            return None
        self._final.add(index)
        self._last[index] = time.monotonic()
        return ("tool_progress", {"name": name, "chars": chars, "index": index if isinstance(index, int) else 0})


def usage_numbers(usage: dict | None, dialect: str) -> dict:
    """``{"prompt_tokens", "completion_tokens"}`` from a provider's own usage dict; ``{}`` when it reported none.

    Anthropic: ``prompt_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens`` (with
    ``cache_control`` on, ``input_tokens`` alone is only the uncached tail). OpenAI chat ``prompt_tokens``;
    Ollama ``prompt_eval_count`` / ``eval_count``.
    """
    if not isinstance(usage, dict) or not usage:
        return {}

    def num(*keys) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return None

    if dialect == "anthropic":
        parts = [num("input_tokens"), num("cache_creation_input_tokens"), num("cache_read_input_tokens")]
        prompt = sum(p for p in parts if p) if any(p is not None for p in parts) else None
        completion = num("output_tokens")
    else:
        prompt, completion = num("prompt_tokens", "prompt_eval_count", "input_tokens"), \
            num("completion_tokens", "eval_count", "output_tokens")
    if prompt is None and completion is None:
        return {}
    return {"prompt_tokens": prompt or 0, "completion_tokens": completion or 0}


# Finish reasons the shared normalizer does not know (verified from docs): Anthropic ``model_context_window_exceeded``
# (https://platform.claude.com/docs/en/api/handling-stop-reasons: "treat the response as truncated"), Z.ai
# ``sensitive`` / ``model_context_window_exceeded`` / ``network_error`` (https://docs.z.ai/api-reference/llm/chat-completion).
_EXTRA_STOP_REASONS = {
    "model_context_window_exceeded": "max_tokens", "sensitive": "content_filter", "network_error": "error",
}


def stop_info(raw_reason, usage: dict | None, dialect: str, *, has_calls: bool = False) -> dict:
    """The ``stop_info`` payload. ``reason`` is the normalized enum; a reply with calls and a plain end is ``tool_use``."""
    raw = "" if raw_reason is None else str(raw_reason)
    reason = normalize_stop_reason(raw)
    if reason == "unknown":
        reason = _EXTRA_STOP_REASONS.get(raw.strip().lower(), reason)
    if has_calls and reason == "end_turn":
        reason = "tool_use"
    return {"reason": reason, "raw": raw, "truncated": reason == "max_tokens", "complete": True,
            "usage": usage_numbers(usage, dialect)}


_CONTEXT_ERROR_PATTERNS = (
    r"maximum context length is\s*(\d[\d,]*)",
    r"context (?:length|window) (?:is|of|limit(?: is)?)\s*(\d[\d,]*)",
    r"exceeds? the (?:model'?s? )?(?:maximum )?context (?:window|length)(?: limit)? of\s*(\d[\d,]*)",
    r"prompt is too long:\s*\d[\d,]*\s*tokens?\s*>\s*(\d[\d,]*)",
    r"input length and `?max_tokens`? exceed context limit:\s*\d[\d,]*\s*\+\s*\d[\d,]*\s*>\s*(\d[\d,]*)",
)


def context_window_from_error(lower: str) -> int:
    """The context window a context-length error states (tokens), or 0 when the text states none."""
    if "context" not in lower and "too long" not in lower:
        return 0
    for pattern in _CONTEXT_ERROR_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            try:
                value = int(match.group(1).replace(",", ""))
            except ValueError:
                continue
            if 1024 <= value <= 10_000_000:
                return value
    return 0


def stream_error(message: str, text_for_class: str) -> ProviderError:
    """An in-stream error event as a ProviderError; overloaded / rate-limit events are retriable."""
    return ProviderError(message, retriable=retriable_message(text_for_class), kind="stream")


def error_text(err) -> str:
    """Readable text of an in-stream ``error`` payload (dict or string)."""
    if isinstance(err, dict):
        return str(err.get("message") or err.get("detail") or err.get("code") or err)
    return str(err)

