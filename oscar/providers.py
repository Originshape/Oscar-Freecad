"""Chat providers for Oscar.

One normalized conversation format is used inside the agent; each provider
class converts it to its wire protocol:

- ``Message`` — role ``user``/``assistant``, plain-text content, optional
  tool calls the assistant requested, and optional tool results (with an
  optional PNG screenshot as a base64 string for vision turns).

Providers stream responses as event tuples ``(kind, payload)`` so the panel
can render text as it arrives:

    ("text", str)                  — a chunk of assistant text
    ("reasoning", str)             — a chunk of the model's thinking text
                                     (DeepSeek-R1 style reasoning_content,
                                     Anthropic extended-thinking deltas)
    ("assistant_blocks", [dict])   — the provider's content blocks to replay
                                     unchanged on the next request of a tool
                                     loop (Anthropic thinking + signature)
    ("usage", dict)                — token counts exactly as the server
                                     reports them (never merged or renamed)
    ("stop_reason", str)           — the RAW provider stop reason
    ("stop_info", dict)            — {"reason": normalized enum, "raw", "truncated",
                                     "complete", "usage": {"prompt_tokens",
                                     "completion_tokens"}}
    ("tool_calls", [ToolCall])     — the model wants tools run (end of turn);
                                     calls whose arguments could not be decoded
                                     carry ``args_error``
    ("done", stop_reason)          — turn finished without pending tool calls
    ("error", Exception)           — terminal provider/transport failure
    ("tick", dict)                 — liveness + progress, at least once a second
    ("tool_progress", dict)        — tool-call argument text is streaming
    ("reset", str)                 — partial output of the attempt is void
                                     ("retry" | "stream_dropped" | "adapted")

Order per ``chat()`` call: any number of tick / tool_progress / reasoning /
text / reset, then assistant_blocks?, usage?, stop_reason, stop_info, then
exactly one of tool_calls / done; or a terminal error.

Everything here is blocking and must run on a worker thread (the agent owns
one). The whole attempt (the request AND the stream parse) runs on one
daemon reader thread; ``chat()`` polls ``should_stop`` every 200 ms, so the
user's Stop button works while a local model is still loading.

Text tool mode (``ModelCaps.tools == "text"``, ``oscar.provider_text``): the OpenAI-compatible and
Ollama providers omit the native ``tools`` array, append a compact protocol to the system prompt, serialize the
history as ``<tool_call>`` / ``<tool_result>`` blocks, withhold tool-call markup from the ``text`` events and
deliver the parsed calls as an ordinary ``tool_calls`` event. Inline ``<think>`` spans become ``reasoning`` events.

Only ``requests`` is used — it ships with the bundled Python of the app.
"""

import json
import re
import types
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generator

import requests  # noqa: F401  (tests patch ``providers.requests.post``; the transport lives in provider_chat)

from . import models as _models
from .provider_args import decode_tool_arguments  # noqa: F401  (public: shared by the three parsers)
from .provider_chat import ChatLoop
from .provider_reasoning import (
    REASONING_FIELDS, anthropic_forced_choice_ok, mentions_field, normalize_effort, reasoning_fragment,
)
from .provider_stream import (
    StreamEvent, ToolProgress as _ToolProgress, context_window_from_error as _context_window_from_error,
    error_text as _error_text, looks_like_image_rejection as _looks_like_image_rejection,  # noqa: F401  (tests / callers)
    looks_like_tools_unsupported as _looks_like_tools_unsupported,
    sse_lines as _sse_lines, stop_info as _stop_info, stream_error as _stream_error,
)
from . import provider_text as _text
from .provider_probe import list_models  # noqa: F401  (public: the /models listing; lives with the other metadata calls)
from .provider_text import extract_text_tool_calls  # noqa: F401  (public: the text-mode tool-call parser)
from .provider_transport import (  # noqa: F401  (StopRequested: public name)
    ProviderError, StopRequested, anthropic_base as _anthropic_base,
)
from .stopreason import normalize_stop_reason  # noqa: F401  (public: the shared stop-reason normalizer)
from .turnlog import log_event

_IMAGES_OMITTED_NOTE = "[image attachment(s) omitted — this model cannot process images]"
_SCREENSHOT_OMITTED_NOTE = "[screenshot omitted — this model cannot process images]"
_OLD_SCREENSHOT_NOTE = "[earlier screenshot omitted to save context — newer ones follow]"

# OpenAI-compatible requests carry at most this many images (newest kept).
# Screenshots pile up over a long build; small/local models choke on them
# and most endpoints cap images per request.
MAX_WIRE_IMAGES = 6

# Anthropic tool_use ids must match this pattern; ids minted by other
# providers (a profile switch mid-conversation) are normalized to it.
_ANTHROPIC_ID_BAD = re.compile(r"[^A-Za-z0-9_-]")
_ANTHROPIC_BLOCK_TYPES = frozenset({"text", "tool_use", "thinking", "redacted_thinking"})


def _anthropic_id(raw: str) -> str:
    cleaned = _ANTHROPIC_ID_BAD.sub("_", raw or "")
    return cleaned or "toolu_missing"


def _anthropic_accepts_sampling(model: str) -> bool:
    """Whether a Claude model still accepts ``temperature``.

    Sampling parameters were removed (HTTP 400 when sent) from Claude
    Opus 4.7 onwards — Opus 4.7/4.8, Opus 5.x, Sonnet 5, Fable, Mythos.
    Claude 3.x, 4.0–4.6 and Haiku 4.5 still accept them. Unknown naming
    schemes keep sending it; a 400 that names the parameter adapts anyway.
    """
    lowered = (model or "").lower()
    if any(old in lowered for old in ("claude-3", "claude-2", "claude-instant")):
        return True
    match = re.search(r"claude-(?:opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d+))?", lowered)
    if not match:
        return True
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    if minor > 99:  # a date suffix (claude-opus-4-20250514), not a minor version
        minor = 0
    return major < 4 or (major == 4 and minor <= 6)


def _openai_reasoning_model(model: str) -> bool:
    """OpenAI reasoning models (o1/o3/o4, gpt-5) reject ``max_tokens`` and a
    non-default ``temperature``."""
    lowered = (model or "").lower().rsplit("/", 1)[-1]
    return bool(re.match(r"(o\d|gpt-5)", lowered))


# Adaptations learned from 400 responses, per (provider class, base url,
# model): a new AgentWorker (one per turn) must not re-discover them. The
# facts that matter beyond this process are also persisted through
# ``models.learn`` (settings["model_caps"]).
_LEARNED: dict[tuple[str, str, str], dict[str, Any]] = {}


@dataclass
class ToolCall:
    """One tool call the model requested. ``args_error`` / ``raw_args`` are set when the arguments could not be decoded."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    args_error: str = ""   # "" = arguments decoded fine; else the human reason. When set, arguments == {"_raw": raw}
    raw_args: str = ""     # the raw argument text (clipped to 20_000 chars) when args_error is set


@dataclass
class ToolResult:
    """The app's answer to one ``ToolCall`` (text, an error flag and an optional PNG screenshot for vision models)."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    image_b64: str = ""  # PNG, base64 — attached for vision on the next turn


@dataclass
class Message:
    """One turn of the normalized conversation; every provider converts it to its own wire format."""

    role: str  # "user" | "assistant"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant only
    tool_results: list[ToolResult] = field(default_factory=list)  # user only
    images: list[str] = field(default_factory=list)  # user only; PNG base64
    # assistant only: the provider's raw content blocks for this message
    # (Anthropic thinking blocks with signatures, text, tool_use), replayed
    # verbatim — thinking models reject tool loops whose thinking was dropped
    provider_blocks: list[dict[str, Any]] = field(default_factory=list)
    synthetic: bool = False  # True for app-injected notes (cutoff / gate / watchdog / breaker / steer / wrapup / resume)
    tag: str = ""            # "" | "cutoff" | "gate" | "watchdog" | "breaker" | "steer" | "wrapup" | "resume"


@dataclass
class ToolSpec:
    """A tool offered to the model: name, description and a JSON Schema for its arguments."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


_TOOL_ARGS_CLIP = 20_000
_BUFFERED_ARGS_MIN_CHARS = 400   # tool arguments at least this long that arrive in ONE fragment prove a buffering server


def _make_call(call_id: str, name: str, raw, stop_reason: str = "") -> ToolCall:
    """A ToolCall from the raw argument text, through the tolerant decoder."""
    arguments, error = decode_tool_arguments(raw, stop_reason=stop_reason)
    raw_text = raw if isinstance(raw, str) else ""
    return ToolCall(id=call_id, name=name, arguments=arguments, args_error=error,
                    raw_args=raw_text[:_TOOL_ARGS_CLIP] if error else "")


def _as_dict(value) -> dict:
    """``value`` when it is a dict, else ``{}``: a gateway may send ``null`` (or a scalar) where an object belongs."""
    return value if isinstance(value, dict) else {}


def _as_text(value) -> str:
    """``value`` when it is a string, else ``""`` (``null`` chunks are empty, never ``None`` concatenated to text)."""
    return value if isinstance(value, str) else ""


def _content_parts(content) -> tuple[str, str]:
    """``(text, reasoning)`` of an OpenAI-style ``delta.content``: a string, or a list of typed parts.

    Some gateways (Mistral's reasoning models among them) stream ``content`` as a list of ``{"type": "text" |
    "thinking", ...}`` parts instead of a string; anything else yields nothing.
    """
    if isinstance(content, str):
        return content, ""
    text, thinking = [], []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                text.append(part)
                continue
            part = _as_dict(part)
            if part.get("type") == "thinking":
                inner = part.get("thinking")
                if isinstance(inner, list):
                    thinking.extend(_as_text(_as_dict(p).get("text")) for p in inner)
                else:
                    thinking.append(_as_text(inner))
            else:
                text.append(_as_text(part.get("text")))
    return "".join(text), "".join(thinking)


# Hosts known to accept `stream_options` / `parallel_tool_calls` (many gateways answer "unrecognized request
# argument"); every other host gets them only once a request proved it accepts them (learned `<field>_ok`).
_OPTIONAL_FIELD_HOSTS = frozenset({"openai", "openrouter", "zai", "deepseek", "groq", "together", "fireworks"})
# Optional request fields a 400 may name; each is dropped and learned per (host, model).
_OPTIONAL_FIELDS = (
    "stream_options", "parallel_tool_calls", "tool_choice", *REASONING_FIELDS, "options",
)


class BaseProvider(ChatLoop):
    """Common plumbing: capability access, request adaptation; ``ChatLoop`` supplies ``chat()`` (reader thread, ticks, retries)."""

    name = "base"
    kind = ""                     # the Profile.provider value this class serves (capability resolution for bare profiles)
    # Tools the TEXT parser accepts although the request does not offer them. The agent sets the whole registry: a tool
    # that is hidden in the current phase still runs when a native call names it ("visibility is not permission"), so a
    # well-formed text call to one must not be dropped (and shown to the user as raw markup) either.
    known_tool_names: frozenset = frozenset()
    # Who is asking, for providers that report it to their server (OriginShape's ``x-oscar-*`` headers):
    # ``{"session": <chat id>, "turn": <turn id>}``. The agent ASSIGNS a new dict per turn / round before ``chat()``
    # starts the reader thread; the shared default is read-only, so mutating it in place fails loudly.
    request_context: Mapping[str, str] = types.MappingProxyType({})

    def __init__(self, profile):  # oscar.params.Profile
        self.profile = profile
        # Flips to False when the model rejects image content (HTTP 400);
        # the request is then retried and future turns degrade to text-only
        # with an explicit note, instead of failing the whole conversation.
        # Optimistic until a rejection (or a KNOWN no-vision model, below).
        self._vision_ok = True
        # Request-parameter adaptations (sampling dropped, token-parameter
        # renamed, output cap lowered, optional fields refused) shared by every
        # instance talking to the same model; seeded from what the model name
        # implies and from what earlier runs persisted.
        key = (type(self).__name__, (profile.base_url or "").rstrip("/"), profile.model or "")
        self._learned = _LEARNED.setdefault(key, {})
        self._seed_from_caps()

    # -- capabilities -------------------------------------------------------

    def _caps_subject(self):
        """The profile, or (for the bare profiles tests pass) a view of it that names this provider kind."""
        profile = self.profile
        if getattr(profile, "provider", ""):
            return profile
        return types.SimpleNamespace(
            provider=self.kind, base_url=getattr(profile, "base_url", "") or "",
            model=getattr(profile, "model", "") or "", reasoning=getattr(profile, "reasoning", "auto"),
            caps_override=getattr(profile, "caps_override", None) or {})

    def _caps(self) -> _models.ModelCaps:
        """Resolved capabilities of this model (cache-only, never raises)."""
        try:
            return _models.resolve_caps(self._caps_subject())
        except Exception:  # noqa: BLE001 - capability lookup must never fail a request
            return _models.ModelCaps()

    def _seed_from_caps(self) -> None:
        subject = self._caps_subject()
        try:
            persisted = _models.learned_fields(subject)
        except Exception:  # noqa: BLE001
            persisted = {}
        caps_fields = _models.ModelCaps.__dataclass_fields__
        for name, value in persisted.items():
            if name == "sampling_ok":
                self._learned.setdefault("sampling", value)
            elif name not in caps_fields:
                self._learned.setdefault(name, value)
        self._vision_ok = self._caps().vision is not False

    def _learn(self, **fields) -> None:
        """Persist facts learned from a rejection (``models.learn``); never raises."""
        try:
            _models.learn(self._caps_subject(), **fields)
        except Exception:  # noqa: BLE001
            pass

    def _host(self) -> str:
        return _models.host_class(self._caps_subject())

    @property
    def vision_enabled(self) -> bool:
        """False once this provider degraded to text-only (a rejection was seen or the model is known to lack vision)."""
        return self._vision_ok

    def _set_vision(self, value: bool, evidence: str = "", *, remember: bool = True) -> None:
        """Switch image support for this provider. ``vision=False`` is remembered across sessions only when
        ``evidence`` (the 400 body) is an unambiguous image rejection: a size / base64 error degrades THIS session
        and nothing more (``models.learn`` enforces it). ``remember=False`` degrades this provider only; ``chat()``
        remembers the rejection itself once the text-only retry has been accepted (a retry that fails too could have
        been something else entirely)."""
        self._vision_ok = value
        if not value and remember:
            self._learn(vision=False, evidence=evidence)

    def _max_tokens(self, per_call: int | None = None) -> int:
        """Output cap of this request: the per-call value (default the profile's), bounded by the learned cap."""
        cap = self._learned.get("max_tokens_cap")
        value = int(per_call or self.profile.max_tokens or 4096)
        return min(value, cap) if cap else value

    def _sent_max_tokens(self, sent: dict) -> int:
        """The output cap the rejected request actually carried (a per-call value may differ from the profile's)."""
        for name in ("max_tokens", "max_completion_tokens"):
            value = sent.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return self._max_tokens()

    def _effort(self, per_call) -> str | None:
        """The effective reasoning effort, or None = send NO reasoning parameter.

        A per-call value wins over the profile; ``auto`` (the profile default) without a per-call value
        sends nothing, which is today's wire shape.
        """
        return normalize_effort(per_call) or normalize_effort(getattr(self.profile, "reasoning", "auto"))

    def _optional_ok(self, name: str, hosts: frozenset | None = None) -> bool:
        """Whether the optional request field ``name`` may be sent: learned flag first, else the host rule."""
        flag = self._learned.get(f"{name}_ok")
        if flag is not None:
            return bool(flag)
        return True if hosts is None else self._host() in hosts

    # -- per-attempt facts shared by ``_body`` (consumer thread) and ``_parse_stream`` (reader thread) -------------

    def _note_attempt(self, caps: _models.ModelCaps, tools: list, opts: dict) -> None:
        """Record what the parser needs to know about the request just built: text tool mode, the tool names,
        the call cap. Assigned before the reader thread starts, so the parser sees it."""
        text_mode = caps.tools == "text" and bool(tools) and opts.get("tool_choice") != "none"
        self._attempt = types.SimpleNamespace(
            text_mode=text_mode,
            tool_names=(frozenset(t.name for t in tools) | frozenset(self.known_tool_names or ()))
            if tools and caps.tools != "none" else frozenset(),
            max_calls=1 if opts.get("parallel_tool_calls") is False else 0, caps_tools=caps.tools,
            replay_reasoning=bool(caps.reasoning_replay))
        self._wire_text_mode = caps.tools == "text"
        self._wire_replay = bool(caps.reasoning_replay)

    def _attempt_info(self):
        """The ``_note_attempt`` record, or a neutral one (native mode, no tool names) for a parser fed directly."""
        info = getattr(self, "_attempt", None)
        if info is None:
            info = types.SimpleNamespace(text_mode=False, tool_names=frozenset(), max_calls=0, caps_tools="native",
                                         replay_reasoning=False)
        return info

    def _text_filter(self) -> "_text.TextStreamFilter":
        """The stream filter of the current attempt (reasoning split, tool-markup hold-back, text-mode extraction)."""
        info = self._attempt_info()
        return _text.TextStreamFilter(info.tool_names, text_mode=info.text_mode, max_calls=info.max_calls)

    def _record_args_streaming(self, streams: bool) -> None:
        """Learn whether this server streams a tool call's arguments incrementally (``tool_args_stream``).

        Only a change is written: the watchdog allows long silent stretches for a server that buffers them.
        """
        try:
            if self._caps().tool_args_stream != streams:
                self._learn(tool_args_stream=streams)
        except Exception:  # noqa: BLE001 - learning is best effort
            pass

    def _tool_choice(self, tool_choice, tools: list[ToolSpec], caps: _models.ModelCaps) -> tuple[str | None, bool]:
        """``(choice, drop_tools)``: ``choice`` is None | "none" | "required" | a tool name usable on this wire.

        ``drop_tools`` means "no tools array at all" (no tools offered, ``caps.tools == "none"``, or
        ``tool_choice == "none"`` on a host that cannot express it).
        """
        if not tools or caps.tools == "none":
            return None, True
        want = (tool_choice or "auto") if isinstance(tool_choice, str) else "auto"
        if want == "auto":
            return None, False
        usable = caps.tool_choice_ok and self._learned.get("tool_choice_ok", True)
        if want == "none":
            return ("none", False) if usable else (None, True)
        if not usable:
            return None, False
        if want == "required":
            return "required", False
        return (want, False) if any(t.name == want for t in tools) else (None, False)

    def _adapt_anthropic_thinking(self, lower: str, sent: dict) -> bool:
        """The documented Anthropic thinking 400s (verified from docs .../thinking-troubleshooting), each answered with
        the documented remedy; True when the request changed and a retry is worthwhile.

        * ``"thinking.type.enabled" is not supported`` -> the model is adaptive-only: learn ``anthropic_adaptive``.
        * ``adaptive thinking is not supported`` -> extended-thinking only: learn ``anthropic_budget``.
        * ``"thinking.type.disabled"`` / ``between_tools`` not supported -> ``off`` becomes ``effort: low``.
        * ``Invalid `signature` in `thinking` block`` / ``thinking ... blocks ... cannot be modified`` -> the replayed
          blocks no longer match the model's own output (Opus 5.5 / Sonnet 5.5 / Fable 5.1 bind a block to the
          unchanged system + tools + earlier messages prefix, which phase-dependent tool lists and compaction change):
          strip every thinking block from the history and send no thinking this process (the documented fallback).
        """
        if "thinking" not in lower and "signature" not in lower:
            return False
        thinking = sent.get("thinking") if isinstance(sent.get("thinking"), dict) else {}
        kind = thinking.get("type")
        caps = self._caps()
        if kind == "enabled" and re.search(r"thinking\.type\.enabled\W+is not supported", lower):
            if caps.reasoning_control == "anthropic_adaptive":
                return False
            self._learn(reasoning_control="anthropic_adaptive")
            return self._caps().reasoning_control == "anthropic_adaptive"
        if kind == "adaptive" and "adaptive thinking is not supported" in lower:
            self._learn(reasoning_control="anthropic_budget")
            return self._caps().reasoning_control == "anthropic_budget"
        if kind in ("disabled", "between_tools") and re.search(
                rf"thinking\.type\.{kind}\W+is not supported|send \W?thinking\W+type\W+between_tools", lower):
            if not self._learned.get("thinking_off_ok", True):
                return False
            self._learned["thinking_off_ok"] = False
            return True
        if re.search(r"invalid `?signature`? in `?thinking`? block|(?:thinking|redacted_thinking)`? blocks? in the latest"
                     r" assistant message cannot be modified", lower):
            if self._learned.get("strip_thinking"):
                return False
            self._learned["strip_thinking"] = True
            log_event("provider_adapt", fields=["thinking_blocks"], model=getattr(self.profile, "model", ""))
            return True
        return False

    def _adapt_to_rejection(self, detail: str, body: dict | None = None) -> bool:
        """Adjust the request after an HTTP 400 that names a parameter this
        model does not accept. Returns True when a retry is worthwhile.

        Models keep tightening their request surface (current Claude models
        reject ``temperature``; OpenAI reasoning models reject
        ``max_tokens`` and a custom temperature) — a known rejection must
        degrade the request, never fail the user's turn. The generic rule:
        a 400 naming one of the optional fields this layer adds drops
        exactly that field and learns it per (host, model).
        """
        lower = (detail or "").lower()
        sent = body if isinstance(body, dict) else {}
        if "tools" in sent and _looks_like_tools_unsupported(lower) and self._caps().tools == "native":
            # "does not support tools": switch the model to text tool mode (remembered) and retry. A user override
            # of ``tools`` beats what is learned, so the retry is only worthwhile when the flip took effect.
            self._learn(tools="text")
            if self._caps().tools == "text":
                log_event("provider_adapt", fields=["tools"], model=getattr(self.profile, "model", ""))
                return True
            return False
        if self._adapt_anthropic_thinking(lower, sent):
            return True
        if "think" in sent and re.search(r"does not support think", lower) and self._learned.get("think_ok", True):
            self._learned["think_ok"] = False
            self._learn(think_ok=False, reasoning_control="none")
            return True
        if (
            "max_completion_tokens" in lower
            and "max_tokens" in (sent or {"max_tokens": 0})
            and not self._learned.get("completion_tokens")
        ):
            self._learned["completion_tokens"] = True
            self._learn(completion_tokens=True)
            return True
        if "token" in lower:
            # Anthropic: "max_tokens: 64000 > 8192, which is the maximum…"
            # OpenAI: "…supports at most 16384 completion tokens…"
            # Groq & co: "…must be less than or equal to `8192`…"
            for pattern in (
                r"max_tokens:\s*\d+\s*>\s*(\d+)",
                r"at most\s*(\d+)",
                r"less than or equal to\s*`?(\d+)",
            ):
                cap = re.search(pattern, lower)
                if cap:
                    limit = int(cap.group(1))
                    if 0 < limit < self._sent_max_tokens(sent):
                        self._learned["max_tokens_cap"] = limit
                        self._learn(max_tokens_cap=limit, max_output=limit)
                        return True
                    break
        if (
            ("temperature" in lower or "top_p" in lower)
            and "temperature" in (sent or {"temperature": 0})
            and self._learned.get("sampling", True)
        ):
            self._learned["sampling"] = False
            self._learn(sampling_ok=False)
            return True
        window = _context_window_from_error(lower)
        if window:
            # The prompt did not fit: nothing to retry at this layer (the agent must send less), but the NEXT
            # request must be sized for the real window.
            self._learn(context_window=window)
            return False
        dropped = [name for name in _OPTIONAL_FIELDS
                   if name in sent and mentions_field(lower, name) and self._learned.get(f"{name}_ok", True)]
        # fields nested inside other top-level fields (Anthropic cache_control, disable_parallel_tool_use)
        for name, where in (("cache_control", ("system", "tools")), ("disable_parallel_tool_use", ("tool_choice",))):
            if (mentions_field(lower, name) and self._learned.get(f"{name}_ok", True)
                    and any(name in json.dumps(sent.get(part)) for part in where)):
                dropped.append(name)
        if not dropped:
            return False
        for name in dropped:
            self._learned[f"{name}_ok"] = False
            self._learn(**{f"{name}_ok": False})
            if name == "tool_choice":
                self._learn(tool_choice_ok=False)
            elif name in REASONING_FIELDS:
                self._learn(reasoning_control="none")
        log_event("provider_adapt", fields=dropped, model=getattr(self.profile, "model", ""))
        return True

    # -- subclass API -----------------------------------------------------

    def _endpoint(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _body(self, system: str, messages: list[Message], tools: list[ToolSpec], **opts) -> dict:
        """The request body. Keyword options: ``tool_choice``, ``reasoning``, ``max_tokens``, ``parallel_tool_calls``."""
        raise NotImplementedError

    def _parse_stream(self, response, should_stop) -> Generator[StreamEvent, None, None]:
        raise NotImplementedError

    # -- helpers for subclasses -------------------------------------------

    @staticmethod
    def _body_has_images(body: dict) -> bool:
        try:
            text = json.dumps(body)
        except (TypeError, ValueError):
            return True  # assume the worst; the retry guard still applies
        return "image_url" in text or '"type": "image"' in text or '"type":"image"' in text

    @staticmethod
    def _image_block(b64: str) -> dict[str, Any]:
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        }



class AnthropicProvider(BaseProvider):
    """Claude — Messages API with SSE streaming."""

    name = "anthropic"
    kind = "anthropic"

    def _endpoint(self) -> str:
        return f"{_anthropic_base(self.profile)}/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.profile.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _upstream_model(self) -> str:
        """The model id that decides the wire shape (thinking spelling, sampling, forced tool choice): the profile's
        model."""
        return self.profile.model or ""

    @staticmethod
    def _open_loop_has_thinking(messages: list[Message]) -> bool:
        """True when the open tool loop (after the last real user message) replays thinking blocks."""
        for m in reversed(messages):
            if m.role == "user" and not m.tool_results:
                return False
            if m.role == "assistant" and any(
                b.get("type") in ("thinking", "redacted_thinking") for b in m.provider_blocks
            ):
                return True
        return False

    def _body(self, system, messages, tools, **opts):
        caps = self._caps()
        max_tokens = self._max_tokens(opts.get("max_tokens"))
        cache = {"type": "ephemeral"} if caps.prompt_cache and self._learned.get("cache_control_ok", True) else None
        body: dict[str, Any] = {
            "model": self.profile.model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": cache}] if cache and system else system,
            "messages": [self._to_wire(m) for m in messages],
            "stream": True,
        }
        thinking: dict = {}
        upstream = self._upstream_model()
        effort = self._effort(opts.get("reasoning"))
        control = caps.reasoning_control
        if (
            effort is not None
            and control in ("anthropic_budget", "anthropic_adaptive")
            and self._learned.get("thinking_ok", True)
            and not self._learned.get("strip_thinking")
        ):
            if effort == "off" and caps.reasoning_replay and self._open_loop_has_thinking(messages):
                # Verified from docs (extended-thinking / thinking: "Thinking with tool use"): a tool loop is ONE
                # assistant turn that runs in a single thinking mode; toggling inside it does not error, it silently
                # disables thinking and may strip the replayed blocks (and changing the configuration invalidates the
                # prompt cache). So the loop keeps thinking on, at the cheapest level, and `off` applies to the next turn.
                effort = "low"
            thinking = reasoning_fragment(control, effort, model=upstream, max_tokens=max_tokens)
            kind = (thinking.get("thinking") or {}).get("type")
            if kind in ("disabled", "between_tools") and not self._learned.get("thinking_off_ok", True):
                thinking = {"output_config": {"effort": "low"}}   # this model rejected the off spelling (learned)
            thinking = {k: v for k, v in thinking.items() if self._learned.get(f"{k}_ok", True)}
        # Claude Opus 4.7+ / Sonnet 5 / Fable reject sampling parameters
        # outright (HTTP 400) — send temperature only where it is accepted
        # (and never together with thinking).
        if not thinking and self._learned.get(
            "sampling", _anthropic_accepts_sampling(upstream) and caps.sampling_ok
        ):
            body["temperature"] = self.profile.temperature
        body.update(thinking)
        choice, drop_tools = self._tool_choice(opts.get("tool_choice"), tools, caps)
        if tools and not drop_tools:
            wire_tools = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
            if cache:
                wire_tools[-1]["cache_control"] = cache
            body["tools"] = wire_tools
            tool_choice = self._anthropic_choice(choice, thinking, caps, upstream)
            if opts.get("parallel_tool_calls") is False and caps.parallel_calls \
                    and self._learned.get("disable_parallel_tool_use_ok", True):
                tool_choice = dict(tool_choice or {"type": "auto"})
                tool_choice["disable_parallel_tool_use"] = True
            if tool_choice:
                body["tool_choice"] = tool_choice
        return body

    @staticmethod
    def _anthropic_choice(choice: str | None, thinking: dict, caps: _models.ModelCaps, model: str = "") -> dict | None:
        """The Messages API ``tool_choice``; a forced choice is downgraded to auto where the API rejects it.

        Verified from docs https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools: ``auto`` and
        ``none`` always work; ``any`` / ``tool`` are an error with manual extended thinking
        (``thinking.type == "enabled"``) and on Opus 5.5, Sonnet 5.5, Fable 5.1 and Mythos 5.1 whatever the thinking
        setting is; adaptive thinking allows them elsewhere.
        """
        if choice is None:
            return None
        if choice == "none":
            return {"type": "none"}
        if not anthropic_forced_choice_ok(model):
            return None
        if thinking and not caps.tool_choice_required_with_reasoning:
            return None   # the API rejects "any" / "tool" with thinking; the agent's nudge text carries the instruction
        if choice == "required":
            return {"type": "any"}
        return {"type": "tool", "name": choice}

    def _to_wire(self, m: Message) -> dict[str, Any]:
        if m.role == "assistant":
            if m.provider_blocks:
                # Replay exactly what the model produced — thinking blocks
                # (with their signatures) included. Thinking models reject a
                # tool loop whose earlier thinking was dropped or edited.
                # Blocks of another provider (a profile switch mid-conversation)
                # are not Anthropic content and are left out.
                strip = bool(self._learned.get("strip_thinking"))   # learned from a signature / modified-block 400
                blocks = [dict(b) for b in m.provider_blocks
                          if isinstance(b, dict) and b.get("type") in _ANTHROPIC_BLOCK_TYPES
                          and not (strip and b.get("type") in ("thinking", "redacted_thinking"))]
                for block in blocks:
                    if block.get("type") == "tool_use":
                        block["id"] = _anthropic_id(block.get("id", ""))
                if any(b.get("type") != "thinking" and b.get("type") != "redacted_thinking" for b in blocks):
                    return {"role": "assistant", "content": blocks}
            blocks = []
            if m.content and m.content.strip():
                blocks.append({"type": "text", "text": m.content})
            for call in m.tool_calls:
                blocks.append({
                    "type": "tool_use",
                    "id": _anthropic_id(call.id),
                    "name": call.name,
                    "input": call.arguments if isinstance(call.arguments, dict)
                    and "_raw" not in call.arguments and not call.args_error else {},
                })
            # The API rejects empty text blocks; an assistant turn with
            # nothing to say still needs one non-empty block.
            return {"role": "assistant", "content": blocks or [{"type": "text", "text": "(no reply)"}]}
        blocks = []
        for result in m.tool_results:
            text = result.content or "(no output)"
            if result.image_b64 and not self._vision_ok:
                text = f"{text}\n{_SCREENSHOT_OMITTED_NOTE}"
            content: list[dict[str, Any]] = [{"type": "text", "text": text}]
            if result.image_b64 and self._vision_ok:
                content.append(self._image_block(result.image_b64))
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": _anthropic_id(result.tool_call_id),
                    "content": content,
                    "is_error": result.is_error,
                }
            )
        if m.content and m.content.strip():
            blocks.append({"type": "text", "text": m.content})
        if m.images and self._vision_ok:
            blocks.extend({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b}} for b in m.images)
        elif m.images:
            blocks.append({"type": "text", "text": _IMAGES_OMITTED_NOTE})
        if not blocks:
            blocks.append({"type": "text", "text": "(empty message)"})
        return {"role": "user", "content": blocks}

    def _parse_stream(self, response, should_stop):
        # Content blocks by stream index, in arrival order. Delta events
        # address blocks by stream position (text and thinking blocks
        # included), never by tool id.
        blocks: dict[int, dict[str, Any]] = {}
        order: list[int] = []
        tool_json: dict[int, str] = {}
        stop_reason = "end_turn"
        usage: dict[str, Any] = {}
        complete = False
        progress = _ToolProgress()

        for payload in _sse_lines(response):
            if should_stop():
                raise StopRequested()
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type", "")
            if kind == "message_start":
                message_usage = _as_dict(event.get("message")).get("usage")
                if isinstance(message_usage, dict):
                    usage.update(message_usage)
            elif kind == "content_block_start":
                index = event.get("index", len(order))
                if not isinstance(index, (int, str)):
                    index = len(order)
                block = dict(_as_dict(event.get("content_block")))
                btype = block.get("type")
                if btype == "tool_use":
                    tool_json[index] = ""
                    started = progress.update(index, _as_text(block.get("name")), 0)
                    if started:
                        yield started
                elif btype == "thinking":
                    block["thinking"] = block.get("thinking") or ""
                    block["signature"] = block.get("signature") or ""
                elif btype == "text":
                    block = {"type": "text", "text": block.get("text") or ""}
                blocks[index] = block
                order.append(index)
            elif kind == "content_block_delta":
                index = event.get("index", -1)
                block = blocks.get(index) if isinstance(index, (int, str)) else None
                delta = _as_dict(event.get("delta"))     # a gateway may send null where an object belongs
                dtype = delta.get("type")
                if dtype == "text_delta":
                    chunk = _as_text(delta.get("text"))
                    if block is not None and block.get("type") == "text":
                        block["text"] += chunk
                    yield ("text", chunk)
                elif dtype == "thinking_delta":
                    chunk = _as_text(delta.get("thinking"))
                    if block is not None and block.get("type") == "thinking":
                        block["thinking"] += chunk
                    if chunk:
                        yield ("reasoning", chunk)
                elif dtype == "signature_delta":
                    if block is not None and block.get("type") == "thinking":
                        block["signature"] += _as_text(delta.get("signature"))
                elif dtype == "input_json_delta":
                    if index in tool_json:
                        tool_json[index] += _as_text(delta.get("partial_json"))
                        update = progress.update(index, (block or {}).get("name", ""), len(tool_json[index]))
                        if update:
                            yield update
            elif kind == "content_block_stop":
                index = event.get("index")
                if index in tool_json:
                    done = progress.final(index, (blocks.get(index) or {}).get("name", ""), len(tool_json[index]))
                    if done:
                        yield done
            elif kind == "message_delta":
                stop_reason = _as_text(_as_dict(event.get("delta")).get("stop_reason")) or stop_reason
                if isinstance(event.get("usage"), dict):
                    usage.update(event["usage"])
            elif kind == "error":
                err = event.get("error", {})
                err = err if isinstance(err, dict) else {"message": err}
                yield ("error", _stream_error(f"{err.get('type')}: {err.get('message')}",
                                              f"{err.get('type')} {err.get('message')}"))
                return
            elif kind == "message_stop":
                complete = True
                break

        if not complete:
            yield ("stream_end", {"complete": False})
            return

        calls: list[ToolCall] = []
        replay: list[dict[str, Any]] = []
        for index in order:
            block = blocks[index]
            btype = block.get("type")
            if btype == "tool_use":
                raw = tool_json.get(index, "").strip()
                call = _make_call(block.get("id", ""), block.get("name", ""), raw, stop_reason)
                final = progress.final(index, call.name, len(raw))
                if final:
                    yield final
                calls.append(call)
                replay.append({
                    "type": "tool_use", "id": block.get("id", ""), "name": block.get("name", ""),
                    "input": {} if "_raw" in call.arguments else call.arguments,
                })
            elif btype == "text":
                if block["text"]:
                    replay.append(block)
            elif btype == "thinking":
                if block["signature"]:  # unsigned thinking cannot be replayed
                    replay.append(block)
            elif btype == "redacted_thinking":
                replay.append(block)

        if any(b.get("type") in ("thinking", "redacted_thinking") for b in replay):
            yield ("assistant_blocks", replay)
        if usage:
            yield ("usage", usage)
        yield ("stop_reason", stop_reason)
        yield ("stop_info", _stop_info(stop_reason, usage, "anthropic", has_calls=bool(calls)))
        if calls:
            yield ("tool_calls", calls)
        else:
            yield ("done", stop_reason)
        yield ("stream_end", {"complete": True})


class OpenAICompatProvider(BaseProvider):
    """Any /chat/completions endpoint: OpenAI, Ollama, OpenRouter, Groq,
    DeepSeek, LM Studio, vLLM, llama.cpp…"""

    name = "openai_compat"
    kind = "openai_compat"

    def _endpoint(self) -> str:
        base = (self.profile.base_url or "https://api.openai.com/v1").rstrip("/")
        if not base.endswith("/chat/completions"):
            base = base + "/chat/completions"
        return base

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.profile.api_key:
            headers["Authorization"] = f"Bearer {self.profile.api_key}"
        return headers

    def _is_openai_host(self) -> bool:
        return "api.openai.com" in (self.profile.base_url or "https://api.openai.com")

    def _body(self, system, messages, tools, **opts):
        caps = self._caps()
        self._note_attempt(caps, tools, opts)
        if caps.tools == "text":
            system = _text.system_with_protocol(system, tools, opts.get("tool_choice"), opts.get("parallel_tool_calls"))
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            wire.extend(self._to_wire(m))
        self._limit_images(wire)
        max_tokens = self._max_tokens(opts.get("max_tokens"))
        reasoning = _openai_reasoning_model(self.profile.model) and self._is_openai_host()
        body: dict[str, Any] = {
            "model": self.profile.model,
            "messages": wire,
            "stream": True,
        }
        # OpenAI reasoning models (o-series, gpt-5) reject a custom
        # temperature and the legacy max_tokens name (HTTP 400).
        if self._learned.get("sampling", (not reasoning) and caps.sampling_ok):
            body["temperature"] = self.profile.temperature
            if self._host() == "zai":
                body["temperature"] = min(1.0, max(0.0, float(self.profile.temperature)))   # docs: range 0.0-1.0
        if self._learned.get("completion_tokens", reasoning):
            body["max_completion_tokens"] = max_tokens
        else:
            body["max_tokens"] = max_tokens
        if self._optional_ok("stream_options", _OPTIONAL_FIELD_HOSTS):
            body["stream_options"] = {"include_usage": True}
        effort = self._effort(opts.get("reasoning"))
        if effort is not None:
            fragment = reasoning_fragment(
                caps.reasoning_control, effort, model=self.profile.model or "",
                base_url=self.profile.base_url or "", max_tokens=max_tokens)
            if caps.reasoning_control == "anthropic_budget":
                fragment = {}   # the Anthropic wire format
            for name, value in fragment.items():
                if self._learned.get(f"{name}_ok", True):
                    body[name] = value
        choice, drop_tools = self._tool_choice(opts.get("tool_choice"), tools, caps)
        if tools and not drop_tools and caps.tools != "text":
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            if choice == "required" or choice == "none":
                body["tool_choice"] = choice
            elif choice:
                body["tool_choice"] = {"type": "function", "function": {"name": choice}}
            if opts.get("parallel_tool_calls") is False and caps.parallel_calls \
                    and self._optional_ok("parallel_tool_calls", _OPTIONAL_FIELD_HOSTS):
                body["parallel_tool_calls"] = False
        return body

    @staticmethod
    def _limit_images(wire: list[dict[str, Any]]) -> None:
        """Keep only the newest MAX_WIRE_IMAGES images in the request.

        Every tool screenshot rides along as an extra user turn; over a
        long build they add up to megabytes per request, past what many
        OpenAI-compatible endpoints (and small local models) accept. Older
        screenshot turns become a one-line text note.
        """
        seen = 0
        for message in reversed(wire):
            content = message.get("content")
            if message.get("role") != "user" or not isinstance(content, list):
                continue
            kept = []
            dropped = False
            for part in reversed(content):
                if part.get("type") == "image_url":
                    if seen >= MAX_WIRE_IMAGES:
                        dropped = True
                        continue
                    seen += 1
                kept.append(part)
            if dropped:
                kept.reverse()
                texts = [p.get("text", "") for p in kept if p.get("type") == "text"]
                images = [p for p in kept if p.get("type") == "image_url"]
                note = " ".join(t for t in texts if t) + " " + _OLD_SCREENSHOT_NOTE
                message["content"] = (
                    [{"type": "text", "text": note.strip()}] + images if images else note.strip()
                )

    def _to_wire(self, m: Message) -> list[dict[str, Any]]:
        if getattr(self, "_wire_text_mode", False):
            return _text.openai_text_wire(m, vision_ok=self._vision_ok, images_note=_IMAGES_OMITTED_NOTE,
                                          screenshot_note=_SCREENSHOT_OMITTED_NOTE)
        if m.role == "assistant":
            # content may only be null next to tool_calls; an empty turn (a reasoning-only reply) needs the same
            # placeholder the Anthropic path uses.
            if m.tool_calls:
                content = m.content or None
            else:
                content = m.content if m.content and m.content.strip() else "(no reply)"
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if getattr(self, "_wire_replay", False):
                kept = next((b.get("text") for b in m.provider_blocks
                             if isinstance(b, dict) and b.get("type") == "reasoning_content" and b.get("text")), "")
                if kept:
                    message["reasoning_content"] = kept      # reasoning the model must see again in a tool loop (caps.reasoning_replay)
            if m.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                {} if ("_raw" in (call.arguments or {}) or call.args_error) else call.arguments
                            ),
                        },
                    }
                    for call in m.tool_calls
                ]
            return [message]
        out: list[dict[str, Any]] = []
        screenshots: list[dict[str, Any]] = []
        # Tool results go in discrete "tool" messages, in request order.
        for result in m.tool_results:
            text = result.content or "(no output)"
            if result.image_b64 and not self._vision_ok:
                text = f"{text}\n{_SCREENSHOT_OMITTED_NOTE}" if text else _SCREENSHOT_OMITTED_NOTE
            entry: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": text,
            }
            out.append(entry)
            if result.image_b64 and self._vision_ok:
                # The wire format has no image slot on tool messages; deliver
                # the screenshot as a follow-up user turn the model can see.
                screenshots.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"(screenshot after {result.name})",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{result.image_b64}"},
                            },
                        ],
                    }
                )
        # Strict servers require every tool message to directly follow the
        # assistant's tool_calls, so no user turn may sit between them.
        out.extend(screenshots)
        if m.content or m.images:
            if m.images and self._vision_ok:
                content: list[dict[str, Any]] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                content.extend(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b}"},
                    }
                    for b in m.images
                )
                out.append({"role": "user", "content": content})
            elif m.images:
                out.append({"role": "user", "content": f"{m.content}\n{_IMAGES_OMITTED_NOTE}".strip()})
            else:
                out.append({"role": "user", "content": m.content})
        return out

    def _parse_stream(self, response, should_stop):
        # index -> {"id":…, "name":…, "args": str, "n": argument fragments}; dict preserves order (3.7+)
        pending: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, Any] | None = None
        complete = False
        progress = _ToolProgress()
        flt = self._text_filter()          # inline <think> split, tool-markup hold-back, text-mode extraction
        reasoning_parts: list[str] = []    # the provider's own reasoning_content (replayed when caps.reasoning_replay)

        for payload in _sse_lines(response):
            if should_stop():
                raise StopRequested()
            if payload == "[DONE]":
                complete = True
                break
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]  # final chunk; may carry no choices
            # Gateways report failures that happen after the 200 was sent (rate limit, content filter, upstream
            # timeout) as an in-stream {"error": …} event; skipping it would end the turn as a clean, empty
            # "stop". OpenRouter's carries a choices entry too (finish_reason "error"), so the error is checked
            # BEFORE the choices.
            err = event.get("error")
            if err:
                detail = _error_text(err)
                marker = f"{err.get('type', '')} {err.get('code', '')} {detail}" if isinstance(err, dict) else detail
                yield ("error", _stream_error(f"{self.profile.model}: {detail}", marker))
                return
            choices = event.get("choices")
            choices = choices if isinstance(choices, list) else []
            if not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason") == "error":
                # the server ended the stream with an error and said nothing more: not a clean stop
                yield ("error", _stream_error(f"{self.profile.model}: stream ended with finish_reason=error", ""))
                return
            delta = _as_dict(choice.get("delta"))
            # Reasoning models stream their thinking separately from the
            # final answer: reasoning_content (DeepSeek, Ollama) or plain
            # reasoning (OpenRouter).
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning and isinstance(reasoning, str):
                reasoning_parts.append(reasoning)
                yield ("reasoning", reasoning)
            if delta.get("content"):
                chunk, thinking = _content_parts(delta["content"])   # a string, or a list of typed parts
                if thinking:
                    yield ("reasoning", thinking)
                if chunk:
                    yield from flt.feed(chunk)
            tool_deltas = delta.get("tool_calls")
            for call in tool_deltas if isinstance(tool_deltas, list) else []:
                if not isinstance(call, dict):
                    continue
                index = self._tool_slot_index(pending, call)
                slot = pending.setdefault(index, {"id": "", "name": "", "args": "", "n": 0})
                if call.get("id") and isinstance(call["id"], (str, int)):
                    slot["id"] = str(call["id"])
                fn = _as_dict(call.get("function"))
                name = fn.get("name")
                # Names normally stream once (or split in fragments); some
                # servers repeat the full name in every chunk.
                if name and isinstance(name, str) and name != slot["name"]:
                    slot["name"] += name
                arguments = fn.get("arguments")
                if arguments is not None and not isinstance(arguments, str):
                    arguments = json.dumps(arguments)   # non-standard: an object, not a string
                if arguments:
                    slot["args"] += arguments
                    slot["n"] += 1
                update = progress.update(index, slot["name"], len(slot["args"]))
                if update:
                    yield update
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
                complete = True  # keep reading: a usage-only chunk may still follow before [DONE]

        if not complete:
            events, _ = flt.finish(complete=False)
            yield from events
            yield ("stream_end", {"complete": False})
            return

        calls: list[ToolCall] = []
        for index in sorted(pending):
            slot = pending[index]
            raw = slot["args"].strip()
            final = progress.final(index, slot["name"], len(raw))
            if final and slot["name"]:
                yield final
            calls.append(_make_call(slot["id"] or f"call_{uuid.uuid4().hex[:12]}", slot["name"], raw, finish_reason))
        calls = [c for c in calls if c.name]
        events, extracted = flt.finish(stop_reason=finish_reason, extract=not calls)
        yield from events
        calls = calls or extracted        # a reply whose text IS the call(s): text mode, or a model that ignored the tools array
        self._learn_args_streaming(pending)
        if reasoning_parts and self._attempt_info().replay_reasoning:
            yield ("assistant_blocks", [{"type": "reasoning_content", "text": "".join(reasoning_parts)}])
        if usage:
            yield ("usage", usage)
        yield ("stop_reason", finish_reason)
        yield ("stop_info", _stop_info(finish_reason, usage, "openai", has_calls=bool(calls)))
        if calls:
            yield ("tool_calls", calls)
        else:
            yield ("done", finish_reason)
        yield ("stream_end", {"complete": True})

    def _learn_args_streaming(self, pending: dict[int, dict[str, Any]]) -> None:
        """Learn ``tool_args_stream`` from how the tool-call arguments of this reply arrived.

        A long argument string (>= 400 characters) that arrived in ONE fragment means the server buffers a call
        until it is complete (the watchdog then allows long silent stretches); one that arrived in four or more
        fragments means it streams. Short calls say nothing: a streaming server also sends them whole.
        """
        long_calls = [slot["n"] for slot in pending.values() if len(slot["args"]) >= _BUFFERED_ARGS_MIN_CHARS]
        if not long_calls:
            return
        if all(n <= 1 for n in long_calls):
            self._record_args_streaming(False)
        elif any(n >= 4 for n in long_calls):
            self._record_args_streaming(True)

    @staticmethod
    def _tool_slot_index(pending: dict[int, dict[str, Any]], call: dict[str, Any]) -> int:
        """Slot for a streamed tool-call fragment.

        The spec keys fragments by ``index``; some OpenAI-compatible
        servers omit it and send each complete call with its own id —
        without this, parallel calls would be glued into one invalid call.
        """
        index = call.get("index")
        if isinstance(index, int) and not isinstance(index, bool):
            return index
        call_id = call.get("id")
        call_id = str(call_id) if isinstance(call_id, (str, int)) and call_id else ""
        if call_id:
            for existing, slot in pending.items():
                if slot["id"] == call_id:
                    return existing
            return max(pending, default=-1) + 1
        return max(pending, default=0)


# Provider kinds announced in the settings dialog but not usable yet (``params.COMING_SOON_PROVIDERS``): kind -> name.
# Empty while nothing is announced; the mechanism stays for the next one.
COMING_SOON_KINDS: dict[str, str] = {}


def make_provider(profile):
    """Factory: build the provider matching a profile.

    Only constructs: it never touches the network (it runs on the GUI thread). ``originshape`` (the account) gets
    ``provider_originshape.OriginShapeProvider``; an Ollama host gets the native ``/api/chat`` provider unless
    ``caps_override["ollama_native"]`` is False. A provider that is only announced (``COMING_SOON_KINDS``) raises
    ``ProviderError``.
    """
    kind = getattr(profile, "provider", "")
    if kind in COMING_SOON_KINDS:
        raise ProviderError(f"{COMING_SOON_KINDS[kind]} is coming soon and cannot be used yet. "
                            "Choose another provider in Oscar's settings (⚙).")
    if kind == "anthropic":
        return AnthropicProvider(profile)
    if kind == "originshape":
        from .provider_originshape import OriginShapeProvider   # lazy: it brings in the sign-in package

        return OriginShapeProvider(profile)
    if _models.ollama_native_wanted(profile):
        from .provider_ollama import OllamaNativeProvider

        return OllamaNativeProvider(profile)
    if _models.host_class(profile) == "ollama":
        _models.clear_runtime_caps(profile)     # a /v1 profile must not inherit the native provider's window
    return OpenAICompatProvider(profile)
