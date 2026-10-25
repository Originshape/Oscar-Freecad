"""Native Ollama provider: ``POST <server>/api/chat`` with an NDJSON stream.

Why not ``/v1``: a per-request ``num_ctx`` is ignored over Ollama's OpenAI-compatible endpoint, so Ollama
silently truncates the prompt FROM THE FRONT at its own window (2k-4k tokens by default), dropping the system prompt
and the task with no error. The native endpoint takes ``options.num_ctx``, so this provider always states the
window it works with (``models.ollama_num_ctx``: the user's ``context_window`` override, else 16384 capped by the
model's trained maximum) and publishes it through ``models.set_runtime_caps`` so the agent budgets for it BEFORE
the first request.

Request body: ``{"model", "messages", "tools", "stream": true, "think": <bool | "low" | "medium" | "high">,
"options": {"num_ctx", "temperature", "num_predict"}}``. A user message carries its screenshots in ``images``
(base64, no data-URL prefix); an assistant message's calls are ``tool_calls[].function{name, arguments (object)}``;
a tool result is ``{"role": "tool", "tool_name", "content"}`` (Ollama has no call ids: ids are minted here).
Stream chunks: ``message.content`` / ``message.thinking`` / ``message.tool_calls[]`` (whole calls, arguments already
an object), then a ``done: true`` chunk with ``done_reason``, ``prompt_eval_count`` and ``eval_count``.

Documentation status, verified from docs
https://github.com/ollama/ollama/blob/main/docs/api.md and https://docs.ollama.com/capabilities/thinking: ``/api/chat``
takes ``model``, ``messages``, ``tools``, ``think`` (``true`` / ``false`` or a model-specific level string such as
``"low"|"medium"|"high"``; ``null`` = model default), ``stream``, ``format``, ``options`` (``num_ctx``, ``num_predict``,
``temperature``, ...) and ``keep_alive`` (default 5 minutes); a tool result is ``{"role": "tool", "content",
"tool_name"}``; ``images`` is an array of base64 strings; the stream has ``message.content`` / ``message.thinking`` /
``message.tool_calls[]``, ``done``, ``done_reason`` (``stop`` ... ), ``prompt_eval_count`` and ``eval_count``;
``GET /api/ps`` reports ``context_length`` of a running model. The OpenAI-compatible ``/v1`` accepts
``reasoning_effort`` / ``reasoning.effort`` and ``stream_options.include_usage`` but NOT ``tool_choice``
(https://docs.ollama.com/api/openai-compatibility) and has no context-size setting (Modelfile ``num_ctx`` only), which
is why this provider exists. Live-only: which models accept a level string (a rejection is learned), whether
``/v1`` streams tool arguments incrementally, and the error text for a model without thinking. Every read is
defensive, and a 400 that names ``think`` (or says the model does not support thinking / tools) is learned and
retried (``BaseProvider._adapt_to_rejection``).

Same event contract as every provider (``oscar.providers``), same reader thread; text tool mode, inline ``<think>``
splitting and the stream hold-back work exactly as on the OpenAI-compatible path.
"""

import json
import uuid
from typing import Any

from . import models as _models
from . import provider_text as _text
from .provider_chat import ChatLoop
from .provider_probe import server_root
from .provider_stream import ToolProgress as _ToolProgress
from .provider_stream import error_text as _error_text
from .provider_stream import ndjson_lines, stop_info as _stop_info, stream_error as _stream_error
from .provider_transport import StopRequested
from .providers import (
    MAX_WIRE_IMAGES, BaseProvider, Message, ToolCall, _as_dict, _as_text, _IMAGES_OMITTED_NOTE, _make_call,
    _OLD_SCREENSHOT_NOTE, _SCREENSHOT_OMITTED_NOTE,
)
from .turnlog import log_event

_USAGE_KEYS = ("prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration", "load_duration",
               "total_duration")
_CONTEXT_FULL_RATIO = 0.97   # a prompt this close to num_ctx was probably truncated by the server


class OllamaNativeProvider(BaseProvider):
    """Ollama over its own ``/api/chat`` endpoint (see the module docstring)."""

    name = "ollama"
    kind = "openai_compat"      # the Profile.provider value of Ollama profiles (they differ from OpenAI by host)

    def __init__(self, profile):
        super().__init__(profile)
        self._publish_window()

    # -- the window -------------------------------------------------------

    def _num_ctx(self) -> int:
        """The ``num_ctx`` this provider requests (``models.ollama_num_ctx``)."""
        return _models.ollama_num_ctx(self._caps_subject())

    def _publish_window(self) -> int:
        """Tell the capability layer which window requests use (the agent resolves caps cache-only before chatting)."""
        number = self._num_ctx()
        _models.set_runtime_caps(self._caps_subject(), context_window=number)
        return number

    # -- request ----------------------------------------------------------

    def _endpoint(self) -> str:
        return f"{server_root(self.profile.base_url or 'http://localhost:11434') or 'http://localhost:11434'}/api/chat"

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if getattr(self.profile, "api_key", ""):
            headers["Authorization"] = f"Bearer {self.profile.api_key}"
        return headers

    def _think_value(self, effort: str):
        """The ``think`` field for an effort: a bool for most models, a level string for gpt-oss (docs: levels are
        model-specific; gpt-oss is the documented level model)."""
        if effort == "off":
            return False
        if (self.profile.model or "").lower().startswith("gpt-oss"):
            return effort
        return True

    def _body(self, system, messages, tools, **opts):
        caps = self._caps()
        self._note_attempt(caps, tools, opts)
        if caps.tools == "text":
            system = _text.system_with_protocol(system, tools, opts.get("tool_choice"), opts.get("parallel_tool_calls"))
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            wire.extend(self._to_wire(m))
        self._limit_images(wire)
        num_ctx = self._publish_window()
        options: dict[str, Any] = {"num_ctx": num_ctx, "num_predict": self._max_tokens(opts.get("max_tokens"))}
        if self._learned.get("sampling", caps.sampling_ok):
            options["temperature"] = self.profile.temperature
        body: dict[str, Any] = {"model": self.profile.model, "messages": wire, "stream": True, "options": options}
        effort = self._effort(opts.get("reasoning"))
        if effort is not None and caps.reasoning_control == "ollama_effort" and self._optional_ok("think"):
            body["think"] = self._think_value(effort)
        choice, drop_tools = self._tool_choice(opts.get("tool_choice"), tools, caps)
        if tools and not drop_tools and caps.tools != "text" and choice != "none":
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        return body

    @staticmethod
    def _limit_images(wire: list[dict[str, Any]]) -> None:
        """Keep only the newest MAX_WIRE_IMAGES images in the request (older ones become a one-line note)."""
        seen = 0
        for message in reversed(wire):
            images = message.get("images")
            if not isinstance(images, list) or not images:
                continue
            room = max(0, MAX_WIRE_IMAGES - seen)
            if len(images) <= room:
                seen += len(images)
                continue
            kept = images[len(images) - room:] if room else []
            if kept:
                message["images"] = kept
            else:
                message.pop("images", None)
            message["content"] = f"{message.get('content') or ''} {_OLD_SCREENSHOT_NOTE}".strip()
            seen += len(kept)

    def _to_wire(self, m: Message) -> list[dict[str, Any]]:
        """One ``Message`` as Ollama chat messages (a list: a tool result with a screenshot adds a user message)."""
        text_mode = getattr(self, "_wire_text_mode", False)
        if m.role == "assistant":
            if text_mode:
                return [{"role": "assistant", "content": _text.assistant_text(m) or "(no reply)"}]
            message: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if getattr(self, "_wire_replay", False):
                kept = next((b.get("text") for b in m.provider_blocks
                             if isinstance(b, dict) and b.get("type") == "reasoning_content" and b.get("text")), "")
                if kept:
                    message["thinking"] = kept
            if m.tool_calls:
                message["tool_calls"] = [
                    {"function": {"name": c.name,
                                  "arguments": {} if (c.args_error or not isinstance(c.arguments, dict)
                                                      or "_raw" in c.arguments) else c.arguments}}
                    for c in m.tool_calls
                ]
            return [message]
        out: list[dict[str, Any]] = []
        images: list[str] = []
        if text_mode:
            text = _text.results_text(m, _SCREENSHOT_OMITTED_NOTE, vision_ok=self._vision_ok)
            if self._vision_ok:
                images.extend(r.image_b64 for r in m.tool_results if r.image_b64)
        else:
            screenshots: list[dict[str, Any]] = []
            for result in m.tool_results:
                content = result.content or "(no output)"
                if result.image_b64 and not self._vision_ok:
                    content = f"{content}\n{_SCREENSHOT_OMITTED_NOTE}"
                out.append({"role": "tool", "tool_name": result.name, "content": content})
                if result.image_b64 and self._vision_ok:
                    screenshots.append({"role": "user", "content": f"(screenshot after {result.name})",
                                        "images": [result.image_b64]})
            # chat templates group the tool messages that directly follow the assistant's calls: no user turn may
            # sit between them, so the screenshots come after every result
            out.extend(screenshots)
            text = m.content or ""
        if m.images and self._vision_ok:
            images.extend(m.images)
        elif m.images:
            text = f"{text}\n{_IMAGES_OMITTED_NOTE}".strip()
        if text or images:
            message = {"role": "user", "content": text or "(see the images)"}
            if images:
                message["images"] = images
            out.append(message)
        return out

    # -- adaptation / errors ----------------------------------------------

    @staticmethod
    def _body_has_images(body: dict) -> bool:
        """True when any message carries an ``images`` list (the 400 retry without images needs to know)."""
        messages = body.get("messages") if isinstance(body, dict) else None
        return any(isinstance(m, dict) and m.get("images") for m in messages or [])

    @staticmethod
    def _exhausted(kind: str, error, attempts: int) -> Exception:
        final = ChatLoop._exhausted(kind, error, attempts)
        if kind == "exc":
            return RuntimeError(f"{final}; is Ollama running? Start it with 'ollama serve' and check the base URL.")
        return final

    # -- stream -----------------------------------------------------------

    def _parse_stream(self, response, should_stop):
        flt = self._text_filter()
        progress = _ToolProgress()
        native: list[ToolCall] = []
        thinking_parts: list[str] = []
        usage: dict[str, Any] = {}
        done_reason = "stop"
        complete = False

        for chunk in ndjson_lines(response):
            if should_stop():
                raise StopRequested()
            err = chunk.get("error")
            if err:
                detail = _error_text(err)
                yield ("error", _stream_error(f"{self.profile.model}: {detail}", detail))
                return
            message = _as_dict(chunk.get("message"))
            thinking = _as_text(message.get("thinking"))
            if thinking:
                thinking_parts.append(thinking)
                yield ("reasoning", thinking)
            content = _as_text(message.get("content"))
            if content:
                yield from flt.feed(content)
            calls = message.get("tool_calls")
            for item in calls if isinstance(calls, list) else []:
                fn = _as_dict(_as_dict(item).get("function"))
                name = _as_text(fn.get("name"))
                if not name:
                    continue
                call_id = _as_text(_as_dict(item).get("id")) or f"call_{uuid.uuid4().hex[:12]}"
                arguments = fn.get("arguments")
                if isinstance(arguments, dict):
                    call = ToolCall(id=call_id, name=name, arguments=arguments)
                else:
                    call = _make_call(call_id, name, arguments if isinstance(arguments, str) else "", done_reason)
                native.append(call)
                size = len(json.dumps(call.arguments, default=str)) if not call.args_error else len(call.raw_args)
                final = progress.final(len(native) - 1, name, size)
                if final:
                    yield final
            if chunk.get("done"):
                complete = True
                done_reason = _as_text(chunk.get("done_reason")) or "stop"
                usage = {k: chunk[k] for k in _USAGE_KEYS
                         if isinstance(chunk.get(k), (int, float)) and not isinstance(chunk.get(k), bool)}
                break

        if not complete:
            events, _ = flt.finish(complete=False)
            yield from events
            yield ("stream_end", {"complete": False})
            return

        events, extracted = flt.finish(stop_reason=done_reason, extract=not native)
        yield from events
        calls_out = native or extracted
        self._note_context_pressure(usage)
        if thinking_parts and self._attempt_info().replay_reasoning:
            yield ("assistant_blocks", [{"type": "reasoning_content", "text": "".join(thinking_parts)}])
        if usage:
            yield ("usage", usage)
        yield ("stop_reason", done_reason)
        yield ("stop_info", _stop_info(done_reason, usage, "openai", has_calls=bool(calls_out)))
        if calls_out:
            yield ("tool_calls", calls_out)
        else:
            yield ("done", done_reason)
        yield ("stream_end", {"complete": True})

    def _note_context_pressure(self, usage: dict[str, Any]) -> None:
        """Log when the prompt filled the window: Ollama then drops the oldest tokens without any error."""
        prompt = usage.get("prompt_eval_count")
        window = self._num_ctx()
        if isinstance(prompt, int) and window and prompt >= _CONTEXT_FULL_RATIO * window:
            log_event("ollama_context_full", prompt_tokens=prompt, num_ctx=window,
                      model=getattr(self.profile, "model", ""))
