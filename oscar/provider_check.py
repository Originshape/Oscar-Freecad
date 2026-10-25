"""The connection test of the settings dialog: can Oscar really work with this model?

``run_connection_test(profile)`` is blocking (run it on a worker thread: ``settings_model.run_in_background``) and
performs, through the real provider:

1. ``probe_metadata`` (OpenRouter / Ollama / LM Studio / Anthropic metadata; a no-op for other hosts);
2. a plain reply ("ready"): credentials, base URL, model name, reachability;
3. ONE tool call: a tiny ``echo`` tool the model is asked to call (``tool_choice="required"``, so a capable model
   that merely preferred to answer in text is not demoted). Native tool calling first; when no call comes back, or
   the endpoint refuses tools, text tool mode is tried and, if it works, remembered (``models.learn(tools="text")``).
   A model that does neither cannot drive Oscar;
4. ONE tiny image (a solid red square): a model that names the colour has vision (remembered); one that refuses the
   image content does not (remembered: a controlled, tiny, valid PNG cannot be a size or encoding problem); one
   that answers without naming it stays "unknown" (some gateways silently drop images).

Everything learned goes through ``oscar.models.learn`` so the next turn already uses it. The report is plain data
for the dialog: ``ConnectionReport.lines`` (one per step) and ``summary()``.
"""

import base64
import dataclasses
import json
import struct
import zlib
from dataclasses import dataclass, field
from typing import Callable

from . import models as _models
from .provider_transport import StopRequested

_PING_SYSTEM = "Reply with the single word: ready"
_TOOL_SYSTEM = "You can call tools. When asked to call a tool, call it; do not describe it."
_TOOL_PROMPT = "Call the echo tool with the text \"oscar-ok\" now."
_IMAGE_SYSTEM = "You describe images in one word."
_IMAGE_PROMPT = "What colour is this image? Answer with one word."
_COLOUR_WORDS = ("red", "rojo", "rot", "rouge", "rosso", "vermelho")
_REPLY_CLIP = 80


def tiny_png(rgb=(255, 0, 0), size: int = 64) -> bytes:
    """A valid solid-colour PNG (``size`` x ``size``; 64 px: below ~28 px some vision servers refuse an image), built with the standard library."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    row = b"\x00" + bytes(rgb) * size
    raw = row * size
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def tiny_png_b64() -> str:
    """The test image as base64 text (what ``Message.images`` carries)."""
    return base64.b64encode(tiny_png()).decode("ascii")


@dataclass
class CheckLine:
    """One step of the test: ``ok`` True / False / None (skipped or inconclusive) and a one-line ``text``."""

    step: str
    ok: bool | None
    text: str


@dataclass
class ConnectionReport:
    """The result of ``run_connection_test``. ``ok`` is True when the model can talk AND call tools (natively or as text)."""

    ok: bool = False
    lines: list[CheckLine] = field(default_factory=list)
    reply: str = ""
    tools: str = ""                    # "native" | "text" | "" (not working / not tested)
    vision: bool | None = None         # True / False when the image step decided, else None
    learned: dict = field(default_factory=dict)
    error: str = ""
    provider_name: str = ""

    def summary(self) -> str:
        """The lines as text, one per line, each starting with a check mark, a cross or a dash."""
        marks = {True: "✓", False: "✗", None: "-"}
        return "\n".join(f"{marks[line.ok]} {line.text}" for line in self.lines)


def _echo_tool():
    from .providers import ToolSpec

    return ToolSpec("echo", "Echo the given text back.",
                    {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})


def _clip(text: str) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= _REPLY_CLIP else flat[:_REPLY_CLIP - 3] + "..."


def _payload_text(payload) -> str:
    """Readable text for the payload of an ``("error", payload)`` event.

    Providers yield an exception; a custom provider (or an older stream parser) may yield the raw failure event (a
    dict, possibly with the message nested under ``response.error``), which cannot be raised or shown as it is.
    """
    if isinstance(payload, BaseException):
        return str(payload) or type(payload).__name__
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        response = payload.get("response")
        for candidate in (payload, payload.get("error"), response.get("error") if isinstance(response, dict) else None):
            if isinstance(candidate, str) and candidate:
                return candidate
            if isinstance(candidate, dict) and candidate.get("message"):
                return str(candidate["message"])
        try:
            return json.dumps(payload, default=str)[:300]
        except (TypeError, ValueError):
            pass
    return str(payload)[:300]


def _collect(provider, system, messages, tools, should_stop, **opts):
    """Run one ``chat()`` call; ``(text, calls, error)``. A stop request raises ``StopRequested`` to the caller.

    ``error`` is always an exception (a non-exception error payload is wrapped, its message kept)."""
    text, calls, error = "", [], None
    for kind, payload in provider.chat(system, messages, tools, should_stop, **opts):
        if kind == "text":
            text += payload
        elif kind == "tool_calls":
            calls = list(payload)
        elif kind == "error":
            error = payload if isinstance(payload, BaseException) else RuntimeError(_payload_text(payload))
    return text, calls, error


def _with_tool_mode(profile, mode: str):
    """A copy of ``profile`` that forces tool mode ``mode`` (same provider / base URL / model, so the same caps key)."""
    override = dict(getattr(profile, "caps_override", None) or {})
    override["tools"] = mode
    try:
        return dataclasses.replace(profile, caps_override=override)
    except TypeError:
        import copy

        clone = copy.copy(profile)
        clone.caps_override = override
        return clone


def _transient(error) -> bool:
    """True for a failure that is about the connection, not the model: an HTTP status the retry policy retries (429,
    5xx), an overload / rate-limit event in the stream, a transport failure that survived every retry."""
    if getattr(error, "retriable", False):
        return True
    text = str(error).lower()
    return "request failed after" in text or "stream ended early" in text or "timed out" in text


def _tool_step(profile, make, should_stop, report: ConnectionReport) -> None:
    """Step 3: one tool call, native first, then text mode; fills ``report.tools`` and the learned facts."""
    from .providers import Message

    messages = [Message(role="user", content=_TOOL_PROMPT)]
    forced = (getattr(profile, "caps_override", None) or {}).get("tools")
    attempts = [forced] if forced in ("native", "text") else ["native", "text"]
    problems: list[str] = []
    for mode in attempts:
        subject = profile if forced == mode else _with_tool_mode(profile, mode)   # an override beats what was learned
        try:
            # ``required``: a capable model that merely chose to answer in text must not be demoted to text mode
            text, calls, error = _collect(make(subject), _TOOL_SYSTEM, messages, [_echo_tool()], should_stop,
                                          tool_choice="required")
        except StopRequested:
            raise
        except Exception as e:  # noqa: BLE001 - a provider bug must not hide the rest of the report
            problems.append(f"{mode}: {type(e).__name__}: {e}")
            continue
        if error is not None:
            problems.append(f"{mode}: {_clip(str(error))}")
            if _transient(error):
                # a rate limit, an overload or a dropped connection says nothing about tool calling: demoting a model
                # to text mode (remembered across sessions) on that evidence would cost it its native tools for good
                report.lines.append(CheckLine("tools", None, f"the tool-call test could not finish ({problems[-1]}); "
                                              "try again in a moment"))
                return
            continue
        good = [c for c in calls if c.name == "echo" and not c.args_error]
        if good:
            report.tools = mode
            if not forced:
                if mode == "text":
                    _models.learn(profile, tools="text")
                    report.learned["tools"] = "text"
                elif _models.learned_fields(profile).get("tools") == "text":
                    _models.learn(profile, tools="native")
                    report.learned["tools"] = "native"
            report.lines.append(CheckLine("tools", True, (
                "tool calling works natively" if mode == "native"
                else "tool calling works in text mode (the model has no native tools; Oscar adapts)")))
            return
        problems.append(f"{mode}: the model answered with text instead of a tool call ({_clip(text) or 'no text'})")
    report.lines.append(CheckLine("tools", False, "no tool call came back (" + "; ".join(problems) + "); Oscar needs "
                                  "a model that can call tools - choose another model"))


def _image_step(profile, make, should_stop, report: ConnectionReport) -> None:
    """Step 4: one tiny image; learns ``vision`` when the answer is conclusive."""
    from .providers import Message

    caps = _models.resolve_caps(profile)
    if caps.vision is False:
        report.vision = False
        report.lines.append(CheckLine("vision", None, "images are switched off for this model (screenshots are skipped)"))
        return
    provider = make(profile)
    message = Message(role="user", content=_IMAGE_PROMPT, images=[tiny_png_b64()])
    try:
        text, _calls, error = _collect(provider, _IMAGE_SYSTEM, [message], [], should_stop)
    except StopRequested:
        raise
    except Exception as e:  # noqa: BLE001
        report.lines.append(CheckLine("vision", None, f"image test failed: {type(e).__name__}: {e}"))
        return
    if not getattr(provider, "vision_enabled", True):
        _models.learn(profile, vision=False)             # a valid 16 px PNG was refused: this model takes no images
        report.vision, report.learned["vision"] = False, False
        report.lines.append(CheckLine("vision", False, "the model does not accept images (screenshots are skipped)"))
    elif error is not None:
        report.lines.append(CheckLine("vision", None, f"image test inconclusive: {_clip(str(error))}"))
    elif any(word in (text or "").lower() for word in _COLOUR_WORDS):
        _models.learn(profile, vision=True)
        report.vision, report.learned["vision"] = True, True
        report.lines.append(CheckLine("vision", True, "the model sees images (it named the colour)"))
    else:
        report.lines.append(CheckLine("vision", None, f"images accepted but not confirmed (the model said: "
                                      f"\"{_clip(text) or 'nothing'}\"); screenshots are still tried"))


def run_connection_test(profile, *, should_stop: Callable[[], bool] | None = None, make=None,
                        probe: bool = True) -> ConnectionReport:
    """Test ``profile`` end to end (see the module docstring). Blocking; never raises except ``StopRequested``.

    ``make`` builds the provider (default ``providers.make_provider``; tests pass a fake). ``should_stop`` is polled
    by every request. ``probe=False`` skips the metadata step.
    """
    from .providers import Message, make_provider

    make = make or make_provider
    report = ConnectionReport()
    if probe:
        try:
            found = _models.probe_metadata(profile, refresh=True)
        except Exception:  # noqa: BLE001
            found = {}
        if found:
            facts = ", ".join(f"{k} {v}" for k, v in sorted(found.items()) if k in ("context_window", "max_output", "tools", "vision"))
            report.lines.append(CheckLine("metadata", True, f"the server reports: {facts}" if facts else "metadata read"))
    try:
        provider = make(profile)
        report.provider_name = getattr(provider, "name", "")
        text, _calls, error = _collect(provider, _PING_SYSTEM, [Message(role="user", content="ping")], [], should_stop)
    except StopRequested:
        raise
    except Exception as e:  # noqa: BLE001
        report.error = f"{type(e).__name__}: {e}"
        report.lines.append(CheckLine("connect", False, report.error))
        return report
    if error is not None:
        report.error = str(error)
        report.lines.append(CheckLine("connect", False, f"{report.provider_name or 'provider'} / {profile.model}: {_clip(report.error)}"))
        return report
    report.reply = text.strip()
    report.lines.append(CheckLine("connect", True, f"{report.provider_name or 'provider'} / {profile.model}: “{_clip(report.reply)}”"))
    _tool_step(profile, make, should_stop, report)
    _image_step(profile, make, should_stop, report)
    report.ok = bool(report.tools)
    return report
