"""tests/support/scripted_llm.py - scripted provider for agent-loop tests.

A script is a list of ``Round`` objects (or a callable that returns the next Round from the
request that was just made). ``ScriptedProvider.chat`` has the real provider signature, records every
request, and yields the round's event tuples exactly like ``oscar.providers`` does.

Pseudo-event ``("advance", seconds)``: not yielded; calls ``ScriptedProvider(advance=...)`` (an injected clock's
``advance``) so a round can model "no events for 200 s" without sleeping. ``support.server.ScriptedServer``
ignores it (it has no wire form).
"""

import copy
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from oscar.providers import Message, StopRequested, ToolCall

Event = tuple[str, Any]


@dataclass
class Round:
    """One scripted model response."""

    events: list[Event] = field(default_factory=list)
    delay: float = 0.0                      # seconds slept before the first event
    raise_exc: BaseException | None = None  # raised from chat() instead of yielding
    note: str = ""


def _end(stop: str, tool_calls: list[ToolCall]) -> list[Event]:
    events: list[Event] = [("stop_reason", stop)]
    events.append(("tool_calls", tool_calls) if tool_calls else ("done", stop))
    return events


def say(text: str, *, reasoning: str = "", stop: str = "end_turn", usage: dict | None = None) -> Round:
    events: list[Event] = []
    if reasoning:
        events.append(("reasoning", reasoning))
    if text:
        events.append(("text", text))
    if usage:
        events.append(("usage", usage))
    return Round(events + _end(stop, []))


def call(tool: str, /, call_id: str | None = None, **arguments) -> ToolCall:
    """A tool call. ``tool`` is positional-only so an argument literally named ``name`` (``create_document``,
    ``set_parameter``) can be passed as a keyword."""
    return ToolCall(id=call_id or f"call_{uuid.uuid4().hex[:8]}", name=tool, arguments=arguments)


def calls(*tool_calls: ToolCall, text: str = "", reasoning: str = "", stop: str = "tool_use") -> Round:
    events: list[Event] = []
    if reasoning:
        events.append(("reasoning", reasoning))
    if text:
        events.append(("text", text))
    return Round(events + _end(stop, list(tool_calls)))


def think(chars: int, *, stop: str = "length") -> Round:
    """A reply that is only reasoning and ends at the output-token limit."""
    return Round([("reasoning", "x" * chars), ("stop_reason", stop), ("done", stop)])


def bad_json(name: str, raw: str = '{"code": "pri', call_id: str | None = None, stop: str = "max_tokens") -> Round:
    """A tool call whose arguments could not be decoded (truncated output)."""
    tc = ToolCall(id=call_id or f"call_{uuid.uuid4().hex[:8]}", name=name, arguments={"_raw": raw},
                  args_error=f"truncated (output cut off at the token limit) at char {len(raw)} of {len(raw)}; tail {raw[-60:]!r}",
                  raw_args=raw)
    return Round([("stop_reason", stop), ("tool_calls", [tc])])


def malformed_json(name: str, raw: str = '{"code": "print(1)",}', call_id: str | None = None,
                   stop: str = "tool_use") -> Round:
    """A tool call whose arguments are NOT valid JSON although the output was complete (not truncated)."""
    tc = ToolCall(id=call_id or f"call_{uuid.uuid4().hex[:8]}", name=name, arguments={"_raw": raw},
                  args_error=f"malformed JSON (Expecting property name enclosed in double quotes) at char {len(raw) - 2} "
                             f"of {len(raw)}; tail {raw[-60:]!r}",
                  raw_args=raw)
    return Round([("stop_reason", stop), ("tool_calls", [tc])])


def tick(elapsed: float, *, idle_s: float | None = None, phase: str = "thinking", reasoning_chars: int = 0,
         text_chars: int = 0, tool_chars: int = 0, out_tokens: int | None = None, tool: str | None = None,
         retry: list | None = None) -> Event:
    """A provider ``tick`` event with the exact payload keys of a provider tick (``idle_s`` defaults to ``elapsed``)."""
    return ("tick", {"elapsed": float(elapsed), "reasoning_chars": reasoning_chars, "text_chars": text_chars,
                     "tool_chars": tool_chars, "out_tokens": out_tokens, "phase": phase, "tool": tool,
                     "retry": retry, "idle_s": float(elapsed if idle_s is None else idle_s)})


def advance(seconds: float) -> Event:
    """The ``("advance", seconds)`` pseudo-event (see the module docstring)."""
    return ("advance", float(seconds))


def nameless() -> Round:
    """finish_reason tool_calls, but the provider dropped the nameless call."""
    return Round([("stop_reason", "tool_calls"), ("done", "tool_calls")])


def empty() -> Round:
    return say("")


def provider_error(message: str) -> Round:
    return Round([("error", RuntimeError(message))])


def from_legacy(kind: str, payload: Any) -> Round:
    """Convert the (kind, payload) tuples of tests/test_oscar_regressions.ScriptedProvider."""
    if kind == "text":
        return say(payload)
    if kind == "cutoff":
        return Round([("reasoning", payload), ("stop_reason", "length"), ("done", "length")])
    if kind == "nameless":
        return nameless()
    return calls(*payload)


@dataclass
class RequestRecord:
    n: int
    system: str
    messages: list
    tools: list
    kwargs: dict


class ScriptedProvider:
    vision_enabled = True

    def __init__(self, script, *, strict: bool = True, name: str = "scripted",
                 advance: Callable[[float], None] | None = None):
        self.name = name
        self.strict = strict
        self.advance = advance                # called for ("advance", seconds) pseudo-events (an injected clock)
        self._fn: Callable[[RequestRecord], Round] | None = script if callable(script) else None
        self._rounds: list[Round] = [] if callable(script) else list(script)
        self.requests: list[RequestRecord] = []
        self.systems: list[str] = []          # legacy attribute used by older tests

    @property
    def remaining(self) -> int:
        return len(self._rounds)

    def assert_exhausted(self) -> None:
        assert not self._rounds, f"{len(self._rounds)} scripted round(s) were never requested"

    def _next(self, record: RequestRecord) -> Round:
        if self._fn is not None:
            return self._fn(record)
        if self._rounds:
            return self._rounds.pop(0)
        if self.strict:
            last = next((m for m in reversed(record.messages) if m.role == "user"), None)
            tail = (last.content or "")[:200] if last is not None else ""
            raise AssertionError(f"unexpected extra provider request #{record.n}; last user message: {tail!r}")
        return say("(script exhausted)")

    def chat(self, system, messages, tools, should_stop=None, **kwargs):
        record = RequestRecord(len(self.requests) + 1, system, copy.deepcopy(list(messages)),
                               [t.name for t in tools], dict(kwargs))
        self.requests.append(record)
        self.systems.append(system)
        rnd = self._next(record)
        if rnd.delay:
            time.sleep(rnd.delay)
        if rnd.raise_exc is not None:
            raise rnd.raise_exc
        for event in rnd.events:
            if should_stop is not None and should_stop():
                raise StopRequested()
            if event[0] == "advance":
                if self.advance is not None:
                    self.advance(float(event[1]))
                continue
            yield event
