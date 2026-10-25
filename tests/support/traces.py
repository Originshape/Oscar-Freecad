"""tests/support/traces.py - recorded model traces as test data, and the replay of one through ``ScriptedProvider``.

PURE PYTHON (no FreeCAD): ``tests/fixtures/traces/<name>.json`` files hold the MODEL side of a real failure shape: the
rounds a model produced (reasoning streams with their timing, text, tool calls, truncated calls). ``script_from_trace``
turns a trace into the ``Round`` list that ``ScriptedProvider`` / ``ScriptedServer`` serve, so the NEW flow can be run
against the exact behaviour that broke the old one. Assertions about what the flow must do stay in the tests; the trace
only says what the model did.

Format (``"format": 1``)::

    {"format": 1, "name": ..., "source": "<which audit finding this reproduces>", "prompt": "<what the user typed>",
     "profile": {"base_url": ...}, "caps": {<ModelCaps fields>},
     "rounds": [{"note": ..., "events": [EVENT, ...]}, ...]}

Events (the string ``{doc}`` in any string is replaced by the document name given to ``script_from_trace``)::

    {"type": "reasoning", "chars": N, "repeat": R, "advance_s": S}   R reasoning events of N chars, each after S injected seconds
    {"type": "text", "text": "..."}
    {"type": "tool_calls", "calls": [{"name": ..., "args": {...}}], "stop": "tool_use"}
    {"type": "bad_tool_call", "name": ..., "raw": "..." | "raw_head"/"raw_unit"/"raw_times", "stop": "max_tokens"}
    {"type": "stop", "reason": "length"}
    {"type": "advance", "seconds": S}
    {"type": "tick", "elapsed": S}
"""

import json
import os
import re

from support.scripted_llm import Round, advance, bad_json, call, tick

TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "traces")
FORMAT = 1
EVENT_TYPES = ("reasoning", "text", "tool_calls", "bad_tool_call", "stop", "advance", "tick")


class TraceError(ValueError):
    """A malformed trace file (the message names the file, the round and the event)."""


def list_traces(directory: str | None = None) -> list[str]:
    """The names (file stems) of every trace in ``directory`` (default: the fixtures), sorted."""
    base = directory or TRACE_DIR
    return sorted(f[:-5] for f in os.listdir(base) if f.endswith(".json"))


def load_trace(name: str, directory: str | None = None) -> dict:
    """The trace ``name`` (validated: raises ``TraceError`` for anything the replay could not play)."""
    path = os.path.join(directory or TRACE_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        try:
            trace = json.load(f)
        except json.JSONDecodeError as e:
            raise TraceError(f"{name}: not valid JSON: {e}") from e
    validate(trace, name)
    return trace


def validate(trace, name: str = "trace") -> None:
    """Raise ``TraceError`` unless ``trace`` is a playable ``format`` 1 trace."""
    if not isinstance(trace, dict) or trace.get("format") != FORMAT:
        raise TraceError(f"{name}: 'format' must be {FORMAT}")
    for key in ("name", "source", "prompt", "rounds"):
        if not trace.get(key):
            raise TraceError(f"{name}: missing '{key}'")
    for i, rnd in enumerate(trace["rounds"]):
        events = rnd.get("events") if isinstance(rnd, dict) else None
        if not isinstance(events, list) or not events:
            raise TraceError(f"{name}: round {i} has no events")
        for j, ev in enumerate(events):
            kind = ev.get("type") if isinstance(ev, dict) else None
            if kind not in EVENT_TYPES:
                raise TraceError(f"{name}: round {i} event {j}: unknown type {kind!r}; use one of {', '.join(EVENT_TYPES)}")
            if kind == "tool_calls" and not all(isinstance(c, dict) and c.get("name") for c in ev.get("calls") or ()):
                raise TraceError(f"{name}: round {i} event {j}: every call needs a 'name'")
            if kind == "bad_tool_call" and not (ev.get("raw") or ev.get("raw_head")):
                raise TraceError(f"{name}: round {i} event {j}: a bad call needs 'raw' or 'raw_head'")


def substitute(value, doc: str):
    """``value`` with ``{doc}`` replaced by ``doc`` in every string (recursively through dicts and lists)."""
    if isinstance(value, str):
        return value.replace("{doc}", doc)
    if isinstance(value, list):
        return [substitute(v, doc) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, doc) for k, v in value.items()}
    return value


def _round(events: list, doc: str, index: int) -> Round:
    out: list = []
    pending_calls: list = []
    stop = ""
    for ev in events:
        ev = substitute(ev, doc)
        kind = ev["type"]
        if kind == "reasoning":
            for _ in range(int(ev.get("repeat", 1))):
                if ev.get("advance_s"):
                    out.append(advance(float(ev["advance_s"])))
                out.append(("reasoning", "x" * int(ev.get("chars", 100))))
        elif kind == "text":
            out.append(("text", ev["text"]))
        elif kind == "tool_calls":
            pending_calls.extend(call(c["name"], call_id=f"call_t{index}_{k}", **(c.get("args") or {}))
                                 for k, c in enumerate(ev["calls"]))
            stop = ev.get("stop", "tool_use")
        elif kind == "bad_tool_call":
            raw = ev.get("raw") or ev["raw_head"] + ev.get("raw_unit", "") * int(ev.get("raw_times", 0))
            return Round(out + bad_json(ev["name"], raw=raw, call_id=f"call_bad{index}", stop=ev.get("stop", "max_tokens")).events)
        elif kind == "stop":
            stop = ev.get("reason", "length")
        elif kind == "advance":
            out.append(advance(float(ev["seconds"])))
        elif kind == "tick":
            out.append(tick(float(ev["elapsed"])))
    if pending_calls:
        return Round(out + [("stop_reason", stop or "tool_use"), ("tool_calls", pending_calls)])
    if stop:
        return Round(out + [("stop_reason", stop), ("done", stop)])
    return Round(out + [("stop_reason", "end_turn"), ("done", "end_turn")])


def script_from_trace(trace: dict, doc: str = "TraceDoc") -> list[Round]:
    """The ``Round`` list of ``trace`` for ``ScriptedProvider`` (``("advance", s)`` pseudo-events included: pass
    ``ScriptedProvider(script, advance=clock.advance)``) and ``ScriptedServer`` (which ignores them)."""
    return [_round(rnd["events"], doc, i) for i, rnd in enumerate(trace["rounds"])]


def reasoning_seconds(trace: dict) -> float:
    """The injected seconds of the trace's first round (what a watchdog test compares against its limits)."""
    total = 0.0
    for ev in trace["rounds"][0]["events"]:
        if ev["type"] == "reasoning":
            total += float(ev.get("advance_s", 0)) * int(ev.get("repeat", 1))
        elif ev["type"] == "advance":
            total += float(ev["seconds"])
    return total


def tool_names(trace: dict) -> list[str]:
    """Every tool the trace calls, in order (bad calls included)."""
    names = []
    for rnd in trace["rounds"]:
        for ev in rnd["events"]:
            if ev["type"] == "tool_calls":
                names.extend(c["name"] for c in ev["calls"])
            elif ev["type"] == "bad_tool_call":
                names.append(ev["name"])
    return names


_DOC_TOKEN = re.compile(r"\{doc\}")


def uses_doc(trace: dict) -> bool:
    """Whether any string of the trace mentions ``{doc}``."""
    return bool(_DOC_TOKEN.search(json.dumps(trace)))
