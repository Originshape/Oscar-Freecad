"""Tolerant decoding of tool-call arguments. Pure Python.

Models and gateways emit slightly broken argument JSON: raw newlines inside
strings, a trailing comma, a markdown fence, a double-encoded string, a
Python-style dict. Refusing all of those wastes a round per mistake and
blames the token limit for faults that are not truncation. This module decodes
what can be decoded safely and, when it cannot, produces a precise reason the
model can act on (``truncated`` versus ``malformed``).

Only ``decode_tool_arguments`` is meant for callers; it is re-exported by
``oscar.providers`` (the three stream parsers share it).
"""

import ast
import json
import re

from .stopreason import normalize_stop_reason

_TAIL_CHARS = 60
_LITERAL_EVAL_MAX_CHARS = 1_000_000   # ast.literal_eval on megabytes of text is a stall, not a repair


def _strip_fence(text: str) -> str:
    """Remove a leading and/or trailing markdown code fence (``` or ```json)."""
    out = text.strip()
    if out.startswith("```"):
        out = re.sub(r"^```[A-Za-z0-9_+-]*[ \t]*\r?\n?", "", out, count=1)
    if out.endswith("```"):
        out = out[:-3]
    return out.strip()


def _drop_trailing_commas(text: str) -> str:
    """Remove commas that directly precede ``}`` or ``]``, but only OUTSIDE JSON strings.

    A regex over the whole text would also rewrite ``"code": "x = [1,]"`` and the script that
    runs would then differ from the one the model sent; hence the small string-aware scanner.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    n = len(text)
    for i, ch in enumerate(text):
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            continue
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                continue            # drop this comma
        out.append(ch)
    return "".join(out)


def _balanced(text: str) -> bool:
    """True when every ``{`` / ``[`` is closed and no string is left open at the end of the text."""
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
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
    return depth <= 0 and not in_string


def _looks_like_python_dict(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("{") and stripped.endswith("}") and "'" in stripped


def _json_attempts(text: str) -> list[str]:
    """Candidate texts in the order they are tried: as sent, fence stripped, trailing commas dropped."""
    attempts = [text]
    fenced = _strip_fence(attempts[-1])
    if fenced != attempts[-1]:
        attempts.append(fenced)
    cleaned = _drop_trailing_commas(attempts[-1])
    if cleaned != attempts[-1]:
        attempts.append(cleaned)
    return attempts


def _reject_constant(name: str):
    """``json.loads`` hook: ``NaN`` / ``Infinity`` are Python extensions, not JSON. A non-finite float that got into
    the history would make every later request fail to serialize (``requests`` refuses out-of-range floats), so the
    call is reported as malformed instead and the model re-issues it with a real number."""
    raise ValueError(f"{name} is not valid JSON; use a finite number")


def _finite_float(text: str) -> float:
    """``json.loads`` hook for numbers with a fraction or exponent: ``1e999`` overflows to ``inf`` (see ``_reject_constant``)."""
    value = float(text)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{text} is out of range; use a finite number")
    return value


def _decode_once(text: str, allow_double: bool) -> tuple[dict | None, str, int]:
    """(dict | None, json error message, error position) for one text, using every repair in order."""
    first_msg = ""
    first_pos = 0
    for index, candidate in enumerate(_json_attempts(text)):
        try:
            value = json.loads(candidate, strict=False, parse_constant=_reject_constant, parse_float=_finite_float)
        except RecursionError:
            # thousands of nested brackets (a degenerate generation): not decodable, and must not crash the reader
            if index == 0:
                first_msg = "nesting too deep"
            continue
        except ValueError as e:
            if index == 0:
                first_msg = getattr(e, "msg", str(e))
                first_pos = getattr(e, "pos", 0) or 0
            continue
        if isinstance(value, str) and allow_double:
            # double-encoded: the model sent a JSON string that itself holds the object
            inner, msg, pos = _decode_once(value, False)
            if inner is not None:
                return inner, "", 0
            return None, msg or "expected a JSON object, got a string", pos
        if isinstance(value, dict):
            return value, "", 0
        if index == 0:
            first_msg = f"expected a JSON object, got {type(value).__name__}"
            first_pos = 0
        break  # valid JSON of the wrong shape: more repairs cannot turn it into an object
    last = _json_attempts(text)[-1]
    if (
        _looks_like_python_dict(last)
        and len(last) <= _LITERAL_EVAL_MAX_CHARS
    ):
        try:
            value = ast.literal_eval(last.strip())
        except Exception:  # noqa: BLE001 - SyntaxError, ValueError, MemoryError, RecursionError ...
            value = None
        if isinstance(value, dict):
            return value, "", 0
    return None, first_msg or "invalid JSON", first_pos


def decode_tool_arguments(raw, *, stop_reason: str = "") -> tuple[dict, str]:
    """Decode the argument text of one tool call -> ``(arguments, error)``.

    ``error == ""`` means ``arguments`` is a dict decoded from ``raw`` (empty or whitespace-only
    text is ``{}``). Repairs, in order: ``json.loads(strict=False)`` (raw control characters inside
    strings are accepted), one more decode when the result is a string (double-encoded), a stripped
    ``` fence, trailing commas removed outside strings only, and ``ast.literal_eval`` for text that
    looks like a Python dict literal. The result must be a dict, otherwise it is an error.

    On failure returns ``({"_raw": raw}, error)`` with
    ``error = "<kind> at char <pos> of <n>; tail <last 60 chars>"`` where ``kind`` is
    ``"truncated (output cut off at the token limit)"`` when the normalized ``stop_reason`` is
    ``max_tokens`` or the braces are unbalanced at the end of the text, else
    ``"malformed JSON (<reason>)"``.
    """
    if isinstance(raw, dict):
        return raw, ""
    if raw is None:
        raw = ""
    if not isinstance(raw, str):
        raw = str(raw)
    text = raw.lstrip("﻿")
    if not text.strip():
        return {}, ""
    value, message, pos = _decode_once(text, True)
    if value is not None:
        return value, ""
    n = len(raw)
    if normalize_stop_reason(stop_reason) == "max_tokens" or not _balanced(text):
        kind = "truncated (output cut off at the token limit)"
        pos = n
    else:
        kind = f"malformed JSON ({message})"
    return {"_raw": raw}, f"{kind} at char {pos} of {n}; tail {raw[-_TAIL_CHARS:]!r}"
