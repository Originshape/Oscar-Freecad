"""Normalized stop reasons (pure Python: no FreeCAD, no Qt, no oscar imports).

Providers keep yielding the RAW provider value in ``("stop_reason", raw)`` (existing tests pin it)
and add ``("stop_info", {"reason": <normalized>, ...})``. The agent loop and the providers both use
the two functions below, so there is exactly one mapping.
"""

STOP_REASONS = ("end_turn", "tool_use", "max_tokens", "refusal", "content_filter", "error", "unknown")

_MAP = {
    # Anthropic
    "end_turn": "end_turn", "stop_sequence": "end_turn", "pause_turn": "end_turn",
    "tool_use": "tool_use", "max_tokens": "max_tokens", "refusal": "refusal",
    # OpenAI chat completions
    "stop": "end_turn", "tool_calls": "tool_use", "function_call": "tool_use",
    "length": "max_tokens", "content_filter": "content_filter",
    # OpenRouter's normalized finish reasons add this one
    "error": "error",
}


def normalize_stop_reason(raw) -> str:
    """Map any provider stop/finish reason to one of ``STOP_REASONS`` ("" / None -> "unknown")."""
    return _MAP.get(str(raw or "").strip().lower(), "unknown")


def is_cut_off(raw) -> bool:
    """True when generation stopped at the output-token limit (raw ``length`` / ``max_tokens`` or normalized)."""
    return normalize_stop_reason(raw) == "max_tokens"
