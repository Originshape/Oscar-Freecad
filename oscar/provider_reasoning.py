"""Reasoning-effort mapper: ``Profile.reasoning`` / per-call effort -> request fragment.

Pure Python (no FreeCAD, no Qt, no network). Each provider's ``_body`` merges the returned dict into
its request; an empty dict means "send nothing". The knob styles are the ``ModelCaps.reasoning_control``
values of ``oscar.models``. The mapper is only reached when an effort was asked for explicitly, and an
HTTP 400 that names the field drops it again (``BaseProvider._adapt_to_rejection``).

Documentation status (each line is "verified from docs <URL>" or "live-only"):

* Anthropic: verified from docs https://platform.claude.com/docs/en/build-with-claude/extended-thinking and
  .../thinking-troubleshooting. ``thinking: {"type": "enabled", "budget_tokens": N}`` (``anthropic_budget``) needs
  ``N >= 1024`` and ``N < max_tokens`` (the interleaved-thinking exception is not used) and is the ONLY mode of
  Claude 4.5 and earlier; it is deprecated on 4.6 and answered with HTTP 400 ``"thinking.type.enabled" is not
  supported for this model`` from 4.7 on. Those models use ``thinking: {"type": "adaptive"}`` plus
  ``output_config: {"effort": "low|medium|high"}`` (``anthropic_adaptive``; effort needs no beta header). Thinking is
  OFF by default on Opus 4.6/4.7/4.8 and Sonnet 4.6, ON by default on Opus 5 / Sonnet 5 / Sonnet 5.5, always on for
  Opus 5.5, Fable and Mythos 5.x. ``thinking: {"type": "disabled"}`` is accepted by Opus 4.6-4.8, Sonnet 4.6,
  Sonnet 5 and Opus 5 (effort high or below) and rejected (400) by Opus 5.5 / Fable / Mythos 5.x; Sonnet 5.5 takes
  ``{"type": "between_tools"}`` instead. The models that cannot be switched off get ``effort: low`` for ``off``.
* OpenAI chat completions ``reasoning_effort``: low / medium / high are the schema enum (verified from the OpenAI
  OpenAPI spec https://github.com/openai/openai-openapi); ``minimal`` (gpt-5), ``none`` and ``xhigh`` are per-model values
  of the reasoning guide (https://platform.openai.com/docs/guides/reasoning, fetch blocked here: HTTP 403), hence the
  400 adaptation for models that reject our ``off`` spelling.
* Z.ai ``thinking: {"type": "enabled|disabled", "clear_thinking": bool}`` verified from docs
  https://docs.z.ai/api-reference/llm/chat-completion.
* OpenRouter ``reasoning: {"effort": "max|xhigh|high|medium|low|minimal|none", "max_tokens", "exclude", "enabled"}``
  (``effort`` and ``max_tokens`` are mutually exclusive; ``reasoning_effort`` is NOT a top-level OpenRouter field;
  ``effort: "none"`` and ``enabled: false`` both disable) verified from docs
  https://openrouter.ai/docs/use-cases/reasoning-tokens.
* Qwen / DashScope: top-level ``enable_thinking`` (the Python SDK's ``extra_body`` is merged into the top level) and
  ``thinking_budget`` (1..32768) verified from docs https://www.alibabacloud.com/help/en/model-studio/deep-thinking.
  ``chat_template_kwargs.enable_thinking`` for self-hosted servers (vLLM / SGLang) is a server convention: live-only.
* Ollama: ``/v1`` accepts ``reasoning_effort`` and ``reasoning.effort`` (verified from docs
  https://docs.ollama.com/api/openai-compatibility); the value ``none`` and the gemini / xai spellings are live-only.
* DeepSeek: ``thinking: {"type": "enabled"}`` / ``reasoning_effort`` low|high|max are documented at
  https://api-docs.deepseek.com/guides/thinking_mode; the table does not set a control for it (live-only).
"""

import re

EFFORTS = ("off", "low", "medium", "high")
ANTHROPIC_EFFORTS = ("low", "medium", "high")   # the effort levels every adaptive-thinking Claude model accepts
EXPLICIT_EFFORTS = frozenset(EFFORTS)

# Anthropic thinking budgets (tokens) per effort; ``off`` omits the ``thinking`` block entirely.
ANTHROPIC_BUDGETS = {"low": 1024, "medium": 4096, "high": 16000}
ANTHROPIC_MIN_BUDGET = 1024
# At or below this ``max_tokens`` no valid ``budget_tokens`` exists (>= 1024 and < max_tokens with headroom).
ANTHROPIC_MIN_MAX_TOKENS = 2048


def normalize_effort(value) -> str | None:
    """``"off" | "low" | "medium" | "high"`` for an explicit value, ``None`` for ``"auto"``, ``None`` and junk.

    ``None`` means "no explicit choice": a provider called without a per-call value and with
    ``Profile.reasoning == "auto"`` sends no reasoning / thinking parameter at all.
    """
    text = str(value or "").strip().lower()
    return text if text in EXPLICIT_EFFORTS else None


def _last_segment(model: str) -> str:
    return (model or "").lower().rsplit("/", 1)[-1]


_CLAUDE_ID = re.compile(r"claude-(opus|sonnet|haiku|fable|mythos)-(?:(\d+)(?:[-.](\d{1,2})(?!\d))?|(preview))")


def claude_version(model: str) -> tuple[str, int, int] | None:
    """``(family, major, minor)`` of a Claude model id, ``None`` when it is not one.

    ``claude-opus-4-5-20251101`` -> ``("opus", 4, 5)``, ``claude-sonnet-5`` -> ``("sonnet", 5, 0)``,
    ``claude-sonnet-5-5`` / ``anthropic/claude-sonnet-5.5`` -> ``("sonnet", 5, 5)``, ``claude-opus-4-20250514`` ->
    ``("opus", 4, 0)`` (an 8-digit date is never a minor version), ``claude-mythos-preview`` -> ``("mythos", 0, 0)``.
    """
    match = _CLAUDE_ID.search((model or "").lower())
    if not match:
        return None
    family = match.group(1)
    if match.group(4):
        return family, 0, 0
    return family, int(match.group(2)), int(match.group(3) or 0)


# Verified from docs https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools ("Forcing tool use"):
# these models answer HTTP 400 to tool_choice any / tool whatever the thinking setting is.
_NO_FORCED_TOOL_CHOICE = frozenset({("opus", 5, 5), ("sonnet", 5, 5), ("fable", 5, 1), ("mythos", 5, 1)})


def anthropic_forced_choice_ok(model: str) -> bool:
    """False for the Claude models that reject ``tool_choice`` ``any`` / ``tool`` outright (unknown ids: True)."""
    return claude_version(model) not in _NO_FORCED_TOOL_CHOICE


def anthropic_thinking_off(model: str) -> str:
    """The ``thinking.type`` that switches thinking off for ``model``: ``"disabled"``, ``"between_tools"`` (Sonnet 5.5)
    or ``""`` when the model cannot be switched off (Opus 5.5, Fable, Mythos 5.x: ``disabled`` is a 400).

    Verified from docs https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting (table
    "Rejected with 400"). Unknown ids get ``"disabled"`` (a 400 is learned by the provider).
    """
    version = claude_version(model)
    if version is None:
        return "disabled"
    family, major, minor = version
    if (family, major, minor) == ("sonnet", 5, 5):
        return "between_tools"
    if family in ("fable", "mythos") or (family, major, minor) == ("opus", 5, 5):
        return ""
    return "disabled"


def anthropic_budget(effort: str, max_tokens: int) -> int:
    """Thinking budget for ``effort`` under ``max_tokens``; 0 when thinking must be omitted.

    ``budget_tokens`` must be at least 1024 and below ``max_tokens``: ``budget = min(table budget,
    max_tokens - 1024)``, and nothing is sent when ``max_tokens <= 2048`` (a request the API would
    reject in a way the 400 adaptation cannot repair).
    """
    table = ANTHROPIC_BUDGETS.get(effort)
    if not table or int(max_tokens or 0) <= ANTHROPIC_MIN_MAX_TOKENS:
        return 0
    return max(ANTHROPIC_MIN_BUDGET, min(table, int(max_tokens) - 1024))


def reasoning_fragment(control: str, effort: str | None, *, model: str = "", base_url: str = "",
                       max_tokens: int = 0) -> dict:
    """The request fields that set reasoning ``effort`` for knob style ``control``; ``{}`` = send nothing.

    ``effort`` is ``off|low|medium|high`` (``None`` / anything else -> ``{}``). ``model`` / ``base_url``
    pick the Anthropic ``off`` spelling (``anthropic_adaptive``), the OpenAI ``off`` spelling (``minimal`` for gpt-5, ``low`` for o-series) and the Qwen field
    placement (DashScope takes a top-level ``enable_thinking``, other servers a
    ``chat_template_kwargs``); ``max_tokens`` bounds the Anthropic budget. A knob style that does not
    belong to the caller's wire format simply produces fields the caller never merges.
    """
    level = normalize_effort(effort)
    if level is None or not control or control == "none":
        return {}
    if control == "anthropic_budget":
        budget = anthropic_budget(level, max_tokens)
        return {"thinking": {"type": "enabled", "budget_tokens": budget}} if budget else {}
    if control == "anthropic_adaptive":
        if level == "off":
            kind = anthropic_thinking_off(model)
            return {"thinking": {"type": kind}} if kind else {"output_config": {"effort": "low"}}
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": level}}
    if control == "openai_effort":
        if level == "off":
            level = "minimal" if _last_segment(model).startswith("gpt-5") else "low"
        return {"reasoning_effort": level}
    if control == "zai_thinking":
        return {"thinking": {"type": "enabled" if level in ("medium", "high") else "disabled"}}
    if control == "ollama_effort":
        return {"reasoning_effort": "none" if level == "off" else level}
    if control == "qwen_enable_thinking":
        enabled = level in ("medium", "high")
        if "dashscope" in (base_url or "").lower():
            return {"enable_thinking": enabled}
        return {"chat_template_kwargs": {"enable_thinking": enabled}}
    if control == "openrouter_reasoning":
        return {"reasoning": {"enabled": False}} if level == "off" else {"reasoning": {"effort": level}}
    if control == "gemini_effort":
        return {"reasoning_effort": "none" if level == "off" else level}
    if control == "xai_effort":
        return {"reasoning_effort": "high" if level in ("medium", "high") else "low"}
    return {}


# Request fields the mapper (and the other optional request features) can add; used by the 400 adaptation.
REASONING_FIELDS = ("reasoning_effort", "reasoning", "thinking", "enable_thinking", "chat_template_kwargs", "think",
                    "output_config")


def mentions_field(text_lower: str, field: str) -> bool:
    """True when ``field`` appears in ``text_lower`` as a whole identifier.

    ``reasoning`` must not match inside ``reasoning_effort`` and ``store`` must not match inside
    ``restore``; the boundary test is "no identifier character on either side".
    """
    return re.search(rf"(?<![a-z0-9_]){re.escape(field)}(?![a-z0-9_])", text_lower) is not None
