"""Model capabilities: what a given (host, model) can do, resolved in layers.

PURE module: no FreeCAD, no Qt and no network at import time. ``oscar.params`` (which imports FreeCAD)
is imported lazily inside the functions that persist or read the settings file, and every one of those
degrades to "no persisted layers" when that import or the file is unavailable, so ``resolve_caps`` and
``learn`` never raise.

Resolution order (each layer overrides ONLY the fields it specifies; the first one listed wins per field)::

    1. profile.caps_override        user's Advanced settings (keys are ModelCaps fields)
    2. runtime                      in-process only, set by a provider for the life of the process (the window the
                                    native Ollama provider requests); reported as source "learned"
    3. learned                      runtime events, settings["model_caps"][key]["learned"] (+ the in-process mirror)
    4. api                          provider metadata, settings["model_caps"][key]["api"] (``probe_metadata``)
    5. MODEL_CAP_TABLE              (host_class, model regex) rows, first match wins; an OriginShape alias without
                                    a row reads the server's cached ``/v1/models`` instead (``_catalog_fields``,
                                    reported as source "catalog")
    6. ModelCaps()                  defaults

Persistence (additive; old builds ignore the key)::

    "model_caps": {"<provider>|<base_url>|<model>": {"learned": {...}, "api": {...}, "api_ts": "<iso>"}}
"""

import dataclasses
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass

# Allowed values of the enumerated fields. Anything else coming from a settings file is ignored.
TOOL_MODES = ("native", "text", "none")
REASONING_KINDS = ("none", "stream", "summary", "hidden", "inline")
REASONING_CONTROLS = (
    "none", "anthropic_budget", "anthropic_adaptive", "openai_effort", "zai_thinking", "ollama_effort",
    "qwen_enable_thinking", "openrouter_reasoning", "gemini_effort", "xai_effort",
)
API_LAYER_MAX_AGE_S = 24 * 3600
OLLAMA_DEFAULT_NUM_CTX = 16384       # the window the native Ollama provider requests unless the user overrides it
_OLLAMA_MIN_NUM_CTX = 512


@dataclass(frozen=True)
class ModelCaps:
    """What one (host, model) pair can do. Frozen: derive variants with ``dataclasses.replace``."""

    tools: str = "native"             # "native" | "text" | "none"
    vision: bool | None = None        # None = UNKNOWN: the provider stays optimistic (sends screenshots, exactly as today) and learns a rejection; True/False = known. A table or probe may only say True/False for models it really knows, otherwise every custom / OpenRouter / Together / Groq model would silently lose its screenshots
    reasoning: str = "none"           # what the model streams: "none" | "stream" | "summary" | "hidden" | "inline"
    reasoning_control: str = "none"   # knob style, see REASONING_CONTROLS
    context_window: int = 32768       # tokens
    max_output: int = 4096            # tokens the model can emit in one reply
    sampling_ok: bool = True          # temperature accepted
    tool_choice_ok: bool = True
    tool_choice_required_with_reasoning: bool = True   # False for Anthropic thinking
    parallel_calls: bool = True
    tool_args_stream: bool = True     # False (learned) when a tool call's arguments are buffered until the call is complete: the watchdog then allows long silent stretches
    prompt_cache: bool = False        # Anthropic cache_control honoured
    reasoning_replay: bool = False    # reasoning blocks must be replayed in a tool loop (Anthropic thinking)
    source: str = "default"           # highest-precedence layer that set any field: "override" | "learned" | "api" | "catalog" | "table" | "default"

    def to_dict(self) -> dict:
        """Every field as a plain dict (JSON-able)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelCaps":
        """Build from a dict; unknown keys and invalid values are ignored (the default stays)."""
        return cls(**_clean_fields(d or {}, with_source=True))


_FIELD_NAMES = tuple(f.name for f in dataclasses.fields(ModelCaps))
_BOOL_FIELDS = frozenset({
    "sampling_ok", "tool_choice_ok", "tool_choice_required_with_reasoning", "parallel_calls",
    "tool_args_stream", "prompt_cache", "reasoning_replay",
})
_INT_FIELDS = frozenset({"context_window", "max_output"})


def _coerce(name: str, value):
    """``value`` as the field's type, or the sentinel ``_BAD`` when it is not acceptable."""
    if name == "tools":
        return value if value in TOOL_MODES else _BAD
    if name == "reasoning":
        return value if value in REASONING_KINDS else _BAD
    if name == "reasoning_control":
        return value if value in REASONING_CONTROLS else _BAD
    if name == "vision":
        if value is None or isinstance(value, bool):
            return value
        return _BAD
    if name in _BOOL_FIELDS:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        return _BAD
    if name in _INT_FIELDS:
        if isinstance(value, bool):
            return _BAD
        try:
            number = int(float(value))
        except (TypeError, ValueError, OverflowError):
            return _BAD
        return number if 1 <= number <= 100_000_000 else _BAD
    if name == "source":
        return value if isinstance(value, str) else _BAD
    return _BAD


_BAD = object()


def _clean_fields(layer, *, with_source: bool = False) -> dict:
    """The valid ModelCaps fields of one layer (``source`` only when ``with_source``)."""
    out: dict = {}
    if not isinstance(layer, dict):
        return out
    for name, value in layer.items():
        if name not in _FIELD_NAMES or (name == "source" and not with_source):
            continue
        coerced = _coerce(name, value)
        if coerced is not _BAD:
            out[name] = coerced
    return out


# -- table -------------------------------------------------------------------------------------------------

# (host_class or "*", model regex, ModelCaps fields); the FIRST matching row wins and replaces the defaults
# for every field it names. The rows up to the `*` row are the seed rows; rows marked "extra" only
# add well-known facts for hosts the seeds do not cover. Each row's provenance is stated beside it:
# "verified from docs <URL>" = read in the vendor's official documentation,
# "live-only" = no documentation settles it (a 400 that names a field switches it off per model at runtime).
# Claude 4.6+ (Opus/Sonnet), Fable 5.x, Mythos: adaptive thinking + ``output_config.effort``. Opus/Sonnet 4.7+ and
# every 5.x model REJECT ``thinking.type = "enabled"`` with HTTP 400; 4.6 still accepts it (deprecated). Verified from
# docs https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting (per-model table).
_THINKS_ADAPTIVE = {
    "vision": True, "reasoning": "stream", "reasoning_control": "anthropic_adaptive",
    "context_window": 200000, "max_output": 64000, "prompt_cache": True, "reasoning_replay": True}
_CLAUDE_SONNET_5_5 = {**_THINKS_ADAPTIVE, "max_output": 128000}   # Sonnet 5.5: max_tokens ceiling 128000 (docs: effort page)
# Claude 4.5 and earlier (and unknown Claude ids): extended thinking with ``budget_tokens`` is the ONLY mode there
# (``adaptive`` answers HTTP 400). Forced ``tool_choice`` is rejected while thinking is on (docs: define-tools).
_THINKS_BUDGET = {
    "vision": True, "reasoning": "stream", "reasoning_control": "anthropic_budget",
    "context_window": 200000, "max_output": 64000, "prompt_cache": True, "reasoning_replay": True,
    "tool_choice_required_with_reasoning": False}
MODEL_CAP_TABLE: list[tuple[str, str, dict]] = [
    ("anthropic", r"claude-sonnet-5[-.]5", _CLAUDE_SONNET_5_5),
    ("anthropic", r"claude-(?:(?:opus|sonnet)-(?:4[-.][6-9]|[5-9])|(?:fable|mythos)-(?:[5-9]|preview))", _THINKS_ADAPTIVE),
    # The OriginShape account's models have no rows: their capabilities come from the server's /v1/models
    # (``_catalog_fields``).
    ("anthropic", r"claude-(opus|sonnet|haiku|fable|mythos)", _THINKS_BUDGET),   # Claude 4.5 and earlier (above)
    ("anthropic", r"^claude-3", {                                     # extra: Claude 3.x has vision and caching, no thinking
        "vision": True, "context_window": 200000, "max_output": 8192, "prompt_cache": True}),
    ("openai", r"^(o\d|gpt-5)", {
        "vision": True, "reasoning": "hidden", "reasoning_control": "openai_effort", "sampling_ok": False,
        "context_window": 400000, "max_output": 32768}),
    ("openai", r"^gpt-4\.1", {"vision": True, "context_window": 1000000}),
    ("openai", r"^gpt-4(o|\.1)", {"vision": True, "context_window": 128000}),
    # live-only (no official page read): the gemini / xai reasoning knob spellings below.
    ("gemini", r"^gemini", {
        "vision": True, "reasoning": "hidden", "reasoning_control": "gemini_effort",
        "context_window": 1000000, "max_output": 65536}),
    # Z.ai: ``tool_choice`` supports only "auto" and ``tool_stream`` (streamed tool arguments) defaults to false, so
    # arguments arrive in one piece; ``max_tokens`` 1..131072. Verified from docs
    # https://docs.z.ai/api-reference/llm/chat-completion.
    ("zai", r"^glm-(4\.[5-9]|5)", {
        "vision": False, "reasoning": "stream", "reasoning_control": "zai_thinking", "tool_choice_ok": False,
        "tool_args_stream": False, "context_window": 128000, "max_output": 32768}),
    # DeepSeek thinking mode: with ``tools`` in the request every earlier ``reasoning_content`` MUST be passed back
    # (HTTP 400 otherwise), without tools it is ignored. Verified from docs https://api-docs.deepseek.com/guides/thinking_mode.
    ("deepseek", r"reasoner", {"reasoning": "stream", "reasoning_replay": True, "context_window": 128000,
                               "max_output": 8192}),
    ("qwen", r"qwen3", {
        "reasoning": "stream", "reasoning_control": "qwen_enable_thinking", "context_window": 131072}),
    ("xai", r"grok-3-mini", {                                          # extra: the only xAI family with a reasoning_effort knob
        "reasoning": "hidden", "reasoning_control": "xai_effort", "context_window": 131072}),
    # Ollama's OpenAI-compatible endpoint does not support ``tool_choice`` (verified from docs
    # https://docs.ollama.com/api/openai-compatibility); the native /api/chat has no such field either.
    ("ollama", r"^(llama3|qwen2\.5|qwen3|mistral|command-r|hermes3|granite3|devstral|smollm2|firefunction)", {
        "tools": "native", "tool_args_stream": False, "tool_choice_ok": False, "context_window": 4096}),
    ("ollama", r".*", {"tools": "text", "tool_args_stream": False, "tool_choice_ok": False, "context_window": 4096}),
    # LM Studio documents that function names and arguments stream "in pieces" in ``delta.tool_calls`` (verified from
    # docs https://lmstudio.ai/docs/developer/openai-compat/tools), so the default (True) stands; a server that
    # buffers anyway is still learned at runtime.
    ("lmstudio", r".*", {}),
    ("*", r".*", {}),
]
_TABLE_RE: dict[str, "re.Pattern[str]"] = {}


def _regex(pattern: str) -> "re.Pattern[str]":
    compiled = _TABLE_RE.get(pattern)
    if compiled is None:
        compiled = _TABLE_RE[pattern] = re.compile(pattern)
    return compiled


def _table_fields(host: str, model: str) -> dict:
    """The fields of the first matching table row (``{}`` when nothing matches or the row names nothing)."""
    lowered = (model or "").strip().lower()
    names = {lowered, lowered.rsplit("/", 1)[-1]}   # "vendor/model" ids match on the model part too
    for row_host, pattern, fields in MODEL_CAP_TABLE:
        if row_host not in (host, "*"):
            continue
        rx = _regex(pattern)
        if any(rx.search(name) for name in names):
            return _clean_fields(fields)
    return {}


def _catalog_fields(profile) -> dict:
    """An OriginShape alias, from the server's ``/v1/models`` (cached by ``oscar.auth.account``, no network); ``{}``
    for any other host, or an alias the cached list doesn't name.

    Only what the server declares counts: ``capabilities.reasoning`` (``adaptive`` or ``budget`` thinking on the
    Messages wire; anything else asks for no thinking and leaves it to the server), then its vision, window, output
    cap and caching."""
    if host_class(profile) != "originshape":
        return {}
    try:
        from .auth import account

        info = account.cached_model(_text(profile, "model"))
    except Exception:  # noqa: BLE001 - no settings file, an unusable server URL: the defaults apply
        return {}
    if info is None:
        return {}
    caps = info.capabilities
    fields = dict({"adaptive": _THINKS_ADAPTIVE, "budget": _THINKS_BUDGET}.get(caps.get("reasoning"),
                                                                             {"reasoning_control": "none"}))
    if isinstance(caps.get("vision"), bool):
        fields["vision"] = caps["vision"]
    if info.context_window:
        fields["context_window"] = info.context_window
    if info.max_output_tokens:
        fields["max_output"] = info.max_output_tokens
    if caps.get("prompt_caching") is True:
        fields["prompt_cache"] = True
    return _clean_fields(fields)


# -- identity ----------------------------------------------------------------------------------------------


def _text(profile, name: str) -> str:
    value = getattr(profile, name, "")
    return value if isinstance(value, str) else ""


def caps_key(profile) -> str:
    """The persistence key ``"<provider>|<base_url without a trailing '/'>|<model>"`` (missing attributes read as "")."""
    return f"{_text(profile, 'provider')}|{_text(profile, 'base_url').rstrip('/')}|{_text(profile, 'model')}"


# Order matters: first substring found in the host name wins.
_HOST_SUBSTRINGS = (
    ("api.openai.com", "openai"),
    ("generativelanguage.googleapis.com", "gemini"),
    ("openrouter.ai", "openrouter"),
    ("api.z.ai", "zai"),
    ("bigmodel.cn", "zai"),
    ("dashscope", "qwen"),
    ("api.deepseek.com", "deepseek"),
    ("api.groq.com", "groq"),
    ("api.x.ai", "xai"),
    ("together.xyz", "together"),
    ("together.ai", "together"),
    ("fireworks.ai", "fireworks"),
    ("api.mistral.ai", "mistral"),
    ("cerebras.ai", "cerebras"),
    ("api.perplexity.ai", "perplexity"),
    ("ollama", "ollama"),
    ("lmstudio", "lmstudio"),
    ("lm-studio", "lmstudio"),
)


def host_class(profile) -> str:
    """Which family of host serves this profile (derived from provider + base URL host, NOT the model name).

    One of ``anthropic originshape openai gemini openrouter zai qwen ollama lmstudio deepseek groq xai together
    fireworks mistral cerebras perplexity custom``. ``originshape`` (the OriginShape account) has no base URL of its
    own: its server comes from ``oscar.auth.config``.
    """
    provider = _text(profile, "provider")
    if provider in ("anthropic", "originshape"):
        return provider
    base = _text(profile, "base_url").strip()
    if not base:
        return "openai" if provider in ("openai_compat", "") else "custom"
    parsed = urllib.parse.urlparse(base if "://" in base else "//" + base)
    host = (parsed.hostname or "").lower()
    for needle, name in _HOST_SUBSTRINGS:
        if needle in host:
            return name
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port == 11434:
        return "ollama"
    if port == 1234:
        return "lmstudio"
    return "custom"


# -- persistence (lazy, never raises) -----------------------------------------------------------------------

_MIRROR_LOCK = threading.Lock()
_LEARNED: dict[str, dict] = {}   # in-process mirror of settings["model_caps"][key]["learned"]; wins over the file


def _params():
    """``oscar.params`` or None when FreeCAD (and so the settings store) is unavailable."""
    try:
        from . import params
    except Exception:  # noqa: BLE001 - ImportError of FreeCAD, anything else at import
        return None
    return params


def _stored_entry(key: str) -> dict:
    params = _params()
    if params is None:
        return {}
    try:
        entry = params.get_model_caps(key)
    except Exception:  # noqa: BLE001
        return {}
    return entry if isinstance(entry, dict) else {}


def learned_fields(profile) -> dict:
    """Everything learned at runtime for this (provider, base URL, model): the settings layer under the in-process mirror.

    Includes request-level facts that are not ModelCaps fields (``stream_options_ok``, ``max_tokens_cap``,
    ``completion_tokens``, ...). Returns a copy; ``{}`` when nothing was learned.
    """
    key = caps_key(profile)
    stored = _stored_entry(key).get("learned")
    merged = dict(stored) if isinstance(stored, dict) else {}
    with _MIRROR_LOCK:
        merged.update(_LEARNED.get(key, {}))
    return merged


# Phrases of a 400 body that state, without room for another reading, that the model / endpoint does not accept
# image content. Anything else that merely mentions an image (a size limit, a base64 problem, a media type) is an
# AMBIGUOUS rejection: this session degrades to text, but ``vision=False`` is never persisted for it.
_IMAGE_UNSUPPORTED_PHRASES = (
    "does not support image", "doesn't support image", "do not support image", "does not support vision",
    "doesn't support vision", "not support multimodal", "does not support multimodal", "does not accept image",
    "image input is not supported", "image inputs are not supported", "image input not supported",
    "images are not supported", "image is not supported", "image content is not supported",
    "image_url is not supported", "image_url are not supported", "unknown variant `image_url`",
    "unknown variant \"image_url\"", "unknown variant 'image_url'", "missing data required for image input",
    "no endpoints found that support image", "text-only model", "text only model", "content.type is invalid",
    "content type is invalid", "multimodal input is not supported", "does not support images",
    "vision is not supported", "vision input is not supported", "model does not support images",
    "not a multimodal", "image_url is only supported by",
)


def unambiguous_image_rejection(detail: str) -> bool:
    """True when ``detail`` (an HTTP 400 body) says the model itself cannot take images.

    ``False`` for size / dimension / base64 / media-type complaints: those are about THIS image, and persisting
    ``vision=False`` for them would silently disable screenshots for the model for good.
    """
    lower = " ".join((detail or "").lower().split())
    return any(phrase in lower for phrase in _IMAGE_UNSUPPORTED_PHRASES)


def learn(profile, *, evidence: str = "", **fields) -> None:
    """Persist runtime-learned facts for this (provider, base URL, model) and update the in-process mirror.

    Fields are ModelCaps fields (``vision=False``, ``tools="text"``, ``tool_choice_ok=False``,
    ``reasoning_control="none"``, ``context_window=8192``, ``max_output=...``) or request-level markers
    (``stream_options_ok=False``, ``max_tokens_cap=1024``, ...). ModelCaps fields with an invalid value are
    dropped; other markers must be bool / int / str. Never raises: a settings file that cannot be written
    still leaves the in-process mirror updated.

    ``evidence`` is the error text that made the caller learn. ``vision=False`` given WITH evidence is kept only
    when that text is an unambiguous image rejection (``unambiguous_image_rejection``): a size or base64 error
    must not permanently disable screenshots. Without evidence the call is taken at face value (the connection
    test learns it from a controlled tiny image).
    """
    clean: dict = {}
    for name, value in fields.items():
        if name in _FIELD_NAMES:
            coerced = _coerce(name, value)
            if coerced is not _BAD and name != "source":
                clean[name] = coerced
        elif isinstance(value, (bool, int, str)):
            clean[name] = value
    if evidence and clean.get("vision") is False and not unambiguous_image_rejection(evidence):
        del clean["vision"]
    if not clean:
        return
    key = caps_key(profile)
    with _MIRROR_LOCK:
        _LEARNED.setdefault(key, {}).update(clean)
    params = _params()
    if params is None:
        return

    def mutate(entry: dict) -> dict:
        learned = entry.get("learned")
        learned = dict(learned) if isinstance(learned, dict) else {}
        learned.update(clean)
        entry["learned"] = learned
        return entry

    try:
        params.update_model_caps(key, mutate)
    except Exception:  # noqa: BLE001 - disk full, locked file, read-only settings
        pass


# -- resolution --------------------------------------------------------------------------------------------


def _override_fields(profile) -> dict:
    override = getattr(profile, "caps_override", None)
    return _clean_fields(override) if isinstance(override, dict) else {}


def _api_fields(entry: dict, probe: dict | None) -> dict:
    stored = entry.get("api")
    fields = _clean_fields(stored) if isinstance(stored, dict) else {}
    if probe:
        fields.update(_clean_fields(probe))
    return fields


_RUNTIME: dict[str, dict] = {}   # in-process only: facts a provider imposes on itself (never persisted)


def set_runtime_caps(profile, **fields) -> None:
    """Pin ModelCaps fields for ``profile`` for the life of this process (the ``runtime`` layer).

    The native Ollama provider uses it to publish the window it requests (``context_window=num_ctx``) so the
    agent, which resolves caps cache-only BEFORE the first request, budgets for the real window. Invalid values are
    dropped; nothing is written to the settings file.
    """
    clean = _clean_fields(fields)
    if not clean:
        return
    with _MIRROR_LOCK:
        _RUNTIME.setdefault(caps_key(profile), {}).update(clean)


def clear_runtime_caps(profile) -> None:
    """Forget the runtime layer of ``profile`` (a profile switched away from the provider that set it)."""
    with _MIRROR_LOCK:
        _RUNTIME.pop(caps_key(profile), None)


def resolve_caps(profile, *, probe: dict | None = None) -> ModelCaps:
    """Resolve the capabilities of ``profile`` from the layers (module docstring). Cache-only: no network, never raises.

    ``probe`` is a ``probe_metadata`` result that is applied as the API layer for this call (over the stored one).
    ``ModelCaps.source`` names the highest-precedence layer that set at least one field.
    """
    key = caps_key(profile)
    entry = _stored_entry(key)
    stored_learned = entry.get("learned")
    learned = _clean_fields(stored_learned) if isinstance(stored_learned, dict) else {}
    with _MIRROR_LOCK:
        learned.update(_clean_fields(_LEARNED.get(key, {})))
        runtime = dict(_RUNTIME.get(key, {}))
    table = _table_fields(host_class(profile), _text(profile, "model"))
    layers = (
        ("table", table),
        ("catalog", {} if table else _catalog_fields(profile)),
        ("api", _api_fields(entry, probe)),
        ("learned", learned),
        ("learned", runtime),
        ("override", _override_fields(profile)),
    )
    merged: dict = {}
    source = "default"
    for name, fields in layers:
        if fields:
            merged.update(fields)
            source = name
    return ModelCaps(**merged, source=source)


# -- native Ollama window ----------------------------------------------------------------------------------


def ollama_native_wanted(profile) -> bool:
    """True when ``profile`` is served by the native Ollama provider (``/api/chat``): an Ollama host, unless
    ``caps_override["ollama_native"]`` is False (the user insists on the OpenAI-compatible ``/v1`` endpoint)."""
    if _text(profile, "provider") not in ("openai_compat", ""):
        return False
    if host_class(profile) != "ollama":
        return False
    override = getattr(profile, "caps_override", None)
    return not (isinstance(override, dict) and override.get("ollama_native") is False)


def stored_api_extra(profile, name: str):
    """A non-ModelCaps value stored in the API layer by ``probe_metadata`` (``context_max``, ...), or None."""
    api = _stored_entry(caps_key(profile)).get("api")
    return api.get(name) if isinstance(api, dict) else None


def ollama_num_ctx(profile) -> int:
    """The ``num_ctx`` the native Ollama provider requests: the user's ``context_window`` override, else 16384
    capped by the model's trained maximum (``context_max`` from ``/api/show``, never taken as the active window)."""
    override = _override_fields(profile).get("context_window")
    if override:
        return max(int(override), _OLLAMA_MIN_NUM_CTX)
    number = OLLAMA_DEFAULT_NUM_CTX
    trained = stored_api_extra(profile, "context_max")
    if isinstance(trained, int) and not isinstance(trained, bool) and trained >= _OLLAMA_MIN_NUM_CTX:
        number = min(number, trained)
    return number


def lite_prompt_wanted(caps: ModelCaps) -> bool:
    """True when the request should use the small prompt and tool set: a window of 16384 tokens or less, text tool mode, or no parallel calls."""
    return caps.context_window <= 16384 or caps.tools == "text" or not caps.parallel_calls


_PROBE_FAIL_TTL_S = 60.0
_PROBE_FAILED: dict[str, float] = {}     # caps key -> monotonic time of the last failed probe (negative cache)


def _fresh_api_layer(entry: dict) -> bool:
    stamp = entry.get("api_ts")
    if not isinstance(stamp, str) or not stamp:
        return False
    try:
        age = time.time() - time.mktime(time.strptime(stamp, "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, OverflowError):
        return False
    return 0 <= age <= API_LAYER_MAX_AGE_S


def probe_metadata(profile, *, timeout: float = 8.0, refresh: bool = False) -> dict:
    """``{ModelCaps field: value}`` from the provider's metadata endpoints; never raises; ``{}`` on failure.

    A stored API layer younger than 24 hours is returned as is. Otherwise (or with ``refresh=True``) the host's
    endpoint is queried (``oscar.provider_probe``: OpenRouter ``/models``, Ollama ``/api/show`` + ``/api/ps``,
    LM Studio ``/api/v0/models``, Anthropic ``/v1/models/<id>``; every other host has no probe and costs no
    request) and the result is stored in ``settings["model_caps"][key]["api"]`` (with ``api_ts``) so the next
    ``resolve_caps`` sees it. Values the endpoint did not state are not invented; a failed probe is remembered for
    a minute so a stopped local server does not cost a timeout per turn. Blocking network call: never on the GUI
    thread. ``timeout`` is the per-request timeout.
    """
    try:
        key = caps_key(profile)
        entry = _stored_entry(key)
        if not refresh and _fresh_api_layer(entry):
            return _api_fields(entry, None)
        with _MIRROR_LOCK:
            failed_at = _PROBE_FAILED.get(key)
        if not refresh and failed_at is not None and time.monotonic() - failed_at < _PROBE_FAIL_TTL_S:
            return {}
        from . import provider_probe

        found = provider_probe.probe(profile, timeout=timeout)
        fields = _clean_fields(found.get("fields")) if isinstance(found, dict) else {}
        extra = found.get("extra") if isinstance(found, dict) else None
        if not fields and not extra:
            with _MIRROR_LOCK:
                _PROBE_FAILED[key] = time.monotonic()
            return {}
        with _MIRROR_LOCK:
            _PROBE_FAILED.pop(key, None)
        stored = dict(fields)
        if isinstance(extra, dict):
            stored.update({k: v for k, v in extra.items() if isinstance(v, (bool, int, str))})
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        def mutate(old: dict) -> dict:
            old["api"] = stored
            old["api_ts"] = stamp
            return old

        params = _params()
        if params is not None:
            try:
                params.update_model_caps(key, mutate)
            except Exception:  # noqa: BLE001 - a read-only settings file still leaves the probe result usable
                pass
        return fields
    except Exception:  # noqa: BLE001 - never raises
        return {}
