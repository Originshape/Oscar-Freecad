"""Provider metadata probing: what a host says about a model, as ``ModelCaps`` fields.

``probe(profile, timeout=...)`` queries the metadata endpoint of the profile's host class and returns
``{"fields": {ModelCaps field: value}, "extra": {...}}`` (``{}`` when the host has no probe, the request failed or
the model is not listed). ``oscar.models.probe_metadata`` stores the result in the API layer; nothing here writes
settings. All network goes through ``http_json`` (``requests.get`` / ``requests.post`` are looked up at call time, so
tests patch them), every parser is a pure function of a JSON payload, and nothing here ever raises.

Hosts and endpoints:

* OpenRouter: ``GET <base>/models`` (public). ``context_length``, ``top_provider.max_completion_tokens``,
  ``supported_parameters`` and ``architecture.input_modalities``.
* Ollama: ``POST <root>/api/show`` (``capabilities``, and ``model_info.*.context_length`` = the TRAINED maximum,
  stored only as ``extra["context_max"]``, never as the window) and ``GET <root>/api/ps`` (``context_length`` of a
  LOADED model = the window the server actually runs).
* LM Studio: ``GET <root>/api/v0/models`` (``type``, ``state``, ``loaded_context_length``, ``max_context_length``,
  ``capabilities``).
* Anthropic: ``GET <base>/v1/models/<id>`` (``max_input_tokens``, ``max_tokens``, ``capabilities.image_input.supported``,
  ``capabilities.thinking.{supported, types.{adaptive,enabled}.supported}``; shape verified from docs
  https://platform.claude.com/docs/en/api/models/retrieve).

``list_models`` (the ``/models`` listing the settings dialog offers) lives here too. The OriginShape account has no
probe and its listing costs no request: it answers the model aliases (``_originshape_models``), so nothing here ever
sends the account's token, or a profile's key, to another host.

Documentation status: ``/api/ps`` lists running models with ``context_length`` and
``/api/show`` returns ``model_info`` / ``capabilities`` (verified from docs
https://github.com/ollama/ollama/blob/main/docs/api.md); LM Studio's ``/api/v0/models`` documents ``id``, ``type``,
``state`` and ``max_context_length`` (verified from docs https://lmstudio.ai/docs/developer/rest/endpoints), while
``loaded_context_length`` and ``capabilities`` are not in its published examples (live-only; LM Studio 0.4 adds a
``/api/v1`` REST API whose schema is not read here). Every parser reads defensively and states only what the payload
states. A value that is absent stays unset, so the table default applies.
"""

import re

import requests

from . import models as _models
from .provider_transport import anthropic_base

_USER_AGENT = "Oscar-FreeCAD"


def http_json(method: str, url: str, *, headers: dict | None = None, body: dict | None = None,
              timeout: float = 8.0) -> tuple[int, object]:
    """``(status, parsed JSON or None)``; ``(0, None)`` on any transport or decoding failure. Never raises."""
    try:
        merged = {"User-Agent": _USER_AGENT, **(headers or {})}
        if method == "POST":
            response = requests.post(url, headers=merged, json=body, timeout=(min(timeout, 5.0), timeout))
        else:
            response = requests.get(url, headers=merged, timeout=(min(timeout, 5.0), timeout))
        status = int(getattr(response, "status_code", 0) or 0)
        if status >= 400:
            return status, None
        return status, response.json()
    except Exception:  # noqa: BLE001 - connection refused, timeout, invalid JSON, a test fake without .json
        return 0, None


def _int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number > 0 else None


def _same_model(listed, wanted: str) -> bool:
    """Ollama-style model-name equality: ``name`` equals ``name:latest``, case-insensitive."""
    if not isinstance(listed, str) or not wanted:
        return False

    def norm(text: str) -> str:
        text = text.strip().lower()
        return text[:-len(":latest")] if text.endswith(":latest") else text

    return norm(listed) == norm(wanted)


def server_root(base_url: str) -> str:
    """The server root of an OpenAI-compatible base URL: ``http://h:11434/v1`` -> ``http://h:11434`` (no ``/v1``, ``/api``, ``/chat/completions``)."""
    root = (base_url or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/v1", "/api/v1", "/api/chat", "/api"):
        if root.endswith(suffix):
            root = root[:-len(suffix)].rstrip("/")
    return root


# -- pure parsers ----------------------------------------------------------------------------------------------

def parse_openrouter(payload, model: str) -> dict:
    """``{"fields": ...}`` from an OpenRouter ``/models`` payload for ``model`` (``{}`` when it is not listed)."""
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {}
    wanted = (model or "").strip()
    entry = next((e for e in entries if isinstance(e, dict) and e.get("id") == wanted), None)
    if entry is None and ":" in wanted:
        base = wanted.split(":", 1)[0]
        entry = next((e for e in entries if isinstance(e, dict) and e.get("id") == base), None)
    if entry is None:
        return {}
    fields: dict = {}
    top = entry.get("top_provider") if isinstance(entry.get("top_provider"), dict) else {}
    window = _int(top.get("context_length")) or _int(entry.get("context_length"))
    if window:
        fields["context_window"] = window
    output = _int(top.get("max_completion_tokens"))
    if output:
        fields["max_output"] = output
    supported = entry.get("supported_parameters")
    if isinstance(supported, list) and supported:
        names = {str(x) for x in supported}
        fields["tools"] = "native" if "tools" in names else "text"
        fields["tool_choice_ok"] = "tool_choice" in names if "tools" in names else True
        fields["sampling_ok"] = "temperature" in names
        if "reasoning" in names or "include_reasoning" in names:
            fields["reasoning"] = "stream"
            fields["reasoning_control"] = "openrouter_reasoning"
    arch = entry.get("architecture") if isinstance(entry.get("architecture"), dict) else {}
    modalities = arch.get("input_modalities")
    if isinstance(modalities, list) and modalities:
        fields["vision"] = "image" in {str(x).lower() for x in modalities}
    return {"fields": fields} if fields else {}


def parse_ollama(show, ps, model: str) -> dict:
    """``{"fields", "extra"}`` from Ollama ``/api/show`` and ``/api/ps`` payloads (either may be None)."""
    fields: dict = {}
    extra: dict = {}
    if isinstance(show, dict):
        capabilities = show.get("capabilities")
        if isinstance(capabilities, list) and capabilities:
            names = {str(x).lower() for x in capabilities}
            fields["tools"] = "native" if "tools" in names else "text"
            fields["vision"] = "vision" in names
            if "thinking" in names:
                fields["reasoning"] = "stream"
                fields["reasoning_control"] = "ollama_effort"
        info = show.get("model_info")
        if isinstance(info, dict):
            trained = next((_int(v) for k, v in info.items() if str(k).endswith(".context_length") and _int(v)), None)
            if trained:
                extra["context_max"] = trained
    models_loaded = ps.get("models") if isinstance(ps, dict) else None
    if isinstance(models_loaded, list):
        for item in models_loaded:
            if isinstance(item, dict) and (_same_model(item.get("name"), model) or _same_model(item.get("model"), model)):
                active = _int(item.get("context_length"))
                if active:
                    fields["context_window"] = active
                    extra["loaded"] = True
                break
    out: dict = {}
    if fields:
        out["fields"] = fields
    if extra:
        out["extra"] = extra
    return out


def parse_lmstudio(payload, model: str) -> dict:
    """``{"fields", "extra"}`` from an LM Studio ``/api/v0/models`` payload (field names: documentation memory)."""
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {}
    entry = next((e for e in entries if isinstance(e, dict) and e.get("id") == model), None)
    if entry is None:
        return {}
    fields: dict = {}
    extra: dict = {}
    kind = str(entry.get("type") or "").lower()
    if kind in ("vlm", "llm"):
        fields["vision"] = kind == "vlm"
    loaded = _int(entry.get("loaded_context_length"))
    if loaded and str(entry.get("state") or "").lower() == "loaded":
        fields["context_window"] = loaded
    trained = _int(entry.get("max_context_length"))
    if trained:
        extra["context_max"] = trained
    capabilities = entry.get("capabilities")
    if isinstance(capabilities, list) and "tool_use" in {str(x).lower() for x in capabilities}:
        fields["tools"] = "native"
    out: dict = {}
    if fields:
        out["fields"] = fields
    if extra:
        out["extra"] = extra
    return out


def parse_anthropic(payload) -> dict:
    """``{"fields"}`` from an Anthropic ``/v1/models/<id>`` payload (``max_input_tokens`` / ``max_tokens`` / image support)."""
    if not isinstance(payload, dict):
        return {}
    fields: dict = {}
    window = _int(payload.get("max_input_tokens"))
    if window:
        fields["context_window"] = window
    output = _int(payload.get("max_tokens"))
    if output:
        fields["max_output"] = output
    capabilities = payload.get("capabilities")
    if isinstance(capabilities, dict):
        image = capabilities.get("image_input")
        if isinstance(image, dict) and isinstance(image.get("supported"), bool):
            fields["vision"] = image["supported"]
        thinking = capabilities.get("thinking")
        if isinstance(thinking, dict) and isinstance(thinking.get("supported"), bool):
            types = thinking.get("types") if isinstance(thinking.get("types"), dict) else {}

            def _ok(name):
                entry = types.get(name)
                return isinstance(entry, dict) and entry.get("supported") is True

            if not thinking["supported"]:
                fields["reasoning_control"] = "none"
            elif _ok("adaptive"):
                fields["reasoning_control"] = "anthropic_adaptive"   # the documented replacement; ``enabled`` may be deprecated
            elif _ok("enabled"):
                fields["reasoning_control"] = "anthropic_budget"
    return {"fields": fields} if fields else {}


# -- the network side -------------------------------------------------------------------------------------------

def _probe_openrouter(profile, timeout: float) -> dict:
    base = (getattr(profile, "base_url", "") or "https://openrouter.ai/api/v1").rstrip("/")
    status, payload = http_json("GET", f"{base}/models", timeout=timeout)
    return parse_openrouter(payload, getattr(profile, "model", "") or "") if status else {}


def _probe_ollama(profile, timeout: float) -> dict:
    root = server_root(getattr(profile, "base_url", "") or "http://localhost:11434")
    model = getattr(profile, "model", "") or ""
    headers = {"Authorization": f"Bearer {profile.api_key}"} if getattr(profile, "api_key", "") else {}
    _, show = http_json("POST", f"{root}/api/show", headers=headers, body={"model": model}, timeout=timeout)
    _, ps = http_json("GET", f"{root}/api/ps", headers=headers, timeout=timeout)
    return parse_ollama(show, ps, model)


def _probe_lmstudio(profile, timeout: float) -> dict:
    root = server_root(getattr(profile, "base_url", "") or "http://localhost:1234")
    status, payload = http_json("GET", f"{root}/api/v0/models", timeout=timeout)
    return parse_lmstudio(payload, getattr(profile, "model", "") or "") if status else {}


def _probe_anthropic(profile, timeout: float) -> dict:
    key = getattr(profile, "api_key", "") or ""
    model = getattr(profile, "model", "") or ""
    if not key or not model or not re.fullmatch(r"[A-Za-z0-9._:\-]+", model):
        return {}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    status, payload = http_json("GET", f"{anthropic_base(profile)}/v1/models/{model}", headers=headers,
                                timeout=timeout)
    return parse_anthropic(payload) if status else {}


_PROBES = {"openrouter": _probe_openrouter, "ollama": _probe_ollama, "lmstudio": _probe_lmstudio,
           "anthropic": _probe_anthropic}


def probe(profile, *, timeout: float = 8.0) -> dict:
    """The probe result for ``profile`` (see the module docstring); ``{}`` for a host without a probe or on failure."""
    try:
        probe_fn = _PROBES.get(_models.host_class(profile))
        if probe_fn is None or not getattr(profile, "model", ""):
            return {}
        return probe_fn(profile, timeout) or {}
    except Exception:  # noqa: BLE001 - a probe is advisory
        return {}


def _originshape_models() -> list[str]:
    """The OriginShape model aliases: the account's ``/v1/models`` answer when one was fetched, else the built-in
    ones. Local only (no request, no token)."""
    from .auth import account

    try:
        catalog = account.cached_models()
    except Exception:  # noqa: BLE001 - the built-in list needs nothing
        catalog = None
    names = [m.id for m in catalog.models] if catalog is not None else []
    return names or list(account.LINEUP)


def list_models(profile) -> tuple[list[str], str]:
    """Fetch the model list from a provider's /models endpoint.

    Works for both wire protocols (OpenAI-compatible ``GET {base}/models``
    and Anthropic ``GET {base}/v1/models``). Returns ``(models, error)`` —
    on error the list is empty and the message explains why; no exceptions
    cross into UI code. The OriginShape account answers its aliases without a request.
    """
    if getattr(profile, "provider", "") == "originshape":
        return _originshape_models(), ""
    try:
        if profile.provider == "anthropic":
            url = f"{anthropic_base(profile)}/v1/models"
            headers = {
                "x-api-key": profile.api_key,
                "anthropic-version": "2023-06-01",
            }
        else:
            base = (profile.base_url or "https://api.openai.com/v1").rstrip("/")
            if base.endswith("/chat/completions"):      # chats fine with it, so accept it here too
                base = base[: -len("/chat/completions")].rstrip("/")
            url = f"{base}/models"
            headers = {}
            if profile.api_key:
                headers["Authorization"] = f"Bearer {profile.api_key}"
        response = requests.get(url, headers=headers, timeout=(10, 30))
        if response.status_code >= 400:
            return [], f"HTTP {response.status_code}: {response.text[:200]}"
        payload = response.json()
        # {"data": [...]} (OpenAI, Anthropic), {"models": [...]} (Ollama native) or a bare array (Together AI, a few
        # gateways: objects or plain names); anything else is "no models"
        if isinstance(payload, dict):
            entries, named = payload.get("data") or payload.get("models") or [], False
        elif isinstance(payload, list):
            entries, named = payload, True
        else:
            entries, named = [], False
        if not isinstance(entries, list):
            return [], "endpoint returned an unexpected answer"
        names = [
            (entry.get("id") or entry.get("name")) if isinstance(entry, dict) else (entry if named else None)
            for entry in entries
            if isinstance(entry, (dict, str))
        ]
        names = sorted({n for n in names if n and isinstance(n, str)})
        if not names:
            return [], "endpoint returned no models"
        return names, ""
    except requests.RequestException as e:
        return [], f"connection failed: {e}"
    except ValueError:
        return [], "endpoint did not return JSON"
    except (AttributeError, TypeError):
        return [], "endpoint returned an unexpected answer"
