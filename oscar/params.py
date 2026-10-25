"""Persistence for Oscar's settings.

Profiles and preferences live in Oscar's OWN json file under the user
app data dir — NOT in FreeCAD's user parameter system. Reason: the CAD core loads
its user.cfg under a file lock and, when the lock is contended (another
instance of the application starting at the same moment), continues with an
empty parameter tree and silently DISCARDS every parameter write on exit
(ParameterManager::LoadDocument sets IgnoreSave). API keys saved that way vanish on restart. A file we
write ourselves, atomically (temp file + os.replace), survives crashes,
force-quits and lock races. A file that cannot be understood (truncated,
hand-edited) is copied to ``<name>.corrupt-<timestamp>`` before it is replaced,
and one that cannot be read at all is never replaced.

The public API is unchanged from the parameter-backed version. Tests (and
multi-instance setups) can point the store elsewhere with the
OSCAR_AI_SETTINGS environment variable.

API keys remain plain text in the user config dir; the settings dialog says
so. Move to the OS keychain if that ever stops being acceptable. The
OriginShape account's sign-in is never stored here (``oscar.auth.keystore``):
its profile (``account_profile()``) carries no key and no URL.

Writers never build a file from a read that failed: every read-modify-write
reads strictly (``SettingsUnavailable``, an ``OSError``, when the file exists
but cannot be read right now), and ``<name>.bak`` keeps the previous file
whenever a write changes the profiles.
"""

import errno
import glob
import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from . import context_guard as _context_guard
from . import fsretry as _fsretry

import FreeCAD

_LOCK = threading.Lock()
_MIGRATED = False

# Provider presets shown in the settings dialog. "openai_compat" covers every
# OpenAI-chat-completions endpoint: OpenAI itself, Ollama, OpenRouter, Groq,
# DeepSeek, LM Studio, vLLM, llama.cpp server, ...
#
# Each preset carries the PREDEFINED sampling settings for that provider:
# applied automatically when the preset or a model from its list is chosen,
# and deliberately not editable in the dialog.
#
# The first preset is the OriginShape account ("account": True): no key and no
# base URL (the provider finds its server through ``oscar.auth.config`` and
# signs in with the account), the model is one of ``models``. New installs
# start on it; "Add" in the dialog starts from Anthropic.
ACCOUNT_PROVIDER = "originshape"
ACCOUNT_PROFILE_NAME = "OriginShape"
# The account's model lineup, by the server's alias: "Origin1" (the standard model) and "Origin1 Super" (the most
# capable CAD model; a turn costs 3x Origin1's credits). The UI only ever shows these names. ``oscar-default`` /
# ``oscar-fast`` are the retired aliases of the same two tiers: the server still accepts
# them (so a profile or a workspace task that names one keeps working, ``ACCOUNT_MODELS``), the pickers no longer
# offer them, and a stored profile is read as its tier's current alias (``account_model_alias``, ``_profiles_from``).
ACCOUNT_MODEL_CHOICES = ("origin1", "origin1-super")
LEGACY_ACCOUNT_MODELS: dict[str, str] = {"oscar-default": "origin1-super", "oscar-fast": "origin1"}
ACCOUNT_MODELS = ACCOUNT_MODEL_CHOICES + tuple(LEGACY_ACCOUNT_MODELS)   # every alias the account accepts
ACCOUNT_MODEL_LABELS: dict[str, str] = {"origin1": "Origin1", "origin1-super": "Origin1 Super"}

ORIGINSHAPE_PRESET: dict[str, Any] = {
    "label": "OriginShape (your account)",
    "provider": ACCOUNT_PROVIDER,
    "base_url": "",                     # never stored: the server comes from oscar.auth.config at request time
    "default_model": ACCOUNT_MODEL_CHOICES[0],
    "models": list(ACCOUNT_MODEL_CHOICES),
    "needs_key": False,
    "account": True,                    # the editor hides Base URL and API key: the account signs in instead
    "temperature": 1.0,                 # never sent: the account's models set their own sampling
    "max_tokens": 32000,
}

PROVIDER_PRESETS: list[dict[str, Any]] = [
    ORIGINSHAPE_PRESET,
    {
        "label": "Anthropic (Claude)",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-5",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 32000,
    },
    {
        "label": "OpenAI",
        "provider": "openai_compat",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 16384,
    },
    {
        "label": "Google Gemini",
        "provider": "openai_compat",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.5-pro",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 65536,
    },
    {
        "label": "OpenRouter",
        "provider": "openai_compat",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "anthropic/claude-sonnet-4.5",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    {
        "label": "xAI (Grok)",
        "provider": "openai_compat",
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-4",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 32768,
    },
    {
        "label": "DeepSeek",
        "provider": "openai_compat",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    {
        "label": "Groq",
        "provider": "openai_compat",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    {
        "label": "Together AI",
        "provider": "openai_compat",
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    {
        "label": "Fireworks AI",
        "provider": "openai_compat",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "default_model": "accounts/fireworks/models/llama4-maverick-instruct-basic",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 16384,
    },
    {
        "label": "Mistral AI",
        "provider": "openai_compat",
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    {
        "label": "Cerebras",
        "provider": "openai_compat",
        "base_url": "https://api.cerebras.ai/v1",
        "default_model": "llama-3.3-70b",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 16384,
    },
    {
        "label": "Perplexity",
        "provider": "openai_compat",
        "base_url": "https://api.perplexity.ai",
        "default_model": "sonar-pro",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    {
        "label": "Z.ai (GLM)",
        "provider": "openai_compat",
        "base_url": "https://api.z.ai/api/coding/paas/v4",
        "default_model": "glm-4.6",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    {
        "label": "Alibaba Qwen (DashScope)",
        "provider": "openai_compat",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-max",
        "needs_key": True,
        "temperature": 0.2,
        "max_tokens": 32768,
    },
    {
        "label": "Ollama (local)",
        "provider": "openai_compat",
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen2.5:7b",
        "needs_key": False,
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    {
        "label": "LM Studio (local)",
        "provider": "openai_compat",
        "base_url": "http://localhost:1234/v1",
        "default_model": "",
        "needs_key": False,
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    {
        "label": "Custom (OpenAI-compatible)",
        "provider": "openai_compat",
        "base_url": "",
        "default_model": "",
        "needs_key": False,
        "temperature": 0.2,
        "max_tokens": 8192,
    },
]

# Providers announced before they can be used: ``{"label", "provider", "note"}``. The settings dialog lists them after
# the presets, greyed out and not selectable; ``providers.make_provider`` refuses their kind
# (``providers.COMING_SOON_KINDS``). Empty while nothing is announced; the mechanism stays for the next one.
COMING_SOON_PROVIDERS: list[dict[str, str]] = []

# Model-specific output caps overriding the preset default. Matched by
# substring, first hit wins; unlisted models use the preset's predefined
# max_tokens.
MODEL_LIMITS: list[tuple[str, int]] = [
    ("origin1", 32000),       # the OriginShape aliases (Origin1 / Origin1 Super)
    ("oscar-", 32000),        # their retired spellings (oscar-default / oscar-fast)
    # Claude 3.x caps first; every Claude 4+ model accepts at least 32000
    # (Opus 4/4.1: 32k; Sonnet 4.x, Haiku 4.5, Opus 4.5+: 64k–128k).
    ("claude-3-5-haiku", 8192),
    ("claude-3-5-sonnet", 8192),
    ("claude-3-haiku", 4096),
    ("claude-3-opus", 4096),
    ("claude", 32000),
    ("gpt-5", 32768),
    ("gpt-4.1", 32768),
    ("gpt-4o", 16384),
    ("o1", 32768),
    ("o3", 32768),
    ("o4-mini", 32768),
    ("deepseek-reasoner", 8192),
    ("llama-3.3-70b", 8192),
    # GLM-4.5+ / GLM-5.x think before answering (up to 128K output);
    # the thinking counts against this cap (see the Z.ai preset).
    ("glm-5", 32768),
    ("glm-4.", 32768),
    ("gemini", 65536),
    ("grok", 32768),
    ("qwen", 32768),
    ("sonar", 8192),
]


def preset_for(provider: str, base_url: str = "") -> dict[str, Any]:
    """Best-matching preset for a provider/base-url combination.

    Exact base URL first, then a preset whose ``hosts`` appear in the URL
    (a vendor's alternative endpoints), then the provider's generic preset
    (the one without a base URL).
    """
    candidates = [p for p in PROVIDER_PRESETS if p["provider"] == provider]
    if not candidates:
        return PROVIDER_PRESETS[-1]   # an unknown kind: the generic Custom preset
    if not base_url:
        return candidates[0]
    wanted = base_url.rstrip("/").lower()
    for preset in candidates:
        if preset["base_url"] and preset["base_url"].rstrip("/").lower() == wanted:
            return preset
    for preset in candidates:
        if any(host in wanted for host in preset.get("hosts", ())):
            return preset
    for preset in candidates:
        if not preset["base_url"]:
            return preset
    return candidates[0]


def predefined_limits(provider: str, base_url: str, model: str) -> tuple[float, int]:
    """Predefined (temperature, max_tokens) for a provider+model.

    Model-specific caps override the preset default; the user never edits
    these in the dialog — only the model name is selectable.
    """
    preset = preset_for(provider, base_url)
    max_tokens = int(preset.get("max_tokens", 8192))
    lowered = (model or "").lower()
    for needle, limit in MODEL_LIMITS:
        if needle in lowered:
            max_tokens = limit
            break
    return float(preset.get("temperature", 0.2)), max_tokens


REASONING_LEVELS = ("off", "low", "medium", "high", "auto")


@dataclass
class Profile:
    """One named chat configuration."""

    name: str = "New profile"
    provider: str = "anthropic"          # "anthropic" | "openai_compat" | "originshape" (the account: account_profile())
    base_url: str = "https://api.anthropic.com"
    model: str = "claude-sonnet-4-5"
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 4096
    notes: str = field(default="")
    reasoning: str = field(default="auto")           # "off" | "low" | "medium" | "high" | "auto"
    caps_override: dict = field(default_factory=dict)  # ModelCaps field overrides, e.g. {"tools": "text", "context_window": 8192}

    def __post_init__(self):
        # A hand-edited settings file must not be able to break the provider layer.
        if self.reasoning not in REASONING_LEVELS:
            self.reasoning = "auto"
        if not isinstance(self.caps_override, dict):
            self.caps_override = {}
        # Every provider key loaded from the settings file stays out of the model's context (context_guard).
        _context_guard.remember(self.api_key)


def effective_limits(profile) -> tuple[float, int]:
    """``(temperature, max_tokens)`` for a profile: the predefined limits overridden by ``caps_override``.

    ``predefined_limits(provider, base_url, model)`` supplies the defaults; the profile's
    ``caps_override`` keys ``max_output`` (tokens, >= 1) and ``temperature`` (0..2) replace them when
    valid, so the Advanced overrides survive the per-turn recompute that used to overwrite
    ``profile.temperature`` / ``profile.max_tokens``. Invalid override values are ignored.
    """
    temperature, max_tokens = predefined_limits(
        getattr(profile, "provider", "") or "", getattr(profile, "base_url", "") or "",
        getattr(profile, "model", "") or "")
    override = getattr(profile, "caps_override", None)
    if isinstance(override, dict):
        value = override.get("max_output")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 1:
            max_tokens = int(value)
        value = override.get("temperature")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 2.0:
            temperature = float(value)
    return temperature, max_tokens


# ---------------------------------------------------------------------------
# Settings file


def settings_path() -> str:
    override = os.environ.get("OSCAR_AI_SETTINGS")
    if override:
        return override
    return os.path.join(FreeCAD.getUserAppDataDir(), "oscar_settings.json")


_DEFAULT_STATE: dict[str, Any] = {
    "profiles": [],
    "active_profile": "",
    "prefs": {
        "BridgeAutoStart": False,
        "PanelAutoOpen": False,
        "DockArea": 2,
        "DockFloating": False,
    },
    "model_caps": {},
}


_READ_ATTEMPTS = 3        # a scanner or sync client can hold the file for a moment
_CORRUPT_BACKUPS = 5      # newest copies of unparseable settings files kept
_PREVIOUS_SUFFIX = ".bak"  # the last well-formed file whose profiles a write replaced (owner-only)


class SettingsUnavailable(OSError):
    """The settings file exists but cannot be read right now (a scanner, a sync client, another FreeCAD).

    Raised by every WRITE instead of building a new file on top of a read that failed: a state that did not
    come from the file must never replace it. Readers keep falling back to the defaults."""


def _read_file_state(path: str) -> dict[str, Any] | None:
    """The settings file's parsed content; None when there is no file.

    Raises ValueError when the content is not a settings object and OSError
    when the file exists but cannot be read (after a few short retries).
    """
    for attempt in range(_READ_ATTEMPTS):
        try:
            # utf-8-sig: tolerate the BOM that Windows editors add
            with open(path, "r", encoding="utf-8-sig") as f:
                state = json.load(f)
            break
        except FileNotFoundError:
            return None
        except OSError:
            if attempt == _READ_ATTEMPTS - 1:
                raise
            time.sleep(0.05)
    if not isinstance(state, dict):
        raise ValueError("settings file is not a JSON object")
    return state


def _well_formed(state: dict[str, Any]) -> bool:
    return (isinstance(state.get("profiles", []), list)
            and all(isinstance(p, dict) for p in state.get("profiles", []))
            and isinstance(state.get("prefs", {}), dict)
            and isinstance(state.get("active_profile", ""), str))


def _unavailable(error: OSError) -> SettingsUnavailable:
    return SettingsUnavailable(error.errno or errno.EBUSY, f"Oscar's settings file could not be read ({error}); "
                                                           "nothing was changed")


def _salvage_or_backup(path: str) -> dict[str, Any] | None:
    """A WRITER met content that does not parse. Read once more (a sync client or an editor may have been
    mid-write and the file is whole now); content that still does not parse is copied aside first and None
    (= start from the defaults) is returned."""
    try:
        return _read_file_state(path)
    except ValueError:
        _backup_corrupt(path)
        return None
    except OSError as e:
        raise _unavailable(e) from e


def _read_state(*, for_write: bool = False) -> dict[str, Any]:
    """The settings state.

    Readers (the default) never raise: a file that cannot be read or parsed reads as the defaults, and
    nothing is written. Writers (``for_write=True``) get what the file really holds or an exception: an
    unreadable file raises ``SettingsUnavailable`` (an OSError); unparseable content is re-read once and,
    still unparseable, copied to ``<name>.corrupt-<time>`` before the defaults take its place.
    """
    global _MIGRATED
    path = settings_path()
    missing = False
    try:
        state = _read_file_state(path)
        missing = state is None
    except OSError as e:
        if for_write:
            raise _unavailable(e) from e
        state = None
    except ValueError:
        state = _salvage_or_backup(path) if for_write else None
    if state is None:
        state = json.loads(json.dumps(_DEFAULT_STATE))  # deep copy
        if missing and not _MIGRATED:   # a first run (no file at all), never a read that failed
            _MIGRATED = True
            _migrate_from_user_parameters(state)
    # Valid JSON of the wrong shape must not crash every accessor.
    profiles = state.get("profiles")
    state["profiles"] = ([p for p in profiles if isinstance(p, dict)]
                         if isinstance(profiles, list) else [])
    if not isinstance(state.get("prefs"), dict):
        state["prefs"] = dict(_DEFAULT_STATE["prefs"])
    if not isinstance(state.get("active_profile"), str):
        state["active_profile"] = ""
    if not isinstance(state.get("model_caps"), dict):
        state["model_caps"] = {}
    return state


def _backup_corrupt(path: str) -> None:
    """Keep a copy of a settings file we are about to replace but could not
    understand (hand edit, truncation, another tool's file)."""
    with open(path, "rb") as f:
        raw = f.read()
    stem = f"{path}.corrupt-"
    existing = sorted(glob.glob(glob.escape(stem) + "*"))
    for backup in existing:
        try:
            with open(backup, "rb") as f:
                if f.read() == raw:
                    return  # this exact content is already kept
        except OSError:
            continue
    target = stem + time.strftime("%Y%m%d%H%M%S")
    suffix = 0
    while os.path.exists(target):
        suffix += 1
        target = f"{stem}{time.strftime('%Y%m%d%H%M%S')}-{suffix}"
    # The copy can still hold API keys in clear text: owner-only like the
    # live file (mkstemp makes that one 0600), not the umask default.
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    existing.append(target)
    for old in sorted(existing)[:-_CORRUPT_BACKUPS]:
        try:
            os.unlink(old)
        except OSError:
            pass


def _backup_previous(path: str) -> None:
    """Keep the file a write is about to replace as ``<name>.bak`` (owner-only, atomic), so the previous set of
    profiles and API keys survives whatever replaced it."""
    with open(path, "rb") as f:
        raw = f.read()
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".oscar_ai_", suffix=".bak.tmp")   # 0600 like the live file
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        _fsretry.replace(tmp, path + _PREVIOUS_SUFFIX)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _guard_existing(path: str, new_state: dict[str, Any]) -> None:
    """Never destroy a settings file we could not make sense of.

    Content that is not valid settings JSON is copied aside before it is
    replaced; a file that exists but cannot be read (OSError) is not
    replaced at all — treating it as empty would wipe every profile and
    API key on a transient read problem. A well-formed file whose profiles
    the write changes is kept as ``<name>.bak``.
    """
    try:
        existing = _read_file_state(path)
    except ValueError:
        _backup_corrupt(path)
        return
    if existing is None:
        return
    if not _well_formed(existing):
        _backup_corrupt(path)
    elif existing.get("profiles") != new_state.get("profiles"):
        _backup_previous(path)


def _write_state(state: dict[str, Any]) -> None:
    path = settings_path()
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    _guard_existing(path, state)
    # Atomic replace: a crash mid-write can never truncate the live file.
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".oscar_ai_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)
        _fsretry.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _migrate_from_user_parameters(state: dict[str, Any]) -> None:
    """One-time import of profiles stored by the previous parameter backend."""
    try:
        raw = FreeCAD.ParamGet(
            "User parameter:BaseApp/Preferences/Mod/Oscar"
        ).GetString("Profiles", "")
        if not raw:
            return
        data = json.loads(raw)
        profiles = [item for item in data if isinstance(item, dict)]
        if profiles:
            state["profiles"] = [
                {k: v for k, v in item.items() if k in Profile.__dataclass_fields__}
                for item in profiles
            ]
            active = FreeCAD.ParamGet(
                "User parameter:BaseApp/Preferences/Mod/Oscar"
            ).GetString("ActiveProfile", "")
            state["active_profile"] = active or profiles[0].get("name", "")
            _write_state(state)
            FreeCAD.Console.PrintLog(
                "Oscar AI: migrated profiles from user parameters.\n"
            )
    except Exception:
        pass  # migration is best-effort; defaults are fine


# ---------------------------------------------------------------------------
# Scalar prefs


def get_bool(name: str, default: bool = False) -> bool:
    with _LOCK:
        return bool(_read_state()["prefs"].get(name, default))


def set_bool(name: str, value: bool) -> None:
    with _LOCK:
        state = _read_state(for_write=True)
        state["prefs"][name] = bool(value)
        _write_state(state)


def get_int(name: str, default: int = 0) -> int:
    with _LOCK:
        try:
            return int(_read_state()["prefs"].get(name, default))
        except (TypeError, ValueError):
            return default


def set_int(name: str, value: int) -> None:
    with _LOCK:
        state = _read_state(for_write=True)
        state["prefs"][name] = int(value)
        _write_state(state)


# String pref kept for API compatibility with the old parameter backend.
def get_string(name: str, default: str = "") -> str:
    with _LOCK:
        return str(_read_state()["prefs"].get(name, default))


def set_string(name: str, value: str) -> None:
    with _LOCK:
        state = _read_state(for_write=True)
        state["prefs"][name] = value
        _write_state(state)


# ---------------------------------------------------------------------------
# Profiles


def account_model_alias(alias: str) -> str:
    """The current alias of an account model: a retired one (``oscar-default`` / ``oscar-fast``) becomes its tier's
    (``origin1-super`` / ``origin1``); any other name comes back unchanged."""
    name = (alias or "").strip() if isinstance(alias, str) else ""
    return LEGACY_ACCOUNT_MODELS.get(name, name)


def model_display_name(profile) -> str:
    """The model name the UI shows for ``profile``: the tier's name for an account alias ("Origin1",
    "Origin1 Super", a retired alias as its tier), an alias the server added since by its name in the cached
    ``/v1/models`` (else as it is), the model id of the user's own provider unchanged; ``""`` for no profile."""
    if profile is None:
        return ""
    model = getattr(profile, "model", "") or ""
    if not is_account_profile(profile):
        return model
    alias = account_model_alias(model)
    if alias in ACCOUNT_MODEL_LABELS:
        return ACCOUNT_MODEL_LABELS[alias]
    try:   # a model the server added: its name from the cached /v1/models
        from .auth import account

        info = account.cached_model(alias)
    except Exception:  # noqa: BLE001 - the alias itself is a fine name
        info = None
    return info.display_name if info is not None and info.display_name else alias


def _profiles_from(raw) -> list[Profile]:
    """Profiles from the stored dicts. An account profile that names a retired alias is read as its tier's current one
    (in memory only: the file changes with the next save, through ``save_profiles``)."""
    known = set(Profile.__dataclass_fields__)
    profiles = [Profile(**{k: v for k, v in item.items() if k in known}) for item in raw if isinstance(item, dict)]
    for profile in profiles:
        if is_account_profile(profile) and isinstance(profile.model, str) and profile.model in LEGACY_ACCOUNT_MODELS:
            profile.model = LEGACY_ACCOUNT_MODELS[profile.model]
    return profiles


def account_profile(name: str = ACCOUNT_PROFILE_NAME) -> Profile:
    """A profile on the OriginShape account: the default alias, no API key and no base URL (the provider finds its
    server through ``oscar.auth.config`` and signs in with the account)."""
    return Profile(name=name, provider=ACCOUNT_PROVIDER, base_url="", model=ORIGINSHAPE_PRESET["default_model"],
                   api_key="", temperature=float(ORIGINSHAPE_PRESET["temperature"]),
                   max_tokens=int(ORIGINSHAPE_PRESET["max_tokens"]))


def is_account_profile(profile) -> bool:
    """True for a profile that uses the OriginShape account (``provider == "originshape"``)."""
    return getattr(profile, "provider", "") == ACCOUNT_PROVIDER


def _seed_profile() -> Profile:
    return account_profile()   # new installs start on the account; existing files are never re-seeded


def ensure_account_profile(*, activate: bool) -> str:
    """The name of the OriginShape account profile, adding one (first in the list, named ``OriginShape``, or
    ``OriginShape 2`` ... when the name is taken) when there is none. ``activate`` makes it the active profile; without
    it, the profile that was in use stays in use. Called when the user signs in: the other profiles are kept as they
    are. One locked, strict read-modify-write that writes only when something changes; raises ``SettingsUnavailable``
    (an ``OSError``) instead of writing on top of a read that failed."""
    with _LOCK:
        state = _read_state(for_write=True)
        profiles = state["profiles"]
        names = [p.get("name") if isinstance(p.get("name"), str) else "" for p in profiles]
        active = state["active_profile"]
        in_use = active if active and active in names else next((n for n in names if n), "")   # active_profile_name()
        name = next((n for p, n in zip(profiles, names) if n and p.get("provider") == ACCOUNT_PROVIDER), "")
        changed = False
        if not name:
            taken = {n.casefold() for n in names}
            name, number = ACCOUNT_PROFILE_NAME, 2
            while name.casefold() in taken:
                name, number = f"{ACCOUNT_PROFILE_NAME} {number}", number + 1
            profiles.insert(0, asdict(account_profile(name)))
            changed = True
        if activate:
            wanted = name
        elif changed and in_use:   # pinned by name: the new first profile must not take over from the one in use
            wanted = in_use
        else:
            wanted = active
        if active != wanted:
            state["active_profile"] = wanted
            changed = True
        if changed:
            _write_state(state)
        return name


def load_profiles_checked() -> tuple[list[Profile], bool]:
    """``(profiles, from_file)``. ``from_file`` is False when the list is an in-memory starter profile because the
    settings file exists but cannot be read right now: a caller that would SAVE the list must not."""
    with _LOCK:
        raw = _read_state()["profiles"]
    profiles = _profiles_from(raw)
    if profiles:
        return profiles, True
    seed = _seed_profile()
    try:
        with _LOCK:   # one transaction: re-read strictly, seed only a file that really has no profiles
            state = _read_state(for_write=True)
            stored = _profiles_from(state["profiles"])
            if stored:   # the first read failed and this one did not: never seed over real profiles
                return stored, True
            state["profiles"] = [asdict(seed)]
            state["active_profile"] = seed.name
            _write_state(state)
    except OSError:
        return [seed], False   # unreadable right now: keep the starter profile in memory, the file untouched
    return [seed], True


def load_profiles() -> list[Profile]:
    """Load all profiles, seeding the OriginShape account profile on first run (never over a file that could not be
    read)."""
    return load_profiles_checked()[0]


def save_profiles(profiles: list[Profile]) -> None:
    with _LOCK:
        state = _read_state(for_write=True)
        state["profiles"] = [asdict(p) for p in profiles]
        # Drop the active profile if it no longer exists.
        names = {p.name for p in profiles}
        if state["active_profile"] not in names and profiles:
            state["active_profile"] = profiles[0].name
        _write_state(state)


def active_profile_name() -> str:
    with _LOCK:
        state = _read_state()
    name = state["active_profile"]
    if name and any(p.get("name") == name for p in state["profiles"]):
        return name
    return state["profiles"][0].get("name", "") if state["profiles"] else ""


def active_profile() -> Profile | None:
    name = active_profile_name()
    for p in load_profiles():
        if p.name == name:
            return p
    return None


def set_active_profile(name: str) -> None:
    with _LOCK:
        state = _read_state(for_write=True)
        if any(p.get("name") == name for p in state["profiles"]):
            state["active_profile"] = name
            _write_state(state)



# ---------------------------------------------------------------------------
# Learned / probed model capabilities (oscar.models; additive settings key "model_caps")


def get_model_caps(key: str) -> dict:
    """The stored capability entry for ``key`` (``{"learned": {...}, "api": {...}, "api_ts": "..."}``) or ``{}``.

    Returns a deep copy, so callers may mutate it freely.
    """
    with _LOCK:
        entry = _read_state()["model_caps"].get(key)
    return json.loads(json.dumps(entry)) if isinstance(entry, dict) else {}


def set_model_caps(key: str, data: dict) -> None:
    """Replace the capability entry for ``key`` (an empty / falsy ``data`` removes it); atomic file write."""
    update_model_caps(key, lambda _old: data if data else None)


def update_model_caps(key: str, mutate) -> None:
    """Atomic read-modify-write of one capability entry.

    ``mutate(entry: dict) -> dict | None`` receives a copy of the stored entry (``{}`` when there is none)
    and returns the new entry; ``None`` or an empty dict removes the entry. Two threads learning at the
    same moment therefore cannot drop each other's update.
    """
    with _LOCK:
        state = _read_state(for_write=True)
        caps = state["model_caps"]
        if not isinstance(caps, dict):
            caps = state["model_caps"] = {}
        old = caps.get(key)
        new = mutate(json.loads(json.dumps(old)) if isinstance(old, dict) else {})
        if new:
            caps[key] = new
        else:
            caps.pop(key, None)
        _write_state(state)


# ---------------------------------------------------------------------------
# Backup / restore (used by the smoke test to leave user settings untouched)


def backup_state() -> str:
    """Snapshot the settings file content as a json string."""
    with _LOCK:
        path = settings_path()
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def restore_state(blob: str) -> None:
    """Restore a backup_state() snapshot (empty string removes the file)."""
    with _LOCK:
        path = settings_path()
        if not blob:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            return
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".oscar_ai_", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(blob)
        _fsretry.replace(tmp, path)


# ---------------------------------------------------------------------------
# Network grants and web settings (oscar.netguard / oscar.tools_web; additive settings key "net_grants")


def _clean_grants(raw) -> list[dict]:
    """Well-formed ``{tool, host, granted_at}`` entries only (anything else in the file is dropped), without duplicates."""
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        tool, host = item.get("tool"), item.get("host")
        if not isinstance(tool, str) or not isinstance(host, str) or not tool or not host or (tool, host) in seen:
            continue
        seen.add((tool, host))
        stamp = item.get("granted_at")
        out.append({"tool": tool, "host": host, "granted_at": stamp if isinstance(stamp, str) else ""})
    return out


def get_net_grants() -> list[dict]:
    """The persistent per-(tool, host) network grants, oldest first: ``[{"tool", "host", "granted_at"}]`` (copies)."""
    with _LOCK:
        return _clean_grants(_read_state().get("net_grants"))


def add_net_grant(tool: str, host: str) -> bool:
    """Store ``{tool, host, granted_at}`` (local ISO time); idempotent. True when the grant is new."""
    with _LOCK:
        state = _read_state(for_write=True)
        grants = _clean_grants(state.get("net_grants"))
        if any(g["tool"] == tool and g["host"] == host for g in grants):
            return False
        grants.append({"tool": tool, "host": host, "granted_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        state["net_grants"] = grants
        _write_state(state)
        return True


def remove_net_grant(tool: str, host: str) -> bool:
    """Remove one grant; True when it existed."""
    with _LOCK:
        state = _read_state(for_write=True)
        grants = _clean_grants(state.get("net_grants"))
        kept = [g for g in grants if not (g["tool"] == tool and g["host"] == host)]
        if len(kept) == len(grants):
            return False
        state["net_grants"] = kept
        _write_state(state)
        return True


def clear_net_grants() -> int:
    """Remove every per-(tool, host) grant (the legacy ``AllowNetwork`` pref is NOT touched here); returns how many."""
    with _LOCK:
        state = _read_state(for_write=True)
        count = len(_clean_grants(state.get("net_grants")))
        if count or "net_grants" in state:
            state["net_grants"] = []
            _write_state(state)
        return count
