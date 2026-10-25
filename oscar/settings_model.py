"""The logic behind the settings dialog, kept free of Qt so it is testable without a GUI.

* ``read_advanced`` / ``apply_advanced`` - the per-profile Advanced overrides (max output, temperature, reasoning,
  context window, tool mode, native Ollama) <-> ``Profile.reasoning`` and ``Profile.caps_override``. The keys are the
  ones ``params.effective_limits`` and ``models.resolve_caps`` read, so an override survives the per-turn recompute
  in ``AgentWorker`` (which used to overwrite ``profile.temperature`` / ``profile.max_tokens`` every turn).
* ``describe_effective`` - the one-line "what will actually be used" the dialog shows under the overrides.
* ``run_in_background`` / ``BackgroundJob`` - run a blocking call (model list, connection test, a grant write) on a
  daemon thread and deliver ONE result to a callback; a cancelled job delivers nothing. The dialog
  hands the result to the GUI thread through a Qt signal; nothing blocking ever runs on the GUI thread.
* ``net_grant_rows`` / ``revoke_net_grant`` / ``revoke_all_net_grants`` - the "Network permissions" section: the per-(tool,
  host) grants of ``netguard`` (plus the legacy global 'Always allow' pref as a revocable legacy row) and their revocation;
  ``validate_search_url`` for the web settings. The dialog runs the writes through ``run_in_background``.
"""

import threading
from dataclasses import dataclass
from typing import Any, Callable

from . import models as _models
from . import params as _params

REASONING_CHOICES = ("auto", "off", "low", "medium", "high")
TOOL_MODE_CHOICES = ("auto", "native", "text")

MAX_OUTPUT_RANGE = (256, 1_000_000)
CONTEXT_WINDOW_RANGE = (1024, 10_000_000)
TEMPERATURE_RANGE = (0.0, 2.0)

# -- Advanced overrides ---------------------------------------------------------------------------------------

def read_advanced(profile) -> dict[str, Any]:
    """The Advanced values of ``profile`` for the dialog; ``0`` / ``None`` / ``"auto"`` mean "no override".

    ``{"max_output": int, "temperature": float | None, "reasoning": str, "context_window": int, "tools": str,
    "ollama_native": bool}``; unreadable stored values read as "no override".
    """
    override = getattr(profile, "caps_override", None)
    override = override if isinstance(override, dict) else {}

    def number(name: str, low, high, cast):
        value = override.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return cast(value) if low <= value <= high else None

    tools = override.get("tools")
    reasoning = getattr(profile, "reasoning", "auto")
    return {
        "max_output": number("max_output", *MAX_OUTPUT_RANGE, int) or 0,
        "temperature": number("temperature", *TEMPERATURE_RANGE, float),
        "reasoning": reasoning if reasoning in REASONING_CHOICES else "auto",
        "context_window": number("context_window", *CONTEXT_WINDOW_RANGE, int) or 0,
        "tools": tools if tools in ("native", "text") else "auto",
        "ollama_native": override.get("ollama_native") is not False,
    }


def _field_problems(values: dict[str, Any]) -> dict[str, str]:
    """``{field: problem sentence}`` for every Advanced field in ``values`` that is not acceptable."""
    problems: dict[str, str] = {}
    out = values.get("max_output") or 0
    if out and not MAX_OUTPUT_RANGE[0] <= int(out) <= MAX_OUTPUT_RANGE[1]:
        problems["max_output"] = (f"max output must be between {MAX_OUTPUT_RANGE[0]} and {MAX_OUTPUT_RANGE[1]} "
                                  "tokens (0 = automatic)")
    window = values.get("context_window") or 0
    if window and not CONTEXT_WINDOW_RANGE[0] <= int(window) <= CONTEXT_WINDOW_RANGE[1]:
        problems["context_window"] = (f"context window must be between {CONTEXT_WINDOW_RANGE[0]} and "
                                      f"{CONTEXT_WINDOW_RANGE[1]} tokens (0 = automatic)")
    temperature = values.get("temperature")
    if temperature is not None and not TEMPERATURE_RANGE[0] <= float(temperature) <= TEMPERATURE_RANGE[1]:
        problems["temperature"] = "temperature must be between 0 and 2"
    if values.get("reasoning", "auto") not in REASONING_CHOICES:
        problems["reasoning"] = "reasoning must be one of " + ", ".join(REASONING_CHOICES)
    if values.get("tools", "auto") not in TOOL_MODE_CHOICES:
        problems["tools"] = "tool mode must be one of " + ", ".join(TOOL_MODE_CHOICES)
    return problems


def validate_advanced(values: dict[str, Any]) -> list[str]:
    """Problems with Advanced ``values`` as short sentences (empty when every value is acceptable)."""
    return list(_field_problems(values).values())


def apply_advanced(profile, values: dict[str, Any]) -> list[str]:
    """Write Advanced ``values`` into ``profile`` (``reasoning`` and ``caps_override``); returns the problems found.

    Only the fields present in ``values`` are touched. Invalid values are NOT applied (the profile keeps what it
    had for them). ``0`` / ``None`` / ``"auto"`` remove the matching override so the predefined / learned / probed
    value applies again. Keys of ``caps_override`` that the dialog does not own (a hand-edited ``vision`` override,
    say) are kept.
    """
    bad = _field_problems(values)
    override = dict(profile.caps_override) if isinstance(getattr(profile, "caps_override", None), dict) else {}

    def put(key: str, value) -> None:
        if value in (None, 0, "auto", False):
            override.pop(key, None)
        else:
            override[key] = value

    if "max_output" in values and "max_output" not in bad:
        put("max_output", int(values["max_output"] or 0))
    if "context_window" in values and "context_window" not in bad:
        put("context_window", int(values["context_window"] or 0))
    if "temperature" in values and "temperature" not in bad:
        temperature = values["temperature"]
        override.pop("temperature", None) if temperature is None else override.__setitem__(
            "temperature", round(float(temperature), 3))
    if "tools" in values and "tools" not in bad:
        put("tools", values["tools"])
    if "ollama_native" in values:
        if values["ollama_native"]:
            override.pop("ollama_native", None)
        else:
            override["ollama_native"] = False
    if "reasoning" in values and "reasoning" not in bad:
        profile.reasoning = values["reasoning"]
    profile.caps_override = override
    return list(bad.values())


def describe_effective(profile) -> str:
    """One line on what a request with ``profile`` will use: window, output cap, tool mode, vision, reasoning.

    Cache-only (``resolve_caps`` never touches the network). The tool mode and vision of an unprobed model are
    shown as such: "vision unknown" means screenshots are tried and remembered if the model refuses them.
    """
    try:
        caps = _models.resolve_caps(profile)
        temperature, max_tokens = _params.effective_limits(profile)
    except Exception as e:  # noqa: BLE001 - a description must never break the dialog
        return f"effective settings unavailable ({type(e).__name__})"
    window = caps.context_window
    if _models.ollama_native_wanted(profile):
        window = _models.ollama_num_ctx(profile)
    vision = "unknown" if caps.vision is None else ("yes" if caps.vision else "no")
    reasoning = getattr(profile, "reasoning", "auto")
    return (f"Effective: context {window} tokens, output up to {max_tokens}, temperature {temperature:g}, "
            f"tools {caps.tools}, images {vision}, reasoning {reasoning} (settings source: {caps.source}).")


# -- background jobs --------------------------------------------------------------------------------------------

@dataclass
class JobResult:
    """What a background job delivers: ``ok`` with ``value``, or not ok with an ``error`` sentence."""

    ok: bool
    value: Any = None
    error: str = ""


class BackgroundJob:
    """One blocking call on a daemon thread. ``cancel()`` sets ``cancelled`` (the callable polls it) and suppresses delivery."""

    def __init__(self, fn: Callable[["threading.Event"], Any], deliver: Callable[[JobResult], None], *,
                 name: str = "oscar-settings-job"):
        self._fn = fn
        self._deliver = deliver
        self.cancelled = threading.Event()
        self._thread = threading.Thread(target=self._main, name=name, daemon=True)
        self._lock = threading.Lock()
        self._delivered = False

    def start(self) -> "BackgroundJob":
        self._thread.start()
        return self

    def cancel(self) -> None:
        """Ask the job to stop; whatever it returns afterwards is dropped."""
        self.cancelled.set()

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the thread (tests); True when it ended."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _main(self) -> None:
        try:
            result = JobResult(True, self._fn(self.cancelled))
        except BaseException as e:  # noqa: BLE001 - delivered as an error sentence, never lost
            result = JobResult(False, None, f"{type(e).__name__}: {e}")
        with self._lock:
            if self._delivered or self.cancelled.is_set():
                return
            self._delivered = True
        try:
            self._deliver(result)
        except Exception:  # noqa: BLE001 - a closed dialog must not leave a traceback on the thread
            pass


def run_in_background(fn, deliver, *, name: str = "oscar-settings-job") -> BackgroundJob:
    """Start ``fn(cancelled_event)`` on a daemon thread; ``deliver(JobResult)`` is called once from that thread
    (unless the job was cancelled). Returns the started ``BackgroundJob``."""
    return BackgroundJob(fn, deliver, name=name).start()


# -- saving -----------------------------------------------------------------------------------------------------


# -- network permissions ----------------------------------------------------------------------------------------

LEGACY_TOOL = "(all tools)"
LEGACY_HOST = "(all hosts)"


def net_grant_rows() -> list[dict]:
    """Rows for the Network permissions list: ``{tool, host, granted_at, legacy, label}``.

    The legacy global 'Always allow' pref grants NOTHING any more; it is listed (first) only so the user can see and revoke it.
    """
    from . import netguard

    rows: list[dict] = []
    if netguard.legacy_flag():
        rows.append({"tool": LEGACY_TOOL, "host": LEGACY_HOST, "granted_at": "", "legacy": True,
                     "label": "Legacy 'Always allow' (inactive: grants nothing; Remove clears it)"})
    for grant in netguard.list_grants():
        any_host = grant["host"] == netguard.UNSCOPED_HOST
        where = "any host (scripts and downloads)" if any_host else grant["host"]
        since = f"  (since {grant['granted_at']})" if grant.get("granted_at") else ""
        rows.append({**grant, "legacy": False, "label": f"{grant['tool']}  \u2192  {where}{since}"})
    return rows


def revoke_net_grant(row: dict) -> bool:
    """Revoke the grant a ``net_grant_rows`` row describes (the legacy row clears the old pref); True when something was removed."""
    from . import netguard

    if row.get("legacy"):
        had = netguard.legacy_flag()
        _params.set_bool("AllowNetwork", False)
        return had
    return netguard.revoke_grant(str(row.get("tool", "")), str(row.get("host", "")))


def revoke_all_net_grants() -> int:
    """Revoke every grant and the legacy pref; returns how many entries were removed."""
    from . import netguard

    return netguard.revoke_all()


def validate_search_url(text: str) -> str:
    """'' when ``text`` is empty or an http(s) URL with a host, else the sentence that says what to enter."""
    from urllib.parse import urlsplit

    value = (text or "").strip()
    if not value:
        return ""
    try:
        parts = urlsplit(value)
        ok = parts.scheme.lower() in ("http", "https") and bool(parts.hostname)
    except ValueError:
        ok = False
    return "" if ok else "The search backend must be an http(s) URL such as http://localhost:8080."
