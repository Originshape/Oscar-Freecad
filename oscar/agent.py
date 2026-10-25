"""Oscar's agent loop.

A ``QThread`` that owns one conversation turn: stream a completion, run any requested tools (dispatched to the
GUI thread by the runtime), feed results back (screenshots included, as vision input) and repeat until the model
answers in plain text or a budget ends the turn. The panel renders progress through the signals.

What makes the loop model-independent is that the APP owns the flow (``oscar.flow``): the typed request is
classified and seeded into a plan ledger before the first model call; every tool result is recorded as evidence
by ``tools.run_tool``; the round governor line tells the model where it is; a reply without tool calls must pass
the finish gate before the turn may end; a watchdog stops runaway reasoning; a failure breaker stops repeated
failures and stalls; the final outcome (``answered`` / ``validated`` / ``unverified`` / ``needs_input`` /
``capped`` / ``failed`` / ``needs_help`` / ``stopped``) is decided by code, never by the model's say-so.
``OSCAR_FLOW=0`` switches all of that off inside this same loop (ledger, gate, governor, watchdog,
breaker, ``run_tool`` hooks); recording into the Project and per-round persistence stay on.

The conversation history is plain ``providers.Message`` data; the panel owns it so it survives across turns and
worker instances. It is append-only within a session (the final answer of every turn included, so thinking
models can replay their reasoning); the only edit is compaction at the start of a turn, which drops whole old
turns at a clean boundary. App-injected notes carry ``synthetic=True`` and a ``tag``. What a provider request
contains is DERIVED every round by ``oscar.context.build_request_view`` from that history (never stored back), with
the system prompt, the per-round tool list, the pinned task memory, the floor rule and the health baseline wired
in ``oscar.agent_ctx``.

Account errors: a turn that fails because of the OriginShape account (a ``ProviderError`` of ``kind`` ``"auth"`` or
``"account"``: signed out, no credits, waitlisted, rate limited ...) emits ``account_problem`` (JSON) just before
``turn_outcome``; the ``"Provider error: ..."`` text of ``turn_finished`` is unchanged. ``session_id`` (the chat id)
and the turn id reach the provider as ``provider.request_context``, a new dict per turn and per round (the
OriginShape provider sends them as ``x-oscar-*`` headers; the others ignore it).

Threading: ``__init__`` runs on the GUI thread in production (it reads the workspace there); the turn runs on
the worker thread; ``steer``, ``act_now``, ``stop``, ``pause`` and ``resume`` are safe to call from any thread.
Tools reach the loop's context only through ``toolctx.use_context`` and are called as exactly
``_tools.run_tool(name, arguments)``.
"""

import dataclasses
import json
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import FreeCAD
from PySide import QtCore

from . import context as _context
from . import loopcore
from . import params as _params
from . import prompt as _prompt
from . import tools as _tools
from . import turnlog as _turnlog
from .agent_legacy import (  # noqa: F401  (legacy names, re-exported)
    CONTEXT_CHAR_BUDGET, IMAGE_COST_CHARS, KEEP_HISTORY_MESSAGES, KEEP_SCREENSHOTS, MAX_CUTOFF_RETRIES, MAX_HISTORY_MESSAGES,
    MAX_TOOL_ROUNDS, PERSONA, STRATEGY, TOOL_RESULT_KEEP_CHARS, _BAD_ARGUMENTS, _BUDGET_NOTE, _CUTOFF_NOTE,
    _APP_NOTE_PREFIXES, _CUTOFF_PLACEHOLDER, _EXECUTION_NOTE, _EXECUTION_WORDS, _TRIMMED_NOTE, _WORD_RE, _cut_off,
    _keep_last_screenshot, _message_size, _summarize_arguments, _users_own_words, compact_history,
    needs_execution_nudge, stub_oversized_history)
from .agent_ctx import ContextTooSmall, ContextWiring, default_validator
from .flow_intent import is_goahead
from .flow import (AgentConfig, FailureBreaker, Ledger, TERMINAL, TurnState, WORK, Watchdog, classify_intent,  # noqa: F401
                   compose_capped, compose_failed, compose_needs_help, compose_unverified, finish_gate,
                   format_governor, health_label, is_continuation, tool_category)
from .providers import Message, StopRequested, ToolResult, make_provider
from .toolctx import ToolContext, ToolOutcome, use_context
from .turnlog import log_event

__all__ = ["AgentWorker", "AgentConfig", "TurnResult", "build_system_prompt", "workspace_snapshot"]

_WATCHDOG_PLACEHOLDER = "(this reply was stopped by the app before it contained a tool call)"
_STEER_PREFIX = "\n[USER, sent while you were working]: "
_FAIL_LINE = re.compile(r"^\s*(FAIL|FAILED|ERROR)\b", re.M)
_SLOW_HOSTS = frozenset({"ollama", "lmstudio", "custom"})
_CALL_BLOCK_TYPES = frozenset({"tool_use", "server_tool_use"})   # provider blocks that carry a tool call


def build_system_prompt(caps=None, *, lite: bool | None = None, playbooks=(), tool_names=None) -> str:
    """The static system prompt: ``prompt.build_system_prompt``; no argument = the full text.

    Deliberately static: it never embeds live state. The system prompt is the first thing in every request; any
    change to it invalidates the model's cached prefix and, on current Claude models, the thinking blocks replayed
    in a tool loop. Live document state travels in the user message instead (see ``workspace_snapshot``). The
    loop passes the caps, the lite switch, the playbooks and the turn-start intent's tool names once per turn.
    """
    return _prompt.build_system_prompt(caps, lite=lite, playbooks=playbooks, tool_names=tool_names)


def workspace_snapshot(max_docs: int = 8, max_objects: int = 40) -> str:
    """Open documents and their objects, as a compact text block (``context.workspace_snapshot``).

    Call on the GUI thread (the panel constructs the worker there).
    """
    return _context.workspace_snapshot(max_docs, max_objects)


# ---------------------------------------------------------------------------------------------------
# Small helpers


def _json(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def _new_turn_id() -> str:
    return "T" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def _flow_wanted(flag: bool | None) -> bool:
    """Whether flow enforcement is on: an explicit ``flow`` argument wins, then ``OSCAR_FLOW=0``, then the pref."""
    if flag is not None:
        return bool(flag)
    if os.environ.get("OSCAR_FLOW", "1").strip() == "0":
        return False
    try:
        return bool(_params.get_bool("OscarFlow", True))
    except Exception:  # noqa: BLE001
        return True


def _script_failure(text: str) -> str:
    """First line of a script's output that starts with FAIL / FAILED / ERROR ("" when none); health suffix ignored."""
    body = (text or "").split("\n\n[health]", 1)[0]
    match = _FAIL_LINE.search(body)
    if not match:
        return ""
    line = body[match.start():].strip().splitlines()[0] if body[match.start():].strip() else ""
    return loopcore.first_line(line, 200)


def _validation_dict(validation: Any) -> dict | None:
    """A ValidationResult / FakeValidation / dict as a plain dict; ``None`` stays ``None``."""
    if validation is None:
        return None
    if isinstance(validation, dict):
        return dict(validation)
    fields = ("ok", "doc", "health", "gate_errors", "fit_rows", "fit_ran", "fail_rows", "dfm", "coverage", "ran",
              "not_validated", "text", "after_step")
    return {name: getattr(validation, name) for name in fields if hasattr(validation, name)}


@dataclass
class TurnResult:
    """What one turn produced (the return value of ``AgentWorker.run_turn``)."""

    ok: bool
    text: str
    outcome: str
    reasons: list[str] = field(default_factory=list)
    rounds: int = 0                  # provider rounds
    tool_rounds: int = 0
    geometry_ok: int = 0
    ledger: dict | None = None       # the ledger the panel must CARRY into the next turn
    validation: dict | None = None
    project_dir: str = ""


@dataclass
class _Reply:
    """One provider reply, assembled from its event stream."""

    text: str = ""
    reasoning_chars: int = 0
    calls: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    stop_reason: str = ""
    stop_info: dict | None = None
    usage: dict | None = None
    error: Exception | None = None
    tripped: str = ""                # watchdog reason that cut the reply short
    wall_s: float = 0.0


@dataclass
class _Round:
    """Accounting of one executed tool round."""

    mutating_ok: int = 0
    failed: int = 0
    errors: dict = field(default_factory=dict)       # tool name -> first line of its unresolved error
    verdicts: list = field(default_factory=list)     # BreakerVerdict objects
    fatal: Any = None


class AgentWorker(ContextWiring, QtCore.QThread):
    """Runs one user turn (possibly several tool rounds) off the GUI thread."""

    # ---- legacy signals, unchanged signatures and semantics ----
    text_delta = QtCore.Signal(str)                   # assistant text while streaming (chunk)
    reasoning_delta = QtCore.Signal(str)              # model thinking text while streaming (chunk)
    tool_started = QtCore.Signal(str)                 # tool name
    tool_finished = QtCore.Signal(str, bool, str)     # (f"{name}({args summary})", ok, result text clipped to 400 chars)
    turn_finished = QtCore.Signal(bool, str)          # (ok, final text); ok=False carries an error message
    notice = QtCore.Signal(str)                       # one-way informational notices
    flow_paused = QtCore.Signal(bool)                 # True when held at a round boundary (Pause), False on resume
    # ---- additive signals; payloads are JSON strings ----
    step_started = QtCore.Signal(str)
    step_finished = QtCore.Signal(str)
    plan_changed = QtCore.Signal(str)                 # Ledger JSON
    progress = QtCore.Signal(str)
    round_reset = QtCore.Signal(str)                  # "retry" | "stream_dropped" | "watchdog" | "gate" | "adapted"
    artifacts_changed = QtCore.Signal(str)
    turn_outcome = QtCore.Signal(str, str)            # (outcome, build summary markdown), just BEFORE turn_finished
    account_problem = QtCore.Signal(str)              # {"code","kind","status","message",...}: just BEFORE turn_outcome

    def __init__(self, history: list[Message], parent=None, *, project=None, ledger: "Ledger | dict | None" = None,
                 profile=None, provider=None, caps=None, config: AgentConfig | None = None,
                 validator: "Callable[[str | None, ToolContext], Any] | bool | None" = None,
                 clock: Callable[[], float] | None = None,
                 on_round: Callable[[list[Message]], None] | None = None,
                 request_text: str | None = None, flow: bool | None = None, attachments: Sequence[str] = (),
                 session_id: str = ""):
        """Build a worker for one turn over ``history`` (the panel owns the list).

        ``project`` / ``ledger`` / ``validator`` / ``on_round`` / ``request_text`` / ``flow`` carry the turn's
        build state; ``profile`` and ``provider`` are injectable for tests (both ``None``: the active
        profile and ``make_provider``, which is called by ``run()`` on the worker thread, never here: it can block on
        the network, and a setup failure ends the turn through ``turn_finished(False, message)``); ``caps`` is
        ``models.ModelCaps`` (``None``: resolved cache-only here and
        refreshed from the provider's metadata in ``run()``, else defaults); ``attachments`` are the PATHS of
        attached files (they reach only the intent classifier); ``session_id`` is the chat id the provider may name
        in its requests (``request_context``; default: the bound Project's chat id). ``validator`` is
        ``validator(doc, ctx)``:
        ``None`` = the real ``tools_verify.run_validation`` (no claim while that module is absent), a callable =
        that function (tests inject fakes), ``False`` = never validate. Never raises when ``provider`` is
        injected, even with no active profile; needs no event loop.
        """
        super().__init__(parent)
        self.history = history
        self._stop = False
        self._vision_notice_sent = False
        self._pause_flag = threading.Event()
        self._act_now = threading.Event()
        self._steer_q: "queue.Queue[dict]" = queue.Queue()
        self._cutoffs = 0                  # thinking-only replies nudged this turn
        self._request_text = request_text
        self._attachments = list(attachments or ())
        self._session_id = session_id if isinstance(session_id, str) else ""
        self._account_problem: dict | None = None    # set when the turn failed for an account reason
        self.project = project
        # a private copy: the kill-switch resolution below must never change a config object the caller shares
        self.config = dataclasses.replace(config) if config is not None else AgentConfig()
        if flow is not None:
            self.config.flow_enforcement = bool(flow)
        else:
            self.config.flow_enforcement = bool(self.config.flow_enforcement) and _flow_wanted(None)
        self._validator = validator
        self._clock = clock or time.monotonic
        self._on_round = on_round
        if profile is None:
            profile = _params.active_profile()
        if profile is None:
            if provider is None:
                raise RuntimeError("No chat profile configured")
            profile = _params.Profile(name="injected", provider="openai_compat", model="injected")
        # Temperature and max tokens are predefined per provider/model; recompute them so profiles saved by an
        # older build (stale caps) pick up current limits. effective_limits keeps the user's Advanced overrides.
        try:
            limits = getattr(_params, "effective_limits", None)
            if callable(limits):
                profile.temperature, profile.max_tokens = limits(profile)
            else:
                profile.temperature, profile.max_tokens = _params.predefined_limits(
                    profile.provider, profile.base_url, profile.model)
        except Exception:  # noqa: BLE001
            pass
        self.profile = profile
        self._provider_injected = provider is not None
        self._caps_injected = caps is not None
        # Built in run(), on the worker thread, when none was injected (this constructor runs on the GUI thread).
        # ``None`` until then.
        self.provider = provider
        self.caps = caps if caps is not None else self._resolve_caps(profile)
        if isinstance(ledger, dict):
            ledger = Ledger.from_dict(ledger)
        self._carried: Ledger | None = ledger if isinstance(ledger, Ledger) else None
        self._ledger: Ledger = self._carried or Ledger()
        self._live = bool(self._carried is not None)
        self._began = False
        self._turn_id = ""
        self._request = ""
        self._summary_text = ""
        self._provider_rounds = 0
        self._tool_round = 0
        self._geometry_ok = 0
        self._cap = self.config.max_tool_rounds
        self._state: dict = {"lite": False}
        self.init_wiring()
        # Constructed on the GUI thread: safe to read documents here.
        self._doc_names = self._open_document_names()
        self._attach_workspace_state()

    @staticmethod
    def _resolve_caps(profile):
        """Cache-only ``models.resolve_caps`` (no network); ``None`` when that module is unavailable or fails."""
        try:
            from . import models as _models

            return _models.resolve_caps(profile)
        except ModuleNotFoundError as e:
            if e.name not in ("oscar.models", f"{__package__}.models"):
                raise
        except Exception:  # noqa: BLE001
            pass
        return None

    def _refresh_caps(self) -> None:
        """Worker thread, before the turn: fold the provider's metadata endpoints into the caps (``models.probe_metadata``,
        which may use the network and never raises). Skipped when the caps or the provider were injected (tests) and
        when the resolution fails: the constructor's cache-only caps stand."""
        if self._caps_injected or self._provider_injected:
            return
        try:
            from . import models as _models

            probe = _models.probe_metadata(self.profile)
            if probe:
                self.caps = _models.resolve_caps(self.profile, probe=probe)
                self.init_wiring()
        except Exception as e:  # noqa: BLE001 - the cache-only caps are good enough
            log_event("caps_probe_error", error=f"{type(e).__name__}: {e}"[:200])

    # -- small API --------------------------------------------------------------------------------

    @property
    def ledger(self) -> Ledger:
        """The live ledger after / while the turn runs (before it: the carried one, or an empty one)."""
        return self._ledger

    @staticmethod
    def _open_document_names() -> list[str]:
        """Names of the open documents (GUI thread, called from the constructor); ``[]`` when they cannot be listed."""
        try:
            return [str(name) for name in FreeCAD.listDocuments()]
        except Exception:  # noqa: BLE001
            return []

    def _attach_workspace_state(self) -> None:
        if not self.history:
            return
        last = self.history[-1]
        if last.role != "user" or last.tool_results or "<workspace_state>" in (last.content or ""):
            return
        snapshot = workspace_snapshot()
        if snapshot:
            last.content = f"{last.content}\n\n<workspace_state>\n{snapshot}\n</workspace_state>"

    def stop(self) -> None:
        """Ask the turn to end: the loop re-checks before every tool call and every provider round."""
        self._stop = True

    def pause(self) -> None:
        """Hold the flow at the next round boundary (between provider calls and tool runs); generation in flight finishes first."""
        self._pause_flag.set()

    def resume(self) -> None:
        """Release a ``pause``."""
        self._pause_flag.clear()

    def act_now(self) -> None:
        """The user pressed "Act now": close the running provider call like a watchdog trip (reason ``user``)."""
        self._act_now.set()

    def steer(self, text: str, note: str = "", images: Sequence[str] = ()) -> None:
        """Queue a message the user sent while the turn runs (thread-safe).

        It is appended to the trailing user message at the start of the next provider round (or at a finish
        attempt, so a message sent while the final answer streamed is not lost). Emits a ``notice``.
        """
        self._steer_q.put({"text": str(text or ""), "note": str(note or ""), "images": list(images or ())})
        self.notice.emit("Message queued; Oscar will see it at the next step")

    def drain_unsent(self) -> list[dict]:
        """Steered messages that arrived too late to be delivered: ``[{"text", "note", "images"}]`` (the panel re-sends them)."""
        out = []
        while True:
            try:
                out.append(self._steer_q.get_nowait())
            except queue.Empty:
                return out

    def _wait_if_paused(self) -> None:
        if not self._pause_flag.is_set():
            return
        self.flow_paused.emit(True)
        self._progress("paused")
        while self._pause_flag.is_set():
            if self._stop or self._should_stop():
                self._pause_flag.clear()
                raise StopRequested()
            time.sleep(0.1)
        self.flow_paused.emit(False)
        self._progress("thinking")

    def _should_stop(self) -> bool:
        return self._stop

    # -- thread entry --------------------------------------------------------------------------------

    def run(self):  # noqa: C901
        """Thread body: run the turn; stop and errors still end with ``turn_outcome`` then ``turn_finished``."""
        if self.provider is not None:
            self._provider_injected = True     # tests / gui_smoke assign ``worker.provider`` ahead of ``run()``
        else:
            try:
                self.provider = make_provider(self.profile)
            except Exception as e:  # noqa: BLE001 - the message reaches the panel
                log_event("turn_end", reason="provider_setup", error=f"{type(e).__name__}: {e}"[:300])
                self._note_account_problem(e)
                self._emit_account_problem()
                self.turn_outcome.emit("failed", "")
                self.turn_finished.emit(False, str(e))
                return
        try:
            self._refresh_caps()
            self.run_turn()
        except StopRequested:
            # ``user_stop`` and ``turn_end`` were logged by run_turn / its teardown (one turn_end per turn).
            self.turn_outcome.emit("stopped", "")
            self.turn_finished.emit(False, "Stopped.")
        except Exception as e:  # noqa: BLE001 - surfaced to the panel
            import traceback

            FreeCAD.Console.PrintError(f"Oscar: turn failed: {e}\n{traceback.format_exc()}\n")
            self._note_account_problem(e)
            self._emit_account_problem()
            self.turn_outcome.emit("failed", "")
            self.turn_finished.emit(False, f"{type(e).__name__}: {e}")

    def _run_turn(self) -> TurnResult:
        """Kept for the existing tests: run the turn synchronously (raises ``StopRequested`` on a stop)."""
        return self.run_turn()

    def run_turn(self) -> TurnResult:
        """Run the whole turn synchronously and return its result; emits every signal.

        Normal end: teardown, then ``plan_changed`` / ``artifacts_changed`` / ``turn_outcome`` / ``turn_finished``.
        A stop raises ``StopRequested`` and any other failure re-raises, both AFTER the teardown (Project
        ``end_turn``, NOTES.md, ``on_round``, ledger save); ``run()`` then emits ``turn_outcome`` and
        ``turn_finished`` for those paths.
        """
        result: TurnResult | None = None
        outcome = "failed"
        self._exit_reason = ""
        if self.provider is None:       # a synchronous caller that never went through run(): build it here
            self.provider = make_provider(self.profile)
        try:
            self._begin_turn()
            result = self._loop()
            outcome = result.outcome
        except StopRequested:
            outcome = "stopped"
            log_event("user_stop", where="loop")
            raise
        except BaseException as e:
            outcome = "failed"
            self._exit_reason = f"{type(e).__name__}: {e}"[:300]
            raise
        finally:
            self._teardown(outcome, result)
        self._emit_account_problem()
        self.turn_outcome.emit(result.outcome, self._summary_text)
        self.turn_finished.emit(result.ok, result.text)
        return result

    # -- account problems and the request context ---------------------------------------------------------------

    def _note_account_problem(self, error: BaseException) -> None:
        """Remember ``error`` for ``account_problem`` when it is about the OriginShape account (a ``ProviderError`` of
        kind ``"auth"`` / ``"account"``); anything else is not an account problem and is ignored."""
        kind = getattr(error, "kind", "")
        if kind not in ("auth", "account"):
            return
        retry_after = getattr(error, "retry_after", None)
        data = getattr(error, "data", None)
        self._account_problem = {
            "code": str(getattr(error, "code", "") or ""), "kind": kind, "status": int(getattr(error, "status", 0) or 0),
            "message": str(error)[:500],
            "retry_after": float(retry_after) if isinstance(retry_after, (int, float)) else None,
            "data": data if isinstance(data, dict) else {},
            # nothing reached the model yet: the panel may give the typed message back
            "first_request": self._provider_rounds <= 1 and self._tool_round == 0,
        }

    def _emit_account_problem(self) -> None:
        """``account_problem`` once, just before ``turn_outcome`` (queued signals from one thread keep their order, so
        the panel has it when the turn's end arrives)."""
        problem, self._account_problem = self._account_problem, None
        if problem is None:
            return
        try:
            payload = json.dumps(problem)
        except (TypeError, ValueError):   # ``data`` held something JSON cannot carry
            payload = json.dumps({**problem, "data": {}})
        self.account_problem.emit(payload)

    def _session_value(self) -> str:
        return self._session_id or self._chat_id()

    def _set_request_context(self, **extra) -> None:
        """``provider.request_context`` for this turn (and round): a NEW dict every time, never one edited in place (the
        reader thread of an earlier attempt may still hold the previous one). Providers that ignore it are fine."""
        try:
            self.provider.request_context = {"session": self._session_value(), "turn": self._turn_id, **extra}
        except Exception:  # noqa: BLE001 - a provider that refuses the attribute simply sends no context
            pass

    # -- turn start / end ----------------------------------------------------------------------------

    def _begin_turn(self) -> None:
        """Per-turn state: ledger (seeded, carried or throwaway), watchdog, breaker, tool-context state, Project turn."""
        cfg = self.config
        self._turn_id = _new_turn_id()
        self._request = (self._request_text if self._request_text is not None
                         else loopcore.real_user_request(self.history)) or ""
        flow_on = cfg.flow_enforcement
        carried = self._carried
        if flow_on:
            if carried is not None and is_continuation(self._request, carried, attachments=self._attachments):
                ledger, live = carried, True
                ledger.round = 0
            elif (carried is not None and carried.steps and carried.request and is_goahead(self._request)
                  and not carried.has_open_required_steps()):
                # A bare "continue" / "yes" after a FINISHED plan is no new request: the throwaway ledger carries the ORIGINAL
                # request, so an escalation (the model starts building) seeds a plan for it and never for the word "continue".
                ledger, live = Ledger.from_request(carried.request, intent="other"), False
            else:
                ledger = Ledger.from_request(self._request, attachments=self._attachments)
                live = ledger.is_work_turn()
                if live and carried is not None and carried.steps:
                    self._archive(carried)
        else:
            ledger, live = Ledger.from_request(self._request, intent="other"), False
        ledger.round, ledger.round_cap = 0, cfg.max_tool_rounds
        self._ledger, self._live = ledger, live
        self._summary_text = ""
        self._finalized = False
        self._provider_rounds = 0
        self._tool_round = 0
        self._cap = cfg.max_tool_rounds
        self._raises = 0
        self._progress_rounds: list[int] = []
        self._wrapup = False
        self._cutoffs = 0
        self._notes = 0
        self._gate_nudges: dict[str, int] = {}
        self._nudges_total = 0
        self._next_tool_choice: str | None = None
        self._force_reasoning = ""
        self._empty_final_nudged = False
        self._nudged_execution = False
        self._tools_this_turn = 0
        self._geometry_ok = 0
        self._deliver_ok = 0
        self._open_errors: list[str] = []
        self._script_fail = ""
        self._validation: dict | None = None
        self._validated_geo = 0
        self._last_phase = ""
        self._last_stop_reason = ""
        self._call_ordinal = 0
        # every tool call id the conversation holds: a reply that repeats one is renamed (loopcore.unique_call_ids)
        self._call_ids: set[str] = {str(c.id) for m in self.history for c in (m.tool_calls or ()) if c.id}
        self._saved_revision = -1
        self._saved_at = 0.0
        self._plan_revision = -1
        self.init_wiring()
        self._note_prior_build()
        self._state = {"lite": self._lite}
        self._turn_started = self._clock()
        self._act_now.clear()
        self._can_lower_effort, self._slow_host = self._watchdog_traits()
        self._watchdog = Watchdog(cfg.watchdog, self._clock, can_lower_effort=self._can_lower_effort,
                                  slow_host=self._slow_host)
        self._breaker = FailureBreaker(cfg.breaker_trips_max, cfg.stall_rounds)
        try:
            _tools.reset_turn_circuits()
        except Exception:  # noqa: BLE001
            pass
        try:
            # text tool mode: a call to a registered tool that this round does not offer still runs, like a native one
            self.provider.known_tool_names = frozenset(entry.name for entry in _tools.get_tools())
        except Exception:  # noqa: BLE001 - a provider that refuses the attribute simply keeps the request's own tools
            pass
        self._set_request_context()
        try:
            from . import netguard as _netguard

            _netguard.reset_turn()
        except Exception:  # noqa: BLE001
            pass
        self._system = self._build_turn_prompt()
        if not flow_on:
            # Kill-switch path only: the new flow keeps the stored history APPEND-ONLY (the derived request view and the
            # pinned task memory, ``context.py``, already give the model what it needs within its window).
            compact_history(self.history)
        setter = getattr(_turnlog, "set_context", None)
        if callable(setter):
            try:
                setter(turn_id=self._turn_id, chat_id=self._chat_id(), model=getattr(self.profile, "model", ""))
            except Exception:  # noqa: BLE001
                pass
        if self.project is not None:
            self._guard("begin_turn", self.project.begin_turn, self._turn_id, self._request,
                        doc_names=self._doc_names, work=bool(ledger.is_work_turn() and live))
        self._began = True
        log_event("turn_start", turn_id=self._turn_id, intent=ledger.intent, flow=flow_on,
                  profile={"name": self.profile.name, "provider": self.profile.provider, "model": self.profile.model,
                           "base_url": self.profile.base_url},
                  caps_source=getattr(self.caps, "source", "none"), round_cap=self._cap, rounds_cap=self._cap,
                  lite=self._lite, playbooks=list(self._playbooks),
                  prompt_chars=len(self._system), prompt_hash=loopcore.args_hash("system", {"p": self._system}))
        if flow_on and live and ledger.steps:
            self._attach_plan_line()
            self._emit_plan(force=True)         # the plan strip first: nothing is blank while the baseline snapshot runs
            self._take_baseline()

    def _chat_id(self) -> str:
        """The chat id of the bound Project for the telemetry context ("" without one).

        ``Project`` keeps it in its manifest (it has no ``chat_id`` attribute); test stubs may expose the attribute.
        """
        project = self.project
        if project is None:
            return ""
        direct = getattr(project, "chat_id", "")
        if isinstance(direct, str) and direct:
            return direct
        manifest = getattr(project, "manifest", None)
        value = manifest.get("chat_id", "") if isinstance(manifest, dict) else ""
        return value if isinstance(value, str) else ""

    def _watchdog_traits(self) -> tuple[bool, bool]:
        """``(can_lower_effort, slow_host)`` for the watchdog from the model caps / host class."""
        caps = self.caps
        can_lower = caps is None or getattr(caps, "reasoning_control", "none") != "none"
        host = ""
        try:
            from . import models as _models

            host = _models.host_class(self.profile)
        except Exception:  # noqa: BLE001
            host = ""
        slow = host in _SLOW_HOSTS or (caps is not None and getattr(caps, "tool_args_stream", True) is False)
        return can_lower, slow

    def _archive(self, old: Ledger) -> None:
        """A new work request replaces the carried ledger: archive the old one in the Project."""
        if self.project is not None:
            self._guard("archive_ledger", self.project.archive_ledger, self._turn_id)

    def _teardown(self, outcome: str, result: TurnResult | None) -> None:
        """Always runs: Project end_turn, NOTES.md for a work turn, the last ``on_round`` and ledger save; never raises."""
        try:
            if self._began:
                work = bool(self._live and self._ledger.is_work_turn())
                if self.project is not None:
                    self._guard("end_turn", self.project.end_turn, self._turn_id, outcome, rounds=self._provider_rounds)
                    if work and not self._finalized:
                        self._guard("write_notes_md", self.project.write_notes_md, self._ledger.to_dict())
                        writer = getattr(self.project, "write_rebuild_py", None)
                        if callable(writer):
                            self._guard("write_rebuild_py", writer)
                self._persist_round()
                self._persist_ledger(force=True)
                if self.config.flow_enforcement and self._live and self._ledger.steps:
                    self._emit_plan(force=True)
                if self.project is not None:
                    self._emit_artifacts()
                log_event("turn_end", outcome=outcome,
                          reasons=(result.reasons if result else ([self._exit_reason] if self._exit_reason else [])),
                          rounds=self._provider_rounds, tool_rounds=self._tool_round, geometry_ok=self._geometry_ok)
        except Exception as e:  # noqa: BLE001 - teardown must never mask the real exit
            log_event("persist_error", where="teardown", error=f"{type(e).__name__}: {e}"[:200])
        finally:
            clearer = getattr(_turnlog, "clear_context", None)
            if callable(clearer):
                try:
                    clearer()
                except Exception:  # noqa: BLE001
                    pass

    # -- persistence / emission helpers ------------------------------------------------------------------

    def _guard(self, what: str, fn: Callable, *args, **kwargs):
        """Call a persistence function; log (never raise) a failure. Returns its result or ``None``."""
        try:
            return fn(*args, **kwargs)
        except StopRequested:
            raise
        except Exception as e:  # noqa: BLE001
            log_event("persist_error", where=what, error=f"{type(e).__name__}: {e}"[:200])
            return None

    def _persist_round(self) -> None:
        if self._on_round is not None:
            self._guard("on_round", self._on_round, self.history)

    def _persist_ledger(self, force: bool = False) -> None:
        """Save ``ledger.json`` on a new revision, debounced to once per second (``force`` at teardown)."""
        if self.project is None or not self._live or not self._ledger.steps:
            return
        now = time.monotonic()
        if self._ledger.revision == self._saved_revision:
            return
        if not force and now - self._saved_at < 1.0:
            return
        self._saved_revision, self._saved_at = self._ledger.revision, now
        self._guard("save_ledger", self.project.save_ledger, self._ledger.to_dict())

    def _emit_plan(self, force: bool = False) -> None:
        """Emit ``plan_changed`` when the ledger revision moved (or ``force``)."""
        ledger = self._ledger
        if not force and ledger.revision == self._plan_revision:
            return
        self._plan_revision = ledger.revision
        self.plan_changed.emit(ledger.to_json())

    def _emit_artifacts(self) -> None:
        project = self.project
        root = getattr(project, "root", "")
        if not root or not os.path.isdir(root):
            return
        files = []
        for name, kind in (("model.FCStd", "fcstd"), ("NOTES.md", "notes"), ("rebuild.py", "rebuild"),
                           ("provenance.json", "provenance")):
            path = os.path.join(root, name)
            if os.path.isfile(path):
                files.append({"kind": kind, "name": name, "path": path})
        for sub, kind, limit in (("exports", "export", 40), ("snapshots", "snapshot", 40)):
            folder = os.path.join(root, sub)
            if os.path.isdir(folder):
                for name in sorted(os.listdir(folder))[-limit:]:
                    if not name.endswith(".FCBak") and os.path.isfile(os.path.join(folder, name)):
                        files.append({"kind": kind, "name": f"{sub}/{name}", "path": os.path.join(folder, name)})
        self.artifacts_changed.emit(_json({"project_dir": root, "files": files}))

    def _progress(self, activity: str, **extra) -> None:
        """Emit a ``progress`` payload; never raises."""
        try:
            ledger = self._ledger
            cur = ledger.current_step() if self._flow() else None
            counts = ledger.counts() if self._flow() else {"done": 0, "total": 0}
            now = self._clock()
            payload = {"round": self._tool_round + 1, "round_cap": self._cap, "phase": ledger.phase if self._flow() else "",
                       "step_id": cur.id if cur else "", "step_title": cur.title if cur else "",
                       "steps_done": counts["done"], "steps_total": counts["total"],
                       "elapsed_round": round(getattr(self._watchdog, "elapsed", 0.0), 1),
                       "elapsed_turn": round(now - self._turn_started, 1), "reasoning_chars": 0, "text_chars": 0,
                       "tool": "", "tool_chars": 0, "activity": activity, "retry": None, "idle_s": 0.0,
                       "started_at": time.time()}
            payload.update(extra)
            self.progress.emit(_json(payload))
        except Exception:  # noqa: BLE001
            pass

    def _flow(self) -> bool:
        return bool(self.config.flow_enforcement)

    # -- the loop ---------------------------------------------------------------------------------------

    def _loop(self) -> TurnResult:  # noqa: C901 - the loop reads clearer linear
        cfg = self.config
        while True:
            if self._stop:
                raise StopRequested()
            self._wait_if_paused()
            self._deliver_steer()
            try:
                reply = self._provider_round()
            except ContextTooSmall as small:
                text = str(small)
                self.notice.emit(text)
                return self._result(False, text, "failed", [text])
            if reply.error is not None:
                self._note_account_problem(reply.error)
                return self._result(False, f"Provider error: {reply.error}", "failed", [f"Provider error: {reply.error}"])
            if reply.tripped:
                done = self._on_trip(reply)
                if done is not None:
                    return done
                continue
            self._last_stop_reason = reply.stop_reason
            self._log_round(reply)
            if not getattr(self.provider, "vision_enabled", True) and not self._vision_notice_sent:
                self._vision_notice_sent = True
                self.notice.emit("Image input disabled for this model — attached images and screenshots are now "
                                 "described as text instead of sent.")
            dropped = False
            if self._wrapup and reply.calls:
                dropped, reply.calls = True, []        # the wrap-up call may not run tools
                # ... and its tool_use blocks must not be replayed: they would have no results
                reply.blocks = [b for b in reply.blocks
                                if not (isinstance(b, dict) and b.get("type") in _CALL_BLOCK_TYPES)]
            text = reply.text
            if (not reply.calls and not text.strip() and _cut_off(reply.stop_reason) and not self._wrapup
                    and self._cutoffs < cfg.cutoff_retries and self._notes < cfg.recovery_notes_max):
                self._cutoff_retry(reply)
                continue
            if reply.calls:
                done = self._tool_round_step(reply)
                if done is not None:
                    return done
                continue
            done = self._finish_attempt(reply, dropped)
            if done is not None:
                return done

    def _chat_options(self) -> dict:
        """Per-call provider options: reasoning effort, tool choice, output cap (``loopcore.call_chat`` drops unsupported ones)."""
        cfg, caps = self.config, self.caps
        opts: dict = {}
        first = self._provider_rounds == 0
        effort = self._force_reasoning
        controllable = caps is not None and getattr(caps, "reasoning_control", "none") != "none"
        if not effort and controllable and getattr(self.profile, "reasoning", "auto") == "auto":
            effort = cfg.reasoning_plan if first else cfg.reasoning_exec
        if effort:
            opts["reasoning"] = effort
        if controllable and not first and effort != "high" and getattr(self.profile, "reasoning", "auto") != "high":
            opts["max_tokens"] = min(int(getattr(self.profile, "max_tokens", 0) or cfg.exec_max_tokens), cfg.exec_max_tokens)
        choice = "none" if self._wrapup else self._next_tool_choice
        if choice and (caps is None or getattr(caps, "tool_choice_ok", True)):
            if choice not in ("none", "auto") and not self._forced_choice_allowed(opts):
                log_event("tool_choice_dropped", choice=choice, why="reasoning_active")
            else:
                opts["tool_choice"] = choice
        return opts

    def _forced_choice_allowed(self, opts: dict) -> bool:
        """Whether a forced tool (``required`` / a tool name) may ride with this request's reasoning setting.

        Some models (Anthropic thinking) reject a forced tool choice while thinking is on. When the request would
        think, reasoning is switched off for this call (``opts["reasoning"] = "off"``); when the model replays its
        thinking blocks in the open tool loop that is not possible, so the forced choice is dropped instead and the
        nudge text alone has to do the work.
        """
        caps = self.caps
        if caps is None or getattr(caps, "tool_choice_required_with_reasoning", True):
            return True
        explicit = str(getattr(self.profile, "reasoning", "auto") or "auto")
        effort = str(opts.get("reasoning") or ("" if explicit == "auto" else explicit))
        if effort in ("", "off"):
            return True
        if getattr(caps, "reasoning_replay", False):
            return False
        opts["reasoning"] = "off"
        return True

    def _provider_round(self) -> _Reply:
        """One provider call: stream events into a ``_Reply``, feeding the watchdog; a trip or ``act_now`` cancels the call."""
        cfg = self.config
        reply = _Reply()
        view, specs = self._request_for_round()      # may raise ContextTooSmall before anything is counted or sent
        opts = self._chat_options()
        self._next_tool_choice = None
        self._act_now.clear()
        flow_on = self._flow()
        watchdog = self._watchdog
        watchdog.start_round(plan_round=self._provider_rounds == 0)
        self._provider_rounds += 1
        self._set_request_context(round=self._provider_rounds)
        started = self._clock()
        text_parts: list[str] = []
        tool_name, tool_chars, text_chars = "", 0, 0
        gen = loopcore.call_chat(self.provider, self._system, view, specs, self._should_stop, **opts)
        try:
            for kind, payload in gen:
                if kind == "reset":
                    text_parts, reply.reasoning_chars, reply.calls, reply.blocks = [], 0, [], []
                    text_chars = tool_chars = 0
                    self.round_reset.emit(str(payload))
                elif kind == "text":
                    text_parts.append(payload)
                    text_chars += len(payload)
                    self.text_delta.emit(payload)
                elif kind == "reasoning":
                    reply.reasoning_chars += len(payload)
                    self.reasoning_delta.emit(payload)
                elif kind == "assistant_blocks":
                    reply.blocks = payload
                elif kind == "stop_reason":
                    reply.stop_reason = payload or ""
                elif kind == "stop_info":
                    reply.stop_info = payload if isinstance(payload, dict) else None
                elif kind == "tool_calls":
                    reply.calls = payload
                elif kind == "usage":
                    reply.usage = payload
                elif kind == "error":
                    reply.error = payload if isinstance(payload, Exception) else RuntimeError(str(payload))
                elif kind == "tool_progress" and isinstance(payload, dict):
                    tool_name, tool_chars = str(payload.get("name") or ""), int(payload.get("chars") or 0)
                if kind == "tick" and isinstance(payload, dict):
                    phase = str(payload.get("phase") or "thinking")
                    self._progress({"connecting": "thinking", "waiting": "thinking"}.get(phase, phase),
                                   reasoning_chars=int(payload.get("reasoning_chars") or reply.reasoning_chars),
                                   text_chars=int(payload.get("text_chars") or text_chars),
                                   tool=str(payload.get("tool") or tool_name), tool_chars=int(payload.get("tool_chars") or tool_chars),
                                   retry=payload.get("retry"), idle_s=float(payload.get("idle_s") or 0.0))
                if flow_on:
                    for note in watchdog.due_notices():
                        log_event("watchdog_skipped", reason="no_reasoning_control")
                        self.notice.emit(note)
                    trip = watchdog.feed(kind, payload if kind in ("reasoning", "tick", "reset") else None)
                    if trip is None and self._act_now.is_set():
                        trip = watchdog.trip("user")
                    if trip:
                        reply.tripped = trip
                        break
                elif self._act_now.is_set():
                    reply.tripped = "user"
                    break
        finally:
            close = getattr(gen, "close", None)
            if callable(close):
                close()
        reply.text = "".join(text_parts)
        reply.wall_s = round(self._clock() - started, 2)
        if reply.calls and not reply.tripped:
            renamed = loopcore.unique_call_ids(reply.calls, self._call_ids, reply.blocks)
            if renamed:
                log_event("call_ids_renamed", count=renamed)
        self._note_usage(reply)
        return reply

    def _log_round(self, reply: _Reply) -> None:
        # Sizes only (never content): enough to answer "why did the turn end with nothing?" from the log.
        log_event("round", n=self._provider_rounds, tool_round=self._tool_round, phase=self._ledger.phase,
                  stop_reason=reply.stop_reason, text_chars=len(reply.text), reasoning_chars=reply.reasoning_chars,
                  tool_calls=[c.name for c in reply.calls], usage=reply.usage, wall_s=reply.wall_s)

    # -- recovery paths ------------------------------------------------------------------------------

    def _add_note(self, text: str, tag: str, *, answer: str = "", blocks: Sequence[dict] = ()) -> None:
        """Append the model's (possibly empty) reply as an assistant message, then an app note as a synthetic user message."""
        content = answer if answer.strip() else loopcore.NO_TEXT_PLACEHOLDER
        kept = list(blocks) if any(b.get("type") == "text" for b in blocks) else []
        self.history.append(Message(role="assistant", content=content, provider_blocks=kept))
        self.history.append(loopcore.make_note(text, tag))
        self._notes += 1
        self._persist_round()

    def _cutoff_retry(self, reply: _Reply) -> None:
        # The whole output budget went into thinking and the reply carries nothing to act on. Ending the turn here
        # would look like a successful empty answer; tell the model what happened and let it act.
        self._cutoffs += 1
        log_event("cutoff_retry", attempt=self._cutoffs, reasoning_chars=reply.reasoning_chars)
        self.notice.emit("The model's thinking hit the output-token limit before it acted — asking it to act now.")
        self.history.append(Message(role="assistant", content=_CUTOFF_PLACEHOLDER))
        self.history.append(loopcore.make_note(_CUTOFF_NOTE, "cutoff"))
        self._notes += 1

    def _next_step_label(self) -> str:
        cur = self._ledger.current_step()
        return f'{cur.id} "{cur.title}"' if cur is not None else "the next open step"

    def _on_trip(self, reply: _Reply) -> TurnResult | None:
        """The watchdog (or the user's Act now) cut a reply short: note, retry the round, or end ``needs_help`` on trip 3."""
        wd = self._watchdog
        reason = reply.tripped
        log_event("watchdog_trip", reason=reason, reasoning_chars=wd.reasoning_chars, elapsed=round(wd.elapsed, 1),
                  round=self._provider_rounds)
        self.round_reset.emit("watchdog")
        if wd.exhausted or self._notes >= self.config.recovery_notes_max:
            if self._notes >= self.config.recovery_notes_max:
                log_event("recovery_cap", where="watchdog")
            return self._needs_help("the model kept thinking instead of acting and the app stopped it "
                                    f"{wd.trips} times")
        self.notice.emit("The model was thinking too long; asking it to act now.")
        note = loopcore.WATCHDOG_NOTE.format(chars=wd.reasoning_chars, secs=int(wd.elapsed), now=self._next_step_label())
        self.history.append(Message(role="assistant", content=_WATCHDOG_PLACEHOLDER))
        self.history.append(loopcore.make_note(note, "watchdog"))
        self._notes += 1
        self._force_reasoning = "off"
        if wd.trips >= 2:
            self._next_tool_choice = "required"
        return None

    # -- tool rounds -----------------------------------------------------------------------------------

    def _tool_round_step(self, reply: _Reply) -> TurnResult | None:
        """Record the assistant's calls, run them, append the results message (+governor / breaker / budget lines)."""
        self._force_reasoning = ""
        self.history.append(Message(role="assistant", content=reply.text, tool_calls=reply.calls,
                                    provider_blocks=reply.blocks))
        self._tools_this_turn += len(reply.calls)
        terminal_before = self._terminal_count()
        results = self._run_tools(reply.calls)
        info: _Round = self._round
        self._tool_round += 1
        n = self._tool_round
        ledger = self._ledger
        ledger.round, ledger.round_cap = n, self._cap
        lines: list[str] = []
        self._open_errors = [f"{name}: {line}" for name, line in info.errors.items()]
        try:
            self._round_notes(info, n, terminal_before, lines)
        except Exception as e:  # noqa: BLE001 - bookkeeping must never leave the tool calls without their results
            log_event("loop_error", where="round_notes", error=f"{type(e).__name__}: {e}"[:200])
        if n >= self._cap and not self._wrapup:
            self._wrapup = True
            lines.append(_BUDGET_NOTE)
        self.history.append(Message(role="user", content="\n".join(lines), tool_results=results))
        self._persist_round()
        self._persist_ledger()
        if info.fatal is not None:
            return self._needs_help(info.fatal.detail or info.fatal.rule)
        return None

    def _round_notes(self, info: _Round, n: int, terminal_before: int, lines: list[str]) -> None:
        """Round bookkeeping after the calls ran: watchdog decay, adaptive cap, stall check, governor and breaker lines.

        Appends the lines for the results message to ``lines`` and may set ``info.fatal``. Only the flow-enforcement
        parts depend on the kill switch; breaker verdicts of a round are always reported.
        """
        cfg, ledger = self.config, self._ledger
        if self._flow():
            if self._terminal_count() > terminal_before:
                self._progress_rounds.append(n)
            if info.failed:
                self._watchdog.note_tool_round_failed()
            else:
                self._watchdog.note_tool_round_ok()
            if (n >= self._cap - 2 and self._cap < cfg.max_rounds_hard and self._raises < 3
                    and any(r >= n - 7 for r in self._progress_rounds)):
                self._cap = min(cfg.max_rounds_hard, self._cap + cfg.raise_step)
                self._raises += 1
                ledger.round_cap = self._cap
                log_event("round_cap_raised", cap=self._cap)
            cur = ledger.current_step()
            stall = self._breaker.end_round(
                mutating_ok=info.mutating_ok, ledger_revision=ledger.revision, track_stall=ledger.has_open_required_steps(),
                next_step=f'{cur.id} "{cur.title}"' if cur is not None else "",
                hint_tools=list(cur.hint_tools) if cur is not None else [])
            if stall is not None:
                info.verdicts.append(stall)
            if ledger.steps:
                lines.append(format_governor(ledger, round_no=n, cap=self._cap,
                                             health=health_label(self._state.get("last_health")),
                                             phase_changed=ledger.phase != self._last_phase,
                                             warn_fraction=cfg.warn_fraction))
                self._last_phase = ledger.phase
        for verdict in info.verdicts:
            if self._notes >= cfg.recovery_notes_max and not verdict.fatal:
                log_event("recovery_cap", where="breaker")
                continue
            self._notes += 1
            log_event("breaker_trip", rule=verdict.rule)
            lines.append(verdict.message)
            if verdict.fatal and info.fatal is None:
                info.fatal = verdict

    def _terminal_count(self) -> int:
        return sum(1 for s in self._ledger.steps if s.required and s.status in TERMINAL)

    def _title_for(self, call) -> str:
        args = call.arguments if isinstance(call.arguments, dict) else {}
        title = str(args.get("step_title") or "").strip()
        if title:
            return title[:80]
        try:
            return _tools.derive_title(call.name, args)
        except Exception:  # noqa: BLE001
            return call.name

    def _make_ctx(self, call_id: str = "") -> ToolContext:
        flow_on = self._flow()
        self._state["flow_enforcement"] = flow_on     # run_tool reads it: the kill switch also reaches its health / ledger hooks
        return ToolContext(project=self.project, ledger=self._ledger if flow_on else None, turn_id=self._turn_id,
                           round=self._tool_round + 1, call_id=call_id, phase=self._ledger.phase if flow_on else "",
                           intent=self._ledger.intent if flow_on else "", state=self._state,
                           should_stop=self._should_stop)

    def _meta(self, name: str) -> tuple[bool, bool, str]:
        """``(mutating, geometry, category)`` of a tool from the registry (unknown tools: all inert)."""
        entry = _tools.tool_meta(name) if hasattr(_tools, "tool_meta") else None
        if entry is None:
            return False, False, tool_category(name)
        return bool(entry.mutating), bool(entry.geometry), str(entry.category)

    def _run_tools(self, tool_calls) -> list[ToolResult]:
        """Run one round's calls. Every call gets exactly one result — even when the user stops or pauses mid-round —
        because a tool call without its result makes every later request invalid. Fills ``self._round``."""
        results: list[ToolResult] = []
        info = self._round = _Round()
        flow_on = self._flow()
        failed_mutation = ""
        try:
            for call in tool_calls:
                if self._stop:
                    raise StopRequested()
                self._wait_if_paused()
                if self._stop:
                    raise StopRequested()
                self._call_ordinal += 1
                mutating, geometry, category = self._meta(call.name)
                title = self._title_for(call)
                summary = _summarize_arguments(call.name, call.arguments)
                started_at = time.time()
                self.tool_started.emit(call.name)
                self._log_tool_start(call)
                self.step_started.emit(_json({
                    "call_id": call.id, "name": call.name, "title": title, "args_summary": summary,
                    "round": self._tool_round + 1, "ordinal": self._call_ordinal, "mutating": mutating,
                    "geometry": geometry, "phase": self._ledger.phase if flow_on else "", "started_at": started_at}))
                self._progress("running_tool", tool=call.name)
                problem = loopcore.args_problem(call)
                skipped = bad_args = False
                outcome: ToolOutcome | None = None
                image = None
                if problem:
                    bad_args = True
                    result_text, is_error = loopcore.bad_arguments_message(call, self._last_stop_reason), True
                    log_event("bad_arguments", name=call.name, reason=problem[:200])
                elif failed_mutation and mutating:
                    skipped = True
                    result_text, is_error = (
                        f"Not executed: an earlier call in this round failed ({failed_mutation}). "
                        "Fix that first, then re-issue this call."), True
                else:
                    if self._stop:
                        raise StopRequested()
                    self._ensure_baseline(call.name)
                    result_text, image, is_error, outcome = self._execute(call, title)
                handler_ok = (not is_error) if outcome is None else bool(outcome.handler_ok)
                if outcome is not None and handler_ok:
                    if outcome.geometry:
                        self._geometry_ok += 1
                    if category == "deliver":
                        self._deliver_ok += 1
                    if outcome.mutating:
                        info.mutating_ok += 1
                if outcome is not None and not handler_ok and outcome.mutating and not failed_mutation:
                    failed_mutation = f"{call.name}: {loopcore.first_line(result_text, 120)}"
                if not skipped:
                    if is_error:
                        info.errors[call.name] = loopcore.first_line(result_text, 160)
                        info.failed += 1
                    else:
                        info.errors.pop(call.name, None)
                    if call.name.startswith("execute_code") and handler_ok:
                        # the script ran: a FAIL line it printed counts even when the health check also flagged the result
                        self._script_fail = _script_failure(result_text)
                if flow_on and not skipped:
                    verdict = self._breaker.record(call, ok=not is_error, handler_ok=handler_ok, text=result_text,
                                                   bad_args=bad_args)
                    if verdict is not None:
                        info.verdicts.append(verdict)
                results.append(ToolResult(tool_call_id=call.id, name=call.name, content=result_text, is_error=is_error,
                                          image_b64=image or ""))
                self._emit_step_finished(call, title, result_text, is_error, skipped, outcome, started_at)
                short = result_text if len(result_text) <= 400 else result_text[:400] + "…"
                self.tool_finished.emit(f"{call.name}({summary})", not is_error, short)
                self._after_call()
                if outcome is not None and outcome.step_index >= 0 and self.project is not None:
                    self._emit_artifacts()          # after plan_changed
        except BaseException as exc:
            done = {r.tool_call_id for r in results}
            reason = ("the user stopped the turn" if isinstance(exc, StopRequested)
                      else "the app hit an internal error")
            for call in tool_calls:
                if call.id not in done:
                    results.append(ToolResult(
                        tool_call_id=call.id, name=call.name, is_error=True,
                        content=f"Not executed: {reason} before this call ran."))
            self.history.append(Message(role="user", tool_results=results))
            self._persist_round()
            raise
        _keep_last_screenshot(results)
        return results

    @staticmethod
    def _log_tool_start(call) -> None:
        """``tool_start{name, arg_names, arg_chars, script_hash}``: names and sizes only, never the code."""
        args = call.arguments if isinstance(call.arguments, dict) else {}
        code = args.get("code")
        log_event("tool_start", name=call.name, arg_names=sorted(str(k) for k in args)[:12],
                  arg_chars=len(_json(args)), script_hash=loopcore.args_hash("code", {"code": code}) if isinstance(code, str) else "")

    def _execute(self, call, title: str):
        """Run one call through ``_tools.run_tool`` inside its ``ToolContext``; returns ``(text, image, is_error, outcome)``."""
        flow_on = self._flow()
        ctx = self._make_ctx(call.id)
        try:
            with use_context(ctx):
                text, image, is_error = _tools.run_tool(call.name, call.arguments)
        except StopRequested:
            raise
        except Exception as e:  # noqa: BLE001 - a crashing tool is a model-visible failure, not the end of the turn
            log_event("tool", name=call.name, ok=False, error=f"{type(e).__name__}: {e}"[:200])
            text, image, is_error = f"{type(e).__name__}: {e}", None, True
        outcome = ctx.outcome
        if outcome is None:
            mutating, geometry, _category = self._meta(call.name)
            outcome = ToolOutcome(
                name=call.name, call_id=call.id, title=title, ok=not is_error, handler_ok=not is_error, text=text,
                mutating=mutating, geometry=geometry, image_b64=image or "",
                args=dict(call.arguments) if isinstance(call.arguments, dict) else {})
            if flow_on and self._ledger is not None:
                try:
                    self._ledger.on_tool_result(outcome)
                except Exception as e:  # noqa: BLE001
                    log_event("ledger_error", hook="ledger", error=f"{type(e).__name__}: {e}"[:200])
        return text, image, is_error, outcome

    def _emit_step_finished(self, call, title: str, text: str, is_error: bool, skipped: bool,
                            outcome: ToolOutcome | None, started_at: float) -> None:
        args = call.arguments if isinstance(call.arguments, dict) else {}
        is_script = call.name in ("execute_code", "execute_code_headless")
        code = str(args.get("code") or "")[:12000] if is_script else ""
        compact = "" if is_script else _json({k: v for k, v in args.items() if k != "_raw"})[:4000]
        o = outcome
        self.step_finished.emit(_json({
            "call_id": call.id, "name": call.name, "title": o.title if o is not None and o.title else title,
            "ok": not is_error, "skipped": skipped, "text": text[:12000],
            "dur_s": o.dur_s if o is not None else round(time.time() - started_at, 2),
            "round": self._tool_round + 1, "ordinal": self._call_ordinal, "code": code, "args_json": compact,
            "delta": (o.delta if o is not None else {}) or {"added": [], "removed": [], "changed": []},
            "health": o.health if o is not None else None, "step_index": o.step_index if o is not None else -1,
            "snapshot_path": o.snapshot_path if o is not None else "", "snapshot_note": o.snapshot_note if o is not None else "",
            "checkpoint": o.checkpoint if o is not None else "", "script_path": o.script_path if o is not None else ""}))

    def _after_call(self) -> None:
        """Plan bookkeeping after a call: ``plan_changed`` on a new revision; a throwaway ledger that escalated becomes live."""
        if not self._flow():
            return
        ledger = self._ledger
        if not self._live and ledger.steps:
            self._live = True            # an `other` / question turn made its first geometry call: the turn is a work turn now
            if self._carried is not None and self._carried.steps:
                self._archive(self._carried)
            log_event("escalated", intent=ledger.intent)
        if self._live:
            self._emit_plan()

    # -- steer ----------------------------------------------------------------------------------------

    def _deliver_steer(self) -> None:
        """Append queued steer messages to the trailing user message (or a new synthetic one), at the start of a round."""
        batch = self.drain_unsent()
        if not batch:
            return
        text = "".join(f"{_STEER_PREFIX}{(item['note'] + ' ') if item['note'] else ''}{item['text']}" for item in batch)
        # what the user said mid-turn rides inside a results message, which the request view may drop with its round:
        # the pinned memory keeps it (``decisions``) so a steer is never lost to context management
        self._steered.extend(item["text"].strip() for item in batch if item["text"].strip())
        images = [img for item in batch for img in item["images"]]
        last = self.history[-1] if self.history else None
        if last is not None and last.role == "user":
            last.content = (last.content or "") + text
            last.images = list(last.images) + images
        else:
            self.history.append(Message(role="user", content=text.lstrip("\n"), images=images, synthetic=True, tag="steer"))
        log_event("steer_delivered", messages=len(batch))

    # -- finishing --------------------------------------------------------------------------------------

    def _empty_reply_message(self, stop_reason: str) -> str:
        if _cut_off(stop_reason):
            # the provider may have learned a lower cap than the profile's
            effective = getattr(self.provider, "_max_tokens", None)
            cap = effective() if callable(effective) else getattr(self.profile, "max_tokens", 0)
            times = f" {self._cutoffs + 1} times in a row" if self._cutoffs else ""
            return (
                f"The model used its whole output budget ({cap} tokens) on "
                f"thinking{times} and produced no answer or tool call. Try a "
                "shorter, more specific request, or pick a model with a larger "
                "output budget or less reasoning in ⚙ settings."
            )
        if stop_reason in ("tool_calls", "tool_use"):
            # The provider drops calls whose name never arrived.
            return (
                "The model tried to call a tool but the call arrived incomplete "
                "(no tool name). Send the request again."
            )
        return (
            "The model returned an empty reply (stop reason: "
            f"{stop_reason or 'unknown'}). Send the request again or rephrase it."
        )

    def _turn_request(self) -> str:
        """What the user typed for this turn (captured once per turn; notes and attachments never reach it)."""
        captured = getattr(self, "_request", None)
        if captured:
            return captured
        return self._request_text or loopcore.real_user_request(self.history)

    def _result(self, ok: bool, text: str, outcome: str, reasons: Sequence[str] = ()) -> TurnResult:
        carried = self._carried
        ledger = self._ledger.to_dict() if (self._flow() and self._live) else (carried.to_dict() if carried is not None else None)
        return TurnResult(ok=ok, text=text, outcome=outcome, reasons=list(reasons), rounds=self._provider_rounds,
                          tool_rounds=self._tool_round, geometry_ok=self._geometry_ok, ledger=ledger,
                          validation=self._validation, project_dir=getattr(self.project, "root", "") or "")

    def _needs_help(self, explanation: str) -> TurnResult:
        text = compose_needs_help(explanation)
        return self._result(False, text, "needs_help", [explanation])

    def _record_answer(self, text: str, blocks: Sequence[dict]) -> None:
        """The answer is part of the conversation: without it the next turn would not know what Oscar said."""
        if text.strip():
            kept = list(blocks) if any(b.get("type") == "text" for b in blocks) else []
            self.history.append(Message(role="assistant", content=text, provider_blocks=kept))
        self._persist_round()

    def _finish(self, text: str, blocks: list[dict], stop_reason: str, tool_calls, rounds: int = 0) -> TurnResult:
        """Plain (non-work) end of a turn: record the answer and return the result. Kept for compatibility."""
        reply = _Reply(text=text, blocks=blocks, stop_reason=stop_reason, calls=list(tool_calls or []))
        return self._plain_finish(reply, bool(tool_calls))

    def _plain_finish(self, reply: _Reply, dropped: bool) -> TurnResult:
        text, blocks, stop = reply.text, reply.blocks, reply.stop_reason
        if stop == "refusal" and not text.strip():
            msg = ("The model declined this request (safety refusal). Rephrase it, or switch to another model in "
                   "⚙ settings.")
            return self._result(False, msg, "failed", [msg])
        if dropped:
            # The budget is spent and the model still wanted tools: nothing more runs this turn.
            text = (text.strip() + "\n\n" if text.strip() else "") + (
                f"(Stopped after {self._tool_round} tool rounds — ask me to continue to keep going.)")
            blocks = []
        elif _cut_off(stop) and text.strip():
            text += "\n\n(The reply was cut off at the output-token limit.)"
            blocks = []
        elif not text.strip():
            # An empty reply must never pass as a successful answer: the panel would render nothing at all.
            msg = self._empty_reply_message(stop)
            return self._result(False, msg, "failed", [msg])
        self._record_answer(text, blocks)
        return self._result(True, text, "answered")

    def _finish_attempt(self, reply: _Reply, dropped: bool) -> TurnResult | None:  # noqa: C901
        """A reply without tool calls: steer first, then the finish gate (work turns) or a plain finish.

        Returns the turn's result, or ``None`` when the loop must go on (a steer message arrived, or a nudge was injected).
        """
        cfg = self.config
        text, blocks = reply.text, reply.blocks
        if not self._steer_q.empty():
            # A message sent while the final answer streamed is delivered, not lost; not a gate round.
            self.history.append(Message(role="assistant", content=text if text.strip() else loopcore.NO_TEXT_PLACEHOLDER,
                                        provider_blocks=list(blocks) if any(b.get("type") == "text" for b in blocks) else []))
            self._persist_round()
            return None
        if not text.strip() and _cut_off(reply.stop_reason):
            msg = self._empty_reply_message(reply.stop_reason)
            return self._result(False, msg, "failed", [msg])
        ledger = self._ledger
        if not self._flow():
            if (text.strip() and not dropped and not self._wrapup and not self._nudged_execution
                    and needs_execution_nudge(self._turn_request(), self._tools_this_turn)):
                # (An empty reply described no plan to execute: ``_plain_finish`` reports it instead.)
                self._nudged_execution = True
                log_event("execution_nudge", text_chars=len(text))
                self.notice.emit("Oscar answered without doing any CAD work — asking it to execute its plan step by step.")
                self._add_note(_EXECUTION_NOTE, "gate", answer=text, blocks=blocks)
                return None
            return self._plain_finish(reply, dropped)
        work = self._live and ledger.is_work_turn()
        if not work:
            return self._plain_finish(reply, dropped)
        return self._work_finish(reply, dropped, cfg)

    def _turn_state(self, validation: dict | None) -> TurnState:
        ledger = self._ledger
        return TurnState(is_work_turn=True, geometry_ok=self._geometry_ok, deliver_ok=self._deliver_ok,
                         open_errors=list(self._open_errors), last_script_fail=self._script_fail, validation=validation,
                         last_geometry_step=ledger.last_geometry_step, gate_nudges=dict(self._gate_nudges),
                         nudges_total=self._nudges_total)

    def _validate(self) -> dict | None:
        """Run the validator (``validator(doc, ctx)``): the injected one, else ``tools_verify.run_validation``.

        ``None`` without a validator (``False`` was injected, or ``tools_verify`` is absent), without a document the turn
        changed (the real validator never guesses "the active document"), on an exception, or when it returns ``None``.
        """
        validator = self._validator if self._validator is not False else None
        if validator is None and self._validator is None:
            validator = default_validator()
        if validator is None or not self.config.auto_validate:
            return None
        ctx = self._make_ctx()
        started = time.monotonic()
        self._progress("validating")
        try:
            with use_context(ctx):
                result = validator(self._state.get("last_doc") or None, ctx)
        except StopRequested:
            raise
        except Exception as e:  # noqa: BLE001 - a broken validator must not kill the turn
            log_event("validation_run", ok=None, error=f"{type(e).__name__}: {e}"[:200])
            return None
        value = _validation_dict(result)
        log_event("validation_run", ok=None if value is None else bool(value.get("ok")),
                  fail_rows=len((value or {}).get("fail_rows") or []), seconds=round(time.monotonic() - started, 2))
        return value

    def _work_finish(self, reply: _Reply, dropped: bool, cfg: AgentConfig) -> TurnResult | None:  # noqa: C901
        ledger = self._ledger
        text, blocks, stop = reply.text, reply.blocks, reply.stop_reason
        ledger.confirm_evidenced_steps()
        pending = [s.id for s in ledger.pending_required()]
        if self._wrapup and pending:
            return self._capped(text, blocks, pending)
        if ledger.last_geometry_step > self._validated_geo:
            # I3: the finish gate always judges from the APP's own run_validation of the last geometry, never from a
            # model-invoked validate_model (which may be health-only) standing in for it. No validator / no document = no claim.
            self._validated_geo = ledger.last_geometry_step
            validation = self._validate()
            if validation is not None:
                self._validation = validation
            ledger.apply_validation(validation)
        verdict = finish_gate(ledger, self._turn_state(self._validation), cfg, reply_nonempty=bool(text.strip()),
                              offered={s.name for s in self._round_specs()})
        if not verdict.ok and not verdict.exhausted and self._notes < cfg.recovery_notes_max:
            for code in verdict.codes:
                self._gate_nudges[code] = self._gate_nudges.get(code, 0) + 1
            self._nudges_total += 1
            log_event("gate_nudge", codes=verdict.codes)
            self.round_reset.emit("gate")
            self.notice.emit("Oscar stopped with unfinished work; asking it to continue.")
            self._add_note(verdict.nudge, "gate", answer=text, blocks=blocks)
            self._next_tool_choice = verdict.tool_choice
            return None
        if not verdict.ok:
            if self._notes >= cfg.recovery_notes_max:
                log_event("recovery_cap", where="gate")
            return self._unverified_or_failed(verdict, text, blocks, stop)
        if verdict.outcome_hint == "needs_input":
            self._record_answer(text, blocks)
            self._finalize("needs_input", finalize_project=False)
            return self._result(True, text, "needs_input", [])
        if not text.strip():
            if not self._empty_final_nudged and self._notes < cfg.recovery_notes_max:
                self._empty_final_nudged = True
                self.round_reset.emit("gate")
                self._add_note("[The work is finished, but your reply was empty. Reply now with a short summary: what "
                               "was built or changed (object names, key dimensions) and anything that needs the user's "
                               "decision.]", "wrapup", answer="", blocks=blocks)
                return None
            msg = self._empty_reply_message(stop)
            return self._result(False, msg, "failed", [msg])
        # The gate passed.
        validation = self._validation
        ok_validation = isinstance(validation, dict) and bool(validation.get("ok")) and not ledger.claimed_only_steps()
        outcome = "validated" if ok_validation else "answered"
        ledger.mark_delivered(self.project is not None)
        summary = self._finalize(outcome)
        model_text = text + ("\n\n(The reply was cut off at the output-token limit.)" if _cut_off(stop) else "")
        self._record_answer(model_text, blocks if model_text == text else [])
        if outcome == "validated" and summary:
            return self._result(True, f"{model_text}\n\n{summary}", outcome)
        return self._result(True, model_text, outcome)

    def _finalize(self, outcome: str, *, finalize_project: bool = True) -> str:
        """``Project.finalize`` (autosave, NOTES.md, rebuild.py, Build Summary); returns the summary markdown ("" without one)."""
        self._summary_text = ""
        if self.project is None or not finalize_project:
            return ""
        built = self._guard("finalize", self.project.finalize, ledger=self._ledger.to_dict(), outcome=outcome,
                            validation=self._validation)
        self._finalized = built is not None
        summary = getattr(built, "markdown", built) if built is not None else ""
        self._summary_text = summary if isinstance(summary, str) else ""
        return self._summary_text

    def _capped(self, text: str, blocks, pending: Sequence[str]) -> TurnResult:
        summary = self._finalize("capped")
        self._record_answer(text, blocks)
        return self._result(True, compose_capped(text, self._tool_round, pending, summary), "capped",
                            [f"{len(pending)} required steps still open: {', '.join(pending)}"])

    def _unverified_or_failed(self, verdict, text: str, blocks, stop: str) -> TurnResult:
        produced = self._geometry_ok + self._deliver_ok
        if produced > 0:
            summary = self._finalize("unverified")
            self._record_answer(text, blocks)
            return self._result(True, compose_unverified(verdict.reasons, text, summary), "unverified", verdict.reasons)
        empty = self._empty_reply_message(stop) if not text.strip() else ""
        return self._result(False, compose_failed(verdict.reasons, empty), "failed", verdict.reasons)

