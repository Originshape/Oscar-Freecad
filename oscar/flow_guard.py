"""Runaway guards of the agent loop: the reasoning watchdog and the failure breaker (private module of ``oscar.flow``).

PURE PYTHON (no FreeCAD, no Qt). Both classes are state machines driven by the agent loop with an injectable clock:

* ``Watchdog``: watches ONE provider round for runaway reasoning, silence and over-long rounds.
* ``FailureBreaker``: watches the stream of tool results for repeated failures, truncated replies and a stalled
  loop of harmless read-only calls.

Everything public here is re-exported by ``oscar.flow``.
"""

import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .loopcore import args_hash, first_line

__all__ = ["WatchdogConfig", "Watchdog", "BreakerVerdict", "FailureBreaker", "WATCHDOG_REASONS"]

WATCHDOG_REASONS = ("reasoning_chars", "silent", "round_seconds", "user")

# Models without a reasoning knob only get notices (never a trip): seconds into the round -> text.
_NOTICE_AT = ((120.0, "The model has been thinking for {n} s; press Act now to stop it."),
              (300.0, "The model has been thinking for {n} s; press Act now to stop it."))


@dataclass
class WatchdogConfig:
    """Thresholds of the reasoning watchdog."""

    reasoning_chars_plan: int = 12000    # plan round = the FIRST provider round of the turn (same definition as the reasoning level `auto`)
    reasoning_chars_exec: int = 6000     # every other round
    silent_s: float = 90.0               # no reasoning/text/tool_progress event for this long
    round_s: float = 240.0               # wall seconds per provider round
    connect_grace_s: float = 120.0       # before the first non-tick event of a round, `silent` only counts after this long
    slow_host_factor: float = 3.5        # silent_s and round_s are multiplied by this for slow hosts (ollama / lmstudio / custom, tool args not streamed)
    trips_max: int = 3                   # CONSECUTIVE trips per turn (the counter is reset after 3 consecutive successful tool rounds)


class Watchdog:
    """Per-round runaway detector for the provider stream.

    Usage: ``start_round(plan_round=...)`` before each provider call, ``feed(kind, payload)`` for every event of
    the call; a non-``None`` return is a trip reason (``reasoning_chars`` | ``silent`` | ``round_seconds`` |
    ``user``) and the caller must cancel the call. A trip increments ``trips`` (CONSECUTIVE trips: three
    successful tool rounds in a row, reported through ``note_tool_round_ok``, reset it), and the same round
    never trips twice.

    * ``can_lower_effort`` (``caps.reasoning_control != "none"``): False means a retry at lower effort cannot change
      the model, so the reasoning-character and round-seconds rules never trip; only ``silent`` can, and
      ``due_notices()`` yields one notice at 120 s and one at 300 s so the user can press Act now.
    * ``slow_host``: ``silent_s`` and ``round_s`` are multiplied by ``cfg.slow_host_factor`` (local servers that buffer
      tool-call arguments produce no event for a whole 40-line script).
    * ``silent`` is measured from the LAST non-tick event once one arrived, before that only after
      ``connect_grace_s`` counted from the start of the current ATTEMPT (a ``reset`` event, i.e. a transport retry
      or a dropped stream, starts a new attempt), and never while the latest tick says ``connecting``, ``waiting``
      or ``retrying`` (the provider is reconnecting or backing off; the round-seconds rule still bounds the call).
    """

    def __init__(self, cfg: WatchdogConfig | None = None, clock: Callable[[], float] = time.monotonic, *,
                 can_lower_effort: bool = True, slow_host: bool = False):
        """``clock`` is injectable (tests); see the class docstring for ``can_lower_effort`` / ``slow_host``."""
        self.cfg = cfg or WatchdogConfig()
        self.clock = clock
        self.can_lower_effort = bool(can_lower_effort)
        self.slow_host = bool(slow_host)
        self.trips = 0               # consecutive-trip counter, survives rounds
        self.reasoning_chars = 0     # current round
        self.last_reason = ""
        self._ok_streak = 0
        self._notices_done: set[int] = set()
        self._reset_round(False)

    # -- round lifecycle ------------------------------------------------------------------------
    def _reset_round(self, plan_round: bool) -> None:
        self._plan_round = bool(plan_round)
        self._round_start = self.clock()
        self._attempt_start = self._round_start
        self._first_event: float | None = None
        self._last_event: float | None = None
        self._tick_phase = ""
        self._tripped = False
        self._notices_done = set()
        self.reasoning_chars = 0

    def start_round(self, *, plan_round: bool) -> None:
        """Begin watching a new provider call (``plan_round``: the first provider round of the turn)."""
        self._reset_round(plan_round)

    @property
    def exhausted(self) -> bool:
        """True once ``cfg.trips_max`` consecutive trips happened: the turn must end ``needs_help``."""
        return self.trips >= self.cfg.trips_max

    @property
    def elapsed(self) -> float:
        """Seconds since the current round started."""
        return max(0.0, self.clock() - self._round_start)

    # -- events -------------------------------------------------------------------------------------
    def feed(self, kind: str, payload=None) -> str | None:
        """Account one provider event; returns a trip reason or ``None``.

        ``reasoning`` adds ``len(payload)`` and counts as progress; every other non-``tick`` event (text,
        tool_progress, tool_calls, usage, stop_reason, ...) also counts as progress; ``tick`` only records the
        provider's phase and runs ``check()``; ``reset`` voids the attempt (counters restart).
        """
        now = self.clock()
        if kind == "tick":
            if isinstance(payload, dict):
                self._tick_phase = str(payload.get("phase") or "")
            return self.check()
        if kind == "reset":
            self.reasoning_chars = 0
            self._first_event = None
            self._last_event = None
            self._attempt_start = now        # a new attempt: it gets its own connect grace
        else:
            if self._first_event is None:
                self._first_event = now
            self._last_event = now
            if kind == "reasoning" and payload:
                self.reasoning_chars += len(payload)
        return self.check()

    def check(self) -> str | None:
        """Evaluate the rules now (the loop calls it on every event and tick); returns a trip reason or ``None``."""
        if self._tripped:
            return None
        now = self.clock()
        scale = self.cfg.slow_host_factor if self.slow_host else 1.0
        reason = None
        if self.can_lower_effort:
            limit = self.cfg.reasoning_chars_plan if self._plan_round else self.cfg.reasoning_chars_exec
            if self.reasoning_chars > limit:
                reason = "reasoning_chars"
            elif now - self._round_start > self.cfg.round_s * scale:
                reason = "round_seconds"
        if reason is None and self._silent(now, scale):
            reason = "silent"
        if reason is not None:
            self._trip(reason)
        return reason

    def _silent(self, now: float, scale: float) -> bool:
        if self._tick_phase in ("connecting", "waiting", "retrying"):
            return False
        if self._last_event is None:
            return now - self._attempt_start > self.cfg.connect_grace_s
        return now - self._last_event > self.cfg.silent_s * scale

    def _trip(self, reason: str) -> None:
        self._tripped = True
        self.trips += 1
        self._ok_streak = 0
        self.last_reason = reason

    def trip(self, reason: str = "user") -> str:
        """Count an externally requested trip (the user pressed Act now) exactly like an automatic one."""
        if not self._tripped:
            self._trip(reason)
        return reason

    def due_notices(self) -> list[str]:
        """Notices that became due this round (once each); only models without a reasoning control get them."""
        if self.can_lower_effort:
            return []
        out = []
        elapsed = self.elapsed
        for index, (at, template) in enumerate(_NOTICE_AT):
            if elapsed >= at and index not in self._notices_done:
                self._notices_done.add(index)
                out.append(template.format(n=int(elapsed)))
        return out

    # -- decay ------------------------------------------------------------------------------------------
    def note_tool_round_ok(self) -> None:
        """A tool round finished without a failure; after three in a row the consecutive-trip counter resets."""
        self._ok_streak += 1
        if self._ok_streak >= 3:
            self.trips = 0
            self._ok_streak = 0

    def note_tool_round_failed(self) -> None:
        """A tool round had a failure: the run of successful rounds starts over."""
        self._ok_streak = 0


@dataclass
class BreakerVerdict:
    """A tripped breaker rule: what to tell the model and whether the turn must end."""

    rule: str       # "repeat" | "streak" | "truncation" | "stall"
    message: str    # appended to the round's results message
    fatal: bool     # True when trips_max (or the second stall) was reached
    detail: str = ""  # human explanation for the final "Oscar is stuck: ..." text


_MESSAGES = {
    "repeat": ("[breaker] You repeated the same failing call. Do not retry it unchanged: call get_objects, remove "
               "leftovers of the failed attempt, then resubmit a smaller, idempotent step (or change the arguments)."),
    "streak": ("[breaker] Three calls in a row failed. Stop and re-plan: read the last error, call get_objects, "
               "and fix the cause before building more."),
    "truncation": ("[breaker] Two replies in a row were cut off. Send much smaller calls: one feature per call, "
                   "under 40 lines per script."),
}

_NUMBER_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight",
                 9: "Nine", 10: "Ten"}


class FailureBreaker:
    """Stops the loop from burning rounds on failures it cannot fix by itself.

    Rules: ``repeat`` = the same ``(name, args)`` call has ``handler_ok == False`` twice in a row;
    ``streak`` = three consecutive calls with ``ok == False`` of any kind (a health failure counts here and ONLY
    here); ``truncation`` = two consecutive calls with ``bad_args``; ``stall`` = ``stall_rounds`` consecutive
    tool rounds with no successful mutating call and no change of ``ledger.revision`` (a model looping on
    read-only calls never fails, so the failure rules never see it). A success resets ``streak`` / ``repeat``.

    Each trip of the first three rules increments ``trips``; three consecutive trips are fatal. ``trips`` resets
    after three consecutive tool rounds without a failure. The first stall injects a note, the second is fatal.
    """

    def __init__(self, trips_max: int = 3, stall_rounds: int = 6):
        """``trips_max`` consecutive trips are fatal; ``stall_rounds`` idle tool rounds make a stall."""
        self.trips_max = trips_max
        self.stall_rounds = stall_rounds
        self.trips = 0
        self._streak = 0
        self._bad_args = 0
        self._prev: tuple[str, bool] | None = None   # (args key, handler_ok) of the previous executed call
        self._round_calls = 0
        self._round_fails = 0
        self._good_rounds = 0
        self._stall = 0
        self._stalls_fired = 0
        self._stall_revision: int | None = None
        self.last_failure = ""

    def _trip(self, rule: str, detail: str) -> BreakerVerdict:
        self.trips += 1
        self._good_rounds = 0
        return BreakerVerdict(rule, _MESSAGES[rule], self.trips >= self.trips_max, detail)

    def record(self, call, *, ok: bool, handler_ok: bool, text: str, bad_args: bool = False) -> BreakerVerdict | None:
        """Account one executed (or answered-with-an-error) call; returns a verdict when a rule trips.

        ``ok`` is what the model saw (False also for health failures); ``handler_ok`` is whether the handler
        itself succeeded; ``bad_args`` marks a call whose arguments could not be decoded (never executed).
        """
        name = getattr(call, "name", "") or "?"
        key = args_hash(name, getattr(call, "arguments", None) or {})
        self._round_calls += 1
        if not ok or not handler_ok or bad_args:
            self._round_fails += 1
            self.last_failure = f"{name}: {first_line(text, 120)}".strip()
        verdict = None
        if bad_args:
            self._bad_args += 1
            self._streak += 1
            self._prev = None
            if self._bad_args >= 2:
                self._bad_args = 0
                self._streak = 0
                verdict = self._trip("truncation", "two replies in a row were cut off before their tool arguments were complete")
            elif self._streak >= 3:
                self._streak = 0
                verdict = self._trip("streak", f"three calls in a row failed (last: {self.last_failure})")
            return verdict
        self._bad_args = 0
        if not handler_ok:
            repeated = self._prev is not None and self._prev == (key, False)
            self._prev = (key, False)
            self._streak += 1
            if repeated:
                self._prev = None
                self._streak = 0
                return self._trip("repeat", f"the same call kept failing ({self.last_failure})")
        else:
            self._prev = (key, True)
            if ok:
                self._streak = 0
            else:
                self._streak += 1
        if self._streak >= 3:
            self._streak = 0
            return self._trip("streak", f"three calls in a row failed (last: {self.last_failure})")
        return None

    def end_round(self, *, mutating_ok: int, ledger_revision: int, next_step: str = "",
                  hint_tools: Sequence[str] = (), track_stall: bool = True) -> BreakerVerdict | None:
        """Called once per tool round: round bookkeeping, trip decay and the stall detector.

        ``mutating_ok`` is the number of successful mutating calls of the round, ``ledger_revision`` the ledger's
        revision after it, ``next_step`` the text after "Next step: " for the stall note (for example
        ``S7 "Air vent slots"``), ``hint_tools`` the tools to suggest. ``track_stall=False`` (a turn with no open
        plan step, i.e. plain chat) disables the stall detector: a read-only exploration has nothing to stall on.
        """
        if self._round_calls and self._round_fails == 0:
            self._good_rounds += 1
            if self._good_rounds >= 3:
                self.trips = 0
                self._good_rounds = 0
        elif self._round_fails:
            self._good_rounds = 0
        self._round_calls = 0
        self._round_fails = 0
        if self._stall_revision is None:
            self._stall_revision = ledger_revision
        if not track_stall:
            self._stall = 0
            self._stall_revision = ledger_revision
            return None
        if mutating_ok > 0 or ledger_revision != self._stall_revision:
            self._stall = 0
            self._stall_revision = ledger_revision
            return None
        self._stall += 1
        if self._stall < self.stall_rounds:
            return None
        self._stall = 0
        self._stalls_fired += 1
        words = _NUMBER_WORDS.get(self.stall_rounds, str(self.stall_rounds))
        tools = ", ".join(hint_tools) if hint_tools else "a building tool"
        step = f" Next step: {next_step}." if next_step else ""
        message = (f"[stall] {words} tool rounds passed without any change to the model or the plan.{step} "
                   f"Call {tools} now.")
        return BreakerVerdict("stall", message, self._stalls_fired >= 2,
                              f"{self.stall_rounds} tool rounds passed without any change to the model or the plan")
