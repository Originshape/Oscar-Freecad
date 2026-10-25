"""oscar.flow_guard: the reasoning Watchdog and the FailureBreaker (pure Python, injected clock)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.flow import FailureBreaker, Watchdog, WatchdogConfig  # noqa: E402
from oscar.providers import ToolCall  # noqa: E402


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class WatchdogTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()

    def dog(self, **kw):
        return Watchdog(WatchdogConfig(), self.clock, **kw)

    def test_defaults(self):
        cfg = WatchdogConfig()
        self.assertEqual((cfg.reasoning_chars_plan, cfg.reasoning_chars_exec), (12000, 6000))
        self.assertEqual((cfg.silent_s, cfg.round_s, cfg.connect_grace_s, cfg.slow_host_factor, cfg.trips_max),
                         (90.0, 240.0, 120.0, 3.5, 3))

    def test_exec_round_trips_above_6000_reasoning_characters(self):
        dog = self.dog()
        dog.start_round(plan_round=False)
        self.assertIsNone(dog.feed("reasoning", "x" * 6000))
        self.assertEqual(dog.feed("reasoning", "x"), "reasoning_chars")
        self.assertEqual((dog.trips, dog.reasoning_chars, dog.last_reason), (1, 6001, "reasoning_chars"))

    def test_the_plan_round_gets_the_larger_budget(self):
        dog = self.dog()
        dog.start_round(plan_round=True)
        self.assertIsNone(dog.feed("reasoning", "x" * 12000))
        self.assertEqual(dog.feed("reasoning", "x"), "reasoning_chars")

    def test_a_round_trips_once(self):
        dog = self.dog()
        dog.start_round(plan_round=False)
        self.assertEqual(dog.feed("reasoning", "x" * 7000), "reasoning_chars")
        self.assertIsNone(dog.feed("reasoning", "x" * 7000))
        self.assertEqual(dog.trips, 1)

    def test_a_new_round_resets_the_counters_but_not_the_trips(self):
        dog = self.dog()
        dog.start_round(plan_round=False)
        dog.feed("reasoning", "x" * 7000)
        dog.start_round(plan_round=False)
        self.assertEqual((dog.reasoning_chars, dog.trips), (0, 1))
        self.assertIsNone(dog.feed("reasoning", "x" * 100))

    def test_reset_event_voids_the_attempt(self):
        dog = self.dog()
        dog.start_round(plan_round=False)
        dog.feed("reasoning", "x" * 5000)
        dog.feed("reset", "retry")
        self.assertEqual(dog.reasoning_chars, 0)
        self.assertIsNone(dog.feed("reasoning", "x" * 5000))

    def test_round_seconds(self):
        dog = self.dog()
        dog.start_round(plan_round=False)
        dog.feed("text", "a")
        self.clock.advance(239)
        self.assertIsNone(dog.feed("text", "b"))
        self.clock.advance(2)
        self.assertEqual(dog.feed("text", "c"), "round_seconds")

    def test_silent_counts_from_the_last_non_tick_event(self):
        dog = self.dog()
        dog.start_round(plan_round=False)
        dog.feed("text", "a")
        self.clock.advance(80)
        self.assertIsNone(dog.feed("tick", {"phase": "thinking"}))        # ticks are liveness, not progress
        self.clock.advance(11)
        self.assertEqual(dog.feed("tick", {"phase": "thinking"}), "silent")

    def test_reasoning_and_tool_progress_count_as_progress(self):
        dog = self.dog()
        dog.start_round(plan_round=False)
        dog.feed("text", "a")
        for kind, payload in (("reasoning", "r"), ("tool_progress", {"name": "x"}), ("text", "t")):
            self.clock.advance(60)
            self.assertIsNone(dog.feed(kind, payload))

    def test_before_the_first_event_silence_only_counts_after_the_connect_grace(self):
        dog = self.dog()
        dog.start_round(plan_round=False)
        self.clock.advance(119)
        self.assertIsNone(dog.feed("tick", {"phase": "thinking"}))
        self.clock.advance(2)
        self.assertEqual(dog.feed("tick", {"phase": "thinking"}), "silent")

    def test_connecting_and_waiting_ticks_disable_the_silent_rule(self):
        for phase in ("connecting", "waiting"):
            dog = self.dog()
            dog.start_round(plan_round=False)
            dog.feed("text", "a")
            self.clock.advance(200)
            self.assertIsNone(dog.feed("tick", {"phase": phase}), phase)

    def test_slow_hosts_get_3_5_times_the_time(self):
        dog = self.dog(slow_host=True)
        dog.start_round(plan_round=False)
        dog.feed("text", "a")
        self.clock.advance(300)
        self.assertIsNone(dog.feed("tick", {"phase": "thinking"}))        # 90 s normally, 315 s here
        self.clock.advance(20)
        self.assertEqual(dog.feed("tick", {"phase": "thinking"}), "silent")
        slow = Watchdog(WatchdogConfig(), self.clock, slow_host=True)
        slow.start_round(plan_round=False)
        slow.feed("text", "a")
        for _ in range(8):
            self.clock.advance(100)
            self.assertIsNone(slow.feed("text", "b"))                     # 800 s < 840 s
        self.clock.advance(100)
        self.assertEqual(slow.feed("text", "c"), "round_seconds")

    def test_models_without_a_reasoning_control_never_trip_on_characters_or_round_time(self):
        dog = self.dog(can_lower_effort=False)
        dog.start_round(plan_round=False)
        self.assertIsNone(dog.feed("reasoning", "x" * 50000))
        self.clock.advance(1000)
        self.assertIsNone(dog.feed("reasoning", "x"))
        self.assertEqual(dog.trips, 0)

    def test_models_without_a_reasoning_control_get_two_notices(self):
        dog = self.dog(can_lower_effort=False)
        dog.start_round(plan_round=False)
        self.assertEqual(dog.due_notices(), [])
        self.clock.advance(125)
        first = dog.due_notices()
        self.assertEqual(len(first), 1)
        self.assertIn("press Act now", first[0])
        self.assertEqual(dog.due_notices(), [])
        self.clock.advance(180)
        self.assertEqual(len(dog.due_notices()), 1)
        self.assertEqual(dog.due_notices(), [])

    def test_controllable_models_get_no_notices(self):
        dog = self.dog()
        dog.start_round(plan_round=False)
        self.clock.advance(400)
        self.assertEqual(dog.due_notices(), [])

    def test_silent_still_trips_without_a_reasoning_control(self):
        dog = self.dog(can_lower_effort=False)
        dog.start_round(plan_round=False)
        dog.feed("text", "a")
        self.clock.advance(100)
        self.assertEqual(dog.feed("tick", {"phase": "thinking"}), "silent")

    def test_exhausted_after_three_consecutive_trips(self):
        dog = self.dog()
        for i in range(3):
            self.assertFalse(dog.exhausted)
            dog.start_round(plan_round=False)
            dog.feed("reasoning", "x" * 7000)
        self.assertTrue(dog.exhausted)

    def test_three_good_tool_rounds_forgive_the_trips(self):
        dog = self.dog()
        for _ in range(2):
            dog.start_round(plan_round=False)
            dog.feed("reasoning", "x" * 7000)
        self.assertEqual(dog.trips, 2)
        dog.note_tool_round_ok()
        dog.note_tool_round_ok()
        self.assertEqual(dog.trips, 2)
        dog.note_tool_round_ok()
        self.assertEqual(dog.trips, 0)

    def test_a_failed_round_restarts_the_good_streak(self):
        dog = self.dog()
        dog.start_round(plan_round=False)
        dog.feed("reasoning", "x" * 7000)
        dog.note_tool_round_ok()
        dog.note_tool_round_ok()
        dog.note_tool_round_failed()
        dog.note_tool_round_ok()
        dog.note_tool_round_ok()
        self.assertEqual(dog.trips, 1)
        dog.note_tool_round_ok()
        self.assertEqual(dog.trips, 0)

    def test_a_trip_resets_the_good_streak(self):
        dog = self.dog()
        dog.note_tool_round_ok()
        dog.note_tool_round_ok()
        dog.start_round(plan_round=False)
        dog.feed("reasoning", "x" * 7000)
        dog.note_tool_round_ok()
        self.assertEqual(dog.trips, 1)

    def test_user_act_now_counts_like_a_trip(self):
        dog = self.dog()
        dog.start_round(plan_round=False)
        self.assertEqual(dog.trip("user"), "user")
        self.assertEqual((dog.trips, dog.last_reason), (1, "user"))
        dog.trip("user")                              # the same round never counts twice
        self.assertEqual(dog.trips, 1)

    def test_default_clock_is_monotonic(self):
        dog = Watchdog()
        dog.start_round(plan_round=True)
        self.assertGreaterEqual(dog.elapsed, 0.0)
        self.assertIsNone(dog.feed("text", "hi"))


def call(name="create_object", **arguments):
    return ToolCall(id="c", name=name, arguments=arguments)


class WatchdogRetryTest(unittest.TestCase):
    """A transport retry or a dropped stream is the provider reconnecting, never a silent model (review defect W1)."""

    def setUp(self):
        self.clock = Clock()
        self.dog = Watchdog(WatchdogConfig(), self.clock)

    def test_a_reset_after_a_long_stream_starts_a_new_attempt_with_its_own_connect_grace(self):
        self.dog.start_round(plan_round=False)
        for _ in range(125):                      # 125 s of healthy text (more than the 120 s connect grace)
            self.clock.advance(1)
            self.assertIsNone(self.dog.feed("text", "x"))
        self.assertIsNone(self.dog.feed("reset", "stream_dropped"))
        self.clock.advance(1)
        self.assertIsNone(self.dog.feed("tick", {"phase": "thinking"}))     # not "silent" one second after the drop
        self.assertEqual(self.dog.trips, 0)
        self.clock.advance(100)                   # 101 s into the new attempt, 227 s into the round
        self.assertIsNone(self.dog.feed("tick", {"phase": "thinking"}))     # still inside the new attempt's 120 s grace

    def test_a_new_attempt_that_stays_mute_does_trip_after_its_grace(self):
        self.dog.start_round(plan_round=False)
        self.clock.advance(100)
        self.dog.feed("reset", "retry")
        self.clock.advance(121)
        self.assertEqual(self.dog.feed("tick", {"phase": "thinking"}), "silent")

    def test_backoff_ticks_never_count_as_silence(self):
        self.dog.start_round(plan_round=True)
        trip = None
        for _ in range(130):                      # two Retry-After waits: nothing arrives, the provider says "retrying"
            self.clock.advance(1)
            trip = self.dog.feed("tick", {"phase": "retrying", "retry": [3, 5]}) or trip
        self.assertIsNone(trip)
        self.assertEqual(self.dog.trips, 0)

    def test_the_round_clock_still_bounds_a_retry_loop(self):
        self.dog.start_round(plan_round=True)
        self.clock.advance(241)
        self.assertEqual(self.dog.feed("tick", {"phase": "retrying", "retry": [4, 5]}), "round_seconds")


class FailureBreakerTest(unittest.TestCase):
    def test_a_repeated_failing_call_trips_repeat(self):
        b = FailureBreaker()
        self.assertIsNone(b.record(call(x=1), ok=False, handler_ok=False, text="Boom"))
        verdict = b.record(call(x=1), ok=False, handler_ok=False, text="Boom")
        self.assertEqual(verdict.rule, "repeat")
        self.assertFalse(verdict.fatal)
        self.assertEqual(verdict.message, "[breaker] You repeated the same failing call. Do not retry it unchanged: call "
                                          "get_objects, remove leftovers of the failed attempt, then resubmit a smaller, "
                                          "idempotent step (or change the arguments).")

    def test_changed_arguments_are_not_a_repeat(self):
        b = FailureBreaker()
        b.record(call(x=1), ok=False, handler_ok=False, text="Boom")
        self.assertIsNone(b.record(call(x=2), ok=False, handler_ok=False, text="Boom"))

    def test_a_success_in_between_resets_repeat(self):
        b = FailureBreaker()
        b.record(call(x=1), ok=False, handler_ok=False, text="Boom")
        b.record(call("get_objects"), ok=True, handler_ok=True, text="[]")
        self.assertIsNone(b.record(call(x=1), ok=False, handler_ok=False, text="Boom"))

    def test_three_failures_in_a_row_trip_streak(self):
        b = FailureBreaker()
        self.assertIsNone(b.record(call(x=1), ok=False, handler_ok=False, text="a"))
        self.assertIsNone(b.record(call(x=2), ok=False, handler_ok=False, text="b"))
        verdict = b.record(call(x=3), ok=False, handler_ok=False, text="c")
        self.assertEqual(verdict.rule, "streak")
        self.assertEqual(verdict.message, "[breaker] Three calls in a row failed. Stop and re-plan: read the last error, call "
                                          "get_objects, and fix the cause before building more.")

    def test_health_failures_count_only_toward_the_streak(self):
        b = FailureBreaker()
        for i in range(2):
            self.assertIsNone(b.record(call(x=1), ok=False, handler_ok=True, text="NOT OK"))   # same call twice: no `repeat`
        self.assertEqual(b.record(call(x=1), ok=False, handler_ok=True, text="NOT OK").rule, "streak")

    def test_a_success_resets_the_streak(self):
        b = FailureBreaker()
        b.record(call(x=1), ok=False, handler_ok=False, text="a")
        b.record(call(x=2), ok=False, handler_ok=False, text="b")
        b.record(call(x=3), ok=True, handler_ok=True, text="fine")
        self.assertIsNone(b.record(call(x=4), ok=False, handler_ok=False, text="c"))

    def test_two_cut_off_replies_in_a_row_trip_truncation(self):
        b = FailureBreaker()
        self.assertIsNone(b.record(call("execute_code"), ok=False, handler_ok=False, text="not valid JSON", bad_args=True))
        verdict = b.record(call("execute_code"), ok=False, handler_ok=False, text="not valid JSON", bad_args=True)
        self.assertEqual(verdict.rule, "truncation")
        self.assertEqual(verdict.message, "[breaker] Two replies in a row were cut off. Send much smaller calls: one feature "
                                          "per call, under 40 lines per script.")

    def test_a_good_call_between_resets_truncation(self):
        b = FailureBreaker()
        b.record(call("execute_code"), ok=False, handler_ok=False, text="x", bad_args=True)
        b.record(call("get_objects"), ok=True, handler_ok=True, text="[]")
        self.assertIsNone(b.record(call("execute_code"), ok=False, handler_ok=False, text="x", bad_args=True))

    def test_three_trips_in_a_row_are_fatal(self):
        b = FailureBreaker()
        verdicts = []
        for round_no in range(3):
            b.record(call(x=round_no), ok=False, handler_ok=False, text="a")
            v = b.record(call(x=round_no), ok=False, handler_ok=False, text="a")
            verdicts.append(v)
            b.end_round(mutating_ok=0, ledger_revision=round_no)
        self.assertEqual([v.fatal for v in verdicts], [False, False, True])
        self.assertIn("kept failing", verdicts[-1].detail)

    def test_three_clean_rounds_forgive_the_trips(self):
        b = FailureBreaker()
        b.record(call(x=1), ok=False, handler_ok=False, text="a")
        b.record(call(x=1), ok=False, handler_ok=False, text="a")
        b.end_round(mutating_ok=0, ledger_revision=0)
        self.assertEqual(b.trips, 1)
        for i in range(3):
            b.record(call("get_objects", i=i), ok=True, handler_ok=True, text="[]")
            b.end_round(mutating_ok=1, ledger_revision=i)
        self.assertEqual(b.trips, 0)

    def test_stall_note_after_six_idle_rounds_then_fatal(self):
        b = FailureBreaker()
        out = None
        for i in range(5):
            b.record(call("get_objects"), ok=True, handler_ok=True, text="[]")
            self.assertIsNone(b.end_round(mutating_ok=0, ledger_revision=3))
        b.record(call("get_objects"), ok=True, handler_ok=True, text="[]")
        out = b.end_round(mutating_ok=0, ledger_revision=3, next_step='S7 "Air vent slots"', hint_tools=["execute_code", "edit_object"])
        self.assertEqual(out.rule, "stall")
        self.assertFalse(out.fatal)
        self.assertEqual(out.message, '[stall] Six tool rounds passed without any change to the model or the plan. '
                                      'Next step: S7 "Air vent slots". Call execute_code, edit_object now.')
        for i in range(5):
            self.assertIsNone(b.end_round(mutating_ok=0, ledger_revision=3))
        second = b.end_round(mutating_ok=0, ledger_revision=3)
        self.assertEqual((second.rule, second.fatal), ("stall", True))

    def test_a_mutating_success_or_a_plan_change_resets_the_stall(self):
        b = FailureBreaker()
        for _ in range(5):
            b.end_round(mutating_ok=0, ledger_revision=1)
        self.assertIsNone(b.end_round(mutating_ok=1, ledger_revision=1))
        for _ in range(5):
            self.assertIsNone(b.end_round(mutating_ok=0, ledger_revision=1))
        self.assertIsNone(b.end_round(mutating_ok=0, ledger_revision=2))          # the ledger moved
        for _ in range(5):
            self.assertIsNone(b.end_round(mutating_ok=0, ledger_revision=2))
        self.assertIsNotNone(b.end_round(mutating_ok=0, ledger_revision=2))

    def test_stall_detector_can_be_switched_off_for_plain_chat(self):
        b = FailureBreaker()
        for _ in range(30):
            self.assertIsNone(b.end_round(mutating_ok=0, ledger_revision=0, track_stall=False))

    def test_stall_note_without_a_known_next_step(self):
        b = FailureBreaker()
        for _ in range(5):
            b.end_round(mutating_ok=0, ledger_revision=0)
        out = b.end_round(mutating_ok=0, ledger_revision=0)
        self.assertEqual(out.message, "[stall] Six tool rounds passed without any change to the model or the plan. "
                                      "Call a building tool now.")

    def test_custom_limits(self):
        b = FailureBreaker(trips_max=1, stall_rounds=2)
        b.record(call(x=1), ok=False, handler_ok=False, text="a")
        self.assertTrue(b.record(call(x=1), ok=False, handler_ok=False, text="a").fatal)
        c = FailureBreaker(stall_rounds=2)
        c.end_round(mutating_ok=0, ledger_revision=0)
        self.assertIn("Two tool rounds", c.end_round(mutating_ok=0, ledger_revision=0).message)

    def test_last_failure_is_remembered_for_the_explanation(self):
        b = FailureBreaker()
        b.record(call("execute_code", code="x"), ok=False, handler_ok=False, text="NameError: x\ntraceback")
        self.assertEqual(b.last_failure, "execute_code: NameError: x")


if __name__ == "__main__":
    unittest.main()
