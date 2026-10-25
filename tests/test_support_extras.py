"""The extensions of the shared test support: ``SignalRecorder`` helpers, ``legacy_provider``, ``temp_project``,
``fake_validation`` with ``gate_errors``, and the isolation sandbox. (The core behaviour stays pinned by test_stage0.)"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from support.fakes import FakeValidation, fake_validation, legacy_provider, temp_project, tiny_png_snapshot  # noqa: E402
from support.scripted_llm import ScriptedProvider, call  # noqa: E402
from support.signal_recorder import SignalRecorder  # noqa: E402


class FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in self.slots:
            slot(*args)


class FakeWorker:
    def __init__(self):
        for name in ("text_delta", "turn_outcome", "turn_finished", "step_finished", "plan_changed", "tool_finished"):
            setattr(self, name, FakeSignal())


class SignalRecorderTest(unittest.TestCase):
    def setUp(self):
        self.worker = FakeWorker()
        self.rec = SignalRecorder(self.worker)

    def test_records_in_emission_order_with_the_declared_arity(self):
        self.worker.text_delta.emit("a")
        self.worker.tool_finished.emit("x", "y", "z")
        self.worker.turn_finished.emit(True, "done")
        self.assertEqual(self.rec.names(), ["text_delta", "tool_finished", "turn_finished"])
        self.assertEqual(self.rec.of("turn_finished"), [(True, "done")])

    def test_unknown_signals_of_a_worker_are_ignored(self):
        self.assertEqual(self.rec.of("flow_paused"), [])       # the fake worker has no such signal: no error

    def test_json_of_decodes_the_first_argument(self):
        self.worker.plan_changed.emit(json.dumps({"steps": [1, 2]}))
        self.assertEqual(self.rec.json_of("plan_changed"), [{"steps": [1, 2]}])

    def test_count_and_index_of(self):
        for _ in range(3):
            self.worker.step_finished.emit("{}")
        self.worker.turn_finished.emit(True, "x")
        self.assertEqual((self.rec.count("step_finished"), self.rec.index_of("turn_finished")), (3, 3))
        self.assertEqual(self.rec.index_of("nope"), -1)
        self.assertEqual(self.rec.index_of("step_finished", start=1), 1)

    def test_ordered_compares_first_emissions(self):
        self.worker.turn_outcome.emit("validated", "{}")
        self.worker.turn_finished.emit(True, "x")
        self.worker.turn_outcome.emit("late", "{}")
        self.assertTrue(self.rec.ordered("turn_outcome", "turn_finished"))
        self.assertFalse(self.rec.ordered("turn_finished", "turn_outcome"))
        self.assertFalse(self.rec.ordered("turn_outcome", "missing"))
        self.rec.assert_ordered("turn_outcome", "turn_finished")

    def test_assert_ordered_failure_shows_the_emissions(self):
        self.worker.turn_finished.emit(True, "x")
        self.worker.turn_outcome.emit("validated", "{}")
        with self.assertRaisesRegex(AssertionError, r"turn_finished.*turn_outcome"):
            self.rec.assert_ordered("turn_outcome", "turn_finished")

    def test_text_concatenates_the_deltas(self):
        self.worker.text_delta.emit("Hel")
        self.worker.text_delta.emit("lo")
        self.assertEqual(self.rec.text(), "Hello")


class FakesTest(unittest.TestCase):
    def test_fake_validation_gate_errors_follow_ok(self):
        good = fake_validation()
        self.assertEqual((good.ok, good.gate_errors, good.fail_rows), (True, [], []))
        bad = fake_validation(ok=False)
        self.assertFalse(bad.ok)
        self.assertEqual(bad.gate_errors[0]["code"], "FRAGMENTS")
        self.assertFalse(bad.health["valid"])

    def test_fake_validation_with_fail_rows(self):
        v = fake_validation(fail_rows=("interference.shell_lid",))
        self.assertEqual((v.ok, v.fit_ran, v.fit_rows), (False, True, ["interference.shell_lid"]))

    def test_fake_validation_defaults_are_independent_per_instance(self):
        a, b = FakeValidation(), FakeValidation()
        a.gate_errors.append({"code": "X"})
        a.ran.append("fit")
        self.assertEqual((b.gate_errors, b.ran), ([], ["health"]))

    def test_legacy_provider_converts_the_old_tuples(self):
        prov = legacy_provider([("text", "hi"), ("tools", [call("get_objects", "c1")]), ("cutoff", "thinking")])
        self.assertIsInstance(prov, ScriptedProvider)
        self.assertEqual(prov.remaining, 3)
        evs = list(prov.chat("s", [], []))
        self.assertEqual([k for k, _ in evs], ["text", "stop_reason", "done"])
        evs = list(prov.chat("s", [], []))
        self.assertEqual(evs[-1][0], "tool_calls")
        evs = list(prov.chat("s", [], []))
        self.assertEqual([k for k, _ in evs][:2], ["reasoning", "stop_reason"])
        with self.assertRaises(AssertionError):
            list(prov.chat("s", [], []))

    def test_tiny_png_is_a_valid_png(self):
        path = os.path.join(isolation.fresh_dir("png_"), "x.png")
        self.assertEqual(tiny_png_snapshot("Doc", path, 480, 360), (True, ""))
        with open(path, "rb") as f:
            self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")


class TempProjectTest(unittest.TestCase):
    def test_a_real_project_in_the_sandbox(self):
        project = temp_project(chat_id="abc123", title="Pi case")
        self.assertTrue(os.path.realpath(project.root).startswith(os.path.realpath(isolation.sandbox_root())))
        self.assertTrue(os.path.isfile(os.path.join(project.root, "project.json")))

    def test_two_projects_never_share_a_folder(self):
        self.assertNotEqual(temp_project().root, temp_project().root)

    def test_deferred_creation(self):
        project = temp_project(create=False)
        self.assertFalse(os.path.exists(os.path.join(project.root, "project.json")))

    def test_uses_the_fake_snapshotter(self):
        project = temp_project()
        self.assertIs(project._snapshot_fn, tiny_png_snapshot)


class IsolationTest(unittest.TestCase):
    def test_every_oscar_directory_points_into_the_sandbox_or_was_pinned_by_the_caller(self):
        root = os.path.realpath(isolation.sandbox_root())
        for var in ("OSCAR_AI_SETTINGS", "OSCAR_PROJECTS_DIR", "OSCAR_LOG_DIR",
                    "OSCAR_CACHE_DIR", "OSCAR_DOWNLOAD_DIR"):
            value = os.environ[var]
            self.assertTrue(var in isolation.PINNED or os.path.realpath(value).startswith(root), f"{var}={value}")

    def test_home_is_never_overridden(self):
        self.assertNotIn(isolation.sandbox_root(), os.environ.get("HOME", ""))

    def test_isolate_process_is_idempotent(self):
        self.assertEqual(isolation.isolate_process(), isolation.isolate_process())

    def test_fresh_dirs_are_unique_and_inside_the_sandbox(self):
        a, b = isolation.fresh_dir(), isolation.fresh_dir("x_")
        self.assertNotEqual(a, b)
        self.assertTrue(os.path.realpath(a).startswith(os.path.realpath(isolation.sandbox_root())))


if __name__ == "__main__":
    unittest.main()
