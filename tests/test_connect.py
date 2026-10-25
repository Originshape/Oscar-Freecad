"""oscar.connect: connect mode's protocol client, event batch and loop, against the fake OriginShape workspace.

Covered: jobs are parsed (and bad ones refused); the batch keeps order, posts the newest reply last, caps a post at
50 and requeues a failed one; every connect call (hello, next, events with the Stop flag, raw uploads, finish, a
deleted task, a refused upload); the loop: a whole turn (events, previews, exports, the reply), polling only while
Oscar is idle (with the hello keep-alive while busy), Stop from the browser, a turn FreeCAD refuses, a task deleted
mid-turn, a refused upload, signed out, and a server that is down and comes back. Attachments: jobs carry them (and
the names of earlier ones in the history), file names are made safe, a download is checked against its size and
checksum (and renewed after a 401), the loop downloads them before the turn (in order, numbering clashing names),
reports a file it can't get and goes on, answers a files-only message whose files all failed without starting a
turn, and removes old downloads. The turn report: plan bookkeeping is not shown, untitled steps get a readable title,
a failed check is a design issue listed in plain words, other failures say whether a tool failed or something was
declined, an approval is one step that changes, and the feature tree goes along after geometry changes and at the
end (an older server that refuses it costs nothing). The preview: a GLB with one node per part, named by its FreeCAD
object, uploaded as ``glb`` (parts without triangles are left out). Models sent from the browser (section 12.2): only
an Oscar that can open them says so in ``hello``; an ``open`` job is parsed (a bad one refused), downloaded and
checked, opened, and reported, including a download that fails and a FreeCAD that can't open the file.
Pure python (stdlib + requests); the keystore runs in memory.
"""
import hashlib
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import connect  # noqa: E402
from oscar.auth import account, discovery, keystore, session  # noqa: E402
from oscar.auth.transport import AuthUnavailable, TokenSet  # noqa: E402
from support.fake_workspace import FakeWorkspace  # noqa: E402

FAST = {"FLUSH_S": 0.01, "KEEPALIVE_S": 0.05, "PREVIEW_INTERVAL_S": 0.0, "HEARTBEAT_S": 0.05,
        "BACKOFF_S": (0.01, 0.01, 0.01), "SIGNED_OUT_RETRY_S": 0.02, "BLOCKED_RETRY_S": 0.02}


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeTurn:
    """A workspace turn as FreeCAD would run it, scripted: ``steps`` progress titles, then the reply, one preview,
    and ``exports`` when it ends ok. ``hold`` keeps it running until ``release()`` or ``stop()``."""

    def __init__(self, job, *, steps=("Pad the plate",), reply="Built the flange.", exports=None, hold=False):
        self.job = job
        self.events = connect.EventBatch()
        self._done = threading.Event()
        self._preview: tuple[int, bytes] | None = None
        self.stopped = False
        self.ok, self.text = True, reply
        self._exports = [("flange.step", "step", b"ISO-10303-21;"), ("flange.stl", "stl", b"solid")] \
            if exports is None else exports
        for title in steps:
            self.events.progress(title, ok=True)
        self.events.reply(reply)
        self._preview = (3, b"\x00" * 84)
        if not hold:
            self._done.set()

    def release(self) -> None:
        self._done.set()

    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout)

    def stop(self) -> None:
        self.stopped = True
        self.ok, self.text = False, "Stopped."
        self._done.set()

    def result(self):
        return self.ok, self.text

    def take_preview(self):
        preview, self._preview = self._preview, None
        return preview

    def exports(self):
        return list(self._exports)


class ConnectCase(unittest.TestCase):
    def setUp(self):
        for reset in (discovery.clear_cache, session.reset_sessions, keystore.reset_memory, account.clear_model_cache):
            reset()
            self.addCleanup(reset)
        patcher = mock.patch.dict(os.environ, {keystore.ENV_STORE: "memory"})
        patcher.start()
        self.addCleanup(patcher.stop)
        for name, value in FAST.items():
            fast = mock.patch.object(connect, name, value)
            fast.start()
            self.addCleanup(fast.stop)
        self.fake = FakeWorkspace()
        self.addCleanup(self.fake.close)
        self.session = session.get_session(self.fake.url)
        data = self.fake.issue_tokens(device_name="Test Box")
        self.session.complete_sign_in(TokenSet(data["access_token"], data["refresh_token"], data["expires_in"],
                                               data["scope"], received_at=time.monotonic()))
        self.client = connect.Client(self.session)
        self.states: list[tuple[str, str]] = []
        self.turns: list[FakeTurn] = []
        self.files_dir = tempfile.mkdtemp(prefix="oscar-connect-test-")
        self.addCleanup(shutil.rmtree, self.files_dir, True)

    def loop(self, *, idle=lambda: True, turn=None, session_for=None, **extra) -> connect.Loop:
        def start_turn(job, files):
            made = (turn or FakeTurn)(job)
            made.files = list(files)
            self.turns.append(made)
            return made

        loop = connect.Loop(session_for or (lambda: self.session), start_turn=start_turn, is_idle=idle,
                            on_state=lambda state, message: self.states.append((state, message)),
                            versions=connect.versions("1.1.3"), files_dir=lambda: self.files_dir, **extra)
        self.addCleanup(loop.stop, 2.0)
        return loop

    def state_names(self) -> list[str]:
        return [state for state, _ in self.states]


class JobAndBatchTests(ConnectCase):
    def test_a_job_is_parsed_from_the_server_answer(self):
        job = connect.Job.parse({"task_id": "t1", "title": "Flange", "model": "origin1",
                                 "message": {"id": "m1", "text": "Make it 10 mm"}, "turn": 2,
                                 "history": [{"role": "user", "text": "A flange"}, {"role": "assistant", "text": "ok"},
                                             {"role": "system", "text": "dropped"}]})
        self.assertEqual((job.task_id, job.model, job.text, job.turn), ("t1", "origin1", "Make it 10 mm", 2))
        self.assertEqual(job.history, (("user", "A flange"), ("assistant", "ok")))

    def test_bad_jobs_are_refused(self):
        for bad in ({}, {"task_id": "t1"}, {"task_id": "t1", "message": {"id": "m", "text": "  "}}, "nope"):
            with self.assertRaises(connect.ConnectError):
                connect.Job.parse(bad)

    def test_the_batch_keeps_order_and_posts_the_newest_reply_last(self):
        batch = connect.EventBatch()
        batch.reply("Bui")
        batch.progress("Sketch", ok=True, phase="BUILD", detail="  two   spaces ")
        batch.reply("Building the plate")
        batch.title("Flange")
        events = batch.drain()
        self.assertEqual([e["kind"] for e in events], ["progress", "title", "reply"])
        self.assertEqual(events[0]["data"], {"title": "Sketch", "detail": "two spaces", "phase": "BUILD", "ok": True})
        self.assertEqual(events[-1]["data"], {"text": "Building the plate"})
        self.assertEqual(batch.drain(), [])

    def test_a_post_holds_at_most_fifty_events(self):
        batch = connect.EventBatch()
        for n in range(60):
            batch.progress(f"step {n}")
        batch.reply("text")
        first, second = batch.drain(), batch.drain()
        self.assertEqual(len(first), 49)
        self.assertEqual([e["kind"] for e in second][-1], "reply")
        self.assertEqual(len(second), 12)

    def test_a_failed_post_is_requeued_in_front(self):
        batch = connect.EventBatch()
        batch.progress("one")
        batch.reply("old")
        failed = batch.drain()
        batch.progress("two")
        batch.requeue(failed)
        self.assertEqual([e["data"].get("title") or e["data"].get("text") for e in batch.drain()], ["one", "two", "old"])


class ClientTests(ConnectCase):
    def test_hello_reports_this_oscar(self):
        self.client.hello(connect.versions("1.1.3"))
        self.assertEqual(self.fake.hellos[-1]["oscar_version"], connect.__version__)
        self.assertEqual(self.fake.hellos[-1]["freecad_version"], "1.1.3")
        self.assertTrue(self.fake.hellos[-1]["platform"])

    def test_next_hands_out_jobs_and_none_without_work(self):
        self.assertIsNone(self.client.next_job(0))
        self.fake.add_job("A flange", task_id="t1", history=[("user", "earlier")])
        job = self.client.next_job(0)
        self.assertEqual((job.task_id, job.text, job.history), ("t1", "A flange", (("user", "earlier"),)))

    def test_events_uploads_and_finish(self):
        self.fake.add_job("A flange", task_id="t1")
        self.client.next_job(0)
        self.assertFalse(self.client.post_events("t1", [{"kind": "progress", "data": {"title": "Pad"}}]))
        self.fake.cancel["t1"] = True
        self.assertTrue(self.client.post_events("t1", []))
        self.client.upload("t1", kind="preview", fmt="stl", name="p.stl", version=4, data=b"\x01\x02")
        upload = self.fake.uploads[-1]
        self.assertEqual((upload["kind"], upload["format"], upload["name"], upload["version"]),
                         ("preview", "stl", "p.stl", "4"))
        self.assertEqual(upload["data"], b"\x01\x02")
        self.assertEqual(upload["content_type"], "application/octet-stream")
        self.client.finish("t1", "Built.", True)
        self.assertEqual(self.fake.finished["t1"], ("Built.", True))

    def test_a_task_that_is_gone_and_a_refused_upload(self):
        self.fake.add_job("A flange", task_id="t1")
        self.client.next_job(0)
        self.fake.refuse_uploads = (409, "conflict", "This task has reached its storage limit.")
        with self.assertRaises(connect.UploadRefused) as refused:
            self.client.upload("t1", kind="export", fmt="step", name="m.step", version=1, data=b"x")
        self.assertIn("storage limit", str(refused.exception))
        self.fake.gone.add("t1")
        for call in (lambda: self.client.post_events("t1", []), lambda: self.client.finish("t1", "x", True)):
            with self.assertRaises(connect.TurnGone):
                call()


class LoopTests(ConnectCase):
    def test_a_whole_turn_is_reported(self):
        self.fake.add_job("A flange", task_id="t1")
        loop = self.loop()
        loop.start()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        self.assertEqual(self.fake.finished["t1"], ("Built the flange.", True))
        kinds = [e["kind"] for e in self.fake.events["t1"]]
        self.assertEqual(kinds, ["progress", "reply"])
        uploads = [(u["kind"], u["format"], u["version"]) for u in self.fake.uploads]
        self.assertEqual(uploads, [("preview", "stl", "3"), ("export", "step", "3"), ("export", "stl", "3")])
        self.assertIn("working", self.state_names())
        self.assertEqual(self.fake.hellos[0]["oscar_version"], connect.__version__)

    def test_an_oscar_older_than_the_server_supports_stays_blocked_until_it_may_connect(self):
        self.fake.me["plugin"] = {"latest_version": "99.0.0", "min_version": "99.0.0"}
        self.fake.add_job("A flange", task_id="t1")
        loop = self.loop()
        loop.start()
        self.assertTrue(wait_until(lambda: "blocked" in self.state_names()))
        self.assertIn("needs Oscar 99.0.0 or newer", dict(self.states)["blocked"])
        self.assertEqual((self.fake.hellos, self.fake.next_calls), ([], 0))   # never online, never took work
        with self.fake.lock:                                                   # the server lowers its minimum
            self.fake.me["plugin"] = {"latest_version": "99.0.0", "min_version": connect.__version__}
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))

    def test_work_is_only_taken_while_oscar_is_idle(self):
        idle = threading.Event()
        self.fake.add_job("A flange", task_id="t1")
        loop = self.loop(idle=idle.is_set)
        loop.start()
        self.assertTrue(wait_until(lambda: len(self.fake.hellos) >= 3))   # the keep-alive while busy
        self.assertEqual(self.fake.next_calls, 0)
        idle.set()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))

    def test_stop_in_the_browser_stops_the_turn(self):
        self.fake.add_job("A flange", task_id="t1")
        self.fake.cancel["t1"] = True
        loop = self.loop(turn=lambda job: FakeTurn(job, hold=True))
        loop.start()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        self.assertTrue(self.turns[0].stopped)
        self.assertEqual(self.fake.finished["t1"], ("Stopped.", False))
        self.assertEqual([u["kind"] for u in self.fake.uploads], ["preview"])   # no exports for a stopped turn

    def test_a_turn_freecad_refuses_is_answered(self):
        def refuse(job):
            raise connect.TurnRefused("Oscar is busy with another chat in FreeCAD.")

        self.fake.add_job("A flange", task_id="t1")
        loop = self.loop(turn=refuse)
        loop.start()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        self.assertEqual(self.fake.finished["t1"], ("Oscar is busy with another chat in FreeCAD.", False))
        self.assertEqual(self.fake.events["t1"][0]["kind"], "error")

    def test_a_task_deleted_mid_turn_stops_it(self):
        self.fake.add_job("A flange", task_id="t1")
        loop = self.loop(turn=lambda job: FakeTurn(job, hold=True))
        loop.start()
        self.assertTrue(wait_until(lambda: self.fake.event_posts["t1"] >= 1))
        self.fake.gone.add("t1")
        self.assertTrue(wait_until(lambda: bool(self.turns) and self.turns[0].stopped))
        self.assertNotIn("t1", self.fake.finished)

    def test_a_refused_upload_is_reported_and_the_turn_still_finishes(self):
        self.fake.add_job("A flange", task_id="t1")
        self.fake.refuse_uploads = (409, "conflict", "This task has reached its storage limit.")
        loop = self.loop()
        loop.start()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        errors = [e["data"]["message"] for e in self.fake.events["t1"] if e["kind"] == "error"]
        self.assertTrue(errors and all("storage limit" in message for message in errors))
        self.assertTrue(self.fake.finished["t1"][1])

    def test_signed_out_waits_without_calling_the_server(self):
        loop = self.loop(session_for=lambda: None)
        loop.start()
        self.assertTrue(wait_until(lambda: "signed_out" in self.state_names()))
        self.assertEqual(self.fake.hellos, [])

    def test_a_server_that_is_down_comes_back(self):
        self.fake.fail("/v1/connect/hello", 503, method="POST", times=2)
        self.fake.add_job("A flange", task_id="t1")
        loop = self.loop()
        loop.start()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        names = self.state_names()
        self.assertIn("offline", names)
        self.assertGreater(names.index("connected"), names.index("offline"))

    def test_stopping_the_loop(self):
        loop = self.loop()
        loop.start()
        self.assertTrue(wait_until(lambda: loop.running))
        loop.stop(2.0)
        self.assertFalse(loop.running)
        self.assertTrue(wait_until(lambda: "stopped" in self.state_names()))


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
PDF = b"%PDF-1.4\n% a drawing\n"
STEP = b"ISO-10303-21;\nHEADER;\nENDSEC;\nEND-ISO-10303-21;\n"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


FIT_TEXT = """CHECK_FIT doc 'RPi4_Housing' (lift_mm=0.5)
PASS interference.shell_lid  Shell vs Lid: common_volume_mm3 = 0.0 (<= 0.05)
FAIL interference.reference_shell  Shell vs reference: common_volume_mm3 = 1.38 (<= 0.05) - the reference overlaps \
Shell by 1.380 mm3 (worst: 2X USB3.0 PORTS, RPi4ModelB, 0.915 mm3)
FIT: 1 PASS, 0 WARN, 1 FAIL, 0 SKIP
FAIL rows (interference.reference_shell) are defects: fix the geometry or the parameter (set_parameter), then run \
check_fit again."""


class StepReportTests(ConnectCase):
    def test_plan_bookkeeping_and_skipped_steps_are_not_shown(self):
        for payload in ({"name": "plan_update", "ok": False, "text": "plan_update: S11 is completed by the app"},
                        {"name": "record_note", "ok": True}, {"name": "plan_set", "ok": True},
                        {"name": "create_object", "ok": True, "skipped": True}, "not a dict"):
            self.assertIsNone(connect.step_event(payload), payload)

    def test_a_step_without_a_title_gets_a_readable_one(self):
        event = connect.step_event({"name": "find_reference_part", "title": "find_reference_part", "ok": True})
        self.assertEqual(event, {"title": "Find reference part", "ok": True})

    def test_a_failed_check_is_a_design_issue_in_plain_words(self):
        event = connect.step_event({"name": "check_fit", "title": "Check fit", "ok": False, "text": FIT_TEXT})
        self.assertEqual(event["title"], "Check fit: 1 problem found")
        self.assertEqual((event["kind"], event["issue"]), ("check", "design"))
        self.assertEqual(event["counts"], {"pass": 1, "fail": 1})   # the "FAIL rows ..." hint is not a row
        self.assertEqual(event["items"], ["Interference (reference shell): the reference overlaps Shell by 1.380 mm³ "
                                          "(worst: 2X USB3.0 PORTS, RPi4ModelB, 0.915 mm³)"])
        self.assertIn("design needs a change", event["detail"])

    def test_model_health_errors_are_problems_too(self):
        text = "header\nERROR [FRAGMENTS] Shell: 2 solids where 1 is expected\nWARN  [THIN] Lid: 0.8 mm wall"
        event = connect.step_event({"name": "validate_model", "title": "Validate model", "ok": False, "text": text})
        self.assertEqual(event["items"], ["Fragments: Shell: 2 solids where 1 is expected"])
        passed = connect.step_event({"name": "check_fit", "title": "Check fit", "ok": True,
                                     "text": "PASS a.b  ok\nPASS a.c  ok\nWARN a.d  thin"})
        self.assertEqual((passed["kind"], passed["counts"]), ("check", {"pass": 2, "warn": 1}))
        self.assertNotIn("issue", passed)

    def test_other_failures_say_what_kind_they_are(self):
        declined = connect.step_event({"name": "execute_code", "title": "Run script (7 lines)", "ok": False,
                                       "text": "The user declined to run this script, so nothing ran."})
        self.assertEqual(declined["issue"], "permission")
        text = "build_enclosure: the stage 'ports' needs the shell first. Run stage shell."
        tool = connect.step_event({"name": "build_enclosure", "title": "Build enclosure: ports", "ok": False,
                                   "text": text})
        self.assertEqual((tool["issue"], tool["detail"]), ("tool", "The stage 'ports' needs the shell first."))

    def test_an_approval_is_one_step_that_changes(self):
        waiting = connect.approval_event(4, "waiting", "Run a script")
        self.assertEqual(waiting, {"title": "Waiting for your approval in FreeCAD", "detail": "Run a script",
                                   "kind": "approval", "key": "approval-4", "state": "waiting"})
        for state, title, ok in (("approved", "Approved in FreeCAD", True), ("declined", "Declined in FreeCAD", False),
                                 ("no_answer", "No answer in FreeCAD", False), ("odd", "No answer in FreeCAD", False)):
            event = connect.approval_event(4, state, "Run a script")
            self.assertEqual((event["title"], event["ok"], event["key"]), (title, ok, "approval-4"))

    def test_an_approval_offered_to_the_browser_carries_its_answers_and_script(self):
        choices = [{"key": "run", "label": "Run", "role": "YesRole"},
                   {"key": "turn", "label": "Run all scripts this turn", "role": "AcceptRole"},
                   {"key": "deny", "label": "Don't run", "role": "RejectRole"}]
        waiting = connect.approval_event(4, "waiting", "Run a script", choices=choices, script="x" * 30_000)
        self.assertEqual(waiting["title"], "Waiting for your approval")
        self.assertEqual([(c["key"], c["tone"]) for c in waiting["choices"]],
                         [("run", "approve"), ("turn", "approve"), ("deny", "decline")])
        self.assertEqual(len(waiting["script"]), connect.SCRIPT_LIMIT)
        approved = connect.approval_event(4, "approved", "Run a script", choices=choices, script="x", by="browser")
        self.assertEqual((approved["title"], approved["by"], approved["ok"]), ("Approved in the browser", "browser", True))
        self.assertNotIn("choices", approved)
        self.assertNotIn("script", approved)
        self.assertEqual(connect.approval_event(4, "declined", "x", by="freecad")["title"], "Declined in FreeCAD")
        no_answer = connect.approval_event(4, "no_answer", "x", by="browser")
        self.assertEqual(no_answer["title"], "No answer in FreeCAD")

    def test_the_browsers_answers_reach_a_turn_waiting_for_one(self):
        class AskingTurn(FakeTurn):
            def __init__(self, job):
                super().__init__(job, hold=True)
                self.answers: list[tuple[str, str]] = []
                self.waiting = True

            def awaiting_answer(self):
                return self.waiting

            def answer(self, key, choice):
                self.answers.append((key, choice))
                self.waiting = False
                self.release()

        self.fake.add_job("A flange", task_id="t1")
        self.fake.answers["t1"] = [{"key": "approval-3", "choice": "run"}, {"key": 5}, "junk"]
        with mock.patch.object(connect, "KEEPALIVE_S", 60.0), mock.patch.object(connect, "ANSWER_POLL_S", 0.01):
            loop = self.loop(turn=AskingTurn)
            loop.start()
            self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        answers = self.turns[0].answers
        self.assertTrue(answers)
        self.assertEqual(set(answers), {("approval-3", "run")})   # malformed answers never reach the turn

    def test_the_batch_posts_the_extra_fields(self):
        batch = connect.EventBatch()
        batch.step(connect.approval_event(1, "waiting", "Network access: github.com"))
        batch.step(None)
        (event,) = batch.drain()
        self.assertEqual(event["data"]["key"], "approval-1")
        self.assertEqual(event["data"]["state"], "waiting")

    def test_the_loop_sends_the_tree_after_geometry_and_when_the_turn_ends(self):
        class TreeTurn(FakeTurn):
            def __init__(self, job, **kwargs):
                super().__init__(job, **kwargs)
                self.trees = [{"document": "D", "objects": [{"name": "Box", "children": []}], "n": 1},
                              {"document": "D", "objects": [{"name": "Box", "children": []}], "n": 2}]

            def take_tree(self):
                return self.trees.pop(0) if self.trees else None

        self.fake.add_job("A box", task_id="t1")
        loop = self.loop(turn=lambda job: TreeTurn(job, hold=True))
        loop.start()
        trees = lambda: [e["data"]["n"] for e in self.fake.events["t1"] if e["kind"] == "tree"]  # noqa: E731
        self.assertTrue(wait_until(lambda: trees() == [1]))   # with the first preview
        self.turns[0].release()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        self.assertEqual(trees(), [1, 2])                     # and once more at the end

    def test_an_upload_the_server_cant_take_now_still_finishes_the_turn(self):
        # connect_gui's log is ``_log(kind, **fields)``: a field called ``kind`` would collide with it.
        logged = []

        def log(kind, **fields):
            logged.append((kind, fields))

        def unavailable(*args, **kwargs):
            raise AuthUnavailable("The server is busy.")

        self.fake.add_job("A flange", task_id="t1")
        with mock.patch.object(connect.Client, "upload", unavailable):
            loop = connect.Loop(lambda: self.session, start_turn=lambda job, files: FakeTurn(job), is_idle=lambda: True,
                                on_state=lambda state, message: None, versions=connect.versions("1.1.3"),
                                files_dir=lambda: self.files_dir, log=log)
            self.addCleanup(loop.stop, 2.0)
            loop.start()
            self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        self.assertEqual(self.fake.finished["t1"], ("Built the flange.", True))
        failed = [fields for kind, fields in logged if kind == "connect_upload_failed"]
        self.assertTrue(failed)
        self.assertEqual(failed[0]["file_kind"], "preview")

    def test_a_glb_preview_is_uploaded_as_glb(self):
        class GlbTurn(FakeTurn):
            def __init__(self, job, **kwargs):
                super().__init__(job, **kwargs)
                self._preview = (5, connect.preview_glb([BOX], "D"), "glb")

        self.fake.add_job("A box", task_id="t1")
        loop = self.loop(turn=GlbTurn)
        loop.start()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        preview = self.fake.uploads[0]
        self.assertEqual((preview["kind"], preview["format"], preview["name"], preview["version"]),
                         ("preview", "glb", "preview-5.glb", "5"))
        self.assertEqual(preview["data"][:4], b"glTF")

    def test_a_server_without_trees_still_gets_the_turn(self):
        class TreeTurn(FakeTurn):
            def take_tree(self):
                return {"document": "D", "objects": []}

        self.fake.refuse_trees = True
        self.fake.add_job("A box", task_id="t1")
        loop = self.loop(turn=TreeTurn)
        loop.start()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        self.assertTrue(self.fake.finished["t1"][1])


# A 10 mm cube as FreeCAD would mesh it: 8 corners, 12 triangles.
CUBE_POINTS = [x for corner in [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (0, 0, 10), (10, 0, 10), (10, 10, 10),
                                (0, 10, 10)] for x in corner]
CUBE_TRIANGLES = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3,
                  4, 7]
BOX = connect.PreviewPart(name="Box", label="Box", positions=CUBE_POINTS, indices=CUBE_TRIANGLES)


def read_glb(data: bytes) -> tuple[dict, bytes]:
    """The JSON and binary chunks of a GLB, after checking its header."""
    import json
    import struct

    magic, version, length = struct.unpack("<4sII", data[:12])
    assert (magic, version, length) == (b"glTF", 2, len(data)), (magic, version, length)
    json_length, json_type = struct.unpack("<I4s", data[12:20])
    assert json_type == b"JSON" and json_length % 4 == 0
    gltf = json.loads(data[20:20 + json_length])
    rest = data[20 + json_length:]
    binary = b""
    if rest:
        bin_length, bin_type = struct.unpack("<I4s", rest[:8])
        assert bin_type == b"BIN\0" and bin_length % 4 == 0 and len(rest) == 8 + bin_length
        binary = rest[8:]
    return gltf, binary


class PreviewTests(unittest.TestCase):
    def test_each_part_is_a_node_named_like_its_freecad_object(self):
        import struct

        lid = connect.PreviewPart(name="Lid_Hooks", label="Lid", positions=[p + (20 if i % 3 == 2 else 0) for i, p in
                                                                             enumerate(CUBE_POINTS)],
                                  indices=CUBE_TRIANGLES, color=(0.2, 0.4, 0.6))
        shell = connect.PreviewPart(name="Shell_Hooks", label="Shell", positions=CUBE_POINTS, indices=CUBE_TRIANGLES)
        gltf, binary = read_glb(connect.preview_glb([shell, lid], "Housing"))

        root, *nodes = gltf["nodes"]
        self.assertEqual(gltf["scenes"], [{"nodes": [0]}])
        self.assertEqual(root["name"], "Housing")
        self.assertEqual(root["children"], [1, 2])
        self.assertAlmostEqual(root["rotation"][0], -(0.5 ** 0.5))   # Z up (FreeCAD) to Y up (glTF)
        self.assertEqual([(n["name"], n["extras"]) for n in nodes],
                         [("Shell_Hooks", {"name": "Shell_Hooks", "label": "Shell"}),
                          ("Lid_Hooks", {"name": "Lid_Hooks", "label": "Lid"})])
        self.assertEqual(gltf["buffers"], [{"byteLength": len(binary)}])
        for node, z in zip(nodes, (0.0, 20.0)):
            primitive = gltf["meshes"][node["mesh"]]["primitives"][0]
            position = gltf["accessors"][primitive["attributes"]["POSITION"]]
            index = gltf["accessors"][primitive["indices"]]
            self.assertEqual((position["count"], position["type"], position["componentType"]), (8, "VEC3", 5126))
            self.assertEqual((position["min"], position["max"]), ([0, 0, z], [10, 10, z + 10]))
            self.assertEqual((index["count"], index["componentType"]), (36, 5125))
            view = gltf["bufferViews"][index["bufferView"]]
            stored = struct.unpack("<36I", binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]])
            self.assertEqual(list(stored), CUBE_TRIANGLES)
        self.assertNotIn("material", gltf["meshes"][0]["primitives"][0])   # no colour: the viewer's own
        material = gltf["materials"][gltf["meshes"][1]["primitives"][0]["material"]]
        self.assertEqual(material["pbrMetallicRoughness"]["baseColorFactor"], [0.2, 0.4, 0.6, 1.0])

    def test_parts_without_triangles_are_left_out(self):
        broken = [connect.PreviewPart(name="Empty", label="", positions=[], indices=[]),
                  connect.PreviewPart(name="Bad", label="", positions=CUBE_POINTS, indices=[0, 1, 99]),
                  connect.PreviewPart(name="Odd", label="", positions=CUBE_POINTS, indices=[0, 1])]
        gltf, binary = read_glb(connect.preview_glb([*broken, BOX], "D"))
        self.assertEqual([n["name"] for n in gltf["nodes"]], ["D", "Box"])
        gltf, binary = read_glb(connect.preview_glb(broken, "D"))
        self.assertEqual(gltf["nodes"], [{"name": "D", "rotation": gltf["nodes"][0]["rotation"]}])
        self.assertEqual(binary, b"")
        self.assertNotIn("meshes", gltf)


class AttachmentTests(ConnectCase):
    def test_a_job_with_files_is_parsed(self):
        job = connect.Job.parse({
            "task_id": "t1", "message": {"id": "m1", "text": "", "attachments": [
                {"id": "a1", "name": "plan.pdf", "format": "pdf", "kind": "pdf", "size": 3, "sha256": sha(b"abc")},
                {"id": "a2", "name": "plan-page-1.png", "format": "png", "kind": "image", "size": 1,
                 "sha256": sha(b"x"), "page": 1}]},
            "history": [{"role": "user", "text": "Use this", "attachments": ["bracket.step"]},
                        {"role": "user", "text": "", "attachments": ["photo.jpg"]}]})
        self.assertEqual([(f.id, f.kind, f.page) for f in job.attachments], [("a1", "pdf", 0), ("a2", "image", 1)])
        self.assertEqual(job.text, "")   # files only is a message too
        self.assertEqual(job.history, (("user", "Use this\n\n[attached: bracket.step]"),
                                       ("user", "[attached: photo.jpg]")))

    def test_files_that_are_not_described_properly_are_refused(self):
        for item in ({"id": "a1", "name": "x.step", "size": 3, "sha256": "nothex"},
                     {"id": "a1", "name": "x.step", "size": -1, "sha256": sha(b"")},
                     {"id": "a1", "name": "x.step"}):
            with self.assertRaises(connect.ConnectError):
                connect.Job.parse({"task_id": "t1", "message": {"id": "m1", "text": "", "attachments": [item]}})

    def test_file_names_are_made_safe(self):
        cases = {"../../etc/passwd": "passwd", "C:\\Users\\x\\part.step": "part.step", "CON.txt": "file-CON.txt",
                 "..hidden.png": "hidden.png", 'a<b>:"c|?*.stl': "a_b___c___.stl", "": "attachment",
                 "plan\x00.pdf": "plan_.pdf", "dessin é.pdf": "dessin é.pdf"}
        for name, expected in cases.items():
            self.assertEqual(connect.safe_filename(name), expected, name)
        self.assertLessEqual(len(connect.safe_filename("x" * 400 + ".step")), 120)
        self.assertTrue(connect.safe_filename("x" * 400 + ".step").endswith(".step"))

    def _job_files(self, *attachments):
        self.fake.add_job("Build it", task_id="t1", attachments=list(attachments))
        return self.client.next_job(0)

    def test_a_download_is_checked_against_its_size_and_checksum(self):
        job = self._job_files(("bracket.step", "cad", STEP), ("photo.png", "image", PNG), ("plan.pdf", "pdf", PDF))
        good, damaged, oversized = job.attachments
        dest = os.path.join(self.files_dir, "bracket.step")
        self.client.download(good, dest)
        with open(dest, "rb") as handle:
            self.assertEqual(handle.read(), STEP)
        self.fake.damaged.add(damaged.id)
        self.fake.oversized.add(oversized.id)
        for item, code in ((damaged, "corrupt"), (oversized, "too_large")):
            target = os.path.join(self.files_dir, item.name)
            with self.assertRaises(connect.ConnectError) as failed:
                self.client.download(item, target)
            self.assertEqual(failed.exception.code, code)
            self.assertFalse(os.path.exists(target))
        self.assertEqual(sorted(os.listdir(self.files_dir)), ["bracket.step"])   # no partial files left behind
        missing = connect.JobFile(id="a-404", name="gone.step", format="step", kind="cad", size=1, sha256=sha(b"x"))
        with self.assertRaises(connect.ConnectError) as gone:
            self.client.download(missing, os.path.join(self.files_dir, "gone.step"))
        self.assertEqual(gone.exception.code, "not_found")

    def test_a_refused_token_is_renewed_for_a_download(self):
        job = self._job_files(("bracket.step", "cad", STEP))
        self.fake.reject_token(self.session.access_token())
        self.client.download(job.attachments[0], os.path.join(self.files_dir, "bracket.step"))
        self.assertEqual(self.fake.counts["token:refresh_token"], 1)
        self.assertEqual(len(self.fake.calls_to("/v1/connect/attachments/a-1")), 2)   # refused, then sent again
        self.assertEqual(self.fake.downloads, ["a-1"])

    def test_the_loop_downloads_the_files_before_the_turn(self):
        self.fake.add_job("Build the bracket", task_id="t1", attachments=[
            ("bracket.step", "cad", STEP), ("plan.pdf", "pdf", PDF), ("plan-page-1.png", "image", PNG, 1),
            ("Bracket.STEP", "cad", STEP + b"\n")])
        loop = self.loop()
        loop.start()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        files = self.turns[0].files
        self.assertEqual([os.path.basename(path) for path in files],
                         ["bracket.step", "plan.pdf", "plan-page-1.png", "Bracket (2).STEP"])
        folder = os.path.join(self.files_dir, "t1", "m-1")
        self.assertTrue(all(os.path.dirname(path) == folder for path in files))
        with open(files[2], "rb") as handle:
            self.assertEqual(handle.read(), PNG)
        first = self.fake.events["t1"][0]
        self.assertEqual((first["kind"], first["data"]["title"]), ("progress", "Reading 3 attached files"))
        self.assertTrue(self.fake.finished["t1"][1])

    def test_a_file_that_cant_be_downloaded_is_reported_and_the_turn_goes_on(self):
        self.fake.add_job("Build the bracket", task_id="t1",
                          attachments=[("bracket.step", "cad", STEP), ("photo.png", "image", PNG)])
        self.fake.damaged.add("a-2")
        loop = self.loop()
        loop.start()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        self.assertEqual([os.path.basename(path) for path in self.turns[0].files], ["bracket.step"])
        errors = [e["data"]["message"] for e in self.fake.events["t1"] if e["kind"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("Couldn’t download photo.png", errors[0])
        self.assertTrue(self.fake.finished["t1"][1])

    def test_a_server_that_drops_a_download_is_asked_again(self):
        self.fake.add_job("Build it", task_id="t1", attachments=[("bracket.step", "cad", STEP)])
        self.fake.fail("/v1/connect/attachments/a-1", 0, method="GET", times=1)
        loop = self.loop()
        loop.start()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        self.assertEqual([os.path.basename(path) for path in self.turns[0].files], ["bracket.step"])
        self.assertFalse([e for e in self.fake.events["t1"] if e["kind"] == "error"])

    def test_files_only_whose_files_all_fail_is_answered_without_a_turn(self):
        self.fake.add_job("", task_id="t1", attachments=[("photo.png", "image", PNG)])
        self.fake.damaged.add("a-1")
        loop = self.loop()
        loop.start()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        self.assertEqual(self.turns, [])
        text, ok = self.fake.finished["t1"]
        self.assertFalse(ok)
        self.assertIn("couldn’t download the attached files", text)

    def test_old_downloads_are_removed(self):
        old, recent = (os.path.join(self.files_dir, name, "m1") for name in ("old-task", "recent-task"))
        for folder in (old, recent):
            os.makedirs(folder)
            with open(os.path.join(folder, "part.step"), "wb") as handle:
                handle.write(STEP)
        now = time.time()
        for path in (old, os.path.dirname(old), os.path.join(old, "part.step")):
            os.utime(path, (now - 40 * 86400, now - 40 * 86400))
        self.assertEqual(connect.prune_files(self.files_dir, now=now), 1)
        self.assertEqual(os.listdir(self.files_dir), ["recent-task"])


class DeliveryTests(ConnectCase):
    """A turn's answer always reaches the workspace."""

    def setUp(self):
        super().setUp()
        self.outbox = connect.Outbox(os.path.join(self.files_dir, "outbox.json"))
        fast = mock.patch.object(connect, "SYNC_S", 0.05)
        fast.start()
        self.addCleanup(fast.stop)

    def test_a_bug_while_reporting_still_ends_with_the_answer(self):
        class BrokenTurn(FakeTurn):
            def exports(self):
                raise TypeError("_log() got multiple values for argument 'kind'")   # the bug that stranded a turn

        self.fake.add_job("A bracket", task_id="t1")
        loop = self.loop(turn=BrokenTurn, outbox=self.outbox)
        loop.start()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        self.assertEqual(self.fake.finished["t1"], ("Built the flange.", True))
        self.assertEqual(self.fake.finish_bodies[0]["message_id"], "m-1")
        self.assertTrue(wait_until(lambda: self.outbox.entries() == []))   # removed once the server answered

    def test_an_answer_the_server_could_not_take_is_delivered_after_reconnecting(self):
        self.fake.add_job("A bracket", task_id="t1")
        self.fake.fail("/v1/connect/tasks/t1/finish", 503, method="POST", times=connect.FINISH_ATTEMPTS)
        loop = self.loop(outbox=self.outbox)
        loop.start()
        self.assertTrue(wait_until(lambda: self.outbox.entries() != []))   # kept on disk meanwhile
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished, timeout=10))
        self.assertEqual(self.fake.finished["t1"], ("Built the flange.", True))
        self.assertTrue(wait_until(lambda: self.outbox.entries() == []))

    def test_a_closed_turn_is_answered_from_the_outbox_or_the_chat(self):
        self.fake.unanswered = [("t1", "m1"), ("t2", "m2"), ("t3", "m3"), ("t4", "m4")]
        self.outbox.add("t1", "m1", "From the outbox", True)
        answers = {"t2": ("found", True, "From the chat"), "t3": ("busy", False, ""), "t4": ("none", False, "")}
        loop = self.loop(outbox=self.outbox, recover=lambda task, message: answers[task])
        loop.start()
        self.assertTrue(wait_until(lambda: {"t1", "t2", "t4"} <= set(self.fake.finished)))
        self.assertEqual(self.fake.finished["t1"], ("From the outbox", True))
        self.assertEqual(self.fake.finished["t2"], ("From the chat", True))
        self.assertEqual(self.fake.finished["t4"], (connect.NO_ANSWER_TEXT, False))
        self.assertNotIn("t3", self.fake.finished)          # FreeCAD still works on it: asked again later
        answers["t3"] = ("found", True, "Finished now")
        self.assertTrue(wait_until(lambda: "t3" in self.fake.finished))
        self.assertTrue(wait_until(lambda: self.outbox.entries() == []))

    def test_local_paths_never_leave_this_computer(self):
        class PathTurn(FakeTurn):
            def __init__(self, job, **kwargs):
                super().__init__(job, reply="Saved to /home/ada/Oscar/Projects/bracket-1/exports/a.step", **kwargs)
                self.events.progress("Export", detail="/home/ada/Oscar/Projects/bracket-1/exports")

        redact = lambda text: text.replace("/home/ada/Oscar/Projects/bracket-1", "Oscar project")  # noqa: E731
        self.fake.add_job("A bracket", task_id="t1")
        loop = self.loop(turn=PathTurn, redact=redact)
        loop.start()
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))
        self.assertEqual(self.fake.finished["t1"][0], "Saved to Oscar project/exports/a.step")
        details = [e["data"].get("detail") for e in self.fake.events["t1"] if e["kind"] == "progress"]
        self.assertIn("Oscar project/exports", details)

    def test_the_outbox_keeps_one_answer_per_turn_for_a_week(self):
        self.outbox.add("t1", "m1", "first", False)
        self.outbox.add("t1", "m1", "second", True)
        self.outbox.add("t2", "m2", "other", True)
        self.assertEqual([(e["task_id"], e["text"]) for e in self.outbox.entries()], [("t1", "second"), ("t2", "other")])
        self.outbox.remove("t1", "m1")
        self.assertEqual(self.outbox.get("t2", "m2")["text"], "other")
        self.assertEqual(self.outbox.entries(now=time.time() + connect.OUTBOX_KEPT_S + 1), [])
        self.assertEqual(connect.Outbox(os.path.join(self.files_dir, "missing", "x.json")).entries(), [])


class RevertTests(ConnectCase):
    """The browser reverting a task's turns in this FreeCAD."""

    TREE = {"document": "Flange", "label": "Flange", "objects": [{"name": "Pad", "label": "Pad"}]}

    def reverter(self, outcome=None):
        calls = []

        def revert(job):
            calls.append(job)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome or connect.RevertOutcome(job.messages, preview=("glb", b"glTF"), tree=self.TREE)

        return revert, calls

    def test_only_an_oscar_that_can_revert_says_so(self):
        revert, _calls = self.reverter()
        loop = self.loop(revert=revert)
        loop.start()
        self.assertTrue(wait_until(lambda: self.fake.hellos))
        self.assertEqual(self.fake.hellos[-1]["capabilities"], [connect.REVERT])
        loop.stop(2.0)
        loop = self.loop(revert=revert, open_model=lambda model, path: (True, "", ""))
        loop.start()
        self.assertTrue(wait_until(lambda: len(self.fake.hellos) >= 2))
        self.assertEqual(self.fake.hellos[-1]["capabilities"], [connect.OPEN_MODEL, connect.REVERT])

    def test_a_revert_job_is_parsed_and_a_bad_one_refused(self):
        self.client.hello({**connect.versions("1.1.3"), "capabilities": [connect.REVERT]})
        revert_id = self.fake.add_revert(["m-2", "m-1"], task_id="t1")
        job = self.client.next_job(0)
        self.assertIsInstance(job, connect.RevertJob)
        self.assertEqual((job.id, job.task_id, job.messages), (revert_id, "t1", ("m-2", "m-1")))
        for bad in ({"id": "r-9", "task_id": "t1", "messages": []}, {"id": "r-9", "messages": [{"id": "m-1"}]},
                    {"id": "r-9", "task_id": "t1", "messages": [{"text": "no id"}]}, {"task_id": "t1"}):
            with self.assertRaises(connect.ConnectError) as refused:
                connect.RevertJob.parse(bad)
            self.assertEqual(refused.exception.code, "bad_job")

    def test_the_loop_reverts_uploads_the_preview_and_reports_then_takes_task_messages(self):
        revert, calls = self.reverter()
        revert_id = self.fake.add_revert(["m-2", "m-1"], task_id="t1")
        self.fake.add_job("A bracket", task_id="t1")
        loop = self.loop(revert=revert)
        loop.start()
        self.assertTrue(wait_until(lambda: revert_id in self.fake.revert_reports and "t1" in self.fake.finished))
        [job] = calls
        self.assertEqual(job.messages, ("m-2", "m-1"))
        self.assertEqual(self.fake.revert_reports[revert_id],
                         {"ok": True, "reverted": ["m-2", "m-1"], "message": "", "tree": self.TREE})
        [upload] = self.fake.revert_uploads
        self.assertEqual((upload["task"], upload["kind"], upload["format"], upload["revert"], upload["data"]),
                         ("t1", "preview", "glb", revert_id, b"glTF"))
        self.assertIn("working", self.state_names())
        self.assertEqual(len(self.turns), 1, "the task message came after the revert")

    def test_a_partial_revert_says_which_turns_and_why(self):
        revert, _calls = self.reverter(connect.RevertOutcome(("m-3",), "A turn made in FreeCAD comes first."))
        revert_id = self.fake.add_revert(["m-3", "m-2"], task_id="t1")
        loop = self.loop(revert=revert)
        loop.start()
        self.assertTrue(wait_until(lambda: revert_id in self.fake.revert_reports))
        self.assertEqual(self.fake.revert_reports[revert_id],
                         {"ok": False, "reverted": ["m-3"], "message": "A turn made in FreeCAD comes first."})
        self.assertEqual(self.fake.revert_uploads, [], "no preview without a tree or a model")

    def test_a_revert_that_breaks_is_reported_as_not_done(self):
        revert, _calls = self.reverter(RuntimeError("boom"))
        revert_id = self.fake.add_revert(["m-1"], task_id="t1")
        loop = self.loop(revert=revert)
        loop.start()
        self.assertTrue(wait_until(lambda: revert_id in self.fake.revert_reports))
        self.assertEqual(self.fake.revert_reports[revert_id],
                         {"ok": False, "reverted": [], "message": "FreeCAD couldn’t revert (RuntimeError)."})

    def test_a_tree_the_server_refuses_does_not_lose_the_report(self):
        self.fake.refuse_revert_trees = True
        revert, _calls = self.reverter()
        revert_id = self.fake.add_revert(["m-1"], task_id="t1")
        loop = self.loop(revert=revert)
        loop.start()
        self.assertTrue(wait_until(lambda: revert_id in self.fake.revert_reports))
        self.assertEqual(self.fake.revert_reports[revert_id], {"ok": True, "reverted": ["m-1"], "message": ""})

    def test_an_oscar_that_cant_revert_gets_no_revert_job(self):
        self.fake.add_revert(["m-1"], task_id="t1")
        self.fake.add_job("A bracket", task_id="t2")
        loop = self.loop()
        loop.start()
        self.assertTrue(wait_until(lambda: "t2" in self.fake.finished))
        self.assertEqual(self.fake.revert_reports, {})
        self.assertEqual(len(self.fake.revert_jobs), 1)


if __name__ == "__main__":
    unittest.main()


class OpenModelTests(ConnectCase):
    """A model built in the browser, opened in this FreeCAD."""

    def opener(self, result=(True, "", "Flange")):
        calls = []

        def open_model(model, path):
            with open(path, "rb") as handle:
                calls.append((model, path, handle.read()))
            if isinstance(result, Exception):
                raise result
            return result

        return open_model, calls

    def test_only_an_oscar_that_can_open_models_says_so(self):
        loop = self.loop()
        loop.start()
        self.assertTrue(wait_until(lambda: self.fake.hellos))
        loop.stop(2.0)
        self.assertNotIn("capabilities", self.fake.hellos[-1])
        open_model, _calls = self.opener()
        loop = self.loop(open_model=open_model)
        loop.start()
        self.assertTrue(wait_until(lambda: len(self.fake.hellos) >= 2))
        self.assertEqual(self.fake.hellos[-1]["capabilities"], [connect.OPEN_MODEL])

    def test_an_open_job_is_parsed_and_a_bad_one_refused(self):
        self.client.hello({**connect.versions("1.1.3"), "capabilities": [connect.OPEN_MODEL]})
        delivery_id = self.fake.add_model(STEP, title="A flange")
        model = self.client.next_job(0)
        self.assertIsInstance(model, connect.OpenModel)
        self.assertEqual((model.id, model.title, model.name, model.format, model.size, model.sha256),
                         (delivery_id, "A flange", "A flange.step", "step", len(STEP), sha(STEP)))
        for bad in ({"id": "d-9", "size": 0, "sha256": sha(STEP), "format": "step"},
                    {"id": "d-9", "size": 4, "sha256": "nope", "format": "step"},
                    {"id": "d-9", "size": 4, "sha256": sha(STEP), "format": "exe"},
                    {"size": 4, "sha256": sha(STEP), "format": "step"}):
            with self.assertRaises(connect.ConnectError) as refused:
                connect.OpenModel.parse(bad)
            self.assertEqual(refused.exception.code, "bad_job")

    def test_the_loop_downloads_opens_and_reports_a_model_then_takes_task_messages(self):
        open_model, calls = self.opener()
        delivery_id = self.fake.add_model(STEP, title="An 80 mm flange")
        self.fake.add_job("A bracket", task_id="t1")
        loop = self.loop(open_model=open_model)
        loop.start()
        self.assertTrue(wait_until(lambda: delivery_id in self.fake.model_reports and "t1" in self.fake.finished))
        self.assertEqual(self.fake.model_reports[delivery_id], {"ok": True, "message": "", "document": "Flange"})
        [(model, path, data)] = calls
        self.assertEqual((model.id, data), (delivery_id, STEP))
        self.assertEqual(path, os.path.join(self.files_dir, connect.MODELS_DIR, delivery_id, "An 80 mm flange.step"))
        self.assertIn("working", self.state_names())
        self.assertEqual(len(self.turns), 1, "the task message came after the model")

    def test_a_model_that_cant_be_downloaded_is_reported_and_not_opened(self):
        open_model, calls = self.opener()
        delivery_id = self.fake.add_model(STEP)
        self.fake.damaged_models.add(delivery_id)
        loop = self.loop(open_model=open_model)
        loop.start()
        self.assertTrue(wait_until(lambda: delivery_id in self.fake.model_reports))
        report = self.fake.model_reports[delivery_id]
        self.assertFalse(report["ok"])
        self.assertIn("couldn’t download the model", report["message"])
        self.assertEqual(calls, [])
        self.assertEqual(os.listdir(os.path.join(self.files_dir, connect.MODELS_DIR, delivery_id)), [])

    def test_a_model_freecad_cant_open_is_reported_and_the_loop_goes_on(self):
        cant = "FreeCAD couldn’t open the model: the file holds no solid"
        open_model, _calls = self.opener(result=(False, cant, ""))
        first = self.fake.add_model(STEP)
        loop = self.loop(open_model=open_model)
        loop.start()
        self.assertTrue(wait_until(lambda: first in self.fake.model_reports))
        self.assertEqual(self.fake.model_reports[first], {"ok": False, "message": cant, "document": ""})
        crash, _ = self.opener(result=RuntimeError("boom"))
        loop.stop(2.0)
        second = self.fake.add_model(STEP)
        loop = self.loop(open_model=crash)
        loop.start()
        self.assertTrue(wait_until(lambda: second in self.fake.model_reports))
        self.assertEqual(self.fake.model_reports[second]["message"],
                         "FreeCAD couldn’t open the model (RuntimeError).")
        self.fake.add_job("A bracket", task_id="t1")
        self.assertTrue(wait_until(lambda: "t1" in self.fake.finished))

    def test_an_open_job_an_oscar_cant_handle_is_still_answered(self):
        """A server that sends one anyway (it shouldn't) gets an answer, not silence."""
        self.fake.add_model(STEP)
        model = connect.OpenModel.parse(self.fake.model_jobs.popleft())
        self.loop()._open(self.client, model)
        self.assertFalse(self.fake.model_reports[model.id]["ok"])
        self.assertIn("can’t open models", self.fake.model_reports[model.id]["message"])
