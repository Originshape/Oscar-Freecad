# SPDX-License-Identifier: Apache-2.0
"""FreeCAD chat records (oscar/records.py) against the fake OriginShape API: a turn record's fields and limits, every
text redacted before it leaves (keys a script printed, remembered tokens, project paths), no record before consent,
for another provider or for a workspace turn, the files of a turn (attachments and the STEP / STL / FCStd it wrote)
sent once per account and only under the size limits, with a Content-Length; retries (network errors, 5xx, 429 with
Retry-After) and drops (other 4xx, a Retry-After too long, another account), 403 terms_required asking for consent
again, 401 renewing the sign-in, a bounded queue, and a bug that never reaches the caller; the turn index a chat keeps
for good (chats.next_turn) and the chat id and turn a rating names. Pure python (stdlib + requests).
"""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import __version__, chats, context_guard, project_paths, rating, records  # noqa: E402
from oscar.auth import account, consent, discovery, keystore, session  # noqa: E402
from oscar.auth.transport import TokenSet  # noqa: E402
from oscar.providers import Message, ToolResult  # noqa: E402
from support.fake_originshape import FakeOriginShape  # noqa: E402

CHAT = "20261102-100000-a1b2c3"
ANTHROPIC_KEY = "sk-ant-api03-" + "A" * 48
ORIGIN_KEY = "os_live_" + "b" * 32
REMEMBERED = "remembered-refresh-token-" + "c" * 24
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyLTEifQ.c2lnbmF0dXJlLXNpZ25hdHVyZQ"


def record(**overrides) -> records.TurnRecord:
    values = {"chat_id": CHAT, "turn": 3, "prompt": "Make an 80 mm flange", "reply": "Done: the flange.",
              "steps": [records.Step("create_object", "Sketch the profile", True, detail="Created Sketch")],
              "model": "origin1", "project": "an-80-mm-flange-a1b2c3", "title": "An 80 mm flange", "ok": True,
              "stopped": False, "started_at": 1_800_000_000.0, "finished_at": 1_800_000_090.0}
    values.update(overrides)
    return records.TurnRecord(**values)


class TurnBodyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="records_", dir=isolation.sandbox_root())

    def test_a_turn_record_carries_the_turn(self):
        script = os.path.join(self.tmp, "step-002.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write("import FreeCAD\n" + "# a long script\n" * 1000)
        body = records.turn_body(record(steps=[
            records.Step("create_object", "Sketch the profile", True, detail="Created Sketch"),
            records.Step("execute_code", "Pad it", False, script="short", script_path=script, detail="Error: x"),
            records.Step("execute_code", "", None, script="print(1)", script_path="/nowhere/step.py"),
        ]))
        self.assertEqual(set(body), {"project", "title", "model", "prompt", "reply", "ok", "stopped", "steps",
                                     "started_at", "finished_at", "client"})
        self.assertEqual((body["prompt"], body["reply"], body["model"], body["title"], body["project"]),
                         ("Make an 80 mm flange", "Done: the flange.", "origin1", "An 80 mm flange",
                          "an-80-mm-flange-a1b2c3"))
        self.assertEqual((body["ok"], body["stopped"]), (True, False))
        self.assertEqual((body["started_at"], body["finished_at"]), ("2027-01-15T08:00:00Z", "2027-01-15T08:01:30Z"))
        self.assertEqual(body["client"], f"oscar-freecad/{__version__}")
        first, second, third = body["steps"]
        self.assertEqual(first, {"tool": "create_object", "title": "Sketch the profile", "ok": True, "script": None,
                                 "detail": "Created Sketch"})
        self.assertTrue(second["script"].startswith("import FreeCAD"))   # the whole recorded script, not the excerpt
        self.assertEqual((second["ok"], second["detail"]), (False, "Error: x"))
        self.assertEqual((third["title"], third["ok"], third["script"], third["detail"]),
                         ("execute_code", False, "print(1)", None))
        json.dumps(body)   # it is JSON
        empty = records.turn_body(records.TurnRecord(CHAT, 0))
        self.assertEqual((empty["prompt"], empty["project"], empty["model"], empty["ok"], empty["started_at"]),
                         ("", None, None, None, None))

    def test_texts_and_steps_are_cut_to_the_servers_limits(self):
        body = records.turn_body(record(prompt="p" * 150_000, reply="r" * 500_000, title="t" * 400,
                                        project="j" * 300, model="m" * 200,
                                        steps=[records.Step("x" * 100, "y" * 400, True, detail="d" * 9000)] * 600))
        self.assertEqual([len(body[k]) for k in ("prompt", "reply", "title", "project", "model")],
                         [100_000, 400_000, 300, 200, 100])
        self.assertEqual(len(body["steps"]), 500)
        self.assertEqual((len(body["steps"][0]["tool"]), len(body["steps"][0]["title"]),
                          len(body["steps"][0]["detail"])), (80, 300, 4000))
        self.assertLessEqual(len(json.dumps(body["steps"]).encode("utf-8")), records.STEPS_BYTES)
        self.assertEqual(len(body["steps"][-1]["detail"]), records.SHORT_DETAIL)   # over 2 MB: the last are short
        big = records.turn_body(record(steps=[records.Step("execute_code", "s", True, script="x" * 12_000)] * 500))
        self.assertLessEqual(len(json.dumps(big["steps"]).encode("utf-8")), records.STEPS_BYTES)
        self.assertEqual(len(big["steps"]), 500)        # every step stays; the later ones go without their script
        self.assertIsNone(big["steps"][-1]["script"])
        self.assertEqual(len(big["steps"][0]["script"]), 12_000)
        small = records.turn_body(record(steps=[records.Step("execute_code", "s", True, script="x" * 100)] * 20))
        self.assertTrue(all(s["script"] for s in small["steps"]))   # under the budget nothing is touched

    def test_credentials_are_redacted_before_anything_leaves(self):
        context_guard.remember(REMEMBERED)
        self.addCleanup(context_guard.forget_all)
        settings = json.dumps({"profiles": [{"api_key": ANTHROPIC_KEY}], "originshape": ORIGIN_KEY})
        script = os.path.join(self.tmp, "show-settings.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(f"print(open('oscar_settings.json').read())  # {ORIGIN_KEY}\n")
        project = os.path.join(project_paths.project_roots()[0], "flange-a1b2c3", "exports", "a.step")
        body = records.turn_body(record(
            prompt=f"Use my key {ANTHROPIC_KEY} please", reply=f"Saved to {project}. Token {JWT}",
            title=f"Key {ORIGIN_KEY}", project=f"p-{REMEMBERED}",
            steps=[records.Step("execute_code", f"print {REMEMBERED}", True, script="print(open(...).read())",
                                script_path=script, detail=settings)]))
        text = json.dumps(body)
        for secret in (ANTHROPIC_KEY, ORIGIN_KEY, REMEMBERED, JWT):
            self.assertNotIn(secret, text)
        self.assertIn(context_guard.REDACTED, body["prompt"])
        self.assertIn(context_guard.REDACTED, body["steps"][0]["script"])
        self.assertIn(context_guard.REDACTED, body["steps"][0]["detail"])
        self.assertIn("Oscar project/exports/a.step", body["reply"])
        self.assertNotIn(project_paths.project_roots()[0], body["reply"])
        # A key right at the limit is redacted before the cut: no half key is left behind.
        cut = records.turn_body(record(prompt="x" * (records.PROMPT_LIMIT - 10) + ANTHROPIC_KEY))
        self.assertNotIn("sk-ant", cut["prompt"])
        self.assertEqual(len(cut["prompt"]), records.PROMPT_LIMIT)

    def test_the_files_a_turn_wrote(self):
        root = os.path.join(self.tmp, "flange-a1b2c3")
        for sub in ("exports", "docs", "snapshots"):
            os.makedirs(os.path.join(root, sub))
        paths = {name: os.path.join(root, name) for name in (
            "model.FCStd", "exports/a.step", "exports/a.stl", "exports/old.step", "exports/notes.txt",
            "exports/.hidden.stl", "docs/Bracket.FCStd", "snapshots/s1.png", "NOTES.md")}
        for path in paths.values():
            with open(path, "wb") as f:
                f.write(b"data")
        since = time.time() - 60
        os.utime(paths["exports/old.step"], (since - 3600, since - 3600))   # written by an earlier turn
        found = [os.path.relpath(p, root).replace(os.sep, "/") for p in records.output_files(root, since)]
        self.assertEqual(found, ["model.FCStd", "docs/Bracket.FCStd", "exports/a.step", "exports/a.stl"])
        self.assertEqual(records.output_files(os.path.join(self.tmp, "missing"), since), [])
        self.assertEqual(records.output_files("", since), [])

    def test_size_limits_and_names(self):
        self.assertTrue(records.file_allowed("a.step", 50_000_000))
        self.assertFalse(records.file_allowed("a.step", 50_000_001))
        self.assertTrue(records.file_allowed("a.STL", 100_000_000))
        self.assertFalse(records.file_allowed("a.stl", 100_000_001))
        self.assertFalse(records.file_allowed("a.step", 0))
        self.assertEqual(records.upload_name("/home/ada/My Parts/flange.step"), "flange.step")
        long = records.upload_name("/x/" + "n" * 300 + ".step")
        self.assertEqual((len(long), long[-5:]), (255, ".step"))
        self.assertEqual(records.upload_name(f"/x/{ORIGIN_KEY}.step"), f"{context_guard.REDACTED}.step")


class GateTest(unittest.TestCase):
    def state(self, me_consent=True, accepted=True, signed_in=True):
        snapshot = {"email": "ada@example.com", "status": "approved", "consent_known": me_consent,
                    "terms_current": accepted, "legal_terms_version": "2026-11-02"}
        return {"api_url": "http://127.0.0.1:9", "signed_in": signed_in, "email": "ada@example.com",
                "snapshot": snapshot}

    def test_only_turns_on_the_account_after_consent_are_recorded(self):
        recorder = records.Recorder(autostart=False)
        cases = [
            ({"account_profile": True, "workspace": False, "state": self.state()}, True),
            ({"account_profile": True, "workspace": False, "state": self.state(accepted=False)}, False),
            ({"account_profile": True, "workspace": False, "state": self.state(me_consent=False)}, False),
            ({"account_profile": True, "workspace": False, "state": self.state(signed_in=False)}, False),
            ({"account_profile": False, "workspace": False, "state": self.state()}, False),   # your own provider
            ({"account_profile": True, "workspace": True, "state": self.state()}, False),     # the workspace's own
            ({"account_profile": True, "workspace": False, "state": None}, False),
        ]
        for kwargs, expected in cases:
            self.assertEqual(records.applies(**kwargs), expected, kwargs)
            self.assertEqual(records.record_turn(record(), into=recorder, **kwargs), expected, kwargs)
        self.assertEqual(recorder.pending(), 1)

    def test_a_record_that_cannot_be_sent_is_refused(self):
        recorder = records.Recorder(autostart=False)
        for bad in (record(chat_id="../x"), record(chat_id=""), record(turn=-1), record(turn=100_001),
                    record(turn=None), record(turn=True)):
            self.assertFalse(recorder.submit(bad, api_url="http://x", account="ada@example.com"))
        self.assertFalse(recorder.submit(record(), api_url="", account="ada@example.com"))
        self.assertFalse(recorder.submit(record(), api_url="http://x", account=""))
        self.assertEqual(recorder.pending(), 0)


class RecorderCase(unittest.TestCase):
    def setUp(self):
        for reset in (discovery.clear_cache, session.reset_sessions, keystore.reset_memory, account.clear_model_cache):
            reset()
            self.addCleanup(reset)
        patcher = mock.patch.dict(os.environ, {keystore.ENV_STORE: "memory"})
        patcher.start()
        self.addCleanup(patcher.stop)
        hook = mock.patch.object(session, "on_change", None)
        hook.start()
        self.addCleanup(hook.stop)
        self.fake = FakeOriginShape()
        self.addCleanup(self.fake.close)
        self.fake.set_consent(accepted="2026-11-02", current="2026-11-02")
        self.tokens = self.fake.issue_tokens()
        self.session = session.get_session(self.fake.url)
        self.session.complete_sign_in(TokenSet(self.tokens["access_token"], self.tokens["refresh_token"],
                                               self.tokens["expires_in"], self.tokens["scope"],
                                               received_at=time.monotonic()))
        self.session.account(force=True)
        self.tmp = tempfile.mkdtemp(prefix="records_", dir=isolation.sandbox_root())
        self.sent_path = os.path.join(self.tmp, "sent-files.json")
        self.waits: list[float] = []
        self.terms: list[str] = []
        self.recorder = self.make()

    def make(self, **kw):
        values = {"sent": records.SentFiles(self.sent_path), "retry_s": (1.0, 2.0, 3.0),
                  "wait": lambda seconds: self.waits.append(seconds) or False,
                  "on_terms_required": self.terms.append}
        values.update(kw)
        recorder = records.Recorder(**values)
        self.addCleanup(recorder.close)
        return recorder

    def state(self):
        return {"api_url": self.fake.url, "signed_in": True, "email": self.session.email,
                "snapshot": self.session.cached_snapshot().to_dict()}

    def send(self, rec=None, recorder=None):
        recorder = recorder or self.recorder
        self.assertTrue(records.record_turn(rec or record(), account_profile=True, workspace=False,
                                            state=self.state(), into=recorder))
        self.assertTrue(recorder.wait_idle(15.0))

    def file(self, name, data=b"solid part\n"):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def turn_path(self, turn=3):
        return f"/v1/freecad/chats/{CHAT}/turns/{turn}"


class RecorderTest(RecorderCase):
    def test_the_turn_and_its_files_are_sent(self):
        attached = self.file("plan.pdf", b"%PDF-1.4 plan")
        output = self.file("flange.step", b"ISO-10303-21; flange")
        self.send(record(attachments=[attached], outputs=[output]))
        body = self.fake.chat_turns[(CHAT, 3)]
        self.assertEqual((body["prompt"], body["model"], body["client"]),
                         ("Make an 80 mm flange", "origin1", f"oscar-freecad/{__version__}"))
        files = [(f["name"], f["kind"], f["turn"], f["size"], f["content_length"]) for f in self.fake.chat_files]
        self.assertEqual(files, [("plan.pdf", "attachment", 3, 13, 13), ("flange.step", "output", 3, 20, 20)])
        put = self.fake.calls_to(f"/v1/freecad/chats/{CHAT}/files")[0]
        self.assertEqual(put.method, "PUT")
        self.assertEqual(put.headers["content-type"], "application/octet-stream")
        self.assertEqual(self.recorder.stats["turn_ok"], 1)

    def test_no_upload_before_consent(self):
        self.fake.set_consent(accepted=None, current="2026-11-02")
        self.session.account(force=True)
        self.assertFalse(records.record_turn(record(), account_profile=True, workspace=False, state=self.state(),
                                             into=self.recorder))
        self.assertTrue(self.recorder.wait_idle(5.0))
        self.assertEqual((self.fake.counts["chat_turn"], self.fake.counts["chat_file"]), (0, 0))

    def test_each_file_once_per_account(self):
        attached = self.file("bracket.stl", b"solid bracket\n" * 10)
        self.send(record(turn=1, attachments=[attached]))
        self.send(record(turn=2, attachments=[attached], outputs=[attached]))
        self.assertEqual(len(self.fake.chat_files), 1)
        self.assertEqual(self.recorder.stats["file_known"], 2)
        # Remembered across a restart, for this account only.
        later = self.make()
        self.send(record(turn=4, attachments=[attached]), recorder=later)
        self.assertEqual(len(self.fake.chat_files), 1)
        sent = records.SentFiles(self.sent_path)
        digest = self.fake.chat_files[0]["sha256"]
        self.assertTrue(sent.has(records.account_key(self.fake.url, "ada@example.com"), digest))
        self.assertTrue(sent.has(records.account_key(self.fake.url, "ADA@example.com"), digest))
        self.assertFalse(sent.has(records.account_key(self.fake.url, "bob@example.com"), digest))
        self.assertNotIn("ada@example.com", open(self.sent_path, encoding="utf-8").read())
        # A changed file is a new file.
        with open(attached, "ab") as f:
            f.write(b"endsolid\n")
        self.send(record(turn=5, attachments=[attached]))
        self.assertEqual(len(self.fake.chat_files), 2)

    def test_files_over_the_limits_are_left_out(self):
        small = self.file("small.step", b"s" * 80)
        big = self.file("big.step", b"b" * 150)
        mesh = self.file("mesh.stl", b"m" * 150)
        huge = self.file("huge.stl", b"h" * 250)
        empty = self.file("empty.step", b"")
        gone = os.path.join(self.tmp, "gone.step")
        with mock.patch.object(records, "MAX_FILE_BYTES", 100), mock.patch.object(records, "MAX_STL_BYTES", 200):
            self.send(record(attachments=[small, big, empty, gone], outputs=[mesh, huge]))
        self.assertEqual(sorted(f["name"] for f in self.fake.chat_files), ["mesh.stl", "small.step"])
        self.assertEqual(self.recorder.stats["file_skipped"], 4)

    def test_5xx_and_429_are_tried_again(self):
        self.fake.fail(self.turn_path(), 503, method="PUT", times=2)
        self.send()
        self.assertIn((CHAT, 3), self.fake.chat_turns)
        self.assertEqual(self.waits, [1.0, 2.0])
        self.waits.clear()
        self.fake.fail(self.turn_path(), 429, {"type": "error", "error": {"code": "rate_limited", "message": "slow"}},
                       method="PUT", headers={"Retry-After": "7"})
        self.send()
        self.assertEqual(self.waits, [7.0])   # Retry-After, honoured
        self.assertEqual(self.recorder.stats["turn_ok"], 2)

    def test_network_errors_are_tried_again(self):
        self.fake.fail(self.turn_path(), 0, method="PUT")   # the connection drops without an answer
        self.send()
        self.assertIn((CHAT, 3), self.fake.chat_turns)
        self.assertEqual(self.waits, [1.0])

    def test_it_gives_up_after_the_retries_or_a_long_retry_after(self):
        self.fake.fail(self.turn_path(), 502, method="PUT", times=10)
        self.send()
        self.assertEqual(len(self.fake.calls_to(self.turn_path())), 4)
        self.assertEqual(self.waits, [1.0, 2.0, 3.0])
        self.assertEqual(self.recorder.stats["turn_dropped"], 1)
        self.waits.clear()
        self.fake.fail(self.turn_path(5), 429, method="PUT", headers={"Retry-After": "3600"})   # the daily limit
        self.send(record(turn=5))
        self.assertEqual((self.waits, len(self.fake.calls_to(self.turn_path(5)))), ([], 1))

    def test_other_4xx_are_dropped_at_once(self):
        output = self.file("a.step")
        for status in (400, 404, 409, 413, 422):
            self.fake.fail(self.turn_path(), status, method="PUT")
        for _ in range(5):
            self.send(record(outputs=[output]))
        self.assertEqual(len(self.fake.calls_to(self.turn_path())), 5)
        self.assertEqual((self.waits, self.recorder.stats["turn_dropped"]), ([], 5))
        self.assertEqual(len(self.fake.chat_files), 1)   # the turn's files are still sent

    def test_terms_required_drops_the_accounts_records_and_asks_again(self):
        recorder = self.make(autostart=False)
        for turn in (1, 2, 3):
            self.assertTrue(records.record_turn(record(turn=turn, outputs=[self.file("a.step")]), account_profile=True,
                                                workspace=False, state=self.state(), into=recorder))
        self.fake.set_consent(accepted="2026-11-02", current="2026-12-01")   # new Terms the account hasn't accepted
        recorder.start()
        self.assertTrue(recorder.wait_idle(15.0))
        self.assertEqual(self.terms, [self.fake.url])
        self.assertEqual(self.fake.counts["chat_turn"], 1)
        self.assertEqual((self.fake.counts["chat_file"], recorder.stats["dropped_terms"]), (0, 2))
        # The real handler: /v1/me again, and the panel then asks for the new version.
        default = self.make(on_terms_required=None)
        self.send(record(turn=4), recorder=default)
        self.assertTrue(consent.needed(self.session.cached_snapshot()))
        self.assertEqual(consent.Prompt.of(self.session.cached_snapshot()).version, "2026-12-01")

    def test_a_401_renews_the_sign_in(self):
        self.fake.reject_token(self.tokens["access_token"])
        self.send()
        self.assertIn((CHAT, 3), self.fake.chat_turns)
        self.assertEqual(self.fake.counts["token:refresh_token"], 1)
        self.assertTrue(self.session.has_session())

    def test_a_record_of_another_account_is_dropped_unsent(self):
        self.assertTrue(self.recorder.submit(record(), api_url=self.fake.url, account="bob@example.com"))
        self.assertTrue(self.recorder.wait_idle(15.0))
        self.assertEqual((self.fake.counts["chat_turn"], self.recorder.stats["dropped_account"]), (0, 1))
        self.session.sign_out()
        self.assertTrue(self.recorder.submit(record(), api_url=self.fake.url, account="ada@example.com"))
        self.assertTrue(self.recorder.wait_idle(15.0))
        self.assertEqual((self.fake.counts["chat_turn"], self.recorder.stats["dropped_account"]), (0, 2))

    def test_the_queue_is_bounded(self):
        recorder = self.make(autostart=False, queue_jobs=3)
        for turn in range(1, 6):
            self.assertTrue(recorder.submit(record(turn=turn), api_url=self.fake.url, account="ada@example.com"))
        self.assertEqual((recorder.pending(), recorder.stats["dropped_full"]), (3, 2))
        recorder.start()
        self.assertTrue(recorder.wait_idle(15.0))
        self.assertEqual(sorted(t for _c, t in self.fake.chat_turns), [3, 4, 5])   # the newest are kept
        by_size = self.make(autostart=False, queue_chars=1000)
        for turn in range(1, 4):
            by_size.submit(record(turn=turn, prompt="p" * 600), api_url=self.fake.url, account="ada@example.com")
        self.assertEqual(by_size.pending(), 1)
        huge = record(prompt="p" * 10_000_000, reply="r" * 10_000_000)   # held cut to twice the limits
        by_size.submit(huge, api_url=self.fake.url, account="ada@example.com")
        self.assertLessEqual(by_size._queue[-1].chars, 2 * (records.PROMPT_LIMIT + records.REPLY_LIMIT) + 1000)

    def test_a_bug_never_reaches_the_caller_nor_stops_the_thread(self):
        calls = []

        def session_for(api_url):
            calls.append(api_url)
            if len(calls) == 1:
                raise RuntimeError("a bug")
            return session.get_session(api_url)

        recorder = self.make(session_for=session_for)
        self.send(record(turn=1), recorder=recorder)
        self.send(record(turn=2), recorder=recorder)
        self.assertEqual(recorder.stats["errors"], 1)
        self.assertEqual(sorted(t for _c, t in self.fake.chat_turns), [2])


class TurnIndexTest(unittest.TestCase):
    def setUp(self):
        self.sandbox = tempfile.mkdtemp(prefix="chats_", dir=isolation.sandbox_root())
        patcher = mock.patch.object(chats, "chats_dir", lambda: self.sandbox)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_turn_index_is_never_given_twice(self):
        chat = chats.new_session()
        self.assertEqual([chats.next_turn(chat), chats.next_turn(chat)], [0, 1])
        chat["history"].append(Message(role="user", content="hi"))
        chats.save(chat)
        loaded = chats.load(chat["id"])
        self.assertEqual(loaded["turns"], 2)
        loaded["history"].clear()   # a cleared chat counts on
        self.assertEqual(chats.next_turn(loaded), 2)

    def test_a_chat_from_before_the_count_starts_after_its_prompts(self):
        chat = chats.new_session()
        del chat["turns"]
        chat["history"] += [Message(role="user", content="one"), Message(role="assistant", content="ok"),
                            Message(role="user", tool_results=[ToolResult("c1", "get_view", "ok")]),
                            Message(role="user", content="two")]
        self.assertEqual(chats.next_turn(chat), 2)
        self.assertEqual(chats.next_turn(chat), 3)


class FeedbackTest(unittest.TestCase):
    def test_a_rating_names_the_chat_and_the_turn(self):
        facts = rating.TurnFacts(prompt="Make a flange " * 30, response="Built it.", mutated=True)
        body = rating.payload(facts, 4, context={"model": "Origin1", **rating.turn_context(CHAT, 3)})
        self.assertEqual((body["context"]["chat_id"], body["context"]["turn"]), (CHAT, 3))
        for chat_id, turn in ((CHAT, None), (None, 3), ("../x", 3), (CHAT, -1), (CHAT, True)):
            self.assertEqual(rating.turn_context(chat_id, turn), {})


if __name__ == "__main__":
    unittest.main()
