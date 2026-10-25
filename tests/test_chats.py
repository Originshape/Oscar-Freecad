"""Chat session store tests: save/load round-trips (including tool calls and results so restored chats can keep
tool-looping, ``synthetic`` / ``tag`` / ``args_error`` and the ``project_dir`` / ``ledger`` / ``blocks_meta`` keys),
listing order and the metadata-only sidecars, deletion, the persisted-copy trim (oversized images, only the newest
tool-result images, clipped tool results), thread-safe saves, ``persist_round`` and ``recovery_notice``."""

import json
import os
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

from oscar import chats  # noqa: E402
from oscar.providers import Message, ToolCall, ToolResult  # noqa: E402


class ChatsCase(unittest.TestCase):
    def setUp(self):
        self.sandbox = tempfile.mkdtemp(prefix="chats_sandbox_", dir=isolation.sandbox_root())
        self._original = chats.chats_dir
        chats.chats_dir = lambda: self.sandbox

    def tearDown(self):
        chats.chats_dir = self._original

    def files(self):
        return sorted(os.listdir(self.sandbox))

    def result_message(self, n: int, image: str = "aW1n", content: str = "ok") -> Message:
        return Message(role="user", tool_results=[
            ToolResult(tool_call_id=f"c{n}", name="get_view", content=content, image_b64=image)])


class ChatsStoreTest(ChatsCase):
    def _session_with_history(self):
        session = chats.new_session("Bracket task")
        session["history"] = [
            Message(role="user", content="Build a bracket."),
            Message(role="assistant", content="On it.",
                    tool_calls=[ToolCall(id="c1", name="create_object",
                                         arguments={"obj_type": "Part::Box"})]),
            Message(role="user", tool_results=[
                ToolResult(tool_call_id="c1", name="create_object",
                           content="Object 'Box' created.",
                           image_b64="aGVsbG8=")]),
        ]
        return session

    def test_roundtrip_preserves_tool_loop(self):
        session = self._session_with_history()
        chats.save(session)
        loaded = chats.load(session["id"])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["title"], "Bracket task")
        self.assertEqual(len(loaded["history"]), 3)
        assistant = loaded["history"][1]
        self.assertEqual(assistant.tool_calls[0].name, "create_object")
        self.assertEqual(assistant.tool_calls[0].arguments,
                         {"obj_type": "Part::Box"})
        result = loaded["history"][2].tool_results[0]
        self.assertEqual(result.tool_call_id, "c1")
        self.assertEqual(result.image_b64, "aGVsbG8=")

    def test_list_orders_newest_first(self):
        first = chats.new_session("first")
        chats.save(first)
        time.sleep(1.1)  # save() stamps updated=now; keep them distinct
        second = chats.new_session("second")
        chats.save(second)
        titles = [s["title"] for s in chats.list_sessions()]
        self.assertEqual(titles.index("second"), 0)
        self.assertIn("first", titles)

    def test_delete_removes_file(self):
        session = chats.new_session("gone")
        chats.save(session)
        chats.delete(session["id"])
        self.assertIsNone(chats.load(session["id"]))

    def test_oversized_images_dropped_at_save(self):
        session = chats.new_session("big")
        session["history"] = [
            Message(role="user", tool_results=[
                ToolResult(tool_call_id="x", name="get_view", content="ok",
                           image_b64="z" * 2_000_000)]),
        ]
        chats.save(session)
        loaded = chats.load(session["id"])
        self.assertEqual(loaded["history"][0].tool_results[0].image_b64, "")

    def test_title_from_prompt(self):
        self.assertEqual(chats.title_from_prompt("  Build a   Pi case\nnow "),
                         "Build a Pi case now")
        long_prompt = "x" * 100
        title = chats.title_from_prompt(long_prompt)
        self.assertTrue(title.endswith("…"))
        self.assertEqual(len(title), 49)

    def test_title_from_prompt_fallback(self):
        self.assertEqual(chats.title_from_prompt("   "), "New chat")
        self.assertEqual(chats.title_from_prompt("", "Other"), "Other")

    def test_new_session_has_the_new_keys(self):
        session = chats.new_session()
        self.assertEqual((session["title"], session["history"]), ("New chat", []))
        self.assertIsNone(session["project_dir"])
        self.assertIsNone(session["ledger"])
        self.assertEqual(session["blocks_meta"], [])
        self.assertRegex(session["id"], r"^\d{8}-\d{6}-[0-9a-f]{6}$")

    def test_load_of_a_missing_or_corrupt_chat_is_none(self):
        self.assertIsNone(chats.load("nope"))
        with open(os.path.join(self.sandbox, "bad.json"), "w") as f:
            f.write("{ torn")
        self.assertIsNone(chats.load("bad"))

    def test_load_of_a_file_without_an_id_uses_the_file_name(self):
        with open(os.path.join(self.sandbox, "weird.json"), "w") as f:
            json.dump({"title": "no id"}, f)
        loaded = chats.load("weird")
        self.assertEqual((loaded["id"], loaded["title"], loaded["history"]), ("weird", "no id", []))

    def test_load_of_a_file_that_is_not_a_chat_object_is_none(self):
        with open(os.path.join(self.sandbox, "list.json"), "w") as f:
            json.dump([1, 2, 3], f)
        self.assertIsNone(chats.load("list"))


class SyntheticAndTagTest(ChatsCase):
    def test_synthetic_tag_and_args_error_round_trip(self):
        session = chats.new_session("flags")
        session["history"] = [
            Message(role="user", content="[app note] you made no change", synthetic=True, tag="gate"),
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="c1", name="execute_code", arguments={"_raw": '{"code": "x'},
                         args_error="the JSON ends early", raw_args='{"code": "x')]),
            Message(role="user", content="hello"),
        ]
        chats.save(session)
        loaded = chats.load(session["id"])
        first, second, third = loaded["history"]
        self.assertEqual((first.synthetic, first.tag), (True, "gate"))
        self.assertEqual((third.synthetic, third.tag), (False, ""))
        call = second.tool_calls[0]
        self.assertEqual((call.args_error, call.raw_args, call.arguments), ("the JSON ends early", '{"code": "x', {"_raw": '{"code": "x'}))

    def test_every_tag_value_survives(self):
        tags = ("cutoff", "gate", "watchdog", "breaker", "steer", "wrapup", "resume")
        session = chats.new_session("tags")
        session["history"] = [Message(role="user", content=f"note {t}", synthetic=True, tag=t) for t in tags]
        chats.save(session)
        self.assertEqual([m.tag for m in chats.load(session["id"])["history"]], list(tags))

    def test_a_message_dict_without_the_new_keys_is_not_synthetic(self):
        message = chats._message_from_dict({"role": "user", "content": "old chat"})
        self.assertEqual((message.synthetic, message.tag), (False, ""))
        self.assertEqual(message.tool_calls, [])

    def test_unknown_keys_are_ignored(self):
        message = chats._message_from_dict({
            "role": "assistant", "content": "x", "from_the_future": 1,
            "tool_calls": [{"id": "c", "name": "n", "arguments": {}, "novel": True}],
            "tool_results": [{"tool_call_id": "c", "name": "n", "content": "r", "extra": 1}]})
        self.assertEqual(message.tool_calls[0].name, "n")
        self.assertEqual(message.tool_results[0].content, "r")

    def test_a_none_tag_is_an_empty_tag(self):
        self.assertEqual(chats._message_from_dict({"role": "user", "tag": None}).tag, "")

    def test_old_files_load_unchanged(self):
        old = {"id": "20260101-000000-abcdef", "title": "Old", "created": "c", "updated": "u",
               "messages": [{"role": "user", "content": "hi", "tool_calls": [], "tool_results": [], "images": [],
                             "provider_blocks": []}]}
        with open(os.path.join(self.sandbox, "20260101-000000-abcdef.json"), "w") as f:
            json.dump(old, f)
        loaded = chats.load("20260101-000000-abcdef")
        self.assertEqual(loaded["history"][0].content, "hi")
        self.assertEqual((loaded["project_dir"], loaded["ledger"], loaded["blocks_meta"]), (None, None, []))


class SessionKeysTest(ChatsCase):
    def test_project_dir_ledger_and_blocks_meta_round_trip(self):
        session = chats.new_session("keys")
        session["project_dir"] = "/Users/someone/Documents/Oscar/Projects/keys-abc123"
        session["ledger"] = {"intent": "enclosure", "steps": [{"id": "S1", "title": "Shell"}], "revision": 3}
        session["blocks_meta"] = [{"kind": "card", "title": "Plan"}, {"kind": "image", "path": "snapshots/001_a.png"}]
        chats.save(session)
        loaded = chats.load(session["id"])
        self.assertEqual(loaded["project_dir"], session["project_dir"])
        self.assertEqual(loaded["ledger"], session["ledger"])
        self.assertEqual(loaded["blocks_meta"], session["blocks_meta"])

    def test_unset_keys_are_not_written_and_load_as_none(self):
        session = chats.new_session("plain")
        chats.save(session)
        raw = json.load(open(os.path.join(self.sandbox, session["id"] + ".json"), encoding="utf-8"))
        for key in ("project_dir", "ledger", "blocks_meta"):
            self.assertNotIn(key, raw)
        loaded = chats.load(session["id"])
        self.assertEqual((loaded["project_dir"], loaded["ledger"], loaded["blocks_meta"]), (None, None, []))

    def test_a_cleared_key_is_removed_on_the_next_save(self):
        session = chats.new_session("clear")
        session["project_dir"] = "/p"
        chats.save(session)
        session["project_dir"] = None
        chats.save(session)
        self.assertIsNone(chats.load(session["id"])["project_dir"])

    def test_a_name_the_user_gave_is_remembered(self):
        session = chats.new_session("Workspace: Box")
        chats.save(session)
        self.assertFalse(chats.load(session["id"])["named"])
        self.assertNotIn("named", json.load(open(os.path.join(self.sandbox, session["id"] + ".json"), encoding="utf-8")))
        session["title"], session["named"] = "Lid for the box", True
        chats.save(session)
        loaded = chats.load(session["id"])
        self.assertEqual((loaded["title"], loaded["named"]), ("Lid for the box", True))

    def test_unicode_titles_and_text_round_trip(self):
        session = chats.new_session("Gehäuse für den Pi")
        session["history"] = [Message(role="user", content="Größe 85 × 56 mm ✓")]
        chats.save(session)
        loaded = chats.load(session["id"])
        self.assertEqual(loaded["title"], "Gehäuse für den Pi")
        self.assertEqual(loaded["history"][0].content, "Größe 85 × 56 mm ✓")


class PersistedCopyTrimTest(ChatsCase):
    def saved_images(self, session):
        chats.save(session)
        loaded = chats.load(session["id"])
        return [m.tool_results[0].image_b64 for m in loaded["history"]]

    def test_only_the_newest_four_tool_result_images_are_written(self):
        session = chats.new_session("images")
        session["history"] = [self.result_message(n, image=f"img{n}") for n in range(7)]
        self.assertEqual(self.saved_images(session), ["", "", "", "img3", "img4", "img5", "img6"])

    def test_the_in_memory_history_is_never_touched(self):
        session = chats.new_session("untouched")
        long_text = "r" * 30_000
        session["history"] = [self.result_message(n, image=f"img{n}", content=long_text) for n in range(6)]
        chats.save(session)
        self.assertEqual([m.tool_results[0].image_b64 for m in session["history"]], [f"img{n}" for n in range(6)])
        self.assertTrue(all(m.tool_results[0].content == long_text for m in session["history"]))

    def test_messages_without_images_do_not_use_up_the_budget(self):
        session = chats.new_session("mixed")
        session["history"] = [self.result_message(0, image="a"), self.result_message(1, image=""),
                              self.result_message(2, image="c"), self.result_message(3, image=""),
                              self.result_message(4, image="e"), self.result_message(5, image="f"),
                              self.result_message(6, image="g")]
        self.assertEqual(self.saved_images(session), ["", "", "c", "", "e", "f", "g"])

    def test_the_oversize_rule_applies_before_the_count(self):
        session = chats.new_session("oversize")
        session["history"] = [self.result_message(0, image="small0"), self.result_message(1, image="z" * 2_000_000)]
        images = self.saved_images(session)
        self.assertEqual(images, ["small0", ""])

    def test_long_tool_results_are_clipped_in_the_file_only(self):
        session = chats.new_session("clip")
        content = "A" * 8000 + "M" * 5000 + "Z" * 3500
        session["history"] = [self.result_message(0, image="", content=content)]
        chats.save(session)
        saved = chats.load(session["id"])["history"][0].tool_results[0].content
        self.assertTrue(saved.startswith("A" * 8000))
        self.assertTrue(saved.endswith("Z" * 3500))
        self.assertIn("5000 chars omitted from the saved chat copy", saved)
        self.assertLess(len(saved), len(content))
        self.assertNotIn("M" * 10, saved)

    def test_the_clip_boundary_is_12000_characters(self):
        session = chats.new_session("boundary")
        session["history"] = [self.result_message(0, image="", content="x" * 12000),
                              self.result_message(1, image="", content="y" * 12001)]
        chats.save(session)
        loaded = chats.load(session["id"])["history"]
        self.assertEqual(loaded[0].tool_results[0].content, "x" * 12000)
        self.assertIn("omitted", loaded[1].tool_results[0].content)

    def test_user_attachment_images_stay_below_the_cap_and_oversized_ones_are_dropped(self):
        session = chats.new_session("attach")
        session["history"] = [Message(role="user", content="see", images=["small", "z" * 2_000_000, "other"])]
        chats.save(session)
        self.assertEqual(chats.load(session["id"])["history"][0].images, ["small", "other"])

    def test_message_to_dict_defaults_keep_the_historic_behaviour(self):
        message = self.result_message(0, image="img", content="c" * 20_000)
        plain = chats._message_to_dict(message)
        self.assertEqual(plain["tool_results"][0]["image_b64"], "img")
        self.assertEqual(len(plain["tool_results"][0]["content"]), 20_000)
        trimmed = chats._message_to_dict(message, keep_result_images=False, clip_results=True)
        self.assertEqual(trimmed["tool_results"][0]["image_b64"], "")
        self.assertLess(len(trimmed["tool_results"][0]["content"]), 20_000)
        self.assertEqual(message.tool_results[0].image_b64, "img")   # a copy: the message is untouched

    def test_the_chat_file_does_not_grow_without_bound(self):
        session = chats.new_session("growth")
        for n in range(60):
            session["history"].append(self.result_message(n, image="I" * 100_000, content="c" * 50_000))
        chats.save(session)
        size = os.path.getsize(os.path.join(self.sandbox, session["id"] + ".json"))
        self.assertLess(size, 4 * 100_000 + 60 * 13_000 + 60_000)


class ThreadSafetyTest(ChatsCase):
    def test_concurrent_saves_and_persist_rounds_never_corrupt_the_file(self):
        session = chats.new_session("busy")
        session["history"] = [Message(role="user", content="start")]
        errors, stop = [], threading.Event()

        def appender():
            n = 0
            while not stop.is_set():
                session["history"].append(Message(role="assistant", content=f"m{n}"))   # the worker mutates mid-turn
                n += 1
                time.sleep(0.001)

        def saver(fn):
            try:
                for _ in range(40):
                    fn(session)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=saver, args=(chats.save,)) for _ in range(3)]
        threads += [threading.Thread(target=saver, args=(chats.persist_round,)) for _ in range(3)]
        mutator = threading.Thread(target=appender)
        mutator.start()
        [t.start() for t in threads]
        [t.join() for t in threads]
        stop.set()
        mutator.join()
        self.assertEqual(errors, [])
        loaded = chats.load(session["id"])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["history"][0].content, "start")
        self.assertEqual(self.files(), [session["id"] + ".json", session["id"] + ".meta.json"])   # no stray temp files

    def test_every_write_uses_a_unique_temp_name_in_the_chats_folder(self):
        session = chats.new_session("tmpnames")
        seen = []
        real = tempfile.mkstemp

        def spy(*args, **kwargs):
            fd, path = real(*args, **kwargs)
            seen.append((os.path.dirname(path), os.path.basename(path)))
            return fd, path

        with mock.patch("oscar.chats.tempfile.mkstemp", spy):
            chats.save(session)
            chats.save(session)
        self.assertEqual({d for d, _ in seen}, {self.sandbox})
        self.assertEqual(len({n for _, n in seen}), len(seen))   # all distinct
        self.assertTrue(all(n.endswith(".tmp") for _, n in seen))
        self.assertFalse(any(n == session["id"] + ".json.tmp" for _, n in seen))   # not the old fixed name

    def test_a_failed_replace_leaves_the_old_file_and_no_temp_file(self):
        session = chats.new_session("atomic")
        session["history"] = [Message(role="user", content="v1")]
        chats.save(session)
        session["history"] = [Message(role="user", content="v2")]
        with mock.patch("oscar.chats.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                chats.save(session)
        self.assertEqual(chats.load(session["id"])["history"][0].content, "v1")
        self.assertEqual(self.files(), [session["id"] + ".json", session["id"] + ".meta.json"])

    def test_the_save_lock_is_not_held_after_a_failure(self):
        session = chats.new_session("lock")
        with mock.patch("oscar.chats.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                chats.save(session)
        done = []
        thread = threading.Thread(target=lambda: (chats.save(session), done.append(True)))
        thread.start()
        thread.join(5)
        self.assertEqual(done, [True])


class PersistRoundTest(ChatsCase):
    def test_persist_round_writes_the_same_file_as_save(self):
        session = chats.new_session("rounds")
        session["history"] = [Message(role="user", content="go")]
        chats.persist_round(session)
        self.assertEqual(chats.load(session["id"])["history"][0].content, "go")
        session["history"].append(Message(role="assistant", content="working", tool_calls=[
            ToolCall(id="c1", name="create_object", arguments={"a": 1})]))
        chats.persist_round(session)
        loaded = chats.load(session["id"])
        self.assertEqual(len(loaded["history"]), 2)
        self.assertEqual(loaded["history"][1].tool_calls[0].id, "c1")

    def test_persist_round_applies_the_image_rules(self):
        session = chats.new_session("rounds-images")
        session["history"] = [self.result_message(n, image=f"img{n}") for n in range(6)]
        session["history"].append(self.result_message(6, image="z" * 2_000_000))
        chats.persist_round(session)
        images = [m.tool_results[0].image_b64 for m in chats.load(session["id"])["history"]]
        self.assertEqual(images[-1], "")
        self.assertEqual([i for i in images if i], ["img3", "img4", "img5"])

    def test_persist_round_keeps_the_sidecar_in_step(self):
        session = chats.new_session("rounds-meta")
        session["history"] = [Message(role="user", content="a"), Message(role="assistant", content="b")]
        chats.persist_round(session)
        meta = json.load(open(os.path.join(self.sandbox, session["id"] + ".meta.json"), encoding="utf-8"))
        self.assertEqual((meta["id"], meta["title"], meta["messages"]), (session["id"], "rounds-meta", 2))


class PersistRoundContractTest(ChatsCase):
    """How the worker and the panel use ``persist_round``: the worker calls ``on_round(history)`` on its own thread and the
    panel's adapter is ``lambda history: chats.persist_round(session)`` (``history`` IS ``session["history"]``)."""

    def test_the_signature_takes_the_session_and_nothing_else(self):
        import inspect

        self.assertEqual(list(inspect.signature(chats.persist_round).parameters), ["session"])
        self.assertEqual(list(inspect.signature(chats.save).parameters), ["session"])

    def test_the_on_round_adapter_persists_the_whole_turn_state(self):
        session = chats.new_session("worker turn")
        session["project_dir"] = "/abs/projects/worker-turn-abc123"
        session["ledger"] = {"intent": "enclosure", "steps": [{"id": "S1", "status": "done"}]}
        session["blocks_meta"] = [{"kind": "user", "text": "build"}]
        history = session["history"]
        on_round = lambda h: chats.persist_round(session)   # noqa: E731 - exactly what the panel passes
        history.append(Message(role="user", content="build a case"))
        on_round(history)
        history.append(Message(role="assistant", content="", tool_calls=[
            ToolCall(id="c1", name="build_enclosure", arguments={"stage": "next"})]))
        on_round(history)
        history.append(Message(role="user", tool_results=[
            ToolResult(tool_call_id="c1", name="build_enclosure", content="Stage 'shell' built (3 of 9).")]))
        history.append(Message(role="user", content="The plan has unfinished steps.", synthetic=True, tag="gate"))
        on_round(history)
        loaded = chats.load(session["id"])
        self.assertEqual([m.role for m in loaded["history"]], ["user", "assistant", "user", "user"])
        self.assertEqual((loaded["history"][3].synthetic, loaded["history"][3].tag), (True, "gate"))
        self.assertEqual(loaded["history"][1].tool_calls[0].arguments, {"stage": "next"})
        self.assertEqual(loaded["project_dir"], "/abs/projects/worker-turn-abc123")
        self.assertEqual(loaded["ledger"]["steps"][0]["status"], "done")
        self.assertEqual(loaded["blocks_meta"], [{"kind": "user", "text": "build"}])
        self.assertEqual(self.files(), [session["id"] + ".json", session["id"] + ".meta.json"])

    def test_session_keys_the_panel_adds_do_not_leak_into_the_file(self):
        session = chats.new_session("panel keys")
        session["history"] = [Message(role="user", content="x")]
        session["_project"] = object()      # a live object the panel keeps next to the chat data
        session["blocks"] = [object()]
        chats.persist_round(session)
        with open(os.path.join(self.sandbox, session["id"] + ".json"), encoding="utf-8") as f:
            stored = json.load(f)
        self.assertEqual(sorted(stored), ["created", "id", "messages", "title", "updated"])

    def test_the_history_instead_of_the_session_is_refused_with_the_fix_in_the_message(self):
        session = chats.new_session("wrong arg")
        session["history"] = [Message(role="user", content="x")]
        for bad in (session["history"], None, "id", {"id": "x", "title": "t"}):
            with self.assertRaises(TypeError, msg=repr(bad)) as caught:
                chats.persist_round(bad)
            self.assertIn("on_round=lambda history: chats.persist_round(session)", str(caught.exception))
        with self.assertRaises(TypeError):
            chats.save(session["history"])
        self.assertEqual(self.files(), [], "a refused call writes nothing")

    def test_a_changed_size_error_is_retried_once_and_other_failures_propagate(self):
        session = chats.new_session("retry")
        session["history"] = [Message(role="user", content="x")]
        with mock.patch.object(chats, "save", side_effect=[RuntimeError("dictionary changed size during iteration"), None]) as save:
            chats.persist_round(session)
        self.assertEqual(save.call_count, 2)
        with mock.patch.object(chats, "save", side_effect=RuntimeError("changed size")) as save:
            with self.assertRaises(RuntimeError):
                chats.persist_round(session)
        self.assertEqual(save.call_count, 2, "one retry, not a loop")
        with mock.patch.object(chats, "save", side_effect=OSError("disk full")) as save:
            with self.assertRaises(OSError):
                chats.persist_round(session)
        self.assertEqual(save.call_count, 1)

    def test_an_older_snapshot_is_never_written_after_a_newer_one(self):
        """The worker's persist_round and the GUI thread's save run at the same time: the lock must cover serialize + write,
        otherwise a save that serialized EARLIER can be written LATER and the file goes back in time."""
        session = chats.new_session("order")
        session["history"] = [Message(role="user", content="one")]
        entered, release = threading.Event(), threading.Event()
        state = {"first": True}
        real = chats._message_to_dict

        def slow(message, **kwargs):
            if threading.current_thread().name == "slow-saver" and state["first"]:
                state["first"] = False
                entered.set()
                release.wait(10)
            return real(message, **kwargs)

        with mock.patch.object(chats, "_message_to_dict", slow):
            older = threading.Thread(target=chats.persist_round, args=(session,), name="slow-saver")
            older.start()
            self.assertTrue(entered.wait(10), "the first save never started serializing")
            session["history"].append(Message(role="assistant", content="two"))
            newer = threading.Thread(target=chats.save, args=(session,), name="gui-saver")
            newer.start()
            time.sleep(0.3)           # without the lock the newer save is finished (and written) by now
            release.set()
            older.join(10)
            newer.join(10)
        self.assertFalse(older.is_alive() or newer.is_alive())
        loaded = chats.load(session["id"])
        self.assertEqual([m.content for m in loaded["history"]], ["one", "two"])
        self.assertEqual(json.load(open(os.path.join(self.sandbox, session["id"] + ".meta.json"), encoding="utf-8"))["messages"], 2)


class RecoveryNoticeTest(ChatsCase):
    def session(self, *messages):
        session = chats.new_session("recovery")
        session["history"] = list(messages)
        return session

    def test_a_clean_history_has_no_notice(self):
        self.assertEqual(chats.recovery_notice(chats.new_session("empty")), "")
        self.assertEqual(chats.recovery_notice({"history": None}), "")
        self.assertEqual(chats.recovery_notice({}), "")
        self.assertEqual(chats.recovery_notice(self.session(Message(role="user", content="hi"))), "")
        self.assertEqual(chats.recovery_notice(self.session(Message(role="user", content="hi"),
                                                            Message(role="assistant", content="done"))), "")

    def test_calls_that_never_reported_back_give_a_notice(self):
        notice = chats.recovery_notice(self.session(
            Message(role="user", content="build"),
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="a", name="create_object", arguments={}), ToolCall(id="b", name="create_object", arguments={}),
                ToolCall(id="c", name="execute_code", arguments={})])))
        self.assertIn("interrupted", notice)
        self.assertIn("3 tool call(s)", notice)
        self.assertIn("create_object, execute_code", notice)   # each name once
        self.assertIn("'continue'", notice)

    def test_results_after_the_calls_clear_the_notice(self):
        history = [Message(role="assistant", content="", tool_calls=[ToolCall(id="a", name="create_object", arguments={})]),
                   Message(role="user", tool_results=[ToolResult(tool_call_id="a", name="create_object", content="ok")])]
        self.assertEqual(chats.recovery_notice(self.session(*history)), "")
        self.assertNotEqual(chats.recovery_notice(self.session(history[0])), "")

    def test_a_restored_session_reports_the_same_notice(self):
        session = self.session(Message(role="assistant", content="", tool_calls=[ToolCall(id="a", name="build_enclosure", arguments={})]))
        chats.save(session)
        self.assertEqual(chats.recovery_notice(chats.load(session["id"])), chats.recovery_notice(session))


class SidecarListingTest(ChatsCase):
    def sidecar(self, session):
        return os.path.join(self.sandbox, session["id"] + ".meta.json")

    def test_save_writes_the_sidecar(self):
        session = chats.new_session("meta")
        session["history"] = [Message(role="user", content="a")] * 3
        chats.save(session)
        meta = json.load(open(self.sidecar(session), encoding="utf-8"))
        self.assertEqual(sorted(meta), ["id", "messages", "title", "updated"])
        self.assertEqual((meta["id"], meta["title"], meta["messages"]), (session["id"], "meta", 3))
        self.assertEqual(meta["updated"], session["updated"])

    def test_list_sessions_reads_the_sidecars_not_the_chat_files(self):
        session = chats.new_session("fast")
        chats.save(session)
        chat_path = os.path.join(self.sandbox, session["id"] + ".json")
        with open(chat_path, "w") as f:
            f.write("{ this would not parse")
        old = time.time() - 100
        os.utime(chat_path, (old, old))   # the sidecar is the newer file: it is trusted
        listed = chats.list_sessions()
        self.assertEqual([s["title"] for s in listed], ["fast"])
        self.assertEqual(sorted(listed[0]), ["id", "messages", "title", "updated"])

    def test_an_old_chat_without_a_sidecar_is_parsed_once(self):
        session = chats.new_session("legacy")
        session["history"] = [Message(role="user", content="a")]
        chats.save(session)
        os.remove(self.sidecar(session))
        with mock.patch.object(chats, "_meta_from_chat_file", wraps=chats._meta_from_chat_file) as parse:
            first = chats.list_sessions()
            second = chats.list_sessions()
        self.assertEqual(parse.call_count, 1)
        self.assertEqual((first[0]["title"], first[0]["messages"]), ("legacy", 1))
        self.assertEqual(first, second)
        self.assertTrue(os.path.isfile(self.sidecar(session)))

    def test_a_chat_newer_than_its_sidecar_is_parsed_again(self):
        session = chats.new_session("stale")
        chats.save(session)
        old = time.time() - 100
        os.utime(self.sidecar(session), (old, old))
        meta = json.load(open(self.sidecar(session)))
        meta["title"] = "outdated title"
        json.dump(meta, open(self.sidecar(session), "w"))
        os.utime(self.sidecar(session), (old, old))
        self.assertEqual(chats.list_sessions()[0]["title"], "stale")

    def test_a_corrupt_sidecar_is_rebuilt(self):
        session = chats.new_session("heal")
        chats.save(session)
        with open(self.sidecar(session), "w") as f:
            f.write("{ torn")
        self.assertEqual(chats.list_sessions()[0]["title"], "heal")
        json.load(open(self.sidecar(session)))   # valid again

    def test_the_sidecar_never_shows_up_as_a_chat(self):
        chats.save(chats.new_session("only one"))
        self.assertEqual(len(chats.list_sessions()), 1)

    def test_unreadable_chat_files_are_skipped(self):
        chats.save(chats.new_session("good"))
        with open(os.path.join(self.sandbox, "broken.json"), "w") as f:
            f.write("{ torn")
        self.assertEqual([s["title"] for s in chats.list_sessions()], ["good"])

    def test_a_chat_whose_messages_are_all_unreadable_is_not_loaded_as_an_empty_chat(self):
        for name, text in (("allbad", '{"title":"x","messages":["hi", 5, null]}'),
                           ("badcall", '{"title":"x","messages":[{"role":"assistant","tool_calls":[{"name":"n"}]}]}')):
            with open(os.path.join(self.sandbox, name + ".json"), "w") as f:
                f.write(text)
            self.assertIsNone(chats.load(name), name)
        with open(os.path.join(self.sandbox, "somegood.json"), "w") as f:
            f.write('{"title":"x","messages":["hi",{"role":"user","content":"kept"}]}')
        self.assertEqual([m.content for m in chats.load("somegood")["history"]], ["kept"])
        with open(os.path.join(self.sandbox, "empty.json"), "w") as f:
            f.write('{"title":"x","messages":[]}')
        self.assertEqual(chats.load("empty")["history"], [])

    def test_is_unused_only_for_a_chat_with_nothing_in_it(self):
        empty = chats.new_session("New chat")
        chats.save(empty)
        self.assertTrue(chats.is_unused(empty["id"]))
        used = chats.new_session("used")
        used["history"] = [Message(role="user", content="hi")]
        chats.save(used)
        self.assertFalse(chats.is_unused(used["id"]))
        planned = chats.new_session("planned")
        planned["ledger"] = {"v": 1}
        chats.save(planned)
        self.assertFalse(chats.is_unused(planned["id"]))                # a plan is worth keeping
        recorded = chats.new_session("recorded")
        recorded["blocks_meta"] = [{"v": 1}]
        chats.save(recorded)
        self.assertFalse(chats.is_unused(recorded["id"]))               # so are turn records

    def test_is_unused_never_guesses(self):
        self.assertFalse(chats.is_unused("never-existed"))
        self.assertFalse(chats.is_unused("../escape"))
        self.assertFalse(chats.is_unused(None))
        for name, text in (("torn", "{ torn"), ("list", "[]"), ("damaged", '{"messages": ["x"]}'),
                           ("dict", '{"messages": {}}')):
            with open(os.path.join(self.sandbox, name + ".json"), "w") as f:
                f.write(text)
            self.assertFalse(chats.is_unused(name), name)               # unreadable / damaged / not empty: kept
        with open(os.path.join(self.sandbox, "nomsg.json"), "w") as f:
            f.write('{"title": "x"}')
        self.assertTrue(chats.is_unused("nomsg"))                       # no messages at all: nothing to lose

    def test_the_limit_applies(self):
        for n in range(5):
            chats.save(chats.new_session(f"chat {n}"))
        self.assertEqual(len(chats.list_sessions(limit=3)), 3)
        self.assertEqual(len(chats.list_sessions()), 5)

    def test_delete_removes_both_files_and_ignores_missing_ones(self):
        session = chats.new_session("bye")
        chats.save(session)
        self.assertEqual(len(self.files()), 2)
        chats.delete(session["id"])
        self.assertEqual(self.files(), [])
        chats.delete(session["id"])   # a second delete is harmless
        chats.delete("never-existed")
        self.assertEqual(chats.list_sessions(), [])

    def test_a_missing_chats_folder_lists_nothing(self):
        gone = os.path.join(self.sandbox, "not-created")
        chats.chats_dir = lambda: gone
        self.assertEqual(chats.list_sessions(), [])


if __name__ == "__main__":
    unittest.main()
