"""The panel and the OriginShape account, against the fake OriginShape API (tests/support/fake_originshape.py).

Covered: the header badge (hidden for your own provider while signed out; *Sign in* / *42.5 credits* / *0 credits* in
the warning colour / *Waitlist*; its menu; signing out from it), refreshed from the process-wide ``AccountHub`` with
no network on the GUI thread (coalesced, and never undone by a late answer); the onboarding card of the empty chat;
the send gate (local facts only: no turn for the account profile without a sign-in, or a fresh waitlisted snapshot;
an unknown sign-in or a stale snapshot lets the turn go; the typed text stays); a failed turn's ``account_problem``
(the card replaces the raw provider error; when the very first request failed the message goes back to the input,
with its attachments and the plan before it); the ``oscar://account/*`` and ``oscar://settings`` links (pages from the
account, never from the link); Settings closing after a sign-in; the Sign in / Account commands; ``session_id``.
Needs FreeCAD and Qt (offscreen); the keystore runs in memory.
"""

import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

try:
    from PySide import QtWidgets  # noqa: E402

    from oscar import chats, commands, netguard, panel, params, settings_ui, updates  # noqa: E402
    from oscar.auth import account, discovery, keystore, session  # noqa: E402
    from oscar.auth.transport import TokenSet  # noqa: E402
    from oscar.panel import Block  # noqa: E402
    from oscar.params import Profile  # noqa: E402
    from support.fake_originshape import FakeOriginShape  # noqa: E402
    _IMPORT_ERROR = None
except ImportError as e:  # FreeCAD / PySide missing: the whole module skips
    _IMPORT_ERROR = e


class _Sig:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


class RecordingWorker:
    """Stands in for ``AgentWorker`` in ``_send``: records the constructor keywords, runs nothing."""

    instances: list = []

    def __init__(self, history, parent=None, **kw):
        self.history, self.kw = history, kw
        for name in ("text_delta", "reasoning_delta", "tool_started", "tool_finished", "turn_finished", "notice",
                     "flow_paused", "turn_outcome", "account_problem"):
            setattr(self, name, _Sig())
        type(self).instances.append(self)

    def start(self):
        pass

    def wait(self):
        return True

    def deleteLater(self):
        pass

    def stop(self):
        pass

    def resume(self):
        pass


def _own(name="Mine", **over):
    values = {"provider": "openai_compat", "base_url": "http://llm.example.test/v1", "model": "m-own", "api_key": "k"}
    values.update(over)
    return Profile(name=name, **values)


class _PanelCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # never open a window on the user's screen
            cls.app = QtWidgets.QApplication(sys.argv[:1])

    def setUp(self):
        backup = params.backup_state()
        self.addCleanup(params.restore_state, backup)
        for reset in (discovery.clear_cache, session.reset_sessions, keystore.reset_memory, account.clear_model_cache,
                      updates.clear_cache):
            reset()
            self.addCleanup(reset)
        self.addCleanup(setattr, session, "on_change", session.on_change)
        self.addCleanup(setattr, netguard, "on_permission_wait", netguard.on_permission_wait)
        settings_ui.reset_account_hub()
        self.addCleanup(settings_ui.reset_account_hub)
        self.fake = FakeOriginShape()
        self.addCleanup(self.fake.close)
        env = mock.patch.dict(os.environ, {"OSCAR_API_URL": self.fake.url, keystore.ENV_STORE: "memory"})
        env.start()
        self.addCleanup(env.stop)
        self.tmp = tempfile.mkdtemp(prefix="oscar_account_panel_")
        for patcher in (mock.patch.object(chats, "chats_dir", lambda: self.tmp),
                        mock.patch.object(panel, "AgentWorker", RecordingWorker),
                        mock.patch("oscar.cad.get_engine", return_value=mock.Mock(**{
                            "begin_turn_snapshot.return_value": {"success": True, "docs": {}}}))):
            patcher.start()
            self.addCleanup(patcher.stop)
        RecordingWorker.instances = []
        self.use([params.account_profile()])

    def use(self, profiles, active=None):
        params.save_profiles(profiles)
        params.set_active_profile(active or profiles[0].name)

    def panel(self):
        p = panel.OscarPanel()

        def dispose():
            p._status_timer.stop()
            p._render_timer.stop()
            p.worker = None
            p.deleteLater()

        self.addCleanup(dispose)
        return p

    def wait(self, predicate, timeout=15.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            QtWidgets.QApplication.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        QtWidgets.QApplication.processEvents()
        return predicate()

    def sign_in(self) -> None:
        data = self.fake.issue_tokens()
        session.get_session(self.fake.url).complete_sign_in(
            TokenSet(data["access_token"], data["refresh_token"], data["expires_in"], data["scope"]))

    def signed_in_panel(self, badge="42.5 credits"):
        self.sign_in()
        p = self.panel()
        self.assertTrue(self.wait(lambda: p.account_btn.text() == badge), p.account_btn.text())
        return p

    @staticmethod
    def menu_labels(p) -> list[str]:
        return [a.text() for a in p.account_menu.actions() if not a.isSeparator()]

    @staticmethod
    def onboarding(p) -> list:
        return [b for b in p.blocks if b.kind == Block.ACCOUNT and b.detail == "onboarding"]

    @staticmethod
    def accounts(p, kind=None) -> list:
        return [b for b in p.blocks if b.kind == Block.ACCOUNT and (kind is None or b.detail == kind)]


class BadgeTest(_PanelCase):
    def test_your_own_provider_signed_out_shows_no_badge_and_asks_no_server(self):
        self.use([_own()])
        p = self.panel()
        self.assertTrue(p.account_btn.isHidden())
        self.assertFalse(p.subtitle.isHidden())
        self.wait(lambda: False, timeout=0.2)
        self.assertEqual(self.fake.calls, [])

    def test_a_newer_oscar_shows_without_the_badge_and_is_asked_about_once_a_day(self):
        # Your own provider, never signed in: the public update check still says a newer Oscar is out.
        self.use([_own()])
        self.fake.plugin = {"latest_version": "9.0.0", "min_version": "0.1.0", "download_url": ""}
        with mock.patch.dict(os.environ, {updates.ENV: "1"}):
            p = self.panel()
            self.assertTrue(self.wait(lambda: "9.0.0" in p.subtitle.text()), p.subtitle.text())
            self.assertIn("oscar://account/download", p.subtitle.text())
            self.assertTrue(p.account_btn.isHidden())
            self.assertEqual(settings_ui.account_hub().links()["download"], self.fake.url + "/download")
            p._refresh_account()                                    # the next look asks nothing more today
            self.wait(lambda: False, timeout=0.2)
        self.assertEqual([c.path for c in self.fake.calls], ["/v1/plugin"])   # and no sign-in was read for it

    def test_the_account_profile_signed_out_reads_sign_in(self):
        p = self.panel()
        self.assertEqual(p.account_btn.objectName(), "oscarAccountBadge")
        self.assertFalse(p.account_btn.isHidden())
        self.assertTrue(p.subtitle.isHidden())                    # the 280 px dock has no room for both
        self.assertEqual(p.account_btn.text(), "Sign in")
        self.assertEqual(self.menu_labels(p), ["Sign in with browser…", "Sign in with a code…"])
        self.assertEqual(self.fake.calls, [])

    def test_the_model_label_names_the_account_model_never_its_alias(self):
        for model, shown in (("origin1", "Origin1"), ("origin1-super", "Origin1 Super"), ("oscar-default", "Origin1 Super"),
                             ("oscar-fast", "Origin1")):
            with self.subTest(model=model):
                self.use([Profile(**{**vars(params.account_profile()), "model": model}), _own()])
                p = self.panel()
                self.assertEqual(p.model_label.text(), shown)
                p.profile_combo.setCurrentIndex(1)                  # your own provider: its model id as it is
                self.assertEqual(p.model_label.text(), "m-own")

    def test_signed_in_it_shows_the_credits_and_the_account_menu(self):
        p = self.signed_in_panel()
        self.assertIn("ada@example.com", p.account_btn.toolTip())
        self.assertIn("Trial plan", p.account_btn.toolTip())
        labels = self.menu_labels(p)
        self.assertEqual(labels[0], "Signed in as ada@example.com")
        self.assertFalse(p.account_menu.actions()[0].isEnabled())
        for label in ("Choose a plan…", "Manage account…", "Refresh", "Sign out"):   # a trial: packs need a plan
            self.assertIn(label, labels)
        self.assertNotIn("Buy credits…", labels)
        self.assertEqual(p.account_btn.styleSheet(), "")

    def test_no_credits_left_is_a_warning(self):
        self.fake.set_account(balance=0)
        p = self.signed_in_panel("0 credits")
        self.assertIn("#d9902b", p.account_btn.styleSheet())

    def test_a_waitlisted_account(self):
        self.fake.set_account(status="waitlisted")
        p = self.signed_in_panel("Waitlist")
        self.assertIn("Your account is on the waitlist", self.menu_labels(p))

    def test_a_signed_in_user_of_another_provider_sees_it_too(self):
        self.use([_own()])
        p = self.signed_in_panel()
        self.assertFalse(p.account_btn.isHidden())

    def test_signing_out_from_the_menu(self):
        p = self.signed_in_panel()
        action = next(a for a in p.account_menu.actions() if a.objectName() == "oscarAccount_signout")
        action.trigger()
        self.assertTrue(self.wait(lambda: p.account_btn.text() == "Sign in"))
        self.assertEqual(self.fake.counts["revoke"], 1)
        self.assertFalse(session.get_session(self.fake.url).has_session())

    def test_the_end_of_an_account_turn_asks_for_the_credits_again(self):
        p = self.signed_in_panel()
        before = self.fake.counts["me"]
        p.input.setPlainText("make a cube")
        p._send()
        p._on_turn_outcome("answered", "")
        p._on_turn_finished(True, "Done.")
        self.assertTrue(self.wait(lambda: self.fake.counts["me"] > before))

    def test_the_credits_follow_the_spend_during_an_account_turn_at_most_every_5_s(self):
        p = self.signed_in_panel()
        p.input.setPlainText("make a cube")
        p._send()
        before = self.fake.counts["me"]
        p._on_tool_started("make_box")
        p._on_tool_finished("make_box done", True, "")
        self.assertTrue(self.wait(lambda: self.fake.counts["me"] > before))
        after_first = self.fake.counts["me"]
        p._on_tool_started("fillet")
        p._on_tool_finished("fillet done", True, "")   # within 5 s of the last one: no new fetch
        self.assertFalse(self.wait(lambda: self.fake.counts["me"] > after_first, timeout=0.5))


class HubTest(_PanelCase):
    def test_refreshes_are_coalesced_into_one_fetch_in_flight(self):
        self.sign_in()
        hub = settings_ui.account_hub()
        hub.refresh()
        self.assertTrue(self.wait(lambda: bool(hub.state["snapshot"])))
        before = self.fake.counts["me"]
        self.fake.delays = {"/v1/me": 0.4}
        for _ in range(5):
            hub.refresh(force=True)
        self.assertTrue(self.wait(lambda: hub._thread is None and hub._again is None and
                                  self.fake.counts["me"] >= before + 1))
        self.wait(lambda: False, timeout=0.6)
        self.assertLessEqual(self.fake.counts["me"] - before, 2)  # the one in flight, plus one for the rest

    def test_a_late_answer_never_undoes_a_sign_out(self):
        self.sign_in()
        hub = settings_ui.account_hub()
        hub.refresh()
        self.assertTrue(self.wait(lambda: bool(hub.state["snapshot"])))
        self.fake.delays = {"/v1/me": 0.4}
        hub.refresh(force=True)                                    # in flight ...
        session.get_session(self.fake.url).sign_out()               # ... when the user signs out
        self.assertFalse(hub.state["signed_in"])
        self.assertTrue(self.wait(lambda: hub._thread is None))
        self.assertFalse(hub.state["signed_in"])
        self.assertIsNone(hub.state["snapshot"])

    def test_a_sign_in_on_another_thread_reaches_the_gui_thread(self):
        hub = settings_ui.account_hub()
        hub.refresh()
        seen = []
        hub.changed.connect(lambda payload: seen.append((threading.get_ident(), json.loads(payload)["signed_in"])))
        worker = threading.Thread(target=self.sign_in)
        worker.start()
        worker.join(10)
        self.assertTrue(self.wait(lambda: hub.state["signed_in"]))
        self.assertTrue(seen)
        self.assertEqual({ident for ident, _signed_in in seen}, {threading.get_ident()})


class OnboardingCardTest(_PanelCase):
    def test_the_empty_chat_offers_the_account(self):
        p = self.panel()
        self.assertEqual(p.blocks[0].text, panel.WELCOME)
        cards = self.onboarding(p)
        self.assertEqual(len(cards), 1)
        self.assertIs(p.blocks[1], cards[0])
        markup = cards[0].html()
        for target in ("oscar://account/signin", "oscar://account/device", "oscar://settings"):
            self.assertIn(target, markup)
        self.assertIn("Sign in to OriginShape", p.log.toPlainText())

    def test_signing_in_removes_it(self):
        p = self.panel()
        self.sign_in()
        self.assertTrue(self.wait(lambda: not self.onboarding(p)))
        self.assertNotIn("Sign in to OriginShape", p.log.toPlainText())

    def test_an_anthropic_profile_without_its_key_is_offered_the_account(self):
        self.use([Profile(name="Claude", provider="anthropic", api_key="")])
        p = self.panel()
        cards = self.onboarding(p)
        self.assertEqual(len(cards), 1)
        self.assertIn("no API key", cards[0].text)

    def test_your_own_provider_with_its_key_gets_none(self):
        self.use([_own()])
        self.assertEqual(self.onboarding(self.panel()), [])

    def test_choosing_the_account_profile_adds_it(self):
        self.use([_own(), params.account_profile()], active="Mine")
        p = self.panel()
        self.assertEqual(self.onboarding(p), [])
        p.profile_combo.setCurrentIndex(p.profile_combo.findText("OriginShape"))
        self.assertEqual(len(self.onboarding(p)), 1)
        self.assertFalse(p.account_btn.isHidden())

    def test_a_new_chat_has_it_and_the_first_message_removes_it(self):
        p = self.panel()
        p._new_chat()
        self.assertEqual(len(self.onboarding(p)), 1)
        self.sign_in()
        self.assertTrue(self.wait(lambda: not self.onboarding(p)))
        p.input.setPlainText("make a cube")
        p._send()
        self.assertEqual(self.onboarding(p), [])
        self.assertEqual(len(RecordingWorker.instances), 1)


class SendGateTest(_PanelCase):
    def test_signed_out_the_account_profile_starts_no_turn_and_keeps_the_text(self):
        p = self.panel()
        attachment = os.path.join(self.tmp, "notes.txt")
        with open(attachment, "w") as f:
            f.write("x")
        p.add_attachments([attachment])
        for _ in range(3):                                          # pressing Send again never stacks cards
            p.input.setPlainText("make a cube")
            p._send()
        self.assertEqual(RecordingWorker.instances, [])
        self.assertEqual(p.input.toPlainText(), "make a cube")
        self.assertEqual(p.pending, [attachment])
        self.assertEqual(len(self.onboarding(p)), 1)
        self.assertEqual(p.history, [])

    def test_a_fresh_waitlisted_snapshot_shows_the_waitlist_and_checks_again(self):
        self.fake.set_account(status="waitlisted")
        p = self.signed_in_panel("Waitlist")
        before = self.fake.counts["me"]
        p.input.setPlainText("make a cube")
        p._send()
        self.assertEqual(RecordingWorker.instances, [])
        self.assertEqual(len(self.accounts(p, "waitlisted")), 1)
        self.assertIn("oscar://account/refresh", self.accounts(p, "waitlisted")[0].html())
        self.assertEqual(p.input.toPlainText(), "make a cube")
        self.assertTrue(self.wait(lambda: self.fake.counts["me"] > before))     # approved meanwhile?

    def test_a_stale_waitlisted_snapshot_lets_the_server_decide(self):
        self.fake.set_account(status="waitlisted")
        p = self.signed_in_panel("Waitlist")
        with mock.patch.object(panel, "_WAITLIST_FRESH_S", 0.0):
            p.input.setPlainText("make a cube")
            p._send()
        self.assertEqual(len(RecordingWorker.instances), 1)

    def test_signed_in_the_turn_goes_and_names_the_chat(self):
        p = self.signed_in_panel()
        p.input.setPlainText("make a cube")
        p._send()
        self.assertEqual(len(RecordingWorker.instances), 1)
        self.assertEqual(RecordingWorker.instances[0].kw["session_id"], p._current_session()["id"])
        self.assertEqual(p.input.toPlainText(), "")

    def test_a_sign_in_that_is_not_known_yet_is_left_to_the_turn(self):
        p = self.panel()
        hub = settings_ui.account_hub()
        hub.state = {**hub.state, "known": False}                   # the OS keychain is still being read
        p.input.setPlainText("make a cube")
        p._send()
        self.assertEqual(len(RecordingWorker.instances), 1)

    def test_an_anthropic_profile_without_its_key(self):
        self.use([Profile(name="Claude", provider="anthropic", api_key="")])
        p = self.panel()
        p.input.setPlainText("hello")
        p._send()
        self.assertEqual(RecordingWorker.instances, [])
        self.assertEqual(p.input.toPlainText(), "hello")
        self.assertEqual(len(self.onboarding(p)), 1)

    def test_a_refused_server_url_says_where_to_fix_it(self):
        os.environ["OSCAR_API_URL"] = "http://example.com"           # plain http to another host: refused
        p = self.panel()
        self.assertEqual(p.account_btn.text(), "Sign in")
        self.assertIn("#d9902b", p.account_btn.styleSheet())
        p.input.setPlainText("hello")
        p._send()
        self.assertEqual(RecordingWorker.instances, [])
        errors = [b for b in p.blocks if b.kind == Block.ERROR]
        self.assertEqual(len(errors), 1)
        self.assertIn("https", errors[0].text)
        self.assertIn("Server field", errors[0].text)


class AccountProblemTest(_PanelCase):
    def send(self, p, text="make a cube"):
        p.input.setPlainText(text)
        p._send()
        return p.history[-1]

    def finish(self, p, problem, text):
        if problem is not None:
            p._on_account_problem(json.dumps(problem))
        p._on_turn_outcome("failed", "")
        p._on_turn_finished(False, text)

    @staticmethod
    def problem(code, kind="account", status=402, first=True):
        return {"code": code, "kind": kind, "status": status, "message": "m", "retry_after": None, "data": {},
                "first_request": first}

    def test_no_credits_on_the_first_request_shows_the_card_and_gives_the_message_back(self):
        p = self.signed_in_panel()
        attachment = os.path.join(self.tmp, "notes.txt")
        with open(attachment, "w") as f:
            f.write("x")
        p.add_attachments([attachment])
        session_ = p._current_session()
        session_["ledger"] = {"request": "before"}
        message = self.send(p)
        session_["ledger"] = {"request": "the plan this turn seeded"}
        self.finish(p, self.problem("insufficient_credits"),
                    "Provider error: You have no OriginShape credits left (HTTP 402). Add credits or choose a plan "
                    "to continue.")
        self.assertEqual([b for b in p.blocks if b.kind == Block.ERROR], [])          # no raw provider error line
        cards = self.accounts(p, "no_credits")
        self.assertEqual(len(cards), 1)
        self.assertIn("back in the input", cards[0].text)
        for target in ("oscar://account/subscribe", "oscar://settings"):   # a trial account: a plan, not packs
            self.assertIn(target, cards[0].html())
        self.assertNotIn("oscar://account/buy", cards[0].html())
        self.assertEqual(p.input.toPlainText(), "make a cube")
        self.assertNotIn(message, p.history)
        self.assertEqual([b for b in p.blocks if b.kind == Block.USER], [])
        self.assertEqual(p.pending, [attachment])
        self.assertEqual(session_["ledger"], {"request": "before"})
        self.assertEqual(chats.load(session_["id"])["history"], [])
        self.assertEqual(session_.get("blocks_meta") or [], [])                     # nothing happened: no record

    def test_after_a_tool_round_the_message_stays(self):
        p = self.signed_in_panel()
        message = self.send(p)
        self.finish(p, self.problem("insufficient_credits", first=False), "Provider error: no credits")
        card = self.accounts(p, "no_credits")[0]
        self.assertNotIn("back in the input", card.text)
        self.assertIn(message, p.history)
        self.assertEqual(p.input.toPlainText(), "")

    def test_signed_out_and_waitlisted_have_their_cards(self):
        p = self.signed_in_panel()
        self.send(p)
        self.finish(p, self.problem("signed_out", kind="auth", status=401), "Provider error: sign in again")
        self.assertIn("oscar://account/signin", self.accounts(p, "signed_out")[0].html())
        p.input.clear()
        self.send(p)
        self.finish(p, self.problem("waitlisted", status=403), "Provider error: waitlisted")
        self.assertEqual(len(self.accounts(p, "waitlisted")), 1)

    def test_a_problem_without_a_card_keeps_the_error_line_and_still_gives_the_message_back(self):
        p = self.signed_in_panel()
        self.send(p)
        self.finish(p, self.problem("rate_limited", status=429), "Provider error: OriginShape is limiting requests")
        errors = [b for b in p.blocks if b.kind == Block.ERROR]
        self.assertEqual([e.text for e in errors], ["Provider error: OriginShape is limiting requests"])
        self.assertIn("back in the input", p.log.toPlainText())
        self.assertEqual(p.input.toPlainText(), "make a cube")

    def test_an_ordinary_failure_is_unchanged(self):
        p = self.signed_in_panel()
        message = self.send(p)
        self.finish(p, None, "Provider error: HTTP 500 server error")
        self.assertEqual([b.text for b in p.blocks if b.kind == Block.ERROR], ["Provider error: HTTP 500 server error"])
        self.assertEqual(self.accounts(p, "no_credits"), [])
        self.assertIn(message, p.history)
        self.assertEqual(p.input.toPlainText(), "")

    def test_text_typed_meanwhile_is_kept_after_the_returned_message(self):
        p = self.signed_in_panel()
        self.send(p)
        p.input.setPlainText("and make it red")
        self.finish(p, self.problem("insufficient_credits"), "Provider error: no credits")
        self.assertEqual(p.input.toPlainText(), "make a cube\n\nand make it red")


class AnchorTest(_PanelCase):
    def opened(self, p):
        calls = []
        p._open_external = calls.append
        return calls

    def test_buy_subscribe_and_manage_open_the_accounts_own_pages(self):
        p = self.signed_in_panel()
        calls = self.opened(p)
        links = self.fake.me["links"]
        for target in ("oscar://account/buy", "oscar://account/subscribe", "oscar://account/manage",
                       "oscar://account/waitlist"):
            p._on_anchor_target(target)
        self.assertEqual(calls, [links["billing"], links["billing"], links["manage"], links["waitlist"]])

    def test_the_link_never_chooses_the_page(self):
        p = self.signed_in_panel()
        calls = self.opened(p)
        p._on_anchor_target("oscar://account/buy?url=https://evil.example/x")
        p._on_anchor_target("oscar://account/https://evil.example")
        p._on_anchor_target("oscar://account/signout")              # never a link: signing out is a menu entry
        self.assertEqual(calls, [self.fake.me["links"]["billing"]])
        self.assertTrue(session.get_session(self.fake.url).has_session())
        self.assertIn("Unknown account link", p.log.toPlainText())

    def test_sign_in_links_open_settings_and_start_that_sign_in(self):
        p = self.panel()
        started = []
        p._open_settings = lambda account_action="": started.append(account_action)
        for target in ("oscar://account/signin", "oscar://account/device", "oscar://settings"):
            p._on_anchor_target(target)
        self.assertEqual(started, ["signin", "device", ""])

    def test_refresh_checks_the_account_again(self):
        p = self.signed_in_panel()
        before = self.fake.counts["me"]
        p._on_anchor_target("oscar://account/refresh")
        self.assertTrue(self.wait(lambda: self.fake.counts["me"] > before))

    def test_without_a_known_page_nothing_opens(self):
        p = self.panel()                                            # signed out: no account links known
        calls = self.opened(p)
        p._on_anchor_target("oscar://account/buy")
        self.assertEqual(calls, [])
        self.assertIn("does not know the OriginShape website yet", p._status)

    def test_a_link_in_the_log_reaches_the_handler(self):
        from PySide import QtCore

        p = self.panel()
        started = []
        p._open_settings = lambda account_action="": started.append(account_action)
        p._on_anchor(QtCore.QUrl("oscar://account/signin"))
        self.assertEqual(started, ["signin"])


class SettingsAndCommandsTest(_PanelCase):
    def fake_dialog(self, accepted=0, changed=False):
        test = self

        class Dialog:
            def __init__(self, parent=None):
                self.account_changed = changed
                test.dialogs.append(self)
                self.started = []

            def start_account_action(self, action):
                self.started.append(action)

            def exec(self):
                return accepted

        self.dialogs = []
        return Dialog

    def test_a_cancelled_dialog_that_signed_in_refreshes_the_panel(self):
        p = self.panel()
        reloads = []
        p._reload_profiles = lambda: reloads.append(1)
        with mock.patch.object(settings_ui, "SettingsDialog", self.fake_dialog(accepted=0, changed=True)):
            p._open_settings(account_action="signin")
        self.assertEqual(self.dialogs[0].started, ["signin"])
        self.assertEqual(reloads, [1])

    def test_a_cancelled_dialog_without_changes_reloads_nothing(self):
        p = self.panel()
        reloads = []
        p._reload_profiles = lambda: reloads.append(1)
        with mock.patch.object(settings_ui, "SettingsDialog", self.fake_dialog()):
            p._open_settings()
        self.assertEqual(self.dialogs[0].started, [])
        self.assertEqual(reloads, [])

    def test_the_sign_in_command_starts_the_browser_sign_in(self):
        window = QtWidgets.QMainWindow()
        self.addCleanup(window.deleteLater)
        gui = types.SimpleNamespace(getMainWindow=lambda: window)
        with mock.patch.object(commands, "Gui", gui), \
                mock.patch.object(settings_ui, "SettingsDialog", self.fake_dialog(accepted=0, changed=True)), \
                mock.patch.object(commands, "refresh_panel_profiles") as refresh:
            commands.SignInCommand().Activated()
            commands.AccountCommand().Activated()
        self.assertEqual([d.started for d in self.dialogs], [["signin"], []])
        self.assertEqual(refresh.call_count, 2)                    # account_changed refreshes, even after Cancel

    def test_the_account_commands_are_registered_and_in_the_menu(self):
        added = []
        gui = types.SimpleNamespace(listCommands=lambda: [], addCommand=lambda name, cmd: added.append(name))
        with mock.patch.object(commands, "Gui", gui):
            commands.register_all()
        self.assertIn("Oscar_SignIn", added)
        self.assertIn("Oscar_Account", added)
        self.assertIn("Oscar_CheckUpdates", added)
        for cls in (commands.SignInCommand, commands.AccountCommand, commands.CheckUpdatesCommand):
            resources = cls().GetResources()
            self.assertTrue(resources["MenuText"].endswith("…"))
        with open(os.path.join(os.path.dirname(__file__), "..", "InitGui.py"), encoding="utf-8") as f:
            source = f.read()
        self.assertIn('"Oscar_SignIn", "Oscar_Account", "Oscar_CheckUpdates"', source)

    def check_for_updates(self) -> dict:
        """Oscar → Check for Updates… with the message boxes caught: ``{"info": [...], "warning": [...], "box": [...]}``
        (the texts shown) once the background check has reported."""
        window = QtWidgets.QMainWindow()
        self.addCleanup(window.deleteLater)
        shown: dict = {"info": [], "warning": [], "box": []}
        gui = types.SimpleNamespace(getMainWindow=lambda: window)
        box = QtWidgets.QMessageBox
        with mock.patch.object(commands, "Gui", gui), \
                mock.patch.object(box, "information", lambda parent, title, text: shown["info"].append(text)), \
                mock.patch.object(box, "warning", lambda parent, title, text: shown["warning"].append(text)), \
                mock.patch.object(box, "exec_", lambda self: shown["box"].append(self.text())):
            commands.CheckUpdatesCommand().Activated()
            self.assertTrue(self.wait(lambda: not commands._PENDING), "the check never reported")
        return shown

    def test_check_for_updates_says_you_have_the_latest(self):
        self.fake.plugin = {"latest_version": panel.__version__, "min_version": "0.1.0", "download_url": ""}
        shown = self.check_for_updates()
        self.assertEqual(shown["info"], [f"You have the latest Oscar ({panel.__version__})."])
        self.assertEqual(self.fake.counts["plugin"], 1)

    def test_check_for_updates_offers_the_newer_oscar_and_the_panel_shows_it(self):
        self.fake.plugin = {"latest_version": "9.0.0", "min_version": "0.1.0", "download_url": ""}
        p = self.panel()
        shown = self.check_for_updates()                           # asked even with the daily check off
        self.assertEqual(len(shown["box"]), 1)
        self.assertIn(f"Oscar 9.0.0 is available (you have {panel.__version__}).", shown["box"][0])
        self.assertTrue(self.wait(lambda: settings_ui.account_hub().state.get("update") == "available"))
        self.assertIn(("Update Oscar to 9.0.0…"), self.menu_labels(p))   # the account profile's badge menu

    def test_check_for_updates_says_when_the_server_cannot_be_reached(self):
        os.environ["OSCAR_API_URL"] = "http://127.0.0.1:9"
        shown = self.check_for_updates()
        self.assertEqual(len(shown["warning"]), 1)
        self.assertTrue(shown["warning"][0].startswith("Oscar couldn't check for updates:"), shown["warning"])


if __name__ == "__main__":
    unittest.main()
