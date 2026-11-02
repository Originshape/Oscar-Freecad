# SPDX-License-Identifier: Apache-2.0
"""OriginShape's Terms in Oscar (oscar/auth/consent.py with the account snapshot and the session): what /v1/me says
about them (a server from before the Terms asks nothing), the prompt's logic (the box ticked by default; Continue
posts accept_terms, the current version and marketing_emails null; an unticked box sends nothing and keeps the account
blocked for chat, never another provider), POST /v1/me/consent against the fake server (200 updates the snapshot with
no second request; 409 fetches /v1/me again and prompts for the new version; other answers are errors), a 403
terms_required marking the Terms as needed again, and the panel's pure pieces (the consent card, the badge menu and
tooltip, the card a failed turn calls for). Pure python (stdlib + requests); the keystore runs in memory.
"""

import json
import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import panel_render as pr  # noqa: E402
from oscar.auth import account, consent, discovery, keystore, session  # noqa: E402
from oscar.auth.transport import AuthError, AuthUnavailable, TokenSet  # noqa: E402
from support.fake_originshape import FakeOriginShape, default_me  # noqa: E402

APP = "https://app.originshape.design"
NOTICE = ("OriginShape keeps your chats, prompts, files and feedback to run and improve the service. They are never "
          "used to train AI models.")


def me(*, accepted="2026-11-02", current="2026-11-02", status="approved", terms_url=APP + "/terms",
       privacy_url=APP + "/privacy"):
    """A /v1/me body of a server that knows the Terms."""
    body = default_me(APP)
    body["user"]["status"] = status
    body["consent"] = {"terms_version": accepted, "terms_accepted_at": "2026-11-01T09:00:00Z" if accepted else None,
                       "terms_current": accepted == current, "marketing_emails": None}
    body["legal"] = {"terms_version": current, "terms_url": terms_url, "privacy_url": privacy_url}
    return body


class SnapshotTest(unittest.TestCase):
    def test_me_carries_what_was_accepted_and_what_is_current(self):
        snap = account.parse_me(me(accepted="2026-10-01"))
        self.assertTrue(snap.consent_known)
        self.assertEqual((snap.terms_version, snap.terms_current, snap.legal_terms_version),
                         ("2026-10-01", False, "2026-11-02"))
        self.assertEqual((snap.link_terms, snap.link_privacy), (APP + "/terms", APP + "/privacy"))
        self.assertEqual(account.AccountSnapshot.from_dict(snap.to_dict()), snap)
        # The pages open in the browser: only https (or http on this computer) is kept.
        odd = account.parse_me(me(terms_url="http://evil.example/terms", privacy_url="javascript:alert(1)"))
        self.assertEqual((odd.link_terms, odd.link_privacy), ("", ""))

    def test_a_server_from_before_the_terms_asks_nothing_and_records_nothing(self):
        snap = account.parse_me(default_me(APP))
        self.assertFalse(snap.consent_known)
        self.assertFalse(consent.needed(snap))
        self.assertFalse(consent.recorded(snap))
        self.assertFalse(consent.blocks_chat(snap, account_profile=True))
        self.assertIsNone(consent.Prompt.of(snap))

    def test_needed_until_the_current_version_is_accepted(self):
        for body, needed, recorded in ((me(accepted=None), True, False), (me(accepted="2026-10-01"), True, False),
                                       (me(), False, True), (me(accepted=None, status="suspended"), False, False)):
            snap = account.parse_me(body)
            for form in (snap, snap.to_dict()):   # the session's snapshot, or the panel's account state
                self.assertEqual((consent.needed(form), consent.recorded(form)), (needed, recorded), body["consent"])

    def test_only_a_chat_on_the_account_waits_for_the_terms(self):
        pending = account.parse_me(me(accepted=None))
        self.assertTrue(consent.blocks_chat(pending, account_profile=True))
        self.assertFalse(consent.blocks_chat(pending, account_profile=False))   # an API key of your own, Ollama ...
        self.assertFalse(consent.blocks_chat(account.parse_me(me()), account_profile=True))


class PromptTest(unittest.TestCase):
    def setUp(self):
        self.prompt = consent.Prompt.of(account.parse_me(me(accepted=None)))

    def test_the_box_is_ticked_by_default_and_continue_accepts_the_current_terms(self):
        self.assertTrue(self.prompt.checked)
        self.assertEqual(self.prompt.body(), {"accept_terms": True, "terms_version": "2026-11-02",
                                              "marketing_emails": None})
        self.assertEqual(self.prompt.note(), "")

    def test_unticked_sends_nothing_and_says_the_account_is_blocked(self):
        unticked = consent.with_box(self.prompt, False)
        self.assertIsNone(unticked.body())
        self.assertEqual(unticked.note(), consent.BLOCKED)
        for words in ("can't be used for chat", "own API key", "Ollama"):
            self.assertIn(words, consent.BLOCKED)
        self.assertTrue(self.prompt.checked)   # the prompt itself is unchanged

    def test_the_words_and_the_links(self):
        self.assertEqual(consent.NOTICE, NOTICE)
        self.assertEqual(consent.AGREE, "I agree to the Terms and the Privacy Policy")
        markup = self.prompt.agree_html()
        self.assertIn(f"<a href='{APP}/terms'>Terms</a>", markup)
        self.assertIn(f"<a href='{APP}/privacy'>Privacy Policy</a>", markup)
        self.assertEqual(markup.count("<a "), 2)
        plain = consent.Prompt("2026-11-02").agree_html()   # pages not known: the words, no links
        self.assertEqual(plain, consent.AGREE)
        quoted = consent.Prompt("v", terms_url="https://x.example/t?a='b'").agree_html()
        self.assertNotIn("'b'", quoted)

    def test_nothing_to_prompt_once_accepted(self):
        self.assertIsNone(consent.Prompt.of(account.parse_me(me())))
        self.assertEqual(consent.Prompt.of(account.parse_me(me(accepted=None)), updated=True).updated, True)


class AcceptCase(unittest.TestCase):
    def setUp(self):
        for reset in (discovery.clear_cache, session.reset_sessions, keystore.reset_memory, account.clear_model_cache):
            reset()
            self.addCleanup(reset)
        patcher = mock.patch.dict(os.environ, {keystore.ENV_STORE: "memory"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.events = []
        hook = mock.patch.object(session, "on_change", lambda payload: self.events.append(json.loads(payload)))
        hook.start()
        self.addCleanup(hook.stop)
        self.fake = FakeOriginShape()
        self.addCleanup(self.fake.close)
        self.fake.set_consent(accepted=None, current="2026-11-02")
        data = self.fake.issue_tokens()
        self.session = session.get_session(self.fake.url)
        self.session.complete_sign_in(TokenSet(data["access_token"], data["refresh_token"], data["expires_in"],
                                               data["scope"], received_at=time.monotonic()))
        self.session.account(force=True)
        self.events.clear()

    def prompt(self, **kw):
        return consent.Prompt.of(self.session.cached_snapshot(), **kw)


class AcceptTest(AcceptCase):
    def test_continue_posts_the_right_body_and_the_snapshot_follows(self):
        before = self.fake.counts["me"]
        snapshot = consent.submit(self.prompt(), self.session)
        self.assertEqual(self.fake.consent_posts, [{"accept_terms": True, "terms_version": "2026-11-02",
                                                    "marketing_emails": None}])
        self.assertTrue(consent.recorded(snapshot))
        self.assertTrue(consent.recorded(self.session.cached_snapshot()))
        self.assertEqual(self.fake.counts["me"], before)                 # the answer's consent, no second request
        self.assertEqual([e["event"] for e in self.events], ["account"])   # the panel hears about it
        self.assertIs(self.events[0]["snapshot"]["terms_current"], True)
        self.assertIsNone(self.prompt())                                 # nothing to ask any more

    def test_an_unticked_box_sends_nothing_and_the_account_stays_blocked(self):
        self.assertIsNone(consent.submit(consent.with_box(self.prompt(), False), self.session))
        self.assertEqual(self.fake.consent_posts, [])
        self.assertEqual(self.fake.counts["consent"], 0)
        self.assertTrue(consent.blocks_chat(self.session.cached_snapshot(), account_profile=True))
        self.assertFalse(consent.blocks_chat(self.session.cached_snapshot(), account_profile=False))

    def test_terms_that_changed_meanwhile_are_fetched_again_and_prompted_for(self):
        shown = self.prompt()
        self.fake.set_consent(accepted=None, current="2026-12-01")   # a new version while the prompt was open
        before = self.fake.counts["me"]
        with self.assertRaises(consent.TermsChanged) as caught:
            consent.submit(shown, self.session)
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(self.fake.counts["me"], before + 1)
        self.assertEqual(caught.exception.snapshot.legal_terms_version, "2026-12-01")
        again = consent.Prompt.of(caught.exception.snapshot, updated=True)
        self.assertEqual((again.version, again.updated, again.checked), ("2026-12-01", True, True))
        self.assertEqual(self.session.cached_snapshot().legal_terms_version, "2026-12-01")
        self.assertTrue(consent.recorded(consent.submit(again, self.session)))
        self.assertEqual([p["terms_version"] for p in self.fake.consent_posts], ["2026-11-02", "2026-12-01"])

    def test_a_refused_or_failed_answer_is_an_error(self):
        self.fake.fail("/v1/me/consent", 400, {"type": "error", "error": {"code": "invalid_request", "message": "x"}})
        with self.assertRaises(AuthError) as caught:
            consent.submit(self.prompt(), self.session)
        self.assertEqual(caught.exception.code, "invalid_request")
        self.fake.fail("/v1/me/consent", 503)
        with self.assertRaises(AuthUnavailable):
            consent.submit(self.prompt(), self.session)
        self.assertTrue(consent.needed(self.session.cached_snapshot()))

    def test_an_api_key_sign_in_is_sent_to_the_website(self):
        key = self.fake.add_api_key()
        self.session.sign_out()
        self.session.use_api_key(key)
        with self.assertRaises(AuthError) as caught:
            consent.submit(self.prompt(), self.session)
        self.assertEqual((caught.exception.code, caught.exception.status), ("forbidden", 403))
        self.assertIn("OriginShape website", str(caught.exception))
        self.assertTrue(consent.needed(self.session.cached_snapshot()))

    def test_terms_required_marks_the_terms_needed(self):
        consent.submit(self.prompt(), self.session)
        self.assertTrue(consent.recorded(self.session.cached_snapshot()))
        # The server changed the Terms: /v1/me says so, and the prompt is for the new version.
        self.fake.set_consent(accepted="2026-11-02", current="2026-12-01")
        consent.mark_needed(self.session)
        self.assertTrue(consent.needed(self.session.cached_snapshot()))
        self.assertEqual(consent.Prompt.of(self.session.cached_snapshot()).version, "2026-12-01")
        # A server whose /v1/me still says "accepted" while a record was refused: marked needed all the same.
        consent.submit(consent.Prompt.of(self.session.cached_snapshot()), self.session)
        consent.mark_needed(self.session)
        self.assertTrue(consent.needed(self.session.cached_snapshot()))
        self.assertEqual(consent.Prompt.of(self.session.cached_snapshot()).version, "2026-12-01")

    def test_the_answer_never_lands_on_another_account(self):
        other = account.parse_me({**default_me(APP), "user": {**default_me(APP)["user"], "email": "bob@example.com"}})
        with mock.patch.object(self.session, "_snapshot", other):
            self.assertIsNone(self.session.note_consent({"terms_current": True}, email="ada@example.com"))
        self.assertIsNone(self.session.note_consent("junk"))


class PanelPiecesTest(unittest.TestCase):
    def state(self, **consent_kw):
        return {"known": True, "signed_in": True, "email": "ada@example.com",
                "snapshot": account.parse_me(me(**consent_kw)).to_dict()}

    def test_the_consent_card_says_what_is_kept_and_offers_the_prompt(self):
        text, actions = pr.account_card("consent")
        self.assertIn(NOTICE, " ".join(text.split()))
        self.assertEqual([t for t, _label in actions], ["oscar://account/consent", "oscar://settings"])
        self.assertTrue(all(t in pr.ACCOUNT_TARGETS for t, _label in actions))
        declined, _actions = pr.account_card("consent", reason="declined")
        for words in ("can't be used for chat", "Ollama", "own API key"):
            self.assertIn(words, declined)
        self.assertIn("back in the input", pr.account_card("consent", returned=True)[0])
        markup = pr.account_card_html(*pr.account_card("consent"), kind="consent")
        self.assertIn("href='oscar://account/consent'", markup)

    def test_the_badge_menu_and_tooltip_while_the_terms_wait(self):
        pending, accepted = self.state(accepted=None), self.state()
        self.assertIn(("consent", "Review the Terms\u2026"), pr.account_menu(pending))
        self.assertNotIn("consent", [a for a, _label in pr.account_menu(accepted)])
        self.assertIn("accept the Terms", pr.account_badge(pending)[2])
        self.assertNotIn("Terms", pr.account_badge(accepted)[2])
        before_terms = {"known": True, "signed_in": True, "snapshot": account.parse_me(default_me(APP)).to_dict()}
        self.assertNotIn("consent", [a for a, _label in pr.account_menu(before_terms)])
        self.assertTrue(pr.terms_pending(pending["snapshot"]))
        self.assertFalse(pr.terms_pending(accepted["snapshot"]))
        self.assertFalse(pr.terms_pending(None))

    def test_a_turn_refused_for_the_terms_shows_the_consent_card(self):
        self.assertEqual(pr.account_problem_card({"kind": "account", "code": "terms_required"}), "consent")


if __name__ == "__main__":
    unittest.main()
