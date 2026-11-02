# SPDX-License-Identifier: Apache-2.0
"""The panel, OriginShape's Terms and the chat records, against the fake OriginShape API (consent_gui, panel_turn).

Covered: after a sign-in the prompt opens once, its box ticked, with the notice word for word, and the consent card
stays in the chat; Continue with the box ticked accepts the current Terms and the card goes; unticked, nothing is sent
and the account can't chat (the send shows the card and the prompt again, the text stays) while another provider can;
a 409 opens the prompt again on the new version; a finished turn on the account is recorded once the Terms are
accepted (the chat id, the turn index, the prompt, the model, the attached file) and a rating names it; nothing is
recorded for another provider or a server from before the Terms. Needs FreeCAD and Qt (offscreen); the keystore runs
in memory.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from test_account_panel import _IMPORT_ERROR, RecordingWorker, _own, _PanelCase  # noqa: E402

if _IMPORT_ERROR is None:
    from PySide import QtWidgets  # noqa: E402

    from oscar import consent_gui, params, rating_gui, records  # noqa: E402
    from oscar.auth import consent, session  # noqa: E402

NOTICE = ("OriginShape keeps your chats, prompts, files and feedback to run and improve the service. They are never "
          "used to train AI models.")


class _ConsentCase(_PanelCase):
    def setUp(self):
        super().setUp()
        for kept in (consent_gui._asked, consent_gui._declined):
            kept.clear()
            self.addCleanup(kept.clear)
        recorder = records.Recorder(sent=records.SentFiles(os.path.join(self.tmp, "sent-files.json")))
        self.addCleanup(recorder.close)
        patcher = mock.patch.object(records, "_RECORDER", recorder)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.recorder = recorder
        self.shown: list[dict] = []
        self.answers: list = []   # per dialog: (checked, accept) or a callable run before answering

    def answer(self, dialog):
        """Stands in for ``ConsentDialog.exec``: note what it shows, then answer as scripted (default: close it)."""
        self.shown.append({"checked": dialog.checked(), "version": dialog._prompt.version,
                           "updated": dialog._prompt.updated,
                           "notice": dialog.findChild(QtWidgets.QLabel, "oscarConsentNotice").text(),
                           "agree": dialog.findChild(QtWidgets.QLabel, "oscarConsentAgreeLabel").text()})
        script = self.answers.pop(0) if self.answers else (True, False)
        if callable(script):
            script = script()
        checked, accept = script
        dialog._box.setChecked(checked)
        return QtWidgets.QDialog.Accepted if accept else QtWidgets.QDialog.Rejected

    def dialogs(self):
        patcher = mock.patch.object(consent_gui.ConsentDialog, "exec", lambda dialog: self.answer(dialog))
        patcher.start()
        self.addCleanup(patcher.stop)

    def cards(self, p, reason=None):
        return [b for b in self.accounts(p, "consent") if reason is None or b.reason == reason]

    def finish(self, p, text="Done."):
        p._on_turn_outcome("answered", "")
        p._on_turn_finished(True, text)


class PromptTest(_ConsentCase):
    def test_after_a_sign_in_the_prompt_asks_once_with_the_box_ticked(self):
        self.fake.set_consent(accepted=None)
        self.dialogs()
        p = self.signed_in_panel()
        self.assertTrue(self.wait(lambda: len(self.shown) == 1))
        shown = self.shown[0]
        self.assertEqual((shown["checked"], shown["version"], shown["updated"]), (True, "2026-11-02", False))
        self.assertEqual(shown["notice"], NOTICE)
        self.assertIn(f"href='{self.fake.url}/terms'", shown["agree"])
        self.assertIn(f"href='{self.fake.url}/privacy'", shown["agree"])
        self.assertEqual(len(self.cards(p)), 1)                       # closed: the card stays in the chat
        self.assertIn("oscar://account/consent", self.cards(p)[0].html())
        self.assertIn("Review the Terms\u2026", self.menu_labels(p))
        p._refresh_account(force=True)                                # no nagging within the session
        self.wait(lambda: False, timeout=0.5)
        self.assertEqual(len(self.shown), 1)
        self.assertEqual(self.fake.consent_posts, [])

    def test_continue_with_the_box_ticked_accepts_and_the_card_goes(self):
        self.fake.set_consent(accepted=None)
        self.dialogs()
        self.answers.append((True, True))
        p = self.signed_in_panel()
        self.assertTrue(self.wait(lambda: self.fake.consent_posts and not self.cards(p)))
        self.assertEqual(self.fake.consent_posts, [{"accept_terms": True, "terms_version": "2026-11-02",
                                                    "marketing_emails": None}])
        self.assertTrue(self.wait(lambda: consent.recorded(p._account_state()["snapshot"])))
        self.assertNotIn("Review the Terms\u2026", self.menu_labels(p))
        p.input.setPlainText("make a cube")
        p._send()
        self.assertEqual(len(RecordingWorker.instances), 1)

    def test_unticked_the_account_cant_chat_but_another_provider_can(self):
        self.fake.set_consent(accepted=None)
        self.dialogs()
        self.answers += [(False, True), (False, True)]
        p = self.signed_in_panel()
        self.assertTrue(self.wait(lambda: len(self.cards(p, "declined")) == 1))
        self.assertEqual(self.fake.consent_posts, [])
        p.input.setPlainText("make a cube")
        p._send()
        self.assertTrue(self.wait(lambda: len(self.shown) == 2))       # the send asks again, at once
        self.assertEqual(RecordingWorker.instances, [])
        self.assertEqual(p.input.toPlainText(), "make a cube")
        self.assertIn("can't be used for chat", self.cards(p, "declined")[-1].text)
        self.assertEqual(self.fake.consent_posts, [])
        self.use([params.account_profile(), _own()], active="Mine")   # your own provider is never held back
        p._reload_profiles()
        p._send()
        self.assertEqual(len(RecordingWorker.instances), 1)

    def test_terms_that_changed_meanwhile_are_asked_for_again(self):
        self.fake.set_consent(accepted=None)
        self.dialogs()

        def new_terms():
            self.fake.set_consent(accepted=None, current="2026-12-01")
            return True, True

        self.answers += [new_terms, (True, True)]
        p = self.signed_in_panel()
        self.assertTrue(self.wait(lambda: len(self.shown) == 2 and len(self.fake.consent_posts) == 2))
        self.assertEqual((self.shown[1]["version"], self.shown[1]["updated"]), ("2026-12-01", True))
        self.assertEqual([b["terms_version"] for b in self.fake.consent_posts], ["2026-11-02", "2026-12-01"])
        self.assertTrue(self.wait(lambda: not self.cards(p)))

    def test_a_server_from_before_the_terms_asks_nothing(self):
        self.dialogs()
        p = self.signed_in_panel()
        self.wait(lambda: False, timeout=0.3)
        self.assertEqual((self.shown, self.cards(p)), ([], []))
        p.input.setPlainText("make a cube")
        p._send()
        self.assertEqual(len(RecordingWorker.instances), 1)


class RecordTest(_ConsentCase):
    def test_a_finished_turn_on_the_account_is_recorded_and_a_rating_names_it(self):
        self.fake.set_consent()
        p = self.signed_in_panel()
        attachment = os.path.join(self.tmp, "plan.txt")
        with open(attachment, "w") as f:
            f.write("an 80 mm flange")
        p.add_attachments([attachment])
        p.input.setPlainText("make a flange")
        p._send()
        chat = p._current_session()["id"]
        self.finish(p, "Built the flange.")
        self.assertTrue(self.recorder.wait_idle(15.0))
        body = self.fake.chat_turns[(chat, 0)]
        self.assertEqual((body["prompt"], body["reply"], body["model"], body["ok"], body["stopped"]),
                         ("make a flange", "Built the flange.", "origin1", True, False))
        self.assertEqual(body["title"], p._current_session()["title"])
        self.assertTrue(body["started_at"].endswith("Z") and body["finished_at"].endswith("Z"))
        self.assertEqual([(f["name"], f["kind"], f["turn"]) for f in self.fake.chat_files],
                         [("plan.txt", "attachment", 0)])
        p.input.setPlainText("make it 6 mm thick")
        p._send()
        self.finish(p)
        self.assertTrue(self.recorder.wait_idle(15.0))
        self.assertIn((chat, 1), self.fake.chat_turns)
        self.assertEqual(p._turn_ref, {"chat_id": chat, "turn": 1})
        context = rating_gui._context(p)
        self.assertEqual((context["chat_id"], context["turn"]), (chat, 1))

    def test_nothing_is_recorded_for_another_provider_or_an_old_server(self):
        self.fake.set_consent()
        self.use([_own(), params.account_profile()])
        p = self.signed_in_panel()
        p.input.setPlainText("make a cube")
        p._send()
        self.finish(p)
        self.assertTrue(self.recorder.wait_idle(5.0))
        self.assertEqual(self.fake.counts["chat_turn"], 0)
        del self.fake.me["consent"], self.fake.me["legal"]        # a server from before the Terms
        self.use([params.account_profile()])
        p._reload_profiles()
        p._refresh_account(force=True)
        self.assertTrue(self.wait(lambda: not p._account_state()["snapshot"].get("consent_known")))
        p.input.setPlainText("make a cube")
        p._send()
        self.finish(p)
        self.assertTrue(self.recorder.wait_idle(5.0))
        self.assertEqual(self.fake.counts["chat_turn"], 0)
        self.assertTrue(session.get_session(self.fake.url).has_session())


if __name__ == "__main__":
    unittest.main()
