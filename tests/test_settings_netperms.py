"""Settings > Network permissions (F-26): the grant list with Remove / Revoke all, the legacy entry, the web settings.

The model functions are pure python over the isolated settings file; the dialog tests run a real ``SettingsDialog`` on an offscreen
QApplication (nothing is shown, no network)."""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import netguard, params  # noqa: E402
from oscar import settings_model as model  # noqa: E402
from oscar.params import Profile  # noqa: E402

try:
    from PySide import QtWidgets
    _QT_ERROR = ""
except Exception as exc:  # pragma: no cover
    _QT_ERROR = str(exc)


def _reset():
    params.clear_net_grants()
    params.set_bool("AllowNetwork", False)
    params.set_string("search_backend_url", "")
    params.set_string("web_allowlist_extra", "")
    netguard.reset_turn()


class ModelTest(unittest.TestCase):
    def setUp(self):
        _reset()
        self.addCleanup(_reset)

    def test_no_grants_means_no_rows(self):
        self.assertEqual(model.net_grant_rows(), [])

    def test_rows_describe_each_grant_and_the_legacy_flag_comes_first_and_inactive(self):
        netguard.add_grant("fetch_url", "datasheets.raspberrypi.com")
        netguard.add_grant("network", "*")
        params.set_bool("AllowNetwork", True)
        rows = model.net_grant_rows()
        self.assertEqual([r["legacy"] for r in rows], [True, False, False])
        self.assertIn("inactive", rows[0]["label"])
        self.assertIn("fetch_url", rows[1]["label"])
        self.assertIn("datasheets.raspberrypi.com", rows[1]["label"])
        self.assertIn("any host", rows[2]["label"])

    def test_revoke_one_grant_and_the_legacy_row(self):
        netguard.add_grant("fetch_url", "a.example.com")
        netguard.add_grant("web_search", "b.example.com")
        params.set_bool("AllowNetwork", True)
        self.assertTrue(model.revoke_net_grant({"tool": "fetch_url", "host": "a.example.com", "legacy": False}))
        self.assertTrue(model.revoke_net_grant({"legacy": True}))
        self.assertFalse(params.get_bool("AllowNetwork", False))
        self.assertFalse(model.revoke_net_grant({"legacy": True}), "a second revoke has nothing to remove")
        self.assertEqual([(r["tool"], r["host"]) for r in model.net_grant_rows()], [("web_search", "b.example.com")])

    def test_revoke_all(self):
        netguard.add_grant("fetch_url", "a.example.com")
        netguard.add_grant("web_search", "b.example.com")
        params.set_bool("AllowNetwork", True)
        self.assertEqual(model.revoke_all_net_grants(), 3)
        self.assertEqual(model.net_grant_rows(), [])
        self.assertEqual(model.revoke_all_net_grants(), 0)

    def test_validate_search_url(self):
        for good in ("", "  ", "http://localhost:8080", "https://searx.example.org/search"):
            self.assertEqual(model.validate_search_url(good), "", good)
        for bad in ("ftp://host", "localhost:8080", "http://", "javascript:alert(1)", "not a url"):
            self.assertIn("http(s)", model.validate_search_url(bad), bad)


@unittest.skipIf(_QT_ERROR, f"PySide unavailable: {_QT_ERROR}")
class DialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            cls.app = QtWidgets.QApplication(sys.argv[:1])

    def setUp(self):
        _reset()
        self.addCleanup(_reset)
        params.save_profiles([Profile(name="Alpha", provider="openai_compat", base_url="http://alpha.test/v1", model="m", api_key="k")])
        params.set_active_profile("Alpha")

    def dialog(self):
        from oscar.settings_ui import SettingsDialog

        dialog = SettingsDialog()
        self.addCleanup(dialog.deleteLater)
        return dialog

    def finish_job(self, dialog, timeout=5.0):
        """Wait for the worker thread the way the dialog's own poller does (drain the mailbox on this, the GUI, thread)."""
        deadline = time.monotonic() + timeout
        while "netgrants" in dialog._jobs and time.monotonic() < deadline:
            time.sleep(0.01)
            dialog._drain_mailbox()
            self.app.processEvents()
        self.assertNotIn("netgrants", dialog._jobs, "the revoke job did not finish")

    def labels(self, dialog):
        return [dialog.net_list.item(i).text() for i in range(dialog.net_list.count())]

    def test_the_section_and_its_widgets_are_addressable(self):
        dialog = self.dialog()
        for name, kind in (("oscarNetPermissions", QtWidgets.QGroupBox), ("oscarNetGrantList", QtWidgets.QListWidget),
                           ("oscarNetGrantRemove", QtWidgets.QPushButton), ("oscarNetGrantRevokeAll", QtWidgets.QPushButton),
                           ("oscarSearchBackendUrl", QtWidgets.QLineEdit), ("oscarWebAllowlistExtra", QtWidgets.QLineEdit)):
            self.assertIsNotNone(dialog.findChild(kind, name), name)
        self.assertEqual(dialog.findChild(QtWidgets.QGroupBox, "oscarNetPermissions").title(), "Network permissions")

    def test_it_lists_the_grants_and_both_buttons_start_disabled_when_there_are_none(self):
        dialog = self.dialog()
        self.assertEqual(self.labels(dialog), [])
        self.assertFalse(dialog.net_remove_btn.isEnabled())
        self.assertFalse(dialog.net_revoke_all_btn.isEnabled())

    def test_remove_revokes_the_selected_grant_on_a_worker_thread(self):
        netguard.add_grant("fetch_url", "a.example.com")
        netguard.add_grant("web_search", "b.example.com")
        dialog = self.dialog()
        self.assertEqual(len(self.labels(dialog)), 2)
        self.assertFalse(dialog.net_remove_btn.isEnabled(), "nothing selected yet")
        dialog.net_list.setCurrentRow(0)
        self.assertTrue(dialog.net_remove_btn.isEnabled())
        dialog.net_remove_btn.click()
        self.assertIn("netgrants", dialog._jobs)
        self.assertFalse(dialog.net_remove_btn.isEnabled(), "buttons wait while the job runs")
        self.finish_job(dialog)
        self.assertEqual([(g["tool"], g["host"]) for g in netguard.list_grants()], [("web_search", "b.example.com")])
        self.assertEqual(len(self.labels(dialog)), 1)
        self.assertIn("b.example.com", self.labels(dialog)[0])

    def test_revoke_all_clears_everything_including_the_legacy_entry(self):
        netguard.add_grant("fetch_url", "a.example.com")
        params.set_bool("AllowNetwork", True)
        dialog = self.dialog()
        self.assertEqual(len(self.labels(dialog)), 2)
        self.assertIn("Legacy", self.labels(dialog)[0])
        dialog.net_revoke_all_btn.click()
        self.finish_job(dialog)
        self.assertEqual(self.labels(dialog), [])
        self.assertEqual(netguard.list_grants(), [])
        self.assertFalse(params.get_bool("AllowNetwork", False))

    def test_removing_the_legacy_entry_clears_only_the_old_flag(self):
        netguard.add_grant("fetch_url", "a.example.com")
        params.set_bool("AllowNetwork", True)
        dialog = self.dialog()
        dialog.net_list.setCurrentRow(0)
        dialog.net_remove_btn.click()
        self.finish_job(dialog)
        self.assertFalse(params.get_bool("AllowNetwork", False))
        self.assertEqual(len(netguard.list_grants()), 1)

    def test_a_failing_write_is_reported_and_the_dialog_stays_usable(self):
        from unittest import mock

        netguard.add_grant("fetch_url", "a.example.com")
        dialog = self.dialog()
        with mock.patch.object(model, "revoke_all_net_grants", side_effect=OSError("locked")):
            dialog.net_revoke_all_btn.click()
            self.finish_job(dialog)
        self.assertIn("locked", dialog.save_label.text())
        self.assertFalse(dialog.save_label.isHidden())
        self.assertEqual(len(self.labels(dialog)), 1)
        self.assertTrue(dialog.net_revoke_all_btn.isEnabled())

    def test_the_web_settings_load_and_save_with_the_dialog(self):
        params.set_string("search_backend_url", "http://localhost:8080")
        params.set_string("web_allowlist_extra", "docs.vendor.io")
        dialog = self.dialog()
        self.assertEqual(dialog.search_url_edit.text(), "http://localhost:8080")
        self.assertEqual(dialog.allow_extra_edit.text(), "docs.vendor.io")
        dialog.search_url_edit.setText("  https://searx.example.org  ")
        dialog.allow_extra_edit.setText("a.example.com, b.example.com")
        dialog._save()
        self.assertEqual(params.get_string("search_backend_url", ""), "https://searx.example.org")
        self.assertEqual(params.get_string("web_allowlist_extra", ""), "a.example.com, b.example.com")

    def test_an_invalid_search_url_blocks_save_and_says_why(self):
        dialog = self.dialog()
        dialog.search_url_edit.setText("ftp://nope")
        dialog._save()
        self.assertFalse(dialog.save_label.isHidden())
        self.assertIn("http(s)", dialog.save_label.text())
        self.assertEqual(params.get_string("search_backend_url", ""), "")

    def test_saving_does_not_touch_grants(self):
        netguard.add_grant("fetch_url", "a.example.com")
        dialog = self.dialog()
        dialog._save()
        self.assertEqual(len(netguard.list_grants()), 1)


if __name__ == "__main__":
    unittest.main()
