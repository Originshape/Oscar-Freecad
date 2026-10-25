"""Per-(tool, host) network grants (F-26): the scoped permission dialog, grant persistence, the legacy flag, revocation, audit.

The dialog is built from fakes (as in test_netguard); grants live in the isolated settings file."""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)
if "FreeCADGui" not in sys.modules:
    import types

    try:
        import FreeCADGui  # noqa: F401
    except Exception:
        sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")

from oscar import netguard, params  # noqa: E402

os.environ.pop("OSCAR_ALLOW_NET", None)


class FakeBox:
    Warning, NoButton, YesRole, AcceptRole, NoRole = 1, 0, 2, 3, 4
    instances: list = []
    click = "deny"          # "deny" | "once" | "always"

    def __init__(self, icon, title, text, buttons, parent):
        self.title, self.text, self.name, self.informative = title, text, "", ""
        self.buttons, self.default, self._clicked = {}, None, None
        type(self).instances.append(self)

    def setObjectName(self, name):
        self.name = name

    def setInformativeText(self, text):
        self.informative = text

    def addButton(self, label, role):
        button = object()
        self.buttons[label] = button
        return button

    def setDefaultButton(self, button):
        self.default = button

    def setModal(self, value):
        pass

    def show(self):
        pass

    def raise_(self):
        pass

    def activateWindow(self):
        pass

    def exec(self):
        wanted = type(self).click
        for label, button in self.buttons.items():
            if (wanted == "deny" and label == "Deny") or (wanted == "once" and label == "Allow once") \
                    or (wanted == "always" and label.startswith("Always allow")):
                self._clicked = button

    def clickedButton(self):
        return self._clicked


class FakeApp:
    window = object()

    @staticmethod
    def instance():
        return FakeApp

    @staticmethod
    def activeWindow():
        return FakeApp.window


class GrantCase(unittest.TestCase):
    def setUp(self):
        from PySide import QtWidgets
        from oscar.cad import dispatch

        os.environ.pop("OSCAR_ALLOW_NET", None)
        params.clear_net_grants()
        params.set_bool("AllowNetwork", False)
        netguard.reset_turn()
        FakeBox.instances = []
        self.events = []
        for patch in (mock.patch.object(QtWidgets, "QMessageBox", FakeBox),
                      mock.patch.object(QtWidgets, "QApplication", FakeApp),
                      mock.patch.object(sys.modules["FreeCADGui"], "getMainWindow", create=True, return_value=object()),
                      mock.patch.object(dispatch, "dispatch", side_effect=lambda fn, timeout=0, op="", **kw: (fn(), False)),
                      mock.patch.object(netguard, "on_permission_wait", lambda waiting: None),
                      mock.patch("oscar.turnlog.log_event", lambda name, **f: self.events.append((name, f)))):
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(params.clear_net_grants)
        self.addCleanup(params.set_bool, "AllowNetwork", False)
        self.addCleanup(netguard.reset_turn)

    def ask(self, click, tool="fetch_url", host="datasheets.raspberrypi.com"):
        FakeBox.click = click
        return netguard.request_permission("summary", "detail", tool=tool, host=host)


class ScopedDialogTest(GrantCase):
    def test_three_buttons_name_the_host_and_tool_and_deny_is_default(self):
        self.ask("deny")
        (box,) = FakeBox.instances
        self.assertEqual(sorted(box.buttons), ["Allow once", "Always allow datasheets.raspberrypi.com for fetch_url", "Deny"])
        self.assertIs(box.default, box.buttons["Deny"])
        self.assertEqual(box.name, "oscarPermission")

    def test_allow_once_stores_nothing_and_covers_only_this_turn(self):
        self.assertEqual(self.ask("once"), "allow")
        self.assertEqual(netguard.list_grants(), [])
        self.assertEqual(self.ask("deny"), "allow", "the same turn reuses the cached decision without a second dialog")
        self.assertEqual(len(FakeBox.instances), 1)
        netguard.reset_turn()
        self.assertEqual(self.ask("deny"), "deny", "a new turn asks again")
        self.assertEqual(len(FakeBox.instances), 2)

    def test_always_stores_a_grant_with_a_timestamp_and_later_calls_skip_the_dialog(self):
        self.assertEqual(self.ask("always"), "always")
        (grant,) = netguard.list_grants()
        self.assertEqual((grant["tool"], grant["host"]), ("fetch_url", "datasheets.raspberrypi.com"))
        self.assertRegex(grant["granted_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$")
        netguard.reset_turn()
        FakeBox.instances = []
        self.assertEqual(self.ask("deny"), "allow")
        self.assertEqual(FakeBox.instances, [], "a stored grant answers without a dialog")
        self.assertTrue(any(f.get("decision") == "allow_saved" for name, f in self.events if name == "net_permission"))

    def test_the_grant_is_per_tool_and_per_host(self):
        self.ask("always")
        netguard.reset_turn()
        self.assertEqual(self.ask("deny", tool="web_search"), "deny", "another tool is asked")
        netguard.reset_turn()
        self.assertEqual(self.ask("deny", host="evil.example.com"), "deny", "another host (a redirect target) is asked")
        netguard.reset_turn()
        self.assertEqual(self.ask("deny", host="DATASHEETS.raspberrypi.com."), "allow", "host match ignores case and the trailing dot")

    def test_deny_is_cached_for_the_turn_per_pair_not_globally(self):
        self.assertEqual(self.ask("deny"), "deny")
        self.assertEqual(self.ask("once", host="other.example.com"), "allow")
        self.assertEqual(self.ask("once"), "deny")

    def test_a_grant_is_not_stored_twice(self):
        self.assertTrue(netguard.add_grant("fetch_url", "a.example.com"))
        self.assertFalse(netguard.add_grant("fetch_url", "A.example.com"))
        self.assertEqual(len(netguard.list_grants()), 1)

    def test_the_env_override_still_allows_unattended_runs_and_stores_nothing(self):
        with mock.patch.dict(os.environ, {"OSCAR_ALLOW_NET": "1"}):
            self.assertEqual(self.ask("deny"), "allow")
        self.assertEqual((FakeBox.instances, netguard.list_grants()), ([], []))

    def test_dialog_logs_decision_with_tool_and_host_only(self):
        self.ask("always")
        decisions = [f for name, f in self.events if name == "net_permission"]
        self.assertEqual((decisions[-1]["decision"], decisions[-1]["tool"], decisions[-1]["host"]),
                         ("always", "fetch_url", "datasheets.raspberrypi.com"))

    def test_always_still_allows_the_turn_when_the_settings_file_is_locked(self):
        """The grant cannot be saved right now (params.SettingsUnavailable): the click is not turned into a deny."""
        from oscar.cad import dispatch

        def dispatch_like_the_gui(fn, timeout=0, op="", **kw):   # the real dispatcher reports an exception as a failure
            try:
                return fn(), False
            except BaseException as e:  # noqa: BLE001
                return f"{type(e).__name__}: {e}", True

        with mock.patch.object(dispatch, "dispatch", side_effect=dispatch_like_the_gui), \
                mock.patch.object(params, "add_net_grant", side_effect=params.SettingsUnavailable(16, "locked")):
            self.assertEqual(self.ask("always"), "always")
        self.assertEqual(netguard.list_grants(), [])
        self.assertTrue(any(f.get("decision") == "grant_not_saved" for name, f in self.events if name == "net_permission"))

    def test_no_dialog_in_a_console_session_means_deny_and_nothing_is_stored(self):
        from PySide import QtWidgets

        with mock.patch.object(QtWidgets.QApplication, "instance", staticmethod(lambda: None), create=True):
            self.assertEqual(netguard.request_permission("s", tool="fetch_url", host="a.example.com"), "deny")
        self.assertEqual(netguard.list_grants(), [])


class UnscopedAndLegacyTest(GrantCase):
    """The script gate and download_file name no host; the legacy global flag grants nothing any more."""

    def test_unscoped_always_stores_the_visible_network_star_grant(self):
        FakeBox.click = "always"
        self.assertEqual(netguard.request_permission("wants the network", "x"), "always")
        self.assertEqual([(g["tool"], g["host"]) for g in netguard.list_grants()], [("network", "*")])
        (box,) = FakeBox.instances
        self.assertIn("Always allow", box.buttons)
        self.assertTrue(netguard.network_granted())

    def test_the_legacy_flag_grants_nothing(self):
        params.set_bool("AllowNetwork", True)
        self.assertTrue(netguard.legacy_flag())
        self.assertFalse(netguard.network_granted())
        FakeBox.click = "deny"
        self.assertEqual(netguard.request_permission("wants the network", "x"), "deny")
        self.assertEqual(len(FakeBox.instances), 1, "the legacy flag no longer skips the dialog")
        self.assertEqual(netguard.request_permission("s", tool="fetch_url", host="a.example.com"), "deny")

    def test_a_scoped_grant_does_not_unlock_the_script_gate(self):
        netguard.add_grant("fetch_url", "a.example.com")
        self.assertFalse(netguard.network_granted())

    def test_unscoped_turn_decision_still_feeds_network_granted(self):
        netguard._TURN_DECISION["download"] = "allow"
        self.assertTrue(netguard.network_granted())
        netguard._TURN_DECISION["download"] = "deny"
        self.assertFalse(netguard.network_granted())


class RevokeTest(GrantCase):
    def test_revoke_one_removes_it_and_the_turn_cache(self):
        self.ask("always")
        self.ask("always", host="b.example.com")
        self.assertTrue(netguard.revoke_grant("fetch_url", "datasheets.raspberrypi.com"))
        self.assertEqual([g["host"] for g in netguard.list_grants()], ["b.example.com"])
        FakeBox.instances = []
        self.assertEqual(self.ask("deny"), "deny", "revoked mid-turn: the cached allow is gone and the user is asked again")
        self.assertEqual(len(FakeBox.instances), 1)
        self.assertFalse(netguard.revoke_grant("fetch_url", "never.example.com"))

    def test_revoke_all_clears_grants_the_legacy_flag_and_the_turn_cache(self):
        self.ask("always")
        self.ask("once", host="c.example.com")
        params.set_bool("AllowNetwork", True)
        self.assertEqual(netguard.revoke_all(), 2)
        self.assertEqual(netguard.list_grants(), [])
        self.assertFalse(netguard.legacy_flag())
        self.assertEqual(netguard._TURN_DECISION, {})
        self.assertEqual(netguard.revoke_all(), 0)

    def test_revocation_is_logged(self):
        netguard.add_grant("fetch_url", "a.example.com")
        netguard.revoke_grant("fetch_url", "a.example.com")
        self.assertTrue(any(name == "net_grant_revoked" and f["host"] == "a.example.com" for name, f in self.events))


class PersistenceTest(unittest.TestCase):
    def setUp(self):
        params.clear_net_grants()
        self.addCleanup(params.clear_net_grants)

    def test_grants_survive_in_the_settings_file_as_tool_host_granted_at(self):
        params.add_net_grant("fetch_url", "a.example.com")
        with open(params.settings_path(), encoding="utf-8") as fh:
            stored = json.load(fh)["net_grants"]
        self.assertEqual([set(g) for g in stored], [{"tool", "host", "granted_at"}])

    def test_garbage_entries_in_the_file_are_ignored(self):
        params.add_net_grant("fetch_url", "ok.example.com")
        with open(params.settings_path(), encoding="utf-8") as fh:
            state = json.load(fh)
        state["net_grants"] += ["x", 3, {"tool": 1, "host": "h"}, {"tool": "t"}, {"tool": "", "host": "h"},
                                {"tool": "fetch_url", "host": "ok.example.com"}]
        with open(params.settings_path(), "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        self.assertEqual([g["host"] for g in params.get_net_grants()], ["ok.example.com"])
        state["net_grants"] = "nonsense"
        with open(params.settings_path(), "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        self.assertEqual(params.get_net_grants(), [])

    def test_clear_and_remove_report_what_they_did(self):
        params.add_net_grant("a", "h1")
        params.add_net_grant("b", "h2")
        self.assertTrue(params.remove_net_grant("a", "h1"))
        self.assertFalse(params.remove_net_grant("a", "h1"))
        self.assertEqual(params.clear_net_grants(), 1)
        self.assertEqual(params.clear_net_grants(), 0)


class AuditTest(unittest.TestCase):
    def test_audit_use_logs_the_host_and_never_the_url_path_or_query(self):
        events = []
        with mock.patch("oscar.turnlog.log_event", lambda name, **f: events.append((name, f))):
            netguard.audit_use("fetch_url", "https://User:pw@Docs.Example.com:8443/a/b?token=SECRET#frag", outcome="ok", bytes=5)
            netguard.audit_use("fetch_url", "http://[bad")
        self.assertEqual(events[0], ("net_use", {"tool": "fetch_url", "host": "docs.example.com", "outcome": "ok", "bytes": 5}))
        self.assertNotIn("SECRET", json.dumps(events))
        self.assertNotIn("pw", json.dumps(events))
        self.assertEqual(events[1][1]["host"], "?")


if __name__ == "__main__":
    unittest.main()
