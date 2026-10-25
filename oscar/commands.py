"""Application GUI commands for the Oscar AI workbench.

Registered once at GUI startup (InitGui imports this module), so the Oscar
panel and bridge are reachable from the workbench menu, the Python console
(``Gui.runCommand("Oscar_ShowPanel")``) and the keyboard-shortcut editor —
without switching to the workbench first.

``Oscar_SignIn`` opens Settings and starts the browser sign-in to OriginShape (when signed out);
``Oscar_Account`` opens Settings on the OriginShape account (email, plan, credits, Manage account, Sign out);
``Oscar_CheckUpdates`` asks the OriginShape server whether a newer Oscar is out (``oscar.updates``) and says so.
A sign-in or sign-out takes effect without Save, so the open panel is refreshed after any of these dialogs when it
was saved OR its account changed (``SettingsDialog.account_changed``).
"""

import threading

import FreeCAD
import FreeCADGui as Gui
from PySide import QtCore, QtGui, QtWidgets

_RESOURCE_CACHE: dict[str, str] = {}
_PENDING: list = []   # the timer of a running update check (kept alive until it reports)


def _icon(name: str) -> str:
    from . import paths

    return paths.icon(name)


class ShowOscarCommand:
    def GetResources(self):
        return {
            "Pixmap": _icon("Oscar.svg"),
            "MenuText": "Oscar chat panel",
            "ToolTip": "Show or raise the Oscar AI assistant panel",
            "Accel": "Ctrl+Shift+O",
        }

    def Activated(self):
        from . import panel

        panel.show_panel()

    def IsActive(self):
        return True


def refresh_panel_profiles() -> bool:
    """Make an open Oscar panel re-read the profiles (its picker and model label). True when a panel was refreshed.

    Settings saved from the menu command (not from the panel's own gear button) would otherwise leave the panel's
    picker showing a profile name / model that is no longer the active one. Never raises."""
    try:
        from . import panel as panel_mod

        dock = Gui.getMainWindow().findChild(QtWidgets.QDockWidget, panel_mod.DOCK_OBJECT_NAME)
        target = dock.widget() if dock is not None else None
        reload_profiles = getattr(target, "_reload_profiles", None)
        if not callable(reload_profiles):
            return False
        reload_profiles()
        return True
    except Exception as e:  # noqa: BLE001 - a stale label is not worth an error dialog
        FreeCAD.Console.PrintLog(f"Oscar: could not refresh the panel's profiles: {e}\n")
        return False


def run_settings_dialog(account_action: str = "") -> None:
    """The settings dialog, modal; ``account_action`` (``"signin"`` / ``"device"``) starts that sign-in at once. The
    open panel is refreshed when the dialog was saved or its account changed (that applies without Save)."""
    from . import settings_ui

    dialog = settings_ui.SettingsDialog(Gui.getMainWindow())
    if account_action:
        start = getattr(dialog, "start_account_action", None)
        if callable(start):
            start(account_action)
    if dialog.exec() or getattr(dialog, "account_changed", False):
        refresh_panel_profiles()
    dispose = getattr(dialog, "deleteLater", None)   # a closed dialog must not keep following the account hub
    if callable(dispose):
        dispose()


class SettingsCommand:
    def GetResources(self):
        return {
            "Pixmap": _icon("Oscar.svg"),
            "MenuText": "Oscar settings…",
            "ToolTip": "Your OriginShape account, chat providers, models, API keys, network permissions and scripts",
        }

    def Activated(self):
        run_settings_dialog()

    def IsActive(self):
        return True


class SignInCommand:
    def GetResources(self):
        return {
            "Pixmap": _icon("Oscar.svg"),
            "MenuText": "Sign in to OriginShape…",
            "ToolTip": "Sign in with your OriginShape account in the browser, so Oscar can chat without an API key",
        }

    def Activated(self):
        run_settings_dialog("signin")

    def IsActive(self):
        return True


class AccountCommand:
    def GetResources(self):
        return {
            "Pixmap": _icon("Oscar.svg"),
            "MenuText": "OriginShape account…",
            "ToolTip": "Your OriginShape account: email, plan, credits, Manage account and Sign out",
        }

    def Activated(self):
        run_settings_dialog()

    def IsActive(self):
        return True


class BridgeToggleCommand:
    def GetResources(self):
        return {
            "Pixmap": _icon("bridge.svg"),
            "MenuText": "Toggle Oscar MCP bridge",
            "ToolTip": "Start/stop the Oscar MCP bridge for external clients (port 9876)",
        }

    def Activated(self):
        from . import bridge

        if bridge.is_running():
            message = bridge.stop()
        else:
            message = bridge.start()
        try:
            Gui.getMainWindow().statusBar().showMessage(message, 8000)
        except Exception:
            pass

    def IsActive(self):
        return True



class CheckUpdatesCommand:
    def GetResources(self):
        return {
            "Pixmap": _icon("Oscar.svg"),
            "MenuText": "Check for Updates…",
            "ToolTip": "Ask OriginShape whether a newer Oscar is out",
        }

    def Activated(self):
        check_for_updates()

    def IsActive(self):
        return True


def check_for_updates() -> None:
    """Oscar → Check for Updates…: ``updates.fetch`` on a daemon thread (FreeCAD stays responsive), then a message
    with the answer; a newer Oscar also shows in the panel (the account hub takes the answer in)."""
    from . import __version__, updates
    from .auth.transport import AuthError

    result: dict = {}

    def work() -> None:
        try:
            result["release"] = updates.fetch()
        except AuthError as e:                       # unreachable, refused: the message is safe to show
            result["error"] = str(e)
        except Exception as e:  # noqa: BLE001 - a bug: only its type
            result["error"] = f"unexpected {type(e).__name__}"

    thread = threading.Thread(target=work, name="oscar-update-check-now", daemon=True)
    thread.start()
    timer = QtCore.QTimer()
    timer.setInterval(100)

    def done() -> None:
        if thread.is_alive():
            return
        timer.stop()
        _PENDING.remove(timer)
        _show_update_result(result, __version__)

    timer.timeout.connect(done)
    _PENDING.append(timer)
    timer.start()


def _show_update_result(result: dict, current: str) -> None:
    from .settings_ui import account_hub

    parent = Gui.getMainWindow()
    release = result.get("release")
    if release is None:
        QtWidgets.QMessageBox.warning(parent, "Oscar", f"Oscar couldn't check for updates: {result.get('error')}")
        return
    account_hub().note_release()                     # the panel shows the answer too
    update = release.update(current)
    if not update:
        QtWidgets.QMessageBox.information(parent, "Oscar", f"You have the latest Oscar ({current}).")
        return
    text = f"Oscar {release.latest_version} is available (you have {current})."
    if update == "required":
        text += " Your OriginShape account works again once Oscar is updated; your own AI providers work meanwhile."
    box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, "Oscar",
                                text + "\n\nUpdate it from FreeCAD's Addon Manager, or download it.", parent=parent)
    get = box.addButton("Open download page", QtWidgets.QMessageBox.AcceptRole) if release.download_url else None
    box.addButton(QtWidgets.QMessageBox.Close)
    box.exec_()
    if get is not None and box.clickedButton() is get:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(release.download_url))


def register_all() -> None:
    commands = (
        ("Oscar_ShowPanel", ShowOscarCommand),
        ("Oscar_Settings", SettingsCommand),
        ("Oscar_ToggleBridge", BridgeToggleCommand),
        ("Oscar_SignIn", SignInCommand),
        ("Oscar_Account", AccountCommand),
        ("Oscar_CheckUpdates", CheckUpdatesCommand),
    )
    for name, cls in commands:
        if name not in Gui.listCommands():
            Gui.addCommand(name, cls())
