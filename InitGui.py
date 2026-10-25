# SPDX-License-Identifier: Apache-2.0
# InitGui.py: runs at GUI startup in every FreeCAD session that has the Oscar addon installed.
#
# FreeCAD exec()s this file from inside a method with exec-locals separate from globals, so names
# assigned at this level are NOT visible inside class bodies or functions defined here. Everything a
# class or function needs is computed inside it (the upstream Test-workbench pattern).

import os
import sys

try:
    _addon_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:  # FreeCAD runs this file without __file__: find the Mod folder that holds this addon
    import FreeCAD as _App

    _addon_dir = ""
    for _base in (os.path.join(_App.getUserAppDataDir(), "Mod"), os.path.join(_App.getResourceDir(), "Mod")):
        for _name in sorted(os.listdir(_base)) if os.path.isdir(_base) else ():
            _candidate = os.path.join(_base, _name)
            if os.path.isfile(os.path.join(_candidate, "oscar", "cad", "dispatch.py")):
                _addon_dir = _candidate
                break
        if _addon_dir:
            break

# Never load next to a different ``oscar`` package (a second copy of this addon): if one is already
# imported, step aside and say which folder to remove.
_other = sys.modules.get("oscar")
_conflict = ""
if _other is not None and not os.path.realpath(getattr(_other, "__file__", "")).startswith(
        os.path.join(os.path.realpath(_addon_dir), "")):   # trailing separator: Mod/OscarDev is not inside Mod/Oscar
    _conflict = os.path.dirname(os.path.dirname(getattr(_other, "__file__", "?")))

if _conflict:
    import FreeCAD as _App

    _App.Console.PrintWarning(f"Oscar: not loaded from {_addon_dir}: another Oscar is already installed at "
                              f"{_conflict}. Remove one of them.\n")
else:
    if _addon_dir and _addon_dir not in sys.path:
        sys.path.insert(0, _addon_dir)

    # Arm the CAD engine's cross-thread dispatcher on the GUI thread (Qt affinity). Everything heavier (the
    # engine, the tool registry) loads on first use: when the panel opens or the bridge starts. FreeCAD runs
    # this file at every start, Oscar used or not.
    from oscar.cad import dispatch as _dispatch  # noqa: E402

    _dispatch.init()

    import FreeCADGui as Gui  # noqa: E402

    class OscarWorkbench(Workbench):  # noqa: F821 - injected by FreeCAD
        MenuText = "Oscar"
        ToolTip = "Oscar, the AI copilot for FreeCAD"

        def __init__(self):
            from oscar import paths as _paths

            self.__class__.Icon = _paths.icon("Oscar.svg")

        def Initialize(self):
            from oscar import commands as _commands

            _commands.register_all()
            items = ["Oscar_ShowPanel", "Oscar_Settings", "Oscar_ToggleBridge"]
            self.appendToolbar("Oscar", items)
            # the menu also offers the OriginShape account (sign in, and the account itself)
            self.appendMenu(["Oscar"], items + ["Separator", "Oscar_SignIn", "Oscar_Account", "Oscar_CheckUpdates"])

        def Activated(self):
            from oscar import panel

            panel.show_panel()

        def Deactivated(self):
            pass

        def ContextMenu(self, recipient):
            pass

        def GetClassName(self):
            return "Gui::PythonWorkbench"

    Gui.addWorkbench(OscarWorkbench())

    # Commands also register outside Initialize(), so the panel can be opened from the
    # Python console or a shortcut without activating the workbench first.
    import oscar.commands as _oc  # noqa: E402

    _oc.register_all()

    def _oscar_autostart():
        try:
            from oscar import params as _params

            if _params.get_bool("BridgeAutoStart", False):
                from oscar import bridge

                bridge.start()
            if _params.get_bool("PanelAutoOpen", False):
                from oscar import panel

                panel.show_panel()
        except Exception as e:  # never break FreeCAD startup over Oscar
            import FreeCAD

            FreeCAD.Console.PrintWarning(f"Oscar autostart: {e}\n")

    from PySide import QtCore  # noqa: E402

    QtCore.QTimer.singleShot(0, _oscar_autostart)
