"""Follow-the-build of the Oscar panel, as a mixin of ``OscarPanel``.

While a turn builds geometry the viewport can follow it: after each step that added or changed objects those
objects are selected for a moment (so the user sees what the step made) and the camera frames the model. The
camera is the user's: a fit happens only when the camera is exactly where the LAST fit left it, i.e. the user has
not orbited, panned or zoomed since; a camera the user moved is never touched (until an explicit action such as
opening a file or a rewind forces a fit). The preference is persisted in the app settings (``FollowBuild``,
default on) and toggled from the checkable "Follow the build" entry of the panel's header menu.

The mixin relies on the host panel for ``_follow_build`` / ``_fit_sig`` (set by ``_init_follow``) and Qt / FreeCAD.
"""

import FreeCAD as App
import FreeCADGui as Gui
from PySide import QtCore

from . import params as _params

_PREF = "FollowBuild"
_FLASH_MS = 1500


def select_objects(app, gui, names: list[str]) -> list[str]:
    """Select the objects ``names`` of the active (or the first matching) document through the ``app`` / ``gui``
    modules; returns the names that exist (``[]`` when none does or anything fails)."""
    try:
        doc = app.ActiveDocument
        if doc is None or not any(doc.getObject(n) is not None for n in names):
            doc = next((d for d in app.listDocuments().values()
                        if any(d.getObject(n) is not None for n in names)), doc)
        present = [n for n in names if doc is not None and doc.getObject(n) is not None]
        if not present:
            return []
        if app.ActiveDocument is None or app.ActiveDocument.Name != doc.Name:
            gui.setActiveDocument(doc.Name)
        gui.Selection.clearSelection()
        for name in present:
            gui.Selection.addSelection(doc.Name, name)
        return present
    except Exception:  # noqa: BLE001
        return []


class FollowMixin:
    """Follow-the-build methods of the panel (see the module docstring)."""

    def _init_follow(self) -> None:
        """Read the persisted preference (default: on) and forget any fit."""
        try:
            self._follow_build = bool(_params.get_bool(_PREF, True))
        except Exception:  # noqa: BLE001
            self._follow_build = True
        self._fit_sig = None     # camera signature right after our last fit (None: never fit)
        self._flash_names: list[str] = []

    def _set_follow_build(self, on) -> None:
        """Slot of the menu action: change and persist the preference."""
        self._follow_build = bool(on)
        try:
            _params.set_bool(_PREF, self._follow_build)
        except Exception as e:  # noqa: BLE001
            self._log_panel_error("follow_pref", e)

    @staticmethod
    def _camera_signature():
        """A hashable picture of the active 3D view's camera (position, orientation, zoom) or ``None`` when there
        is no 3D view or the camera cannot be read."""
        try:
            doc = Gui.ActiveDocument
            view = doc.ActiveView if doc is not None else None
            if view is None or not hasattr(view, "getCameraNode"):
                return None
            cam = view.getCameraNode()
            sig = [tuple(round(v, 4) for v in cam.position.getValue().getValue()),
                   tuple(round(v, 5) for v in cam.orientation.getValue().getValue())]
            for attr in ("height", "heightAngle"):
                field = getattr(cam, attr, None)
                if field is not None:
                    sig.append(round(float(field.getValue()), 4))
            return tuple(sig)
        except Exception:  # noqa: BLE001
            return None

    def _user_moved_camera(self) -> bool:
        """True when the camera differs from where the last fit left it (never true before the first fit or
        when the camera cannot be read)."""
        now = self._camera_signature()
        return now is not None and self._fit_sig is not None and now != self._fit_sig

    def _fit_camera(self, force: bool = False) -> bool:
        """Frame the model once, if allowed: ``force`` (an explicit user action) always; otherwise only with
        follow-the-build on and a camera the user has not moved since the last fit. True when it fitted."""
        if not force and (not self._follow_build or self._user_moved_camera()):
            return False
        try:
            if Gui.ActiveDocument is None:
                return False
            view = Gui.ActiveDocument.ActiveView
            if view is None or not hasattr(view, "saveImage"):   # not a 3D view
                return False
            from .cad.views import fit_view

            fit_view()
            Gui.updateGui()
        except Exception:  # noqa: BLE001
            return False   # framing is cosmetic: never fail the turn on it
        self._fit_sig = self._camera_signature()
        return True

    def _select_objects(self, names: list[str]) -> list[str]:
        """Select the objects ``names`` (see ``select_objects``); returns the names found."""
        return select_objects(App, Gui, names)

    def _follow_step(self, view) -> bool:
        """After a finished step: flash the objects it created (select now, clear after a moment unless the user
        selected something else meanwhile) and fit the camera if the user has not moved it. True when it ran."""
        if not self._follow_build or not view.ok or view.skipped:
            return False
        names = view.changed_objects()
        if not names:
            return False
        present = self._select_objects(names)
        if present:
            self._flash_names = present
            QtCore.QTimer.singleShot(_FLASH_MS, self._end_flash)
        self._fit_camera()
        return True

    def _end_flash(self) -> None:
        """Clear the flash selection when it is still exactly what the flash selected."""
        names, self._flash_names = self._flash_names, []
        try:
            if names and sorted(o.Name for o in Gui.Selection.getSelection()) == sorted(names):
                Gui.Selection.clearSelection()
        except Exception:  # noqa: BLE001
            pass
