"""Viewport orientation and screenshot capture.

Orientations map to the application's own standard-view commands, so the
result always matches what a user gets from the View menu. Capture goes
through the active 3D view's renderer; non-3D views (sheets, spreadsheets)
report themselves as non-capturable instead of returning a blank image.

Focusing uses visibility toggling rather than camera math: hide everything
except the target, fit, capture, restore. It costs one re-render and works
for any object regardless of nesting.

The user's camera is sacred: every capture saves it first and restores it
afterwards. The agent may capture many times per turn, and without the
restore each one would re-orient and re-fit the user's view mid-build —
the view would fight whoever is navigating it and the zoom would be fit
to a half-built model. The end-of-turn fit (panel) frames the finished
model once, in the user's own orientation.

Two rules keep a picture honest. A console session has no 3D view: ``FreeCADGui`` then has no
``getDocument`` / ``ActiveDocument`` (AttributeError, not None), so everything is gated on
``App.GuiUp`` first and reports "no 3D view (headless session)". And a blank frame is never
evidence: after saving, a 64 x 48 grid of the image is sampled; a sample is background when
every channel is within 6 of the median colour of its row (tolerant of the gradient and the
anti-aliased edge of the default background), and a frame with more than 99.5% background
samples is blank. A blank frame gets one ``fit_view()`` retry, then ``capture_checked``
answers ``(False, "blank frame")``.
"""

import os

import FreeCAD as App
import FreeCADGui as Gui

# orientation name -> standard view command
ORIENTATIONS = {
    "Front": "Std_ViewFront",
    "Back": "Std_ViewRear",
    "Left": "Std_ViewLeft",
    "Right": "Std_ViewRight",
    "Top": "Std_ViewTop",
    "Bottom": "Std_ViewBottom",
    "Isometric": "Std_ViewIsometric",
    "Dimetric": "Std_ViewDimetric",
    "Trimetric": "Std_ViewTrimetric",
}

_LONGEST_EDGE_CAP = 1024


def fit_view() -> bool:
    """Frame the active 3D view on visible geometry, across builds.

    FreeCAD 1.1 renamed the Std command (Std_ViewFit → Std_ViewFitAll);
    the message-based fallback covers builds where neither command is
    registered. Returns False silently when there is nothing to fit
    (console sessions) — framing is cosmetic and must never fail a tool.
    """
    try:
        import FreeCADGui as Gui

        commands = Gui.listCommands()
        for name in ("Std_ViewFitAll", "Std_ViewFit"):
            if name in commands:
                Gui.runCommand(name)
                return True
        Gui.SendMsgToActiveView("ViewFit")
        return True
    except Exception:
        return False



def orient(view, orientation: str) -> None:
    """Switch ``view`` to the named standard orientation (a name of ``ORIENTATIONS``); an unknown name is ignored. GUI-thread only."""
    command = ORIENTATIONS.get(orientation)
    if command:
        Gui.runCommand(command)


def _viewport_size(view) -> tuple[int, int]:
    """Viewport pixels: View3DInventor.getSize(), else width/height
    attributes or methods, else a sane default."""
    try:
        w, h = (int(v) for v in view.getSize())
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    try:
        w, h = getattr(view, "width"), getattr(view, "height")
        w, h = int(w() if callable(w) else w), int(h() if callable(h) else h)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 800, 600


def _image_size(view, width: int | None, height: int | None) -> tuple[int, int]:
    """Output size: the requested one, else the viewport's — with the
    missing side derived from the viewport aspect and the long edge capped
    so screenshots stay cheap for vision turns (a Retina viewport would
    otherwise ship multi-megapixel PNGs with every tool call)."""
    view_w, view_h = _viewport_size(view)
    w = int(width or 0)
    h = int(height or 0)
    if w <= 0 and h <= 0:
        w, h = view_w, view_h
    elif w <= 0:
        w = max(1, round(h * view_w / view_h))
    elif h <= 0:
        h = max(1, round(w * view_h / view_w))
    longest = max(w, h)
    if longest > _LONGEST_EDGE_CAP:
        scale = _LONGEST_EDGE_CAP / longest
        w, h = max(1, int(w * scale)), max(1, int(h * scale))
    return w, h


HEADLESS_REASON = "no 3D view (headless session)"
BLANK_REASON = "blank frame"

_GRID_COLS, _GRID_ROWS = 64, 48
_BACKGROUND_TOLERANCE = 6
_BLANK_FRACTION = 0.995


def is_blank_grid(rows: list[list[tuple[int, int, int]]]) -> bool:
    """True when a sampled image is blank (pure; ``rows`` is a list of rows of ``(r, g, b)``).

    A sample is background when every channel is within 6 of the per-row median colour; the
    frame is blank when more than 99.5% of the samples are background. An empty grid is blank.
    """
    background = total = 0
    for row in rows:
        if not row:
            continue
        medians = [sorted(px[c] for px in row)[len(row) // 2] for c in range(3)]
        for px in row:
            total += 1
            if all(abs(px[c] - medians[c]) <= _BACKGROUND_TOLERANCE for c in range(3)):
                background += 1
    return total == 0 or background / total > _BLANK_FRACTION


def is_blank_image(path: str) -> bool:
    """True when the PNG at ``path`` is blank (see ``is_blank_grid``); unreadable = blank."""
    try:
        from PySide import QtGui

        image = QtGui.QImage(path)
        if image.isNull() or image.width() < 1 or image.height() < 1:
            return True
        rows = []
        for j in range(_GRID_ROWS):
            y = min(int((j + 0.5) * image.height() / _GRID_ROWS), image.height() - 1)
            row = []
            for i in range(_GRID_COLS):
                x = min(int((i + 0.5) * image.width() / _GRID_COLS), image.width() - 1)
                color = image.pixelColor(x, y)
                row.append((color.red(), color.green(), color.blue()))
            rows.append(row)
        return is_blank_grid(rows)
    except Exception:  # noqa: BLE001 - cannot judge: do not discard a possibly good picture
        return False


def _active_view(doc_name: str | None):
    """The 3D view to capture (the named document brought to the front first), or None."""
    if doc_name and Gui.getDocument(doc_name) is not None:
        active = Gui.ActiveDocument
        if active is None or active.Document.Name != doc_name:
            try:
                Gui.setActiveDocument(doc_name)
            except Exception:
                pass
    view = Gui.ActiveDocument.ActiveView if Gui.ActiveDocument else None
    if view is None or not hasattr(view, "saveImage"):
        return None
    return view


def capture_checked(
    path: str,
    orientation: str = "Isometric",
    width: int | None = None,
    height: int | None = None,
    focus: str | None = None,
    doc_name: str | None = None,
    *,
    check_blank: bool = True,
) -> tuple[bool, str]:
    """Capture a 3D view to ``path`` as PNG; returns ``(ok, reason)`` (``reason`` is ``""`` when ok).

    ``doc_name`` names the document to show (the one a tool just changed);
    it is brought to the front first — otherwise the capture would show
    whichever document happened to be active, and the model would conclude
    its change never happened. Reasons: ``"no 3D view (headless session)"`` (no GUI: nothing of
    ``FreeCADGui`` is touched), ``"no 3D view is active (...)"``, ``"blank frame"`` (after one
    ``fit_view()`` retry; skipped with ``check_blank=False``) and ``"capture failed: ..."``.
    GUI-thread only. The user's camera is restored afterwards.
    """
    if not App.GuiUp:
        return False, HEADLESS_REASON
    try:
        view = _active_view(doc_name)
    except AttributeError:
        return False, HEADLESS_REASON
    if view is None:
        App.Console.PrintWarning(
            "Oscar: the active view does not support image capture\n"
        )
        return False, ("no 3D view is active (a TechDraw or Spreadsheet view may be in front, "
                       "or no document is open)")

    # Save the user's camera BEFORE orienting: restoring a camera saved
    # after the orient command would "restore" the isometric view.
    saved_camera = None
    if hasattr(view, "getCamera") and hasattr(view, "setCamera"):
        try:
            saved_camera = view.getCamera()
        except Exception:
            saved_camera = None

    size = _image_size(view, width, height)
    hidden: list = []
    target_was_hidden = None  # the focus target's view object, if it started hidden
    doc = Gui.ActiveDocument.Document
    try:
        orient(view, orientation)
        if focus:
            target = doc.getObject(focus)
            if target is None:
                App.Console.PrintWarning(
                    f"Oscar: cannot focus '{focus}' — no such object\n"
                )
            else:
                for other in doc.Objects:
                    view_object = getattr(other, "ViewObject", None)
                    if other is not target and view_object is not None and view_object.Visibility:
                        view_object.Visibility = False
                        hidden.append(view_object)
                if getattr(target, "ViewObject", None) is not None:
                    if not target.ViewObject.Visibility:
                        target_was_hidden = target.ViewObject
                    target.ViewObject.Visibility = True
        fit_view()
        Gui.updateGui()
        view.saveImage(path, *size)
        if check_blank and is_blank_image(path):
            fit_view()
            Gui.updateGui()
            view.saveImage(path, *size)
            if is_blank_image(path):
                return False, BLANK_REASON
        return True, ""
    except Exception as e:  # noqa: BLE001 - reported, never raised into a tool
        return False, f"capture failed: {type(e).__name__}: {e}"
    finally:
        for view_object in hidden:
            view_object.Visibility = True
        if target_was_hidden is not None:
            try:
                target_was_hidden.Visibility = False
            except Exception:
                pass
        if saved_camera is not None:
            try:
                view.setCamera(saved_camera)
                Gui.updateGui()
            except Exception:
                App.Console.PrintWarning(
                    "Oscar: could not restore the user's camera after capture\n"
                )


def capture(
    path: str,
    orientation: str = "Isometric",
    width: int | None = None,
    height: int | None = None,
    focus: str | None = None,
    doc_name: str | None = None,
) -> bool:
    """Capture a 3D view to ``path`` as PNG. False when unsupported or the frame is blank.

    The boolean form of ``capture_checked`` (which says WHY it failed): never raises in a
    console session, never reports a blank frame as a picture."""
    return capture_checked(path, orientation, width, height, focus, doc_name)[0]


def capture_base64_checked(
    orientation: str = "Isometric",
    width: int | None = None,
    height: int | None = None,
    focus: str | None = None,
    doc_name: str | None = None,
) -> tuple[str | None, str]:
    """Capture to a temp file; returns ``(png_base64 | None, reason)``. GUI-thread only."""
    import base64
    import tempfile

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        ok, reason = capture_checked(tmp_path, orientation, width, height, focus, doc_name)
        if not ok:
            return None, reason
        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def capture_base64(
    orientation: str = "Isometric",
    width: int | None = None,
    height: int | None = None,
    focus: str | None = None,
    doc_name: str | None = None,
) -> str | None:
    """Capture to a temp file and return the PNG as base64 (or None)."""
    return capture_base64_checked(orientation, width, height, focus, doc_name)[0]
