"""The pen next to the paperclip: draw on the 3D view to show Oscar where to change the model.

``start(panel)`` takes a picture of the active 3D view and opens it in a drawing dialog (``MarkupDialog``). The
view holds still while the user draws (the picture is a still: orbiting would move the model from under the marks).
Drag to mark; **Undo** (Ctrl+Z) takes the last stroke back, **Clear** all of them, **Cancel** (Escape) leaves,
**Attach to message** (Enter) attaches. Attaching:

- the stroke points are picked in the model (``View3DInventor.getObjectInfo``, at most ``markup.PICKED_POINTS``):
  the part under each one, the point in global mm and the surface normal there;
- the picture with the strokes drawn over it is saved as ``markup-HHMMSS.png`` and queued as an ordinary image
  attachment, so the model sees it;
- the draft starts with ``markup.describe``'s line naming the parts, the side and the region, and the user writes
  what to change after it.

Qt + FreeCAD GUI; the dialog itself takes the picture and a ``pick`` callable, so tests drive it without a 3D view.
"""

import os
import tempfile
import time
from typing import Callable

from PySide import QtCore, QtGui, QtWidgets

from . import markup

STROKE = "#ff453a"
STROKE_WIDTH = 4        # logical pixels on the screen; scaled to the picture's pixels when attached
MIN_STEP_PX = 2         # a stroke point closer than this to the last one is dropped
NO_VIEW = "Open the model in a 3D view first: the pen draws on the 3D view."

Pick = Callable[[list], list]   # [(u, v)] (0..1, from the picture's top left) -> [markup.MarkHit]


def capture() -> "tuple[QtGui.QImage, Pick] | str":
    """A picture of the active 3D view at its own resolution, and the ``pick`` for points on it; or why not (words).
    GUI thread only."""
    try:
        import FreeCADGui as Gui
    except ImportError:
        return NO_VIEW
    gui_doc = getattr(Gui, "ActiveDocument", None)
    view = getattr(gui_doc, "ActiveView", None) if gui_doc is not None else None
    if view is None or not all(hasattr(view, name) for name in ("saveImage", "getObjectInfo", "getSize")):
        return NO_VIEW
    width, height = (int(value) for value in view.getSize())
    if width <= 0 or height <= 0:
        return NO_VIEW
    handle, path = tempfile.mkstemp(prefix="oscar-view-", suffix=".png")
    os.close(handle)
    try:
        view.saveImage(path, width, height, "Current")
        image = QtGui.QImage(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if image.isNull():
        return "The 3D view couldn't be captured."

    def pick(points: list) -> list:
        hits = []
        for u, v in points:
            # the picture's (u, v) from its top left; the view picks in its own pixels, from the bottom left
            try:
                info = view.getObjectInfo((int(u * width), int(height - 1 - v * height)))
            except Exception:  # noqa: BLE001 - a point off the model is no hit
                info = None
            hit = _hit(info) if info else None
            if hit is not None:
                hits.append(hit)
        return hits

    return image, pick


def _hit(info: dict) -> "markup.MarkHit | None":
    """A ``getObjectInfo`` result as a ``MarkHit``: the object's label, the point, the normal of the face (or mesh
    facet) there."""
    import FreeCAD as App

    try:
        doc = App.getDocument(info.get("Document")) if info.get("Document") else App.ActiveDocument
        obj = doc.getObject(info["Object"])
        point = (float(info["x"]), float(info["y"]), float(info["z"]))
    except Exception:  # noqa: BLE001
        return None
    if obj is None:
        return None
    return markup.MarkHit(part=str(obj.Label or obj.Name), point=point, normal=_normal(obj, str(info.get("Component") or ""), point))


def _normal(obj, element: str, point: tuple) -> "tuple | None":
    """The outward normal (global) of ``obj``'s ``element`` at ``point`` (global), or None."""
    import FreeCAD as App

    try:
        # obj.Shape / obj.Mesh carry the object's own placement; its containers' placement is on top
        outer = obj.getGlobalPlacement().multiply(obj.Placement.inverse())
        if element.startswith("Face"):
            face = obj.Shape.getElement(element)
            local = outer.inverse().multVec(App.Vector(*point))
            u, v = face.Surface.parameter(local)
            normal = face.normalAt(u, v)
        elif element.startswith("Facet") and getattr(obj, "TypeId", "") == "Mesh::Feature":
            normal = obj.Mesh.Facets[int(element[len("Facet"):])].Normal
        else:
            return None
        normal = outer.Rotation.multVec(normal)
        return (normal.x, normal.y, normal.z)
    except Exception:  # noqa: BLE001 - no normal means no "its top"; the region still says where
        return None


class _Canvas(QtWidgets.QWidget):
    """The picture, with the strokes over it. Strokes are kept from the picture's top left in 0..1, so they follow
    the canvas when the dialog is resized."""

    changed = QtCore.Signal()

    def __init__(self, image: QtGui.QImage, parent=None):
        super().__init__(parent)
        self.image = image
        self.strokes: list = []
        self._drawing: "list | None" = None
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setAccessibleName("Drawing area: drag over the model to mark where Oscar should change it")
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def sizeHint(self):
        ratio = self.devicePixelRatioF() or 1.0
        return QtCore.QSize(int(self.image.width() / ratio), int(self.image.height() / ratio))

    def _target(self) -> QtCore.QRectF:
        """Where the picture is drawn: as large as fits, centred, its proportions kept."""
        if self.image.width() <= 0 or self.image.height() <= 0:
            return QtCore.QRectF(self.rect())
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        w, h = self.image.width() * scale, self.image.height() * scale
        return QtCore.QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _uv(self, pos) -> tuple:
        target = self._target()
        u = (pos.x() - target.x()) / max(1.0, target.width())
        v = (pos.y() - target.y()) / max(1.0, target.height())
        return (min(1.0, max(0.0, u)), min(1.0, max(0.0, v)))

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
        self._drawing = [self._uv(event.position() if hasattr(event, "position") else event.pos())]
        self.update()

    def mouseMoveEvent(self, event):
        if self._drawing is None:
            return
        uv = self._uv(event.position() if hasattr(event, "position") else event.pos())
        target = self._target()
        last = self._drawing[-1]
        if abs(uv[0] - last[0]) * target.width() + abs(uv[1] - last[1]) * target.height() < MIN_STEP_PX:
            return
        self._drawing.append(uv)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._drawing is None:
            return
        self.strokes.append(self._drawing)
        self._drawing = None
        self.update()
        self.changed.emit()

    def undo(self):
        if self.strokes:
            self.strokes.pop()
            self.update()
            self.changed.emit()

    def clear(self):
        if self.strokes:
            self.strokes = []
            self.update()
            self.changed.emit()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#1b1e22"))
        target = self._target()
        painter.drawImage(target, self.image)
        _paint_strokes(painter, self.strokes + ([self._drawing] if self._drawing else []), target, STROKE_WIDTH)
        painter.end()

    def composite(self) -> QtGui.QImage:
        """The picture with the strokes drawn over it, at the picture's resolution."""
        out = self.image.convertToFormat(QtGui.QImage.Format_ARGB32)
        target = self._target()
        scale = out.width() / max(1.0, target.width())
        painter = QtGui.QPainter(out)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        _paint_strokes(painter, self.strokes, QtCore.QRectF(0, 0, out.width(), out.height()), STROKE_WIDTH * scale)
        painter.end()
        return out


def _paint_strokes(painter, strokes: list, target: QtCore.QRectF, width: float) -> None:
    pen = QtGui.QPen(QtGui.QColor(STROKE))
    pen.setWidthF(width)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    painter.setPen(pen)
    for stroke in strokes:
        points = [QtCore.QPointF(target.x() + u * target.width(), target.y() + v * target.height()) for u, v in stroke]
        if len(points) == 1:
            painter.drawPoint(points[0])
        else:
            painter.drawPolyline(points)


class MarkupDialog(QtWidgets.QDialog):
    """Draw on ``image`` (a picture of the 3D view); Attach leaves ``marked`` = ``(png_path, note)``."""

    def __init__(self, image: QtGui.QImage, pick: Pick, out_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mark up the model")
        self.setModal(True)
        self.marked: "tuple[str, str] | None" = None
        self._pick = pick
        self._out_dir = out_dir

        self.canvas = _Canvas(image, self)
        self.hint = QtWidgets.QLabel()
        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.clear_btn = QtWidgets.QPushButton("Clear")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        self.attach_btn = QtWidgets.QPushButton("Attach to message")
        self.attach_btn.setDefault(True)
        for button in (self.undo_btn, self.clear_btn, cancel_btn, self.attach_btn):
            button.setAutoDefault(False)
        self.undo_btn.clicked.connect(self.canvas.undo)
        self.clear_btn.clicked.connect(self.canvas.clear)
        cancel_btn.clicked.connect(self.reject)
        self.attach_btn.clicked.connect(self.attach)
        self.canvas.changed.connect(self._sync)
        shortcut_cls = getattr(QtGui, "QShortcut", None) or getattr(QtWidgets, "QShortcut")   # Qt 6 / Qt 5
        shortcut_cls(QtGui.QKeySequence.Undo, self, self.canvas.undo)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.hint, 1)
        for button in (self.undo_btn, self.clear_btn, cancel_btn, self.attach_btn):
            bar.addWidget(button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.canvas, 1)
        self._fit()
        self._sync()

    def _fit(self) -> None:
        """Open at the picture's size, or smaller to fit the screen."""
        screen = QtGui.QGuiApplication.primaryScreen()
        hint = self.canvas.sizeHint()
        if screen is None:
            self.resize(hint.width(), hint.height() + 48)
            return
        room = screen.availableGeometry()
        scale = min(1.0, room.width() * 0.85 / max(1, hint.width()), (room.height() * 0.85 - 48) / max(1, hint.height()))
        self.resize(int(hint.width() * scale), int(hint.height() * scale) + 48)

    def _sync(self) -> None:
        marked = bool(self.canvas.strokes)
        self.hint.setText("Mark more, or attach it to your message" if marked
                          else "Draw on the model where Oscar should change it")
        for button in (self.undo_btn, self.clear_btn, self.attach_btn):
            button.setEnabled(marked)

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.attach()
            return
        super().keyPressEvent(event)   # Escape rejects

    def attach(self) -> None:
        if not self.canvas.strokes:
            return
        points = markup.sample([point for stroke in self.canvas.strokes for point in stroke])
        try:
            hits = self._pick(points)
        except Exception:  # noqa: BLE001 - the picture still says where
            hits = []
        os.makedirs(self._out_dir, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        path = os.path.join(self._out_dir, f"markup-{stamp}.png")
        number = 2
        while os.path.exists(path):
            path = os.path.join(self._out_dir, f"markup-{stamp}-{number}.png")
            number += 1
        if not self.canvas.composite().save(path, "PNG"):
            QtWidgets.QMessageBox.warning(self, "Mark up the model", f"The picture couldn't be saved to {path}.")
            return
        self.marked = (path, markup.describe(hits))
        self.accept()


def start(panel, out_dir: str) -> None:
    """The pen's click: capture the 3D view, let the user draw, then queue the picture and start the draft with the
    region line. Nothing happens on Cancel; a missing 3D view is a notice in the chat."""
    captured = capture()
    if isinstance(captured, str):
        panel._on_notice(captured)
        return
    image, pick = captured
    dialog = MarkupDialog(image, pick, out_dir, panel)
    if dialog.exec() != QtWidgets.QDialog.Accepted or dialog.marked is None:
        return
    path, note = dialog.marked
    panel.add_attachments([path])
    text = panel.input.toPlainText()
    panel.input.setPlainText(f"{note}\n{text}")
    cursor = panel.input.textCursor()
    cursor.movePosition(QtGui.QTextCursor.End)
    panel.input.setTextCursor(cursor)
    panel.input.setFocus()
