"""The pen next to the paperclip (oscar.panel_markup): the drawing dialog on a picture of the view, what Attach leaves
(the picture with the strokes, the region line), and the panel's draft and attachments after it.

Offscreen QApplication (nothing is shown to the user). The dialog takes the picture and a ``pick`` stand-in, so no 3D
view is needed; ``capture`` in the console (no GUI document) says there is no 3D view.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

_IMPORT_ERROR = ""
try:
    from PySide import QtCore, QtGui, QtWidgets
    from oscar import markup, panel_markup
except Exception as exc:   # pragma: no cover - plain interpreter without FreeCAD / Qt
    _IMPORT_ERROR = str(exc)


class _Mouse:
    """The slice of a QMouseEvent the canvas reads."""

    def __init__(self, x, y, button=None):
        self._pos = QtCore.QPointF(x, y)
        self._button = QtCore.Qt.LeftButton if button is None else button

    def position(self):
        return self._pos

    def button(self):
        return self._button


class _Key:
    def __init__(self, key):
        self._key = key

    def key(self):
        return self._key


def _picture(width=400, height=300):
    image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor("#202020"))
    return image


class MarkupDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR:
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # never open a window on the user's screen
            cls.app = QtWidgets.QApplication(sys.argv[:1])

    def setUp(self):
        self.out = tempfile.mkdtemp(prefix="oscar_markup_test_")
        self.picked = []

        def pick(points):
            self.picked.append(points)
            return [markup.MarkHit("Box", (10.0 + 30 * u, 5.0, 10.0), (0.0, 0.0, 1.0)) for u, _v in points]

        self.dialog = panel_markup.MarkupDialog(_picture(), pick, self.out)
        self.addCleanup(self.dialog.deleteLater)
        self.canvas = self.dialog.canvas
        self.canvas.resize(400, 300)   # the picture fills it exactly: canvas pixels are picture pixels

    def stroke(self, points):
        self.canvas.mousePressEvent(_Mouse(*points[0]))
        for point in points[1:]:
            self.canvas.mouseMoveEvent(_Mouse(*point))
        self.canvas.mouseReleaseEvent(_Mouse(*points[-1]))

    def test_nothing_to_attach_until_a_stroke_and_undo_clear_take_strokes_back(self):
        self.assertFalse(self.dialog.attach_btn.isEnabled())
        self.assertIn("Draw on the model", self.dialog.hint.text())
        self.stroke([(100, 150), (150, 150), (200, 150)])
        self.stroke([(100, 200), (200, 200)])
        self.assertEqual(len(self.canvas.strokes), 2)
        self.assertTrue(self.dialog.attach_btn.isEnabled())
        self.assertIn("attach it to your message", self.dialog.hint.text())
        self.dialog.undo_btn.click()
        self.assertEqual(len(self.canvas.strokes), 1)
        self.dialog.clear_btn.click()
        self.assertEqual(self.canvas.strokes, [])
        self.assertFalse(self.dialog.attach_btn.isEnabled())
        self.dialog.attach()
        self.assertIsNone(self.dialog.marked)

    def test_points_closer_than_a_step_are_dropped_and_a_right_click_draws_nothing(self):
        self.stroke([(100, 100), (100.5, 100), (101, 100.4), (110, 100)])
        self.assertEqual(len(self.canvas.strokes[0]), 2)
        self.canvas.mousePressEvent(_Mouse(50, 50, QtCore.Qt.RightButton))
        self.canvas.mouseReleaseEvent(_Mouse(50, 50))
        self.assertEqual(len(self.canvas.strokes), 1)

    def test_attach_saves_the_picture_with_the_strokes_and_the_region_line(self):
        self.stroke([(100, 150), (200, 150), (300, 150)])
        self.dialog.keyPressEvent(_Key(QtCore.Qt.Key_Return))   # Enter attaches
        path, note = self.dialog.marked
        self.assertEqual(os.path.dirname(path), self.out)
        self.assertRegex(os.path.basename(path), r"^markup-\d{6}\.png$")
        saved = QtGui.QImage(path)
        self.assertEqual((saved.width(), saved.height()), (400, 300))
        self.assertEqual(QtGui.QColor(saved.pixel(200, 150)).name(), panel_markup.STROKE)
        self.assertEqual(QtGui.QColor(saved.pixel(200, 50)).name(), "#202020")
        [points] = self.picked
        self.assertTrue(all(0 <= u <= 1 and 0 <= v <= 1 for u, v in points))
        self.assertAlmostEqual(points[0][0], 0.25)
        self.assertAlmostEqual(points[0][1], 0.5)
        self.assertEqual(note, "[Marked on the model (see the attached picture): part “Box”, its top (+Z), "
                               "around x 17.5 to 32.5, y 5, z 10 mm.]")
        self.assertEqual(self.dialog.result(), QtWidgets.QDialog.Accepted)

    def test_a_second_attach_in_the_same_second_gets_its_own_file(self):
        self.stroke([(100, 150), (200, 150)])
        self.dialog.attach()
        first = self.dialog.marked[0]
        self.dialog.attach()
        self.assertNotEqual(self.dialog.marked[0], first)
        self.assertTrue(os.path.exists(first))

    def test_strokes_follow_the_picture_when_the_canvas_is_wider(self):
        self.canvas.resize(800, 300)   # the picture is drawn 400 x 300, centred: x 200..600
        self.stroke([(200, 0), (600, 300)])
        self.assertEqual(self.canvas.strokes[0], [(0.0, 0.0), (1.0, 1.0)])

    def test_escape_cancels(self):
        rejected = []
        self.dialog.rejected.connect(lambda: rejected.append(True))
        self.stroke([(100, 150), (200, 150)])
        event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape, QtCore.Qt.NoModifier)
        self.dialog.keyPressEvent(event)
        self.assertEqual(rejected, [True])
        self.assertIsNone(self.dialog.marked)


class StartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR:
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            cls.app = QtWidgets.QApplication(sys.argv[:1])

    def _panel(self):
        fake = mock.Mock()
        fake.input = QtWidgets.QPlainTextEdit()
        self.addCleanup(fake.input.deleteLater)
        return fake

    def test_without_a_3d_view_the_chat_says_so(self):
        fake = self._panel()
        self.assertEqual(panel_markup.capture(), panel_markup.NO_VIEW)   # the console has no GUI document
        panel_markup.start(fake, tempfile.mkdtemp())
        fake._on_notice.assert_called_once_with(panel_markup.NO_VIEW)
        fake.add_attachments.assert_not_called()

    def test_attaching_queues_the_picture_and_starts_the_draft_with_the_region(self):
        fake = self._panel()
        fake.input.setPlainText("Round this edge.")

        class Accepting:
            def __init__(self, image, pick, out_dir, parent):
                self.marked = ("/tmp/markup-101010.png", "[Marked on the model: see the attached picture.]")

            def exec(self):
                return QtWidgets.QDialog.Accepted

        with mock.patch.object(panel_markup, "capture", return_value=(_picture(), lambda points: [])), \
                mock.patch.object(panel_markup, "MarkupDialog", Accepting):
            panel_markup.start(fake, tempfile.mkdtemp())
        fake.add_attachments.assert_called_once_with(["/tmp/markup-101010.png"])
        self.assertEqual(fake.input.toPlainText(), "[Marked on the model: see the attached picture.]\nRound this edge.")

    def test_cancel_changes_nothing(self):
        fake = self._panel()
        fake.input.setPlainText("draft")

        class Cancelled:
            def __init__(self, *args):
                self.marked = None

            def exec(self):
                return QtWidgets.QDialog.Rejected

        with mock.patch.object(panel_markup, "capture", return_value=(_picture(), lambda points: [])), \
                mock.patch.object(panel_markup, "MarkupDialog", Cancelled):
            panel_markup.start(fake, tempfile.mkdtemp())
        fake.add_attachments.assert_not_called()
        self.assertEqual(fake.input.toPlainText(), "draft")


if __name__ == "__main__":
    unittest.main()
