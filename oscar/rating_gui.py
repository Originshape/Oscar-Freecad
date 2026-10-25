"""The rating card's FreeCAD side (``oscar.rating`` decides when to ask and what is sent).

* ``maybe_offer(panel, answer, outcome, card)`` runs when a turn ends: for a signed-in OriginShape account and a
  turn ``rating.should_ask`` picks, it adds a RATING card under the turn and remembers what the turn was.
* ``on_anchor(panel, target)`` handles the card's links (``oscar://rate/<key>/<1-5|later>``). A link only acts
  for a card this panel offered, by its random key, so a link written by the model does nothing. Stars open ``RatingDialog`` (the
  stars, an optional comment, *Include a picture of the model*, and what is shared); only Send sends.
* The picture is the active 3D view as it is now (no re-orienting), at most 1024 px wide (smaller if that is
  over the server's 2 MB; the card says when it couldn't be included). The request runs on a
  worker thread; a timer brings its result back to the GUI thread.
"""

from __future__ import annotations

import base64
import os
import random
import tempfile
import threading
import time
from typing import Any

from PySide import QtCore, QtWidgets

from . import params as _params
from . import rating
from .panel_blocks import Block

IMAGE_WIDTH = 1024
IMAGE_FALLBACK_WIDTHS = (768, 512)   # tried in turn while the picture is over IMAGE_MAX_BYTES
IMAGE_MAX_BYTES = 2 * 1024 * 1024
SEND_TIMEOUT = (5.0, 30.0)


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def _steps_of(card: Block | None) -> list[dict[str, Any]]:
    """The tool steps of the turn's process card (the model's thought rows are not part of a rating)."""
    if card is None:
        return []
    return [rating.step_record(step.name, (step.view.title if step.view is not None else "") or step.summary,
                               step.ok, step.dur)
            for step in card.steps if not step.is_thought and not step.plain]


def _context(panel: Any) -> dict[str, Any]:
    context: dict[str, Any] = {}
    try:
        from . import __version__

        context["oscar_version"] = __version__
        profile = _params.active_profile()
        context["model"] = _params.model_display_name(profile)
    except Exception:  # noqa: BLE001 - context is a nicety
        pass
    try:
        import FreeCAD as App

        context["freecad_version"] = ".".join(str(part) for part in App.Version()[:3])
    except Exception:  # noqa: BLE001
        pass
    session = panel._current_session() if hasattr(panel, "_current_session") else None
    if isinstance(session, dict) and str(session.get("title") or "").startswith("Workspace:"):
        context["source"] = "workspace"
    else:
        context["source"] = "freecad"
    return context


def maybe_offer(panel: Any, answer: str, outcome: str, card: Block | None) -> Block | None:
    """After a turn: add the rating card when this turn is sampled (``rating.should_ask``)."""
    if not panel._account_state().get("signed_in"):
        return None
    facts = rating.TurnFacts(prompt=str(getattr(panel, "_turn_request", "") or ""), response=answer or "",
                             steps=_steps_of(card), mutated=bool(getattr(panel, "_turn_mutated", False)),
                             ok=True, outcome=outcome)
    today = _today()
    day = _params.get_string(rating.PREF_DAY, "")
    count = _params.get_int(rating.PREF_COUNT, 0)
    if not rating.should_ask(facts, now=time.time(), today=today, last_asked=_params.get_int(rating.PREF_LAST, 0),
                             asked_day=day, asked_count=count, roll=random.random()):
        return None
    block = panel._append(Block.RATING)
    block.detail = rating.OFFERED
    block.key = rating.new_key()
    offers = getattr(panel, "_rating_offers", None)
    if offers is None:
        offers = panel._rating_offers = {}
    offers[block.key] = {"facts": facts, "block": block, "context": _context(panel)}
    day, count = rating.asked(today, day, count)
    _params.set_int(rating.PREF_LAST, int(time.time()))
    _params.set_string(rating.PREF_DAY, day)
    _params.set_int(rating.PREF_COUNT, count)
    return block


def on_anchor(panel: Any, target: str) -> None:
    """A link of a rating card: stars open the dialog, Not now puts the card away."""
    parsed = rating.parse_anchor(target)
    offers = getattr(panel, "_rating_offers", None) or {}
    if parsed is None or parsed[0] not in offers:
        return   # not a card this panel offered (a link in model text, or an old card)
    key, stars = parsed
    offer = offers[key]
    block: Block = offer["block"]
    if block.detail in (rating.SENDING, rating.SENT, rating.SENT_WITHOUT_PICTURE):
        return
    if stars is None:
        block.detail = rating.DISMISSED
        offers.pop(key, None)
        panel._render()
        return
    dialog = RatingDialog(panel, stars)
    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return
    image = capture_view_png() if dialog.include_image() else ""
    offer["picture_missing"] = dialog.include_image() and not image
    body = rating.payload(offer["facts"], dialog.stars(), comment=dialog.comment(), image_png=image,
                          context={**offer["context"], "image": bool(image)})
    block.detail = rating.SENDING
    panel._render()
    _send(panel, key, body)


def _send(panel: Any, key: str, body: dict[str, Any]) -> None:
    result: dict[str, Any] = {}

    def work() -> None:
        try:
            from .auth import config, session

            reply = session.get_session(config.api_url()).authorized_request(
                "POST", "/v1/feedback", json_body=body, timeout=SEND_TIMEOUT)
            result["ok"] = 200 <= reply.status < 300
            result["status"] = reply.status
        except Exception as e:  # noqa: BLE001 - reported on the card
            result["ok"] = False
            result["error"] = type(e).__name__
        finally:
            result["done"] = True

    threading.Thread(target=work, name="oscar-rating", daemon=True).start()
    timer = QtCore.QTimer(panel)

    def check() -> None:
        if not result.get("done"):
            return
        timer.stop()
        timer.deleteLater()
        offers = getattr(panel, "_rating_offers", None) or {}
        offer = offers.get(key)
        if offer is None:
            return
        if result.get("ok"):
            offer["block"].detail = rating.SENT_WITHOUT_PICTURE if offer.get("picture_missing") else rating.SENT
            offers.pop(key, None)
        else:
            offer["block"].detail = rating.FAILED
            try:
                panel._log_panel_error("rating_send", RuntimeError(str(result.get("status") or result.get("error"))))
            except Exception:  # noqa: BLE001
                pass
        panel._render()

    timer.timeout.connect(check)
    timer.start(200)


def capture_view_png(width: int = IMAGE_WIDTH) -> str:
    """The active 3D view as it is, as base64 PNG at most ``width`` px wide (narrower while it is over
    ``IMAGE_MAX_BYTES``); "" without a view, on failure, or when even the smallest is too big."""
    try:
        import FreeCADGui as Gui

        view = Gui.ActiveDocument.ActiveView if Gui.ActiveDocument else None
        if view is None or not hasattr(view, "saveImage"):
            return ""
        size = view.getSize() if hasattr(view, "getSize") else (width, int(width * 0.75))
        full_w, full_h = int(size[0]) or width, int(size[1]) or int(width * 0.75)
        for limit in (width, *(w for w in IMAGE_FALLBACK_WIDTHS if w < width)):
            w, h = full_w, full_h
            if w > limit:
                w, h = limit, max(1, int(h * limit / w))
            data = _save_view(view, w, h)
            if not data.startswith(b"\x89PNG"):
                return ""
            if len(data) <= IMAGE_MAX_BYTES:
                return base64.b64encode(data).decode("ascii")
        return ""
    except Exception:  # noqa: BLE001 - a rating without a picture is still a rating
        return ""


def _save_view(view: Any, w: int, h: int) -> bytes:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        view.saveImage(path, w, h, "Current")
        with open(path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


class RatingDialog(QtWidgets.QDialog):
    """The stars (pre-set to the one clicked), an optional comment, the picture checkbox and what is shared."""

    def __init__(self, parent: Any, stars: int) -> None:
        super().__init__(parent)
        self.setObjectName("oscarRatingDialog")
        self.setWindowTitle("Rate this turn")
        self._stars = max(1, min(5, int(stars)))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("How did Oscar do on this one?"))
        row = QtWidgets.QHBoxLayout()
        self._buttons: list[QtWidgets.QToolButton] = []
        for value in range(1, 6):
            button = QtWidgets.QToolButton()
            button.setObjectName(f"oscarRatingStar{value}")
            button.setAutoRaise(True)
            button.setToolTip(f"{value} of 5")
            button.clicked.connect(lambda _checked=False, v=value: self._set(v))
            self._buttons.append(button)
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        self._comment = QtWidgets.QPlainTextEdit()
        self._comment.setObjectName("oscarRatingComment")
        self._comment.setPlaceholderText("What worked, or what should Oscar have done? (optional)")
        self._comment.setFixedHeight(90)
        layout.addWidget(self._comment)
        self._image = QtWidgets.QCheckBox("Include a picture of the model")
        self._image.setObjectName("oscarRatingImage")
        self._image.setChecked(True)
        layout.addWidget(self._image)
        note = QtWidgets.QLabel(rating.SHARED_TEXT)
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox()
        send = buttons.addButton("Send", QtWidgets.QDialogButtonBox.AcceptRole)
        send.setObjectName("oscarRatingSend")
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._set(self._stars)

    def _set(self, value: int) -> None:
        self._stars = value
        for index, button in enumerate(self._buttons, start=1):
            button.setText("★" if index <= value else "☆")
            button.setStyleSheet("QToolButton { font-size: 22px; color: #d9a21b; }")

    def stars(self) -> int:
        return self._stars

    def comment(self) -> str:
        return self._comment.toPlainText()

    def include_image(self) -> bool:
        return self._image.isChecked()
