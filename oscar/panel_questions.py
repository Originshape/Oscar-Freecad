# SPDX-License-Identifier: Apache-2.0
"""The in-chat question bar of the redesigned panel (chat redesign 2026-11-16).

:class:`QuestionsBar` is the interactive counterpart of ``panel_render.question_html`` (the
transcript embed of an ANSWERED question): while a question is open it offers the choices as
buttons, locks itself the moment one is picked (or a custom answer is submitted) and then shows
the recorded choice. It never answers anything by itself — the panel connects to its signals and
routes the decision where it belongs (``netguard`` approval / clarification resume), exactly like
the ``oscar://answer/`` anchors of the legacy question cards.

Construction (the payload may come straight from a worker signal; every field is optional and
tolerated missing or malformed). With no payload, the bar starts hidden while no question is
pending::

    bar = QuestionsBar()

    bar = QuestionsBar({"text": "Fillet radius?", "options": ["1 mm", "2 mm"],
                        "allow_custom": True, "approve": False})

``approve=True`` payloads get **Accept** / **Reject** buttons (added when the payload carries no
options of its own); ``allow_custom=True`` adds the custom-answer field with its Submit button
(Enter submits too).

Signals
- ``answered(str)`` — an option button (or Accept / Reject) was clicked; the argument is the
  option's label verbatim (``"1 mm"``, ``"Accept"``, ``"Reject"``).
- ``custom_answered(str)`` — the custom field was submitted with non-blank text. The two signals
  are disjoint: one bar answers exactly once.

State
- :meth:`is_locked`, :meth:`recorded` — after any answer the bar is locked (every control
  disabled) and shows the recorded choice.
- :meth:`lock(answer, custom)` — lock programmatically (the answer arrived from elsewhere, e.g. a
  remote decision); sets the same state WITHOUT emitting.

Theme: the widget carries **objectNames only** (``oscarQuestionsBar``, ``oscarQuestionText``,
``oscarQuestionOption``, ``oscarQuestionAccept``, ``oscarQuestionReject``, ``oscarQuestionCustom``,
``oscarQuestionSubmit``, ``oscarQuestionRecorded``) and no colours of its own, so the panel
stylesheet (``panel_theme``) styles it; it renders unstyled but functional anywhere else.

Height: the bar is capped (``max_height``, 200 px by default) so a long question can never
grow the panel/dock — the body lives in an internal vertical-only QScrollArea and wraps its
text; anything taller than the cap scrolls inside the bar (never sideways).

No FreeCAD import: the module needs only PySide and is importable by the pure test suites.
"""

from PySide import QtCore, QtWidgets

_ACCEPT = "Accept"
_REJECT = "Reject"
_DEFAULT_MAX_HEIGHT = 200


class QuestionsBar(QtWidgets.QWidget):
    """One question with option buttons, an optional custom-answer field and a lock-after-answer
    record. See the module docstring for the payload, the signals and the theming contract."""

    answered = QtCore.Signal(str)
    custom_answered = QtCore.Signal(str)

    def __init__(self, payload=None, parent=None, max_height: int | None = None):
        super().__init__(parent)
        self.setObjectName("oscarQuestionsBar")
        self._max_height = max(int(max_height), 1) if max_height else _DEFAULT_MAX_HEIGHT
        self.setMaximumHeight(self._max_height)   # a question never sizes the panel/dock
        data = payload if isinstance(payload, dict) else {}
        self._text = str(data.get("text") or "").strip()
        self._approve = bool(data.get("approve"))
        self._allow_custom = bool(data.get("allow_custom"))
        self._locked = False
        self._recorded = ""
        raw_options = data.get("options") if isinstance(data.get("options"), (list, tuple)) else []
        self._options = [str(option).strip() for option in raw_options if str(option).strip()][:6]
        if self._approve and not self._options:
            self._options = [_ACCEPT, _REJECT]
        self._buttons: list[QtWidgets.QPushButton] = []
        self._body: QtWidgets.QWidget | None = None
        self._root = QtWidgets.QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(0)
        # The body scrolls INSIDE this capped strip (vertical only): long questions wrap and
        # scroll here instead of propagating their height into the panel's size hint.
        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._root.addWidget(self._scroll)
        self._build_body()
        if payload is None:
            self.hide()

    def _build_body(self) -> None:
        """(Re)create the contents for the current ``_text`` / ``_options`` / flags."""
        if self._body is not None:
            self._body.setParent(None)
            self._body.deleteLater()
        self._buttons = []
        body = QtWidgets.QWidget(self)
        self._body = body
        outer = QtWidgets.QVBoxLayout(body)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        self.label = QtWidgets.QLabel(self._text or "Oscar needs your decision")
        self.label.setObjectName("oscarQuestionText")
        self.label.setWordWrap(True)
        outer.addWidget(self.label)

        self.buttons_row = QtWidgets.QWidget()
        self.buttons_row.setObjectName("oscarQuestionOptions")
        row = QtWidgets.QHBoxLayout(self.buttons_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for option in self._options:
            button = QtWidgets.QPushButton(option)
            if self._approve and option == _ACCEPT:
                button.setObjectName("oscarQuestionAccept")
            elif self._approve and option == _REJECT:
                button.setObjectName("oscarQuestionReject")
            else:
                button.setObjectName("oscarQuestionOption")
            button.clicked.connect(lambda _checked=False, text=option: self._choose(text))
            row.addWidget(button)
            self._buttons.append(button)
        row.addStretch(1)
        self.buttons_row.setVisible(bool(self._options))
        outer.addWidget(self.buttons_row)

        self.custom_row = QtWidgets.QWidget()
        self.custom_row.setObjectName("oscarQuestionCustomRow")
        custom = QtWidgets.QHBoxLayout(self.custom_row)
        custom.setContentsMargins(0, 0, 0, 0)
        custom.setSpacing(6)
        self.custom_edit = QtWidgets.QLineEdit()
        self.custom_edit.setObjectName("oscarQuestionCustom")
        self.custom_edit.setPlaceholderText("Type your own answer…")
        self.custom_edit.returnPressed.connect(self._submit_custom)
        custom.addWidget(self.custom_edit, 1)
        self.submit_btn = QtWidgets.QPushButton("Submit")
        self.submit_btn.setObjectName("oscarQuestionSubmit")
        self.submit_btn.clicked.connect(self._submit_custom)
        custom.addWidget(self.submit_btn)
        self.custom_row.setVisible(self._allow_custom)
        outer.addWidget(self.custom_row)

        self.recorded_label = QtWidgets.QLabel("")
        self.recorded_label.setObjectName("oscarQuestionRecorded")
        self.recorded_label.setWordWrap(True)
        self.recorded_label.hide()
        outer.addWidget(self.recorded_label)
        self._scroll.setWidget(body)

    def load(self, payload) -> None:
        """Show ``payload`` (the ``QuestionsBar`` dict shape) as a fresh, unanswered question.

        A blank payload hides the bar. This is how the host presents each new clarification /
        approval: the bar answers exactly once per call, so a new question must reset the lock."""
        data = payload if isinstance(payload, dict) else {}
        text = str(data.get("text") or "").strip()
        if not text:
            self.reset()
            return
        self._text = text
        self._approve = bool(data.get("approve"))
        self._allow_custom = bool(data.get("allow_custom"))
        raw_options = data.get("options") if isinstance(data.get("options"), (list, tuple)) else []
        self._options = [str(option).strip() for option in raw_options if str(option).strip()][:6]
        if self._approve and not self._options:
            self._options = [_ACCEPT, _REJECT]
        self._locked = False
        self._recorded = ""
        self._build_body()
        self.show()

    def reset(self) -> None:
        """Hide the bar and forget the previous question (no signal)."""
        self._locked = False
        self._recorded = ""
        self._text = ""
        self._options = []
        self.hide()

    # -- answering ------------------------------------------------------------------------------

    def _choose(self, text: str) -> None:
        self.lock(text)
        self.answered.emit(text)

    def _submit_custom(self) -> None:
        if not self._allow_custom or self._locked:
            return
        text = self.custom_edit.text().strip()
        if not text:
            return
        self.lock(text, custom=True)
        self.custom_answered.emit(text)

    def lock(self, answer: str, custom: bool = False) -> None:
        """Record ``answer`` and lock the bar (every control disabled, the recorded choice shown).
        Idempotent and silent: the first answer wins, later calls change nothing. No signal is
        emitted — the click paths (:meth:`_choose`, :meth:`_submit_custom`) emit."""
        if self._locked or not str(answer).strip():
            return
        self._locked = True
        self._recorded = str(answer).strip()
        for button in self._buttons:
            button.setEnabled(False)
        self.custom_edit.setEnabled(False)
        self.submit_btn.setEnabled(False)
        self.recorded_label.setText(self._recorded_text(custom))
        self.recorded_label.show()

    def _recorded_text(self, custom: bool) -> str:
        if custom:
            return f"✓ Your answer — {self._recorded}"
        if self._approve and self._recorded == _ACCEPT:
            return "✓ You accepted"
        if self._approve and self._recorded == _REJECT:
            return "✕ You rejected"
        return f"✓ You chose: {self._recorded}"

    # -- state (for the panel and the tests) ----------------------------------------------------

    def is_locked(self) -> bool:
        """True once an answer was recorded (click or :meth:`lock`)."""
        return self._locked

    def recorded(self) -> str:
        """The recorded answer (``""`` while the bar is still open)."""
        return self._recorded

    def options(self) -> list[str]:
        """The option labels the bar shows (approve payloads default to Accept / Reject)."""
        return list(self._options)

    def allows_custom(self) -> bool:
        return self._allow_custom

    def is_approve(self) -> bool:
        return self._approve
