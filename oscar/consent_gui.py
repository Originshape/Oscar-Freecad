# SPDX-License-Identifier: Apache-2.0
"""The Terms prompt's FreeCAD side (``oscar.auth.consent`` decides what it says and what Continue sends).

* ``sync(panel, state)`` runs whenever the panel takes the account state in (``_apply_account_state``). While the
  OriginShape profile is in use and its account has not accepted the current Terms (``consent.needed``), the
  ``consent`` card joins the chat and ``ConsentDialog`` opens: once per account and version of the Terms in this
  FreeCAD run (after a sign-in that is when Settings closes), never while a turn runs or another dialog is open (it
  tries again a second later). Once the Terms are accepted the cards go and nothing is asked again.
* ``open_dialog(panel)``: from the card (``oscar://account/consent``), the badge menu (*Review the Terms…*) and a
  send the gate held back (``_send_blocker`` says ``consent``). Continue with the box ticked runs ``consent.submit``
  on a worker thread and a timer brings the answer back to the GUI thread (the ``rating_gui`` pattern): accepted, the
  cards go; 409 (the Terms changed meanwhile) opens the dialog again on the new version. Continue with the box
  unticked sends nothing: the card says the account can't be used for chat until the Terms are accepted, and the
  user's own providers keep working. Closing the dialog leaves the card.
* ``ConsentDialog``: ``consent.NOTICE`` word for word, the box (ticked by default) whose "Terms" and "Privacy Policy"
  open the account's pages, and Continue.
"""

from __future__ import annotations

import threading
from typing import Any

from PySide import QtCore, QtWidgets

from . import params as _params
from .auth import consent
from .auth.transport import AuthError
from .panel_blocks import Block, _muted

POLL_MS = 200        # how often the answer of the accept request is looked for
RETRY_MS = 1000      # a prompt that had to wait (a turn, another dialog) is tried again after this long
_asked: set[tuple[str, str, str]] = set()      # (server, email, Terms version) prompted in this FreeCAD run
_declined: set[tuple[str, str, str]] = set()   # ... and left unticked


def _key(state: dict) -> tuple[str, str, str]:
    snapshot = state.get("snapshot") if isinstance(state.get("snapshot"), dict) else {}
    email = str(snapshot.get("email") or state.get("email") or "").strip().lower()
    return str(state.get("api_url") or ""), email, str(snapshot.get("legal_terms_version") or "")


def _snapshot(state: dict) -> dict | None:
    snapshot = state.get("snapshot") if state.get("signed_in") else None
    return snapshot if isinstance(snapshot, dict) else None


def _account_in_use() -> bool:
    try:
        return _params.is_account_profile(_params.active_profile())
    except Exception:  # noqa: BLE001 - an unreadable settings file reads as "no profile"
        return False


def _cards(blocks: Any) -> list[Block]:
    return [b for b in blocks if b.kind == Block.ACCOUNT and b.detail == "consent"] if isinstance(blocks, list) else []


def drop_cards(panel: Any) -> bool:
    """Remove the ``consent`` cards from every chat (the Terms were accepted). True when a block list changed."""
    changed = False
    lists = [panel.blocks] + [s.get("blocks") for s in getattr(panel, "sessions", [])]
    seen: set[int] = set()
    for blocks in lists:
        if not isinstance(blocks, list) or id(blocks) in seen:
            continue
        seen.add(id(blocks))
        for card in _cards(blocks):
            blocks.remove(card)
            changed = True
    return changed


def show_card(panel: Any, *, declined: bool | None = None, returned: bool = False) -> None:
    """Put the ``consent`` card at the end of the chat (an earlier one in this chat moves there). ``declined``: the
    box was left unticked (None: as this account last answered in this run). No render."""
    if declined is None:
        declined = _key(panel._account_state()) in _declined
    for card in _cards(panel.blocks):
        panel.blocks.remove(card)
    panel.blocks.append(panel._account_block("consent", reason="declined" if declined else "", returned=returned))


def sync(panel: Any, state: dict) -> bool:
    """The account state changed (GUI thread): drop the cards once the Terms are accepted, or ask for them (see the
    module docstring). True when a block list changed (the caller renders)."""
    snapshot = _snapshot(state)
    if not consent.needed(snapshot):
        if consent.recorded(snapshot):   # accepted: a later change of the Terms is asked about again
            account = _key(state)[:2]
            _asked.difference_update({k for k in _asked if k[:2] == account})
            _declined.difference_update({k for k in _declined if k[:2] == account})
        return drop_cards(panel)
    if not _account_in_use() or _key(state) in _asked:
        return False
    if not _can_ask(panel):
        _retry_later(panel)
        return False
    _asked.add(_key(state))
    show_card(panel)
    QtCore.QTimer.singleShot(0, lambda: open_dialog(panel))
    return True


def _can_ask(panel: Any) -> bool:
    busy = getattr(panel, "_busy", None)
    return not ((callable(busy) and busy()) or getattr(panel, "_consent_open", False)
                or getattr(panel, "_consent_job", None) is not None
                or QtWidgets.QApplication.activeModalWidget() is not None)


def _retry_later(panel: Any) -> None:
    if getattr(panel, "_consent_retry", False):
        return
    panel._consent_retry = True

    def again() -> None:
        panel._consent_retry = False
        try:
            if sync(panel, panel._account_state()):
                panel._render()
        except RuntimeError:   # the panel is gone
            pass

    QtCore.QTimer.singleShot(RETRY_MS, again)


def open_dialog(panel: Any, *, updated: bool = False, snapshot: dict | None = None) -> None:
    """The consent dialog for the account's current Terms (``snapshot``: the account as a 409 just said, else the
    panel's state), then what Continue says. Does nothing while a dialog or an answer is under way."""
    if getattr(panel, "_consent_open", False) or getattr(panel, "_consent_job", None) is not None:
        return
    state = panel._account_state()
    prompt = consent.Prompt.of(snapshot if snapshot is not None else _snapshot(state), updated=updated)
    if prompt is None:
        if drop_cards(panel):
            panel._render()
        if consent.recorded(_snapshot(state)):
            panel._flash_status("Your OriginShape account has accepted the current Terms.")
        return
    key = _key({**state, "snapshot": snapshot}) if snapshot is not None else _key(state)
    _asked.add(key)
    panel._consent_open = True
    dialog = ConsentDialog(panel, prompt)
    try:
        accepted = dialog.exec() == QtWidgets.QDialog.Accepted
        prompt = consent.with_box(prompt, dialog.checked())
    finally:
        panel._consent_open = False
        dialog.deleteLater()
    if not accepted:   # closed: the card stays, and asks again when clicked
        if not _cards(panel.blocks):
            show_card(panel)
            panel._render()
        return
    if prompt.body() is None:
        _declined.add(key)
        show_card(panel, declined=True)
        panel._render()
        return
    _declined.discard(key)
    _accept(panel, prompt, str(state.get("api_url") or ""))


def _accept(panel: Any, prompt: consent.Prompt, api_url: str) -> None:
    """``consent.submit`` on a worker thread; a timer brings the answer back."""
    result: dict[str, Any] = {}

    def work() -> None:
        try:
            from .auth import session as _session

            consent.submit(prompt, _session.get_session(api_url))
        except consent.TermsChanged as e:
            result["changed"] = e.snapshot.to_dict() if e.snapshot is not None else {}
        except AuthError as e:   # the message is safe to show
            result["error"] = str(e)
        except Exception as e:  # noqa: BLE001 - a bug: its text could quote anything, so only its type
            result["error"] = f"unexpected {type(e).__name__}"
        finally:
            result["done"] = True

    panel._consent_job = result
    threading.Thread(target=work, name="oscar-consent", daemon=True).start()
    panel._flash_status("Saving your answer…", 15000)
    timer = QtCore.QTimer(panel)

    def check() -> None:
        if not result.get("done"):
            return
        timer.stop()
        timer.deleteLater()
        panel._consent_job = None
        if "changed" in result:
            panel._flash_status(consent.CHANGED, 6000)
            snapshot = result["changed"] or None
            QtCore.QTimer.singleShot(0, lambda: open_dialog(panel, updated=True, snapshot=snapshot))
        elif result.get("error"):
            panel._flash_status(f"Oscar couldn't save your answer: {result['error']}", 8000)
            if not _cards(panel.blocks):
                show_card(panel)
            panel._render()
        else:
            drop_cards(panel)
            panel._flash_status("Thanks: you can chat with your OriginShape account now.", 5000)
            panel._render()

    timer.timeout.connect(check)
    timer.start(POLL_MS)


class _AgreeLabel(QtWidgets.QLabel):
    """The box's words: its links open their pages, a click anywhere else toggles the box."""

    clicked = QtCore.Signal()

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._hovered = ""
        self.linkHovered.connect(self._hover)

    def _hover(self, link: str) -> None:
        self._hovered = link or ""

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().mouseReleaseEvent(event)
        if not self._hovered and event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()


class ConsentDialog(QtWidgets.QDialog):
    """What OriginShape keeps, the box (ticked by default) and Continue."""

    def __init__(self, parent: Any, prompt: consent.Prompt) -> None:
        super().__init__(parent)
        self.setObjectName("oscarConsentDialog")
        self.setWindowTitle("OriginShape Terms")
        self._prompt = prompt
        layout = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel("OriginShape's Terms have changed" if prompt.updated
                                   else "Before you chat with your OriginShape account")
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)
        notice = QtWidgets.QLabel(consent.NOTICE)
        notice.setObjectName("oscarConsentNotice")
        notice.setTextFormat(QtCore.Qt.PlainText)
        notice.setWordWrap(True)
        layout.addWidget(notice)
        row = QtWidgets.QHBoxLayout()
        self._box = QtWidgets.QCheckBox()
        self._box.setObjectName("oscarConsentAgree")
        self._box.setAccessibleName(consent.AGREE)
        self._box.setChecked(prompt.checked)
        label = _AgreeLabel(prompt.agree_html())
        label.setObjectName("oscarConsentAgreeLabel")
        label.setTextFormat(QtCore.Qt.RichText)
        label.setOpenExternalLinks(True)   # the account's own pages (https, or http on this computer)
        label.setWordWrap(True)
        label.clicked.connect(self._box.toggle)
        row.addWidget(self._box)
        row.addWidget(label, 1)
        layout.addLayout(row)
        self._note = QtWidgets.QLabel("")
        self._note.setObjectName("oscarConsentNote")
        self._note.setWordWrap(True)
        self._note.setStyleSheet("color: #d9902b;")
        layout.addWidget(self._note)
        if prompt.version:
            version = QtWidgets.QLabel(f"Terms version {prompt.version}")
            version.setTextFormat(QtCore.Qt.PlainText)
            version.setStyleSheet(f"color:{_muted()};font-size:small;")
            layout.addWidget(version)
        buttons = QtWidgets.QDialogButtonBox()
        go = buttons.addButton("Continue", QtWidgets.QDialogButtonBox.AcceptRole)
        go.setObjectName("oscarConsentContinue")
        go.setDefault(True)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self._box.toggled.connect(self._update)
        self._update()

    def _update(self, *_args) -> None:
        note = consent.with_box(self._prompt, self._box.isChecked()).note()
        self._note.setText(note)
        self._note.setVisible(bool(note))

    def checked(self) -> bool:
        return self._box.isChecked()
