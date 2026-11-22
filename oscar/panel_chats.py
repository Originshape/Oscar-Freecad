"""Renaming and searching the panel's chats (the ⋯ menu's **Rename chat…** and **Search chats…**), as the OriginShape
studio renames and searches its tasks.

- ``rename(panel)`` asks for a new name for the current chat. A name the user gives is kept: a chat titled after
  its first prompt never renames itself again, and a workspace chat stops following its task's title.
- ``search(panel)`` opens ``ChatSearchDialog`` over the panel's chats (``chatsearch.find``: titles, prompts and
  replies); choosing a result switches to that chat.

Qt; the dialog takes plain ``(title, [(role, text)])`` data, so tests drive it without a panel.
"""

import html

from PySide import QtCore, QtWidgets

from . import chatsearch

MAX_TITLE = 80
_ATTACHED = "\n\n--- attached files ---\n"   # attachments.build_user_message: what follows is the files' content


def chat_texts(sessions: list) -> list:
    """``[(title, [(role, text)])]`` of ``sessions``: what the user typed (without the inlined attachment text) and
    Oscar's replies; the app's own notes are left out."""
    out = []
    for session in sessions:
        messages = []
        for message in session.get("history") or []:
            role = getattr(message, "role", "")
            if role not in ("user", "assistant") or getattr(message, "synthetic", False):
                continue
            content = getattr(message, "content", "")
            if isinstance(content, list):
                content = "\n".join(str(part.get("text") or "") for part in content
                                    if isinstance(part, dict) and part.get("type", "text") == "text")
            text = str(content or "").split(_ATTACHED, 1)[0].strip()
            if text:
                messages.append((role, text))
        out.append((str(session.get("title") or "New chat"), messages))
    return out


def rename(panel) -> None:
    """Ask for the current chat's new name and keep it (an empty or unchanged name changes nothing)."""
    session = panel._current_session()
    if session is None:
        return
    name, ok = QtWidgets.QInputDialog.getText(panel, "Rename chat", "Name of this chat:",
                                              QtWidgets.QLineEdit.Normal, session.get("title") or "")
    name = " ".join(str(name or "").split())[:MAX_TITLE]
    if not ok or not name or name == session.get("title"):
        return
    session["title"] = name
    session["named"] = True
    panel._rebuild_chat_combo()
    panel._persist_current()


class ChatSearchDialog(QtWidgets.QDialog):
    """A search field over the chats and the chats that match, best first; Enter or a double click chooses one
    (``chosen``: its index in the data), the arrows move through the results, Escape closes."""

    def __init__(self, chats: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Search chats")
        self.setModal(True)
        self.resize(460, 420)
        self.chosen: "int | None" = None
        self._chats = chats
        self._found: list = []

        self.field = QtWidgets.QLineEdit()
        self.field.setPlaceholderText("Search chats: titles, prompts and replies")
        self.field.setClearButtonEnabled(True)
        self.field.installEventFilter(self)
        self.results = QtWidgets.QListWidget()
        self.results.setWordWrap(True)
        self.results.setAccessibleName("Chats that match")
        self.hint = QtWidgets.QLabel()
        self.hint.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.field)
        layout.addWidget(self.hint)
        layout.addWidget(self.results, 1)

        self.field.textChanged.connect(self._refresh)
        self.field.returnPressed.connect(self.choose)
        self.results.itemActivated.connect(lambda _item: self.choose())
        self._refresh("")

    def _refresh(self, text: str) -> None:
        self._found = chatsearch.find(self._chats, text)
        self.results.clear()
        for found in self._found:
            item = QtWidgets.QListWidgetItem()
            label = QtWidgets.QLabel(self._html(found))
            label.setTextFormat(QtCore.Qt.RichText)
            label.setWordWrap(True)
            label.setContentsMargins(6, 4, 6, 4)
            item.setSizeHint(label.sizeHint())
            item.setToolTip(found.title)
            self.results.addItem(item)
            self.results.setItemWidget(item, label)
        if self._found:
            self.results.setCurrentRow(0)
        if not chatsearch.terms(text):
            self.hint.setText("Type to search the titles of your chats, what you asked and what Oscar answered.")
        elif not self._found:
            self.hint.setText("No chat matches. Every word has to be in the title or in one message.")
        else:
            count = len(self._found)
            self.hint.setText(f"{count} chat{'' if count == 1 else 's'}. Enter opens the selected one.")

    @staticmethod
    def _html(found) -> str:
        def marked(text: str) -> str:
            return "".join(f"<b>{html.escape(piece)}</b>" if hit else html.escape(piece)
                           for piece, hit in chatsearch.mark(text, found.terms))

        title = f"<b>{html.escape(found.title)}</b>" if not found.snippet else marked(found.title)
        if not found.snippet:
            return title
        who = "You" if found.role == "user" else "Oscar"
        return f"{title}<br><span style='color:palette(mid)'>{who}:</span> {marked(found.snippet)}"

    def eventFilter(self, watched, event):
        if watched is self.field and event.type() == QtCore.QEvent.KeyPress and self._found:
            step = {QtCore.Qt.Key_Down: 1, QtCore.Qt.Key_Up: -1}.get(event.key())
            if step:
                row = min(len(self._found) - 1, max(0, self.results.currentRow() + step))
                self.results.setCurrentRow(row)
                return True
        return super().eventFilter(watched, event)

    def choose(self) -> None:
        row = self.results.currentRow()
        if not (0 <= row < len(self._found)):
            return
        self.chosen = self._found[row].index
        self.accept()


def search(panel) -> None:
    """Search the panel's chats; the chosen one becomes the current chat (not while a turn runs)."""
    sessions = list(panel.sessions)
    dialog = ChatSearchDialog(chat_texts(sessions), panel)
    if dialog.exec() != QtWidgets.QDialog.Accepted or dialog.chosen is None:
        return
    session = sessions[dialog.chosen]
    if session["history"] is panel.history:
        return
    if panel._busy():
        panel._set_status("Wait for the current turn to finish before switching chats.")
        return
    panel._persist_current()
    panel._activate_session(session)
