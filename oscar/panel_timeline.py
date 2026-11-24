# SPDX-License-Identifier: Apache-2.0
"""The LOGS tab as a visual activity timeline (Qt).

``TimelineTab`` renders what :func:`oscar.panel_activity.build_timeline` derives from the
project's ``steps.jsonl`` records: a chronological vertical timeline, newest at the bottom,
grouped into Analysis / Plan & parameters / Build / Verify & save / Export sections.

Data-fed and passive, like the other redesigned views: the host either calls
:meth:`refresh_from` with the chat's ``Project`` (records are re-read, the widget rebuilt) or
feeds live steps with :meth:`append_step` while a turn runs. Nothing here captures snapshots
of its own — a thumbnail appears only where the record itself carries a snapshot path.

Theme-agnostic: colours, dot shapes and fonts come from the shell's stylesheet through the
object names this module assigns (``timeline-entry``, ``timeline-entry-<status>``,
``timeline-dot-<status>``, ``timeline-group-header``, ``timeline-entry-result`` /
``-error`` / ``-raw`` / ``-fullshot``, ``timeline-empty``). Without a stylesheet the tab
still works — the running dot's pulse is a :class:`QGraphicsOpacityEffect`, not CSS.

Qt only: no FreeCAD import (the ``Project`` is duck-typed through ``read_steps()`` /
``root``), so the module runs on any interpreter that has PySide.
"""

import json
import time

from PySide import QtCore, QtGui, QtWidgets

from .panel_activity import GROUP_LABELS, build_timeline, entry_key

_THUMB_BOX = (96, 64)      # max thumbnail cell (aspect preserved inside it)
_FULL_PX = 320             # full snapshot width in the expanded details
_GROW_MS = 150             # details grow animation
_PULSE_MS = 120            # running-dot pulse tick


def _mono() -> QtGui.QFont:
    font = QtGui.QFont()
    font.setFamilies(["Consolas", "Cascadia Mono", "Courier New"])
    font.setStyleHint(QtGui.QFont.Monospace)
    return font


def _load_pixmap(path: str, box: tuple[int, int]) -> "QtGui.QPixmap | None":
    """``path`` as a pixmap scaled to fit ``box`` keeping its aspect (None when the file
    is missing or not an image — a record may carry a path whose capture never landed)."""
    if not path:
        return None
    reader = QtGui.QImageReader(path)
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        return None
    scaled = image.scaled(box[0], box[1], QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    return QtGui.QPixmap.fromImage(scaled)


class _EntryRow(QtWidgets.QWidget):
    """One timeline entry: status dot, timestamp + op + explanation, result / error line,
    snapshot thumbnail, expandable raw details. Identity for reuse is ``entry_key``."""

    toggled = QtCore.Signal(object)   # this row's key — the tab re-layouts, nothing more

    def __init__(self, entry: dict, thumb: "QtGui.QPixmap | None", parent=None):
        super().__init__(parent)
        self.key = entry_key(entry)
        self.entry = entry
        self.status = str(entry.get("status") or "completed")
        self.setObjectName(f"timeline-entry-{self.status}")

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 3, 4, 3)
        root.setSpacing(1)

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(6)
        self.dot = QtWidgets.QLabel()
        self.dot.setObjectName(f"timeline-dot-{self.status}")
        self.dot.setFixedSize(10, 10)
        head.addWidget(self.dot)
        stamp = QtWidgets.QLabel(f"{_short(entry.get('timestamp'))}  ·  {entry.get('op') or ''}")
        stamp.setObjectName("timeline-entry-stamp")
        stamp.setFont(_mono())
        head.addWidget(stamp)
        head.addWidget(_vline())
        explain = QtWidgets.QLabel(str(entry.get("explanation") or ""))
        explain.setObjectName("timeline-entry-explain")
        head.addWidget(explain, 1)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setObjectName("timeline-entry-toggle")
        self.toggle.setText("▸")
        self.toggle.setCheckable(True)
        self.toggle.setToolTip("Show the raw record and the full snapshot")
        self.toggle.toggled.connect(self._on_toggle)
        head.addWidget(self.toggle)
        root.addLayout(head)

        self.result = QtWidgets.QLabel(str(entry.get("result_text") or ""))
        self.result.setObjectName("timeline-entry-result")
        self.result.setWordWrap(False)
        self.result.setVisible(bool(entry.get("result_text")))
        root.addWidget(self.result)
        self.error = QtWidgets.QLabel(str(entry.get("error_text") or ""))
        self.error.setObjectName("timeline-entry-error")
        self.error.setWordWrap(True)
        self.error.setVisible(bool(entry.get("error_text")))
        root.addWidget(self.error)

        if thumb is not None:
            self.thumb_btn = QtWidgets.QToolButton()
            self.thumb_btn.setObjectName("timeline-entry-thumb")
            self.thumb_btn.setIconSize(thumb.size())
            self.thumb_btn.setIcon(QtGui.QIcon(thumb))
            self.thumb_btn.setAutoRaise(True)
            self.thumb_btn.setToolTip(str(entry.get("snapshot_path") or ""))
            self.snapshot_path = str(entry.get("snapshot_path") or "")
            self.thumb_btn.clicked.connect(lambda: self._open())
            root.addWidget(self.thumb_btn, 0, QtCore.Qt.AlignLeft)
        else:
            self.thumb_btn = None
            self.snapshot_path = ""

        # expandable details: raw record dict + the full snapshot
        self.details = QtWidgets.QWidget()
        self.details.setObjectName("timeline-entry-details")
        details_layout = QtWidgets.QVBoxLayout(self.details)
        details_layout.setContentsMargins(0, 2, 0, 2)
        self.raw = QtWidgets.QPlainTextEdit()
        self.raw.setObjectName("timeline-entry-raw")
        self.raw.setFont(_mono())
        self.raw.setReadOnly(True)
        self.raw.setMaximumHeight(180)
        self.raw.setPlainText(_pretty(entry.get("detail")))
        details_layout.addWidget(self.raw)
        self.fullshot = QtWidgets.QLabel()
        self.fullshot.setObjectName("timeline-entry-fullshot")
        full = _load_pixmap(str(entry.get("snapshot_path") or ""), (_FULL_PX, 100000))
        if full is not None:
            self.fullshot.setPixmap(full)
            details_layout.addWidget(self.fullshot)
        self.details.setMaximumHeight(0)
        self.details.hide()
        root.addWidget(self.details)
        self._anim = QtCore.QPropertyAnimation(self.details, b"maximumHeight", self)
        self._anim.setDuration(_GROW_MS)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)

    # -- expansion ---------------------------------------------------------------------------

    def _on_toggle(self, checked: bool) -> None:
        self.toggle.setText("▾" if checked else "▸")
        self.set_expanded(checked)
        self.toggled.emit(self.key)

    def set_expanded(self, expanded: bool, animate: bool = True) -> None:
        """Show / hide the raw details; the 150 ms grow animation can be skipped
        (tests, or an accessibility setting that dislikes motion)."""
        if expanded:
            self.details.setMaximumHeight(0)
            self.details.show()
            target = max(1, self.details.sizeHint().height())
            if animate:
                self._anim.stop()
                self._anim.setStartValue(0)
                self._anim.setEndValue(target)
                self._anim.start()
            else:
                self.details.setMaximumHeight(target)
        else:
            if animate:
                self._anim.stop()
                self._anim.setStartValue(self.details.maximumHeight())
                self._anim.setEndValue(0)
                try:
                    self._anim.finished.disconnect()
                except (RuntimeError, TypeError):
                    pass
                self._anim.finished.connect(self.details.hide)
                self._anim.start()
            else:
                self.details.hide()
                self.details.setMaximumHeight(0)

    def is_expanded(self) -> bool:
        return self.details.isVisible()

    def _open(self) -> None:
        pass   # wired by TimelineTab to its open_snapshot signal

    def signature(self, entry: dict) -> bool:
        """True when this row already shows exactly ``entry`` (no rebuild needed)."""
        return (entry_key(entry) == self.key
                and str(entry.get("status") or "") == self.status
                and str(entry.get("explanation") or "") == str(self.entry.get("explanation") or "")
                and str(entry.get("result_text") or "") == str(self.entry.get("result_text") or "")
                and str(entry.get("error_text") or "") == str(self.entry.get("error_text") or "")
                and str(entry.get("snapshot_path") or "") == self.snapshot_path)


def _vline() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.VLine)
    line.setObjectName("timeline-entry-sep")
    return line


def _short(ts) -> str:
    text = str(ts or "")
    return text.split("T", 1)[1] if "T" in text else text


def _pretty(detail) -> str:
    try:
        return json.dumps(detail, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(detail)


class TimelineTab(QtWidgets.QWidget):
    """The visual activity timeline of one chat's project records.

    The host calls :meth:`refresh_from(project) <refresh_from>` when the tab opens or the
    chat changes, and :meth:`append_step(record) <append_step>` for every step that
    finishes while it is visible. Scrolling stays pinned to the newest entry unless the
    user scrolled up (then nothing moves until they return to the bottom).
    """

    open_snapshot = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("timeline")
        self._records: list[dict] = []
        self._snapshot_dir: str | None = None
        self._entries: list[dict] = []
        self._visible_entries: list[dict] = []
        self._rows: dict = {}          # entry key -> _EntryRow
        self._thumbs: dict = {}        # snapshot path -> QPixmap
        self._pinned = True

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(2, 0, 2, 4)
        toolbar.addWidget(QtWidgets.QLabel("Activity"))
        toolbar.addStretch(1)
        self.status_filter = QtWidgets.QComboBox()
        self.status_filter.setObjectName("timelineStatusFilter")
        self.status_filter.addItem("All statuses", "all")
        for label, status in (("Running", "running"), ("Completed", "completed"),
                              ("Warning", "warning"), ("Failed", "failed"), ("Skipped", "skipped")):
            self.status_filter.addItem(label, status)
        self.status_filter.currentIndexChanged.connect(self._on_filter_changed)
        toolbar.addWidget(self.status_filter)
        root.addLayout(toolbar)
        self.empty = QtWidgets.QLabel("No activity recorded yet — steps appear here as Oscar works.")
        self.empty.setObjectName("timeline-empty")
        self.empty.setProperty("muted", True)
        self.empty.setWordWrap(True)
        root.addWidget(self.empty)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setObjectName("timelineScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.content = QtWidgets.QWidget()
        self.content.setObjectName("timelineContent")
        self.column = QtWidgets.QVBoxLayout(self.content)
        self.column.setContentsMargins(6, 6, 6, 6)
        self.column.setSpacing(2)
        self.column.addStretch(1)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll, 1)

        bar = self.scroll.verticalScrollBar()
        bar.valueChanged.connect(self._on_scrolled)

        self.pulse = QtCore.QTimer(self)
        self.pulse.setInterval(_PULSE_MS)
        self.pulse.timeout.connect(self._tick_pulse)
        self._pulse_t0 = time.monotonic()

    # -- data ----------------------------------------------------------------------------------

    def refresh_from(self, project) -> None:
        """Rebuild from a ``Project``: its ``read_steps()`` records and ``root`` (the
        folder the relative snapshot paths live in)."""
        records = []
        try:
            records = list(project.read_steps())
        except Exception:   # noqa: BLE001 - an unreadable journal shows the empty state
            records = []
        self._records = records
        self._snapshot_dir = getattr(project, "root", None)
        self._rebuild(scroll_bottom=True)

    def load_entries(self, entries: "list[dict]") -> None:
        """Show pre-built :func:`panel_activity.build_timeline` entries (tests, or a host
        that keeps its own record list)."""
        self._entries = list(entries or [])
        self._render(scroll_bottom=True)

    def append_step(self, record) -> None:
        """One finished (or still-running, ``ok`` absent) step record, live: the timeline
        is re-derived, rows whose key did not change keep their widgets, the view stays at
        the bottom unless the user scrolled up."""
        if isinstance(record, dict):
            self._records.append(dict(record))
        else:
            to_line = getattr(record, "to_line", None)
            self._records.append(dict(to_line()) if callable(to_line) else {})
        self._rebuild(scroll_bottom=False)

    def entries(self) -> "list[dict]":
        return list(self._entries)

    def visible_entries(self) -> "list[dict]":
        return list(self._visible_entries)

    def is_pinned(self) -> bool:
        """True when the view follows the newest entry (the user has not scrolled up)."""
        return self._pinned

    def row(self, key):
        """The ``_EntryRow`` of an entry key (None when absent) — for tests."""
        return self._rows.get(key)

    # -- rendering ---------------------------------------------------------------------------

    def _rebuild(self, scroll_bottom: bool) -> None:
        self._entries = build_timeline(self._records, self._snapshot_dir)
        self._render(scroll_bottom=scroll_bottom)

    def _on_filter_changed(self, _index: int) -> None:
        self._render(scroll_bottom=False)

    def _render(self, scroll_bottom: bool) -> None:
        status = str(self.status_filter.currentData() or "all")
        if status == "all":
            visible_entries = list(self._entries)
        else:
            visible_entries = [entry for entry in self._entries if str(entry.get("status") or "") == status]
        self._visible_entries = visible_entries
        self.empty.setText("No activity recorded yet — steps appear here as Oscar works."
                           if not self._entries else "No activity matches this status filter.")
        self.empty.setVisible(not visible_entries)
        # drop rows whose entry disappeared, reuse rows whose entry did not change
        wanted = {}
        for entry in self._entries:
            key = entry_key(entry)
            row = self._rows.get(key)
            if row is not None and row.signature(entry):
                wanted[key] = row
            else:
                if row is not None:
                    row.deleteLater()
                wanted[key] = self._make_row(entry)
        for key, row in list(self._rows.items()):
            if key not in wanted:
                row.deleteLater()
        self._rows = wanted

        # relayout: group sections in order of appearance, newest entry at the bottom
        while self.column.count():
            item = self.column.takeAt(0)
            widget = item.widget()
            if widget is not None:
                self.column.removeWidget(widget)
                if widget.objectName() in ("timeline-group-header", "timeline-turn-header"):
                    widget.deleteLater()
        previous_group = ""
        previous_turn = None
        turn_number = 0
        visible_keys = {entry_key(entry) for entry in visible_entries}
        for key, row in self._rows.items():
            row.setVisible(key in visible_keys)
        for entry in visible_entries:
            turn_id = str(entry.get("turn_id") or "")
            if turn_id and turn_id != previous_turn:
                turn_number += 1
                turn_label = QtWidgets.QLabel(f"TURN {turn_number} · {_short(entry.get('timestamp'))}")
                turn_label.setObjectName("timeline-turn-header")
                self.column.addWidget(turn_label)
                previous_group = ""
            previous_turn = turn_id
            group = str(entry.get("group") or "")
            if group != previous_group:
                header = QtWidgets.QLabel(_group_header(visible_entries, entry))
                header.setObjectName("timeline-group-header")
                self.column.addWidget(header)
                previous_group = group
            self.column.addWidget(self._rows[entry_key(entry)])
        self.column.addStretch(1)   # keep the spacer after the entries, so it cannot push them into each other
        self._sync_pulse()
        if scroll_bottom or self._pinned:
            QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

    def _make_row(self, entry: dict) -> _EntryRow:
        path = str(entry.get("snapshot_path") or "")
        row = _EntryRow(entry, self._thumb(path))
        if row.thumb_btn is not None:
            row.thumb_btn.clicked.disconnect()
            row.thumb_btn.clicked.connect(lambda _=False, r=row: self.open_snapshot.emit(r.snapshot_path))
        return row

    def _thumb(self, path: str):
        if not path:
            return None
        if path not in self._thumbs:
            pixmap = _load_pixmap(path, _THUMB_BOX)
            if pixmap is None:
                self._thumbs[path] = None
            else:
                self._thumbs[path] = pixmap
                while len([p for p in self._thumbs.values() if p]) > 48:
                    for key, value in self._thumbs.items():
                        if value is not None:
                            self._thumbs.pop(key)
                            break
        return self._thumbs[path]

    # -- scrolling ---------------------------------------------------------------------------

    def _on_scrolled(self, value: int) -> None:
        bar = self.scroll.verticalScrollBar()
        self._pinned = value >= bar.maximum() - 2

    def _scroll_to_bottom(self) -> None:
        try:
            bar = self.scroll.verticalScrollBar()
            bar.setValue(bar.maximum())
        except RuntimeError:  # a queued scroll can outlive the dock during panel teardown
            return

    # -- the running pulse -------------------------------------------------------------------

    def _sync_pulse(self) -> None:
        running = [row for row in self._rows.values() if row.status == "running"]
        for row in self._rows.values():
            if row.status == "running":
                if row.dot.graphicsEffect() is None:
                    effect = QtWidgets.QGraphicsOpacityEffect(row.dot)
                    row.dot.setGraphicsEffect(effect)
            else:
                row.dot.setGraphicsEffect(None)
        if running:
            if not self.pulse.isActive():
                self._pulse_t0 = time.monotonic()
                self.pulse.start()
        else:
            self.pulse.stop()

    def _tick_pulse(self) -> None:
        elapsed = time.monotonic() - self._pulse_t0
        opacity = 0.35 + 0.65 * (0.5 + 0.5 * __import__("math").sin(elapsed * 5.0))
        for row in self._rows.values():
            if row.status == "running":
                effect = row.dot.graphicsEffect()
                if effect is not None:
                    effect.setOpacity(opacity)


def _group_header(entries: "list[dict]", entry: dict) -> str:
    """``BUILD · 14`` — the group label and how many entries share it around here."""
    group = str(entry.get("group") or "")
    label = GROUP_LABELS.get(group, group or "Activity")
    count = sum(1 for e in entries if str(e.get("group") or "") == group)
    return f"{label.upper()} · {count}" if count else label.upper()
