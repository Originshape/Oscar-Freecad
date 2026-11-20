"""The Parameters and Logs pages of the redesigned panel (UI redesign 2026-11-15, shell
redesign 2026-11-16) and the SEAM FACTORIES that mount them.

Seam contract (2026-11-16 shell redesign): the pages come from their own modules —
``panel_params.ParamsTab`` and ``panel_timeline.TimelineTab`` — and are mounted through
:func:`make_params_tab` / :func:`make_timeline_tab`. Both factories catch ``ImportError``
and fall back to the built-in views below, so a seam module that is absent (or not yet
landed) never crashes the panel and the page keeps working. A seam module that IS present
but raises while building is NOT masked — real errors surface.

Both fallbacks are DATA-FED and passive: the panel (or a test) pushes data in and
provides callables for the actions. No FreeCAD import at module level — everything
document-related is resolved by the host panel on the GUI thread, so the module stays
importable by the pure test suites.

- :class:`ParamsView` — the built-in Parameters page: the parameter table of the active
  document's ``Params`` / ``Parameters`` sheet exactly as
  :func:`oscar.project_gui.param_table` reads it (name / value / unit / expression /
  source / bound). The Value column of a SAFE row is editable; edits buffer locally
  until Apply (N) / Discard. Derived (expression) and needs-rebuild rows are read-only
  with a tooltip instead of a second, half-working edit path.
- :class:`LogsView` — the built-in Logs page: the execution log (chronological tool
  events plus the chat's workflow milestones, refreshed while a turn runs). Nothing is
  invented: log lines come from the existing Block/Step records; conversation text is
  kept out. Snapshot PNGs decode lazily — at most two per event-loop pass, and only
  while the page is actually visible.
"""

import html as _html

from PySide import QtCore, QtGui, QtWidgets

from . import panel_theme as _theme
from .panel_blocks import LogBrowser, ThumbCache
from .panel_render import thumb_url

_COLS = ("Parameter", "Value", "Unit", "Expression", "Source", "Bound")

# Friendly, honest renderings of the sheet's source column (raw value shown when unknown)
_SOURCE_LABELS = {
    "user_requirement": "User",
    "step_measurement": "Measured",
    "assumption": "Assumed",
    "": "—",
}


def make_params_tab(on_apply=None, parent=None) -> QtWidgets.QWidget:
    """The Parameters page: ``panel_params.ParamsTab`` when that module exists, the
    built-in :class:`ParamsView` (same ``load`` / ``on_apply`` / ``on_refresh`` contract)
    until then. Only ``ImportError`` falls back — an error inside a present module is
    never masked."""
    try:
        from .panel_params import ParamsTab
    except ImportError:
        return ParamsView(on_apply=on_apply, parent=parent)
    apply_fn = None
    if on_apply is not None:
        def apply_fn(changes):
            result = on_apply(changes) or {}
            ok = bool(result.get("ok"))
            text = str(result.get("text") or ("Applied." if ok else "Apply failed."))
            return ok, [] if ok else [text]

    return ParamsTab(apply_fn=apply_fn, parent=parent)


def make_timeline_tab(parent=None) -> QtWidgets.QWidget:
    """The Logs page: ``panel_timeline.TimelineTab`` when that module exists, the
    built-in :class:`LogsView` until then. The mounted widget is fed through
    ``load_events(events, thumbs)`` when it offers it (defensive: the host checks)."""
    try:
        from .panel_timeline import TimelineTab
    except ImportError:
        return LogsView(parent=parent)
    return TimelineTab(parent=parent)


def _source_label(source: str) -> str:
    return _SOURCE_LABELS.get(str(source or "").strip(), str(source or "").strip() or "—")


def _mono_font() -> QtGui.QFont:
    font = QtGui.QFont()
    font.setFamilies(["Consolas", "Cascadia Mono", "Courier New"])
    font.setStyleHint(QtGui.QFont.Monospace)
    return font


class ParamsView(QtWidgets.QWidget):
    """The parameter table with buffered editing and Apply (N) / Discard.

    The host wires: ``on_apply(edits)`` -> ``{"ok", "text"}`` (runs the real write path),
    and calls :meth:`load` with a fresh :func:`project_gui.param_table` dict whenever the
    document may have changed. Pending edits survive a reload when the row still matches.
    """

    def __init__(self, on_apply=None, parent=None):
        super().__init__(parent)
        self._on_apply = on_apply
        self._pending: dict[str, str] = {}       # name -> new value text (as typed)
        self._rows: list[dict] = []

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(4)
        self.search = QtWidgets.QLineEdit()
        self.search.setObjectName("oscarParamsSearch")
        self.search.setPlaceholderText("Filter parameters…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda _text: self._rebuild())
        top.addWidget(self.search, 1)
        self.refresh_btn = QtWidgets.QToolButton()
        self.refresh_btn.setText("↻")
        self.refresh_btn.setToolTip("Re-read the parameter sheet of the active document")
        # the host replaces this no-op with a real re-read
        self.refresh_btn.clicked.connect(self._emit_refresh)
        top.addWidget(self.refresh_btn)
        root.addLayout(top)

        self.table = QtWidgets.QTableWidget(0, len(_COLS))
        self.table.setObjectName("oscarParamsTable")
        self.table.setHorizontalHeaderLabels(list(_COLS))
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked
                                   | QtWidgets.QAbstractItemView.SelectedClicked
                                   | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        for col in range(2, len(_COLS)):
            self.table.horizontalHeader().setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.table, 1)

        self.hint = QtWidgets.QLabel("")
        self.hint.setObjectName("oscarParamsHint")
        self.hint.setProperty("muted", True)
        self.hint.setWordWrap(True)
        self.hint.hide()
        root.addWidget(self.hint)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self.discard_btn = QtWidgets.QPushButton("Discard")
        self.discard_btn.setObjectName("oscarParamsDiscard")
        self.discard_btn.clicked.connect(self.discard)
        self.discard_btn.hide()
        footer.addWidget(self.discard_btn)
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.setObjectName("oscarParamsApply")
        self.apply_btn.clicked.connect(self.apply)
        self.apply_btn.hide()
        footer.addWidget(self.apply_btn)
        root.addLayout(footer)

    # -- data ---------------------------------------------------------------

    def _emit_refresh(self):
        handler = getattr(self, "on_refresh", None)
        if callable(handler):
            handler()

    def load(self, table: dict) -> None:
        """Fill from ``project_gui.param_table`` output: ``{"sheet": str, "rows": [...]}``."""
        rows = list((table or {}).get("rows") or [])
        sheet = str((table or {}).get("sheet") or "")
        self._rows = rows
        if not rows:
            self.table.hide()
            self.hint.setText("No parameter sheet in the active document."
                              if not sheet else
                              f"The '{sheet}' sheet has no aliased parameter rows.")
            self.hint.show()
            return
        self.hint.hide()
        self.table.show()
        # drop pending edits whose row disappeared
        names = {str(r.get("name")) for r in rows}
        self._pending = {n: v for n, v in self._pending.items() if n in names}
        self._rebuild()

    def _rebuild(self) -> None:
        filter_text = self.search.text().strip().lower()
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for row in self._rows:
            name = str(row.get("name") or "")
            if filter_text and filter_text not in name.lower():
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            pending = self._pending.get(name)
            value = pending if pending is not None else self._value_text(row)
            editable = self._editable(row)
            items = (
                name,
                value,
                str(row.get("unit") or ""),
                str(row.get("expr") or ""),
                _source_label(str(row.get("source") or "")),
                str(row.get("bound") or "0") + "×" if int(row.get("bound") or 0) else "—",
            )
            for col, text in enumerate(items):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable
                              if (col != 1 or not editable) else item.flags() | QtCore.Qt.ItemIsEditable)
                if col == 1 and pending is not None:
                    item.setForeground(QtGui.QColor(_theme.TOKENS["accent"]))
                if col in (0, 1):
                    item.setFont(_mono_font())
                self.table.setItem(r, col, item)
            if not editable:
                tip = ("Derived parameter — its value comes from the expression; edit the inputs"
                       if row.get("expr") else
                       "This parameter only takes effect through a rebuild — ask Oscar to change it")
                self.table.item(r, 1).setToolTip(tip)
        self.table.blockSignals(False)
        self._refresh_footer()

    @staticmethod
    def _value_text(row: dict) -> str:
        value = row.get("value")
        unit = str(row.get("unit") or "").strip()
        if isinstance(value, (int, float)):
            return f"{value:g}" + (f" {unit}" if unit else "")
        return str(value if value is not None else "")

    @staticmethod
    def _editable(row: dict) -> bool:
        return not row.get("expr") and not row.get("needs_rebuild")

    # -- editing ------------------------------------------------------------

    def _on_item_changed(self, item):
        if item.column() != 1:
            return
        r = item.row()
        if r >= len(self._rows):
            return
        row = self._rows[r]
        name = str(row.get("name") or "")
        if not self._editable(row):
            return
        self._pending[name] = item.text().strip()
        self._refresh_footer()

    def _refresh_footer(self) -> None:
        count = len(self._pending)
        self.apply_btn.setVisible(count > 0)
        self.discard_btn.setVisible(count > 0)
        self.apply_btn.setText(f"Apply ({count})" if count else "Apply")

    def discard(self) -> None:
        self._pending.clear()
        self._rebuild()

    def apply(self) -> None:
        if not self._pending or not callable(self._on_apply):
            return
        edits = [{"name": n, "value": v} for n, v in self._pending.items()]
        result = self._on_apply(edits) or {}
        if result.get("ok"):
            self._pending.clear()
        self._rebuild()

    def pending(self) -> dict:
        return dict(self._pending)


class LogsView(QtWidgets.QWidget):
    """The built-in Logs page: execution records and the current chat's workflow
    milestones in one chronological browser (the fallback for ``panel_timeline``).

    ``load_events(events, thumbs)`` is the seam method the host calls; the identical
    ``load_log_events`` name is kept for the tests. Rendering is cached: unchanged
    markup never re-lays-out the document. Snapshot PNGs decode only while this page is
    visible, at most two per event-loop pass (decoding them on every completed geometry
    step stalls the GUI on workflows with many screenshots).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.log_view = LogBrowser()
        self.log_view.setObjectName("oscarLogView")
        self.log_view.setOpenLinks(False)
        root.addWidget(self.log_view)
        self._log_markup = None
        self._thumbs: ThumbCache | None = None
        self._snapshots: list[tuple[str, str]] = []   # (snapshot_key, absolute path)

    # -- data ---------------------------------------------------------------

    def load_events(self, events: "list[dict]", thumbs: ThumbCache | None = None) -> None:
        """Render workflow milestones and execution records without rebuilding unchanged log HTML."""
        if thumbs is not None:
            self._thumbs = thumbs
            self.log_view.register_thumbnails(thumbs)
        rows = []
        for event in events:
            if event.get("kind") == "workflow":
                rows.append(
                    "<div style='font-weight:bold;padding:5px 0 2px;'>"
                    f"{_html.escape(str(event.get('title') or 'Workflow'))}</div>"
                )
                for step in event.get("steps") or []:
                    status = str(step.get("status") or "pending").upper()
                    title = _html.escape(str(step.get("title") or step.get("id") or "Step"))
                    sid = _html.escape(str(step.get("id") or ""))
                    note = str(step.get("note") or "").strip()
                    note_html = ("<br/><span style='color:#999;padding-left:18px;'>"
                                 + _html.escape(note) + "</span>") if note else ""
                    rows.append(
                        "<div style='padding:2px 0 2px 8px;'>"
                        f"{_html.escape(status)} &nbsp; {sid} {title}"
                        f"{note_html}"
                        "</div>"
                    )
                continue
            status = _html.escape(str(event.get("status") or "EVENT").upper())
            timestamp = _html.escape(str(event.get("timestamp") or ""))
            tool = _html.escape(str(event.get("tool") or "operation"))
            title = _html.escape(str(event.get("title") or ""))
            duration = event.get("duration")
            suffix = f" · {float(duration):.1f}s" if duration else ""
            rows.append(
                "<div style='border-bottom:1px solid #3a3a3f;padding:5px 0;'>"
                f"<div>{timestamp} &nbsp; <b>{status}</b> &nbsp; {tool} · {title}{suffix}</div>"
            )
            for label in ("input", "result", "delta", "validation", "warning", "error"):
                value = str(event.get(label) or "").strip()
                if value:
                    rows.append(
                        f"<div style='white-space:pre-wrap;padding:2px 0 2px 12px;'>"
                        f"<span style='color:#a1a1aa;'>{label}: </span>{_html.escape(value)}</div>"
                    )
            snapshot = str(event.get("snapshot") or "").strip()
            if snapshot:
                image = str(event.get("snapshot_key") or "").strip()
                rows.append("<div style='color:#a1a1aa;padding:2px 0 2px 12px;'>"
                            f"snapshot: {_html.escape(snapshot)}</div>")
                if image:
                    rows.append(f"<div style='padding:3px 0 3px 12px;'><img src='{thumb_url(image)}' width='320'></div>")
            rows.append("</div>")
        if not rows:
            rows = [f"<p style='color:{_theme.TOKENS['text_2']};'>No execution events yet — "
                    "they appear while Oscar works.</p>"]
        markup = f"<div style='font-size:12px;'>{''.join(rows)}</div>"
        if markup != self._log_markup:
            self.log_view.setHtml(markup)
            self._log_markup = markup
        self._snapshots = [(str(e["snapshot_key"]), str(e["snapshot"])) for e in events
                           if e.get("snapshot_key") and e.get("snapshot")]
        if self.isVisible():
            self._pump_snapshots()

    load_log_events = load_events   # the historical name (tests)

    # -- lazy snapshots -------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self._pump_snapshots()

    def _pump_snapshots(self) -> None:
        """Decode at most two pending snapshots per event-loop pass, only while visible."""
        thumbs = self._thumbs
        if thumbs is None or not self._snapshots or not self.isVisible():
            return
        pending = [(key, path) for key, path in self._snapshots
                   if not thumbs.has(key) and not thumbs.failed(key)]
        for key, path in pending[:2]:
            thumbs.load(key, path, keep={item[0] for item in self._snapshots})
        self.log_view.register_thumbnails(thumbs)
        self.log_view.viewport().update()
        if len(pending) > 2:
            QtCore.QTimer.singleShot(0, self._pump_snapshots)
