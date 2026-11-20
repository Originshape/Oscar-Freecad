"""The Parameters tab as a dedicated workspace (Issue 003, UI redesign follow-up).

:class:`ParamsTab` shows the driving dimensions of the active document in one
editable table and applies buffered edits through the ONE transactional write
path the tools use. It is DATA-FED and passive, exactly like
``panel_tabs.ParamsView`` (which it supersedes on the Parameters page):

* the host pushes data in through :meth:`ParamsTab.refresh_from` with the
  exact shape of :func:`oscar.cad.health_eval.read_param_table`
  (``{"sheet": str, "rows": [{"name", "value", "unit", "expr", "source",
  "note", "cell", "bound", "needs_rebuild"}]}``; the host reads it on the GUI
  thread, e.g. via ``project_gui.param_table``). This module NEVER imports
  FreeCAD - every document access stays with the host, dispatcher-mediated.
* reconstruction evidence columns F-K of ``recon/param_table`` (F = measured,
  G = deviation, H = tolerance, I = confidence, J = basis, K = status) are
  surfaced when the host merges those keys into the rows; they are read-only.
* Apply writes ONLY changed numeric cells through the injectable
  ``apply_fn(changes) -> (ok, errors)``. The intended wiring is
  ``oscar.tools_kit_params.apply_parameter_edits`` (one undo transaction,
  recompute guard, dependent-stage rebuild, rollback on first failure)::

      from .tools_kit_params import apply_parameter_edits
      out = apply_parameter_edits(doc.Name,
                                  [{"name": c["name"], "value": c["apply_text"]}
                                   for c in changes])
      tab = ParamsTab(apply_fn=lambda changes: (out["ok"], [] if out["ok"] else [out["text"]]))

  (the lambda obviously computes ``out`` per call). Unwired, Apply raises
  ``NotImplementedError`` with this guidance and the error is shown, never
  swallowed. Cells of columns A-E are never written by this module at all -
  the spreadsheet contract (A name | B aliased value | C unit | D source |
  E note) belongs to ``plan/builder.py`` and the write path.
* modified rows carry their state as a per-item flag string
  (``oscarParamModified`` / ``oscarParamApplied`` / ``oscarParamError``) in
  the :data:`FLAG_ROLE` data of the row's items - ``QTableWidgetItem`` is not
  a QObject, so there is no objectName to set on it; a shell item delegate
  reads the role and paints from QSS tokens. No color is set here
  (theme-agnostic).

The decision logic (diffing, change collection, unit parsing, nudging) lives
in the module-level pure functions below; the widget only renders state and
shuffles strings between them. It shows parameters only - the FreeCAD model
tree is never duplicated here.
"""

import math
import re

from PySide import QtCore, QtGui, QtWidgets

__all__ = [
    "FLAG_ROLE", "ParamsTab", "collect_changes", "diff_rows", "flag_state", "format_number",
    "has_evidence", "item_flag", "nudge_text", "normalize_input", "parse_numeric",
    "row_change", "row_is_editable", "row_value_text", "same_as_row",
    "source_label", "FK_FIELDS",
]

# Columns: the base table plus the reconstruction evidence columns (F/J/K of
# the F-K block; G/H/I ride along in the Measured tooltip). recon/param_table.
COLS_BASE = ("Name", "Value", "Unit", "Source")
COLS_FULL = COLS_BASE + ("Measured", "Basis", "Status")

# Row states, stored in the items' FLAG_ROLE data (the objectName-equivalent
# string a shell delegate can key its painting on - items are not QObjects).
FLAG_MODIFIED = "oscarParamModified"   # typed value differs from the refresh
FLAG_APPLIED = "oscarParamApplied"     # changed by the last apply / refresh
FLAG_ERROR = "oscarParamError"         # refused locally or by apply_fn
FLAG_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 1

FK_FIELDS = ("measured", "deviation", "tolerance", "confidence", "basis", "status")

_SOURCE_LABELS = {"user_requirement": "User", "step_measurement": "Measured",
                  "assumption": "Assumed", "": "-"}

_NUMBER_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)\s*$")


# ---------------------------------------------------------------------------
# pure helpers (no Qt state beyond reading table items in collect_changes)
# ---------------------------------------------------------------------------


def source_label(source: str) -> str:
    """Friendly label of the sheet's source column (raw value when unknown)."""
    text = str(source or "").strip()
    return _SOURCE_LABELS.get(text, text or "-")


def format_number(value: float) -> str:
    """Compact text of a number (3.0 -> '3', 0.1 -> '0.1')."""
    return "%g" % float(value)


def parse_numeric(text) -> "tuple[float, str] | None":
    """``(value, unit)`` of a typed cell text, or ``None`` when it is not a
    number with at most one trailing unit word ('4', '2.4 mm', '-2.5e1mm';
    formulas, text and double tokens are refused)."""
    match = _NUMBER_RE.match(str(text if text is not None else ""))
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:  # pragma: no cover - the regex already excludes these
        return None
    if not math.isfinite(value):
        return None
    return value, match.group(2)


def row_value_text(row: dict) -> str:
    """Display text of the Value column: the formula for a derived row (why it
    is read-only stays visible), ``value unit`` for a plain numeric row, the
    raw text otherwise, ``?`` when the sheet holds nothing usable."""
    expr = str(row.get("expr") or "").strip()
    if expr:
        return expr
    value = row.get("value")
    unit = str(row.get("unit") or "").strip()
    if value is None:
        return "?"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return format_number(float(value)) + ((" " + unit) if unit else "")


def row_is_editable(row: dict) -> bool:
    """True when the Value cell may be edited: a finite plain number, not a
    formula, not a count/shape parameter that only a rebuild can apply
    (``needs_rebuild`` rows change nothing until their stage is rebuilt - the
    same policy as the shipped Parameters view)."""
    if str(row.get("expr") or "").strip():
        return False
    if row.get("needs_rebuild"):
        return False
    value = row.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def same_as_row(row: dict, text) -> bool:
    """True when ``text`` denotes the row's current value ('3.0 mm' == a 3 mm
    row; a bare number takes the row's unit; another unit never matches)."""
    parsed = parse_numeric(text)
    if parsed is None:
        return False
    value, unit = parsed
    if unit and unit != str(row.get("unit") or "").strip():
        return False
    base = row.get("value")
    if isinstance(base, bool) or not isinstance(base, (int, float)):
        return False
    return abs(value - float(base)) <= 1e-9 * max(1.0, abs(float(base)))


def row_change(row: dict, text) -> "dict | None":
    """One pending edit as a change or a refusal.

    ``None`` - the text denotes the current value (no change);
    ``{"name", "error"}`` - refused locally (not a number, wrong unit, ...);
    ``{"name", "value", "unit", "apply_text"}`` - a change whose
    ``apply_text`` is exactly what the write path accepts ('4 mm' / '7'),
    with the ROW's unit preserved (a bare typed number never changes it).
    """
    name = str(row.get("name") or "")
    if not row_is_editable(row):
        return {"name": name, "error": "this row is read-only"}
    typed = str(text if text is not None else "").strip()
    if not typed:
        return {"name": name, "error": "the value is empty"}
    if typed.startswith("="):
        return {"name": name, "error": "formulas cannot be typed here - edit the inputs or ask Oscar to bind an expression"}
    parsed = parse_numeric(typed)
    if parsed is None:
        return {"name": name, "error": "'%s' is not a number" % typed}
    value, unit = parsed
    row_unit = str(row.get("unit") or "").strip()
    if unit and unit != row_unit:
        return {"name": name, "error": "unit '%s' does not match this row's '%s'" % (unit, row_unit or "unitless")}
    base = float(row.get("value"))
    if abs(value - base) <= 1e-9 * max(1.0, abs(base)):
        return None
    return {"name": name, "value": value, "unit": row_unit,
            "apply_text": format_number(value) + ((" " + row_unit) if row_unit else "")}


def collect_changes(rows, table) -> "tuple[list, list]":
    """``(changes, errors)`` of every EDITED visible value cell.

    ``rows`` is the last-refresh row list (``read_param_table`` shape),
    ``table`` the QTableWidget; only cells whose text differs from the refresh
    baseline are considered, read-only rows never produce anything, and each
    edited cell becomes exactly one :func:`row_change` result. Pure: nothing
    is written, on the table or the rows.
    """
    index = {}
    for row in rows or []:
        name = str((row or {}).get("name") or "")
        if name and name not in index:
            index[name] = row
    changes: list = []
    errors: list = []
    for r in range(table.rowCount()):
        name_item = table.item(r, 0)
        value_item = table.item(r, 1)
        if name_item is None or value_item is None:
            continue
        row = index.get(name_item.text())
        if row is None or not row_is_editable(row):
            continue
        if value_item.text() == row_value_text(row):
            continue
        result = row_change(row, value_item.text())
        if result is None:
            continue
        (errors if "error" in result else changes).append(result)
    return changes, errors


def _same_value(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def diff_rows(old_rows, new_rows) -> dict:
    """Row-level diff between two refreshes, by name:
    ``{"added": [names], "removed": [names], "changed": [names]}`` (a change
    is a differing value / unit / expr / source)."""
    old = {str((r or {}).get("name") or ""): r for r in old_rows or []}
    new = {str((r or {}).get("name") or ""): r for r in new_rows or []}
    changed = []
    for name, row in new.items():
        if name not in old:
            continue
        prev = old[name]
        if any(not _same_value(prev.get(f), row.get(f)) for f in ("value", "unit", "expr", "source")):
            changed.append(name)
    return {"added": [n for n in new if n not in old],
            "removed": [n for n in old if n not in new],
            "changed": changed}


def nudge_text(row: dict, percent: float, current_text: str = "") -> str:
    """Text of the value cell nudged by ``percent`` (the +/-10 % context
    action). Nudges from the cell's current text when that is a number in the
    row's unit (an unapplied edit), else from the refreshed value - so the
    action is just another edit through the same pipeline. ``""`` when the
    row has no numeric base."""
    row_unit = str(row.get("unit") or "").strip()
    base = None
    parsed = parse_numeric(current_text)
    if parsed and (not parsed[1] or parsed[1] == row_unit):
        base = parsed[0]
    else:
        value = row.get("value")
        if not isinstance(value, bool) and isinstance(value, (int, float)):
            base = float(value)
    if base is None:
        return ""
    nudged = base * (1.0 + float(percent) / 100.0)
    return format_number(nudged) + ((" " + row_unit) if row_unit else "")


def has_evidence(rows) -> bool:
    """True when any row carries reconstruction evidence (an F-K field)."""
    for row in rows or []:
        for field in FK_FIELDS:
            value = (row or {}).get(field)
            if value is not None and str(value).strip() != "":
                return True
    return False


def normalize_input(data) -> "tuple[str, list]":
    """``(sheet, rows)`` of whatever the host pushes: the ``read_param_table``
    dict, a bare row list, or (defensively) anything else -> empty. Rows
    without a name are dropped; a ``sheets`` key (multi-sheet documents) is
    informational and ignored here."""
    if isinstance(data, dict):
        sheet = str(data.get("sheet") or "")
        raw = data.get("rows") or []
    elif isinstance(data, list):
        sheet, raw = "", data
    else:
        sheet, raw = "", []
    rows = [row for row in raw
            if isinstance(row, dict) and str(row.get("name") or "").strip()]
    return sheet, rows


def flag_state(name: str, pending=(), errors=(), applied=()) -> str:
    """The row's flag string: error beats modified beats applied. Pure,
    so tests and the shell can reason about the visible state."""
    errored = {str((e or {}).get("name") or "") for e in errors or [] if isinstance(e, dict)}
    errored.discard("")
    if name in errored:
        return FLAG_ERROR
    if name in set(pending or ()):
        return FLAG_MODIFIED
    if name in set(applied or ()):
        return FLAG_APPLIED
    return ""


def item_flag(item) -> str:
    """The flag string stored on a table item (``""`` when none) - what a
    shell delegate reads to restyle modified / applied / error rows."""
    if item is None:
        return ""
    return str(item.data(FLAG_ROLE) or "")


def _default_apply_fn(changes):
    raise NotImplementedError(
        "ParamsTab was built without apply_fn, so Apply has no write path. Wire it to the "
        "existing transactional path when constructing the tab: "
        "ParamsTab(apply_fn=lambda changes: _apply(doc_name, changes)) with "
        "out = oscar.tools_kit_params.apply_parameter_edits(doc_name, "
        "[{'name': c['name'], 'value': c['apply_text']} for c in changes]) "
        "returning (out['ok'], [] if out['ok'] else [out['text']]). "
        "Parameter writes must go through that undo-transaction path; never write sheet cells directly.")


def _mono_font() -> QtGui.QFont:
    font = QtGui.QFont()
    font.setFamilies(["Consolas", "Cascadia Mono", "Courier New"])
    font.setStyleHint(QtGui.QFont.Monospace)
    return font


# ---------------------------------------------------------------------------
# the widget (thin: renders state, delegates every decision to the helpers)
# ---------------------------------------------------------------------------


class ParamsTab(QtWidgets.QWidget):
    """The Parameters workspace: readable editable table, buffered Apply /
    Discard, visible validation errors, F-K evidence columns when present.

    The host wires (all optional):

    * ``apply_fn(changes) -> (ok, errors)`` - the ONLY write path (see the
      module docstring for the ``apply_parameter_edits`` adapter);
    * ``on_refresh`` - the refresh button hook (the host re-reads the sheet
      and calls :meth:`refresh_from`);
    * :meth:`refresh_from` - with fresh ``read_param_table``-shaped data.

    ``errors`` items are ``{"name": str, "error": str}`` (row-flagged) or
    plain strings (a general line). Nothing is ever applied silently: local
    refusals and ``apply_fn`` failures (exceptions included) land in the
    inline error area with the offending rows flagged.
    """

    def __init__(self, apply_fn=None, parent=None):
        super().__init__(parent)
        self._apply_fn = apply_fn          # None -> _default_apply_fn on Apply
        self._rows: list = []              # the last-refresh snapshot (copies)
        self._row_of_name: dict = {}
        self._pending: dict = {}           # name -> typed value text
        self._errors: list = []            # [{"name", "error"}]
        self._applied: set = set()         # names changed since the previous refresh
        self._sheet = ""

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(4)
        self.sheet_label = QtWidgets.QLabel("")
        self.sheet_label.setObjectName("oscarParamsSheet")
        self.sheet_label.setProperty("muted", True)
        top.addWidget(self.sheet_label)
        top.addStretch(1)
        self.search = QtWidgets.QLineEdit()
        self.search.setObjectName("oscarParamsSearch")
        self.search.setPlaceholderText("Filter parameters...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda _text: self._rebuild())
        top.addWidget(self.search, 1)
        self.refresh_btn = QtWidgets.QToolButton()
        self.refresh_btn.setText("R")   # refresh; the host replaces the hook below
        self.refresh_btn.setToolTip("Re-read the parameter sheet of the active document")
        self.refresh_btn.clicked.connect(self._emit_refresh)
        top.addWidget(self.refresh_btn)
        root.addLayout(top)

        self.table = QtWidgets.QTableWidget(0, len(COLS_BASE))
        self.table.setObjectName("oscarParamsTable")   # styled by the shell's QSS
        self.table.setHorizontalHeaderLabels(list(COLS_BASE))
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked
                                   | QtWidgets.QAbstractItemView.SelectedClicked
                                   | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.table, 1)

        self.hint = QtWidgets.QLabel("")
        self.hint.setObjectName("oscarParamsHint")
        self.hint.setProperty("muted", True)
        self.hint.setWordWrap(True)
        self.hint.hide()
        root.addWidget(self.hint)

        self.error_label = QtWidgets.QLabel("")
        self.error_label.setObjectName("oscarParamsErrorArea")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.clear_errors_btn = QtWidgets.QToolButton()
        self.clear_errors_btn.setObjectName("oscarParamErrorsClear")
        self.clear_errors_btn.setText("Clear")
        self.clear_errors_btn.setToolTip("Dismiss the validation messages")
        self.clear_errors_btn.clicked.connect(self.clear_errors)
        self.clear_errors_btn.hide()
        error_row = QtWidgets.QHBoxLayout()
        error_row.setContentsMargins(0, 0, 0, 0)
        error_row.addWidget(self.error_label, 1)
        error_row.addWidget(self.clear_errors_btn)
        self._error_row_widget = QtWidgets.QWidget()
        self._error_row_widget.setLayout(error_row)
        self._error_row_widget.hide()
        root.addWidget(self._error_row_widget)

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

    def refresh_from(self, data) -> None:
        """Replace the table from ``read_param_table``-shaped data (dict or
        bare row list). Pending edits survive while their row still exists,
        stays editable and the typed value still differs from the fresh one;
        errors are cleared; rows whose value changed since the previous
        refresh are flagged ``oscarParamApplied``."""
        sheet, rows = normalize_input(data)
        rows = [dict(row) for row in rows]        # own snapshot, never the caller's dicts
        diff = diff_rows(self._rows, rows) if self._rows else {"added": [], "removed": [], "changed": []}
        self._sheet = sheet
        self._rows = rows
        self._row_of_name = {str(r["name"]): r for r in rows}
        self._applied = set(diff["changed"])
        self._pending = {name: text for name, text in self._pending.items()
                         if name in self._row_of_name
                         and row_is_editable(self._row_of_name[name])
                         and not same_as_row(self._row_of_name[name], text)}
        self._errors = []
        self._rebuild()

    def rows(self) -> "list[dict]":
        """The last-refresh snapshot (copies; the basis Discard restores from)."""
        return [dict(row) for row in self._rows]

    def pending(self) -> dict:
        """``{name: typed text}`` of the rows modified but not applied yet."""
        return dict(self._pending)

    # -- rendering ----------------------------------------------------------

    def _rebuild(self) -> None:
        filter_text = self.search.text().strip().lower()
        evidence = has_evidence(self._rows)
        cols = COLS_FULL if evidence else COLS_BASE
        self.table.blockSignals(True)
        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(list(cols))
        for row in self._rows:
            name = str(row.get("name") or "")
            if filter_text and filter_text not in name.lower():
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            pending = self._pending.get(name)
            value = pending if pending is not None else row_value_text(row)
            editable = row_is_editable(row)
            items = [name, value, str(row.get("unit") or ""), source_label(row.get("source"))]
            if evidence:
                measured = row.get("measured")
                items.append(format_number(float(measured))
                             if isinstance(measured, (int, float)) and not isinstance(measured, bool)
                             else (str(measured) if measured not in (None, "") else "-"))
                items.append(str(row.get("basis") or "").strip() or "-")
                items.append(str(row.get("status") or "").strip() or "-")
            for col, text in enumerate(items):
                item = QtWidgets.QTableWidgetItem(text)
                if col == 1 and editable:
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                if col in (0, 1):
                    item.setFont(_mono_font())
                self.table.setItem(r, col, item)
            if not editable:
                self.table.item(r, 1).setToolTip(self._readonly_tooltip(row))
            if evidence:
                detail = self._evidence_detail(row)
                if detail:
                    self.table.item(r, 4).setToolTip(detail)
        self.table.blockSignals(False)
        if not self._rows:
            self.table.hide()
            self.hint.setText("No parameter sheet in the active document."
                              if not self._sheet else
                              "The '%s' sheet has no aliased parameter rows." % self._sheet)
            self.hint.show()
        else:
            self.hint.hide()
            self.table.show()
        bound = sum(1 for row in self._rows if row.get("bound"))
        self.sheet_label.setText("Sheet '%s' - %d rows, %d bound" % (self._sheet, len(self._rows), bound)
                                 if self._rows else "")
        self._repaint_flags()
        self._refresh_footer()

    @staticmethod
    def _readonly_tooltip(row: dict) -> str:
        if str(row.get("expr") or "").strip():
            return "Derived - the value comes from the expression; edit its inputs instead"
        if row.get("needs_rebuild"):
            return ("Takes effect only through a rebuild - change it via Oscar (set_parameter) "
                    "so the dependent stages are rebuilt too")
        return "Read-only - the sheet holds text or no numeric value"

    @staticmethod
    def _evidence_detail(row: dict) -> str:
        parts = []
        for field, label in (("deviation", "deviation"), ("tolerance", "tolerance"), ("confidence", "confidence")):
            value = row.get(field)
            if value is not None and str(value).strip() != "":
                parts.append("%s %s" % (label, value))
        return " | ".join(parts)

    def _on_item_changed(self, item):
        if item.column() != 1:
            return
        name_item = self.table.item(item.row(), 0)
        row = self._row_of_name.get(name_item.text()) if name_item is not None else None
        if row is None or not row_is_editable(row):
            return
        name = name_item.text()
        text = item.text().strip()
        if same_as_row(row, text) or text == row_value_text(row):
            self._pending.pop(name, None)
        else:
            self._pending[name] = item.text()
        self._repaint_flags()
        self._refresh_footer()

    def _repaint_flags(self) -> None:
        self.table.blockSignals(True)
        error_of = {str(e.get("name") or ""): str(e.get("error") or "")
                    for e in self._errors if isinstance(e, dict) and e.get("name")}
        for r in range(self.table.rowCount()):
            name_item = self.table.item(r, 0)
            value_item = self.table.item(r, 1)
            if name_item is None or value_item is None:
                continue
            name = name_item.text()
            state = flag_state(name, self._pending, self._errors, self._applied)
            name_item.setData(FLAG_ROLE, state)
            value_item.setData(FLAG_ROLE, state)
            font = value_item.font()
            font.setBold(state in (FLAG_MODIFIED, FLAG_ERROR))
            value_item.setFont(font)
            if state == FLAG_ERROR:
                value_item.setToolTip(error_of.get(name, "refused"))
            elif state == FLAG_MODIFIED:
                value_item.setToolTip("Modified - not applied yet (Apply or Discard)")
            elif state == FLAG_APPLIED:
                value_item.setToolTip("Changed since the previous refresh")
        self.table.blockSignals(False)

    def _refresh_footer(self) -> None:
        count = len(self._pending)
        self.apply_btn.setVisible(count > 0)
        self.discard_btn.setVisible(count > 0)
        self.apply_btn.setText("Apply (%d)" % count if count else "Apply")

    # -- errors --------------------------------------------------------------

    def _show_errors(self, errors) -> None:
        normalized = []
        for item in errors or []:
            if isinstance(item, dict):
                normalized.append({"name": str(item.get("name") or ""),
                                   "error": str(item.get("error") or item.get("message") or item.get("text") or "")})
            else:
                normalized.append({"name": "", "error": str(item)})
        self._errors = normalized
        lines = ["%s - %s" % (e["name"], e["error"]) if e["name"] else e["error"] for e in normalized]
        self.error_label.setText("\n".join(lines))
        self.error_label.setVisible(bool(lines))
        self.clear_errors_btn.setVisible(bool(lines))
        self._error_row_widget.setVisible(bool(lines))
        self._repaint_flags()

    def clear_errors(self) -> None:
        """Dismiss the validation messages (the edits themselves stay pending)."""
        self._errors = []
        self.error_label.hide()
        self.clear_errors_btn.hide()
        self._error_row_widget.hide()
        self._repaint_flags()

    # -- actions ---------------------------------------------------------------

    def apply(self) -> None:
        """Apply the modified visible cells through ``apply_fn`` - the ONLY
        write path. Local validation failures and ``apply_fn`` failures are
        shown in the error area; a success updates the baseline so the rows
        stop being modified (a host refresh confirms with fresh data)."""
        changes, errors = collect_changes(self._rows, self.table)
        if errors:
            self._show_errors(errors)
            return
        self.clear_errors()
        if not changes:
            return
        fn = self._apply_fn if callable(self._apply_fn) else _default_apply_fn
        try:
            result = fn(changes)
            if not (isinstance(result, tuple) and len(result) == 2):
                raise ValueError("apply_fn must return (ok, errors), got %r" % (result,))
            ok, apply_errors = result
        except Exception as exc:  # noqa: BLE001 - an apply failure is data, never a crash
            ok, apply_errors = False, ["%s: %s" % (type(exc).__name__, exc)]
        if not ok:
            self._show_errors(list(apply_errors or []))
            return
        for change in changes:
            name = change["name"]
            row = self._row_of_name.get(name)
            if row is not None:
                row["value"] = change["value"]
            self._pending.pop(name, None)
        self._applied = {change["name"] for change in changes}
        self._rebuild()

    def discard(self) -> None:
        """Restore every cell from the last refresh and drop pending edits."""
        self._pending.clear()
        self.clear_errors()
        self._rebuild()

    # -- the +/-10 % nudge (an edit like any other, through the same pipeline)

    def _context_menu(self, pos):
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        row = self._row_of_name.get(self.table.item(index.row(), 0).text()
                                    if self.table.item(index.row(), 0) else "")
        if row is None or not row_is_editable(row):
            return
        menu = QtWidgets.QMenu(self.table)
        up = menu.addAction("Value +10 %")
        down = menu.addAction("Value -10 %")
        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen is up:
            self._nudge_row(index.row(), 10)
        elif chosen is down:
            self._nudge_row(index.row(), -10)

    def _nudge_row(self, row_index: int, percent: float) -> None:
        """Set the value cell of ``row_index`` to the nudged text - an ordinary
        edit that marks the row modified and flows through Apply."""
        name_item = self.table.item(row_index, 0)
        value_item = self.table.item(row_index, 1)
        row = self._row_of_name.get(name_item.text()) if name_item is not None else None
        if row is None or value_item is None or not row_is_editable(row):
            return
        text = nudge_text(row, percent, value_item.text())
        if text:
            value_item.setText(text)
