"""Writing measured board dimensions into the ``Params`` sheet, and diffing them against it.

GUI-thread only (it touches an ``App.Document``, passed in; this module imports no FreeCAD itself). The sheet has ONE producer, ``kit.params.ensure_sheet`` (get-or-create the object whose
``Name`` is ``Params``, never a ``Params001``), imported through a guarded import so this module works before the kit exists:
the fallback ``_Sheet`` writes the same layout (row 1 header ``name | value | unit | source | note``, alias on column B, numbers
as ``=<value> <unit>`` formulas that are read back).

Rules:

* every row is written with source ``step_measurement``; an existing row whose source is ``user_requirement`` is NEVER overwritten
  (the user's own number wins) and a formula row (``derived``) is left alone; both are reported as skipped with the reason;
* ``pcb_t``, ``tallest_z`` and ``bottom_z`` are ONE group: the kit shifts every connector height above the PCB top by
  ``pcb_t - 1.4``, so a new ``pcb_t`` without the matching heights would make the derived cavity disagree with the connector
  boxes. The group is written together or not at all (one protected row skips the whole group);
* measured heights come straight from the normalized frame (z = 0 at the PCB bottom), which already contains the real PCB
  thickness: ``tallest_z_new = tallest_z_old + (pcb_t_new - pcb_t_old)`` holds by construction.
"""

SOURCE = "step_measurement"
SHEET = "Params"
PROTECTED = ("user_requirement",)
GROUP = ("pcb_t", "tallest_z", "bottom_z")
DIFF_TOL = 0.005                    # mm: smaller differences are not reported
_HEADER = ("name", "value", "unit", "source", "note")
_SCAN_ROWS = 200


def _kit_params():
    """``oscar.kit.params`` or ``None`` while the kit is not delivered (a missing DEPENDENCY inside it is a real bug and raises)."""
    try:
        from ..kit import params as kit_params
    except ModuleNotFoundError as exc:
        if not str(exc.name or "").endswith((".kit", ".kit.params")):
            raise
        return None
    return kit_params


class _Sheet:
    """The minimal ``Params`` writer used when ``kit.params`` is absent: same layout, same read-back check."""

    def __init__(self, doc, create: bool):
        self.doc = doc
        sheet = doc.getObject(SHEET)
        if sheet is None:
            if not create:
                raise LookupError(SHEET)
            sheet = doc.addObject("Spreadsheet::Sheet", SHEET)
            if sheet.Name != SHEET:
                doc.removeObject(sheet.Name)
                raise ValueError(f"FreeCAD named the new sheet '{sheet.Name}' instead of '{SHEET}'; rename or delete the other object called '{SHEET}'")
            for column, text in zip("ABCDE", _HEADER):
                sheet.set(f"{column}1", text)
        self.sheet = sheet

    def _text(self, cell: str) -> str:
        return _strip_quote(self.sheet.getContents(cell) or "")

    def _rows(self) -> dict:
        rows, empty = {}, 0
        for row in range(2, _SCAN_ROWS):
            alias = self.sheet.getAlias(f"B{row}")
            if alias:
                rows[alias], empty = row, 0
            else:
                empty += 1
                if empty >= 5:
                    break
        return rows

    def table(self) -> list:
        out = []
        for name, row in self._rows().items():
            contents = self.sheet.getContents(f"B{row}") or ""
            try:
                value = self.sheet.get(name)
                value = float(getattr(value, "Value", value))
            except Exception:  # noqa: BLE001
                value = None
            out.append({"name": name, "value": value, "unit": self._text(f"C{row}"),
                        "expr": "" if not contents.startswith("=") or _literal(contents) else contents,
                        "source": self._text(f"D{row}"), "note": self._text(f"E{row}")})
        return out

    def set(self, name: str, value: float, *, unit: str, source: str, note: str) -> None:
        rows = self._rows()
        row = rows.get(name)
        if row is None:
            row = (max(rows.values()) + 1) if rows else 2
            self.sheet.set(f"A{row}", name)
            self.sheet.setAlias(f"B{row}", name)
        self.sheet.set(f"B{row}", f"={value:.6g} {unit}".rstrip())
        for column, text in (("C", unit), ("D", source), ("E", note)):
            self.sheet.set(f"{column}{row}", _safe_text(text)) if text else self.sheet.clear(f"{column}{row}")
        self.sheet.recompute()
        stored = self.sheet.get(name)
        if isinstance(stored, str):
            raise ValueError(f"Params: '{name}' was stored as text, not a number; write it as '=<value> <unit>'")


def _safe_text(text: str) -> str:
    """Text for a unit / source / note cell: a leading ``=`` or apostrophe must not become a formula or be swallowed (as ``kit.params``)."""
    return "'" + text if text[:1] in ("=", "'") else text


def _strip_quote(contents: str) -> str:
    return contents[1:] if contents.startswith("'") else contents


def _literal(contents: str) -> bool:
    import re

    return bool(re.match(r"^=\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*(?:mm|deg)?\s*$", contents))


def planned_rows(m) -> list:
    """The rows a measurement can supply: ``[{"name", "value", "unit", "note"}]`` in write order (an empty list without a PCB)."""
    if not m.pcb:
        return []
    where = f"measured from {m.object or 'the reference'}"
    pcb, rows = m.pcb, []
    rows.append({"name": "board_len", "value": pcb["length"], "unit": "mm", "note": f"{where}: PCB length (exact B-rep)"})
    rows.append({"name": "board_w", "value": pcb["width"], "unit": "mm", "note": f"{where}: PCB width (exact B-rep)"})
    rows.append({"name": "pcb_t", "value": pcb["thickness"], "unit": "mm", "note": f"{where}: PCB thickness"})
    if m.tallest_z is not None:
        rows.append({"name": "tallest_z", "value": m.tallest_z, "unit": "mm", "note": f"{where}: highest point above the PCB bottom"})
        rows.append({"name": "bottom_z", "value": m.bottom_z, "unit": "mm", "note": f"{where}: lowest point below the PCB bottom"})
    mount = m.mounting
    if mount:
        rows.append({"name": "hole_d", "value": mount["d"], "unit": "mm", "note": f"{where}: mounting hole diameter ({mount['count']} holes)"})
        pattern = mount.get("pattern")
        if pattern:
            rows.append({"name": "hole_dx", "value": pattern["dx"], "unit": "mm", "note": f"{where}: mounting hole spacing along the length"})
            rows.append({"name": "hole_dy", "value": pattern["dy"], "unit": "mm", "note": f"{where}: mounting hole spacing along the width"})
            if pattern["symmetric"]:
                rows.append({"name": "hole_off", "value": pattern["offset"][0], "unit": "mm",
                             "note": f"{where}: first mounting hole from the board edge"})
    return rows


def _open(doc, create: bool):
    """``(sheet object with table() / set(), is_kit)`` or ``(None, False)`` when there is no sheet and ``create`` is False."""
    kit = _kit_params()
    if kit is not None:
        if create:
            return kit.ensure_sheet(doc), True
        if doc.getObject(SHEET) is None:
            return None, True
        return kit.Params(doc), True
    try:
        return _Sheet(doc, create), False
    except LookupError:
        return None, False


def sync(doc, m, *, write: bool) -> dict:
    """Diff a measurement against the ``Params`` sheet and, with ``write=True``, write it.

    Returns ``{"written": [name, ...], "skipped": [{"name", "why"}], "diff": [{"name", "sheet", "measured", "source"}],
    "names": [every Params name after the call], "sheet": bool}``. A failure of one row (a bad alias, a text cell) is reported
    in ``skipped`` and never aborts the others; with ``write=False`` nothing in the document changes (no sheet is created).
    """
    report: dict = {"written": [], "skipped": [], "diff": [], "names": [], "sheet": False}
    rows = planned_rows(m)
    try:
        sheet, _is_kit = _open(doc, create=write and bool(rows))
    except ValueError as exc:                               # kit.params.ParamsError: the name 'Params' belongs to something else
        report["skipped"].append({"name": SHEET, "why": str(exc)[:200]})
        return report
    if sheet is None:
        return report
    report["sheet"] = True
    existing = {r["name"]: r for r in sheet.table()}
    for row in rows:
        old = existing.get(row["name"])
        if old is not None and old.get("value") is not None and abs(old["value"] - row["value"]) > DIFF_TOL:
            report["diff"].append({"name": row["name"], "sheet": round(old["value"], 4), "measured": row["value"], "source": old.get("source", "")})
    if not write:
        report["names"] = list(existing)
        return report
    blocked_group = next((n for n in GROUP if (existing.get(n) or {}).get("source") in PROTECTED), "")
    wrote = False
    for row in rows:
        name, old = row["name"], existing.get(row["name"])
        if name in GROUP and blocked_group:
            report["skipped"].append({"name": name, "why": f"'{blocked_group}' is a user requirement; pcb_t, tallest_z and bottom_z are written together or not at all"})
            continue
        if old is not None and old.get("source") in PROTECTED:
            report["skipped"].append({"name": name, "why": "the user set it (source user_requirement)"})
            continue
        if old is not None and old.get("expr"):
            report["skipped"].append({"name": name, "why": "it is a formula (derived) row"})
            continue
        try:
            sheet.set(name, row["value"], unit=row["unit"], source=SOURCE, note=row["note"])
        except (ValueError, LookupError) as exc:            # kit.params.ParamsError is a ValueError
            report["skipped"].append({"name": name, "why": str(exc)[:160]})
            continue
        report["written"].append(name)
        wrote = True
    if wrote:
        doc.recompute()
    report["names"] = [r["name"] for r in sheet.table()]
    return report
