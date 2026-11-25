"""§15.1 unified parameter table: the reconstruction F–K block, and nothing else.

Columns A–E (name | value aliased | unit | source kind | note/ref) belong to the
plan builder's sheet contract and are NEVER touched here [V: plan/builder.py
``_upsert_parameters`` / ``_write_source_metadata``]. This module appends the
reconstruction block — F = measured value, G = deviation (proposed − measured),
H = tolerance, I = confidence, J = basis, K = status — to rows located by their
B-cell alias, idempotently (a cell is rewritten only when its content differs).

Basis and status enums are exactly spec §15.2; anything else is rejected BEFORE a
single cell is written, so a bad batch can never leave a half-written block.
"""

BASIS_ENUM = ("measured", "fitted", "inferred", "user", "none")
STATUS_ENUM = ("proposed", "approved", "rejected", "deferred")

# §15.1 column letters — the only columns this module may write.
COLUMNS = (("measured", "F"), ("deviation", "G"), ("tolerance", "H"),
           ("confidence", "I"), ("basis", "J"), ("status", "K"))
FIRST_COLUMN = "F"

_NUMBER_FIELDS = ("measured", "deviation", "tolerance")


def _cell_text(value):
    """The spreadsheet cell content for a value: plain numbers, verbatim text."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return repr(round(float(value), 6))
    return str(value)


def _same(cell_content, text):
    """True when the cell already holds this value (numbers compare numerically; text
    cells read back with a leading apostrophe, like ``_write_source_metadata`` sees)."""
    current = (cell_content or "").lstrip("'").strip()
    if current == text:
        return True
    try:
        return abs(float(current) - float(text)) <= 1e-12 * max(1.0, abs(float(text)))
    except ValueError:
        return False


def _validate(rows):
    for row in rows or []:
        if not isinstance(row, dict) or not str(row.get("parameter") or "").strip():
            raise ValueError("every F–K row needs a non-empty 'parameter'")
        basis, status = row.get("basis"), row.get("status")
        if basis not in BASIS_ENUM:
            raise ValueError(f"{row.get('parameter')}: basis {basis!r} is not one of "
                             f"{BASIS_ENUM} (spec §15.2)")
        if status not in STATUS_ENUM:
            raise ValueError(f"{row.get('parameter')}: status {status!r} is not one of "
                             f"{STATUS_ENUM} (spec §15.2)")
        for field in _NUMBER_FIELDS:
            value = row.get(field)
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, (int, float))):
                raise ValueError(f"{row.get('parameter')}: {field} must be a number or "
                                 f"None, got {value!r}")
        confidence = row.get("confidence")
        if isinstance(confidence, bool) or (confidence is not None
                                            and not isinstance(confidence, (int, float, str))):
            raise ValueError(f"{row.get('parameter')}: confidence must be a number, a "
                             f"word or None, got {confidence!r}")


def append_fk_rows(sheet_writer, rows):
    """Write the §15.1 F–K block for each row onto an ALREADY-OPEN spreadsheet.

    ``rows``: dicts with ``parameter`` (the B-cell alias) plus any of the F–K fields;
    ``None`` fields are skipped (cells are never cleared). Columns A–E are never
    touched — only F..K cells of alias-located rows are written. Idempotent: a cell
    whose content already matches is left alone. Returns
    ``{"written": {parameter: [columns]}, "unchanged": [parameters], "missing":
    [parameters without an alias on the sheet]}``."""
    _validate(rows)
    report = {"written": {}, "unchanged": [], "missing": []}
    for row in rows or []:
        name = str(row["parameter"])
        cell = None
        try:
            cell = sheet_writer.getCellFromAlias(name)
        except Exception:  # noqa: BLE001 - a sheet that cannot resolve aliases reports missing
            cell = None
        if not cell:
            report["missing"].append(name)
            continue
        row_no = str(cell)[1:]          # the alias lives on the B cell → its row
        touched = []
        for field, column in COLUMNS:
            value = row.get(field)
            if value is None:
                continue
            address = f"{column}{row_no}"
            text = _cell_text(value)
            try:
                current = sheet_writer.getContents(address)
            except Exception:  # noqa: BLE001 - unreadable → rewrite honestly
                current = None
            if not _same(current, text):
                sheet_writer.set(address, text)
                touched.append(column)
        if touched:
            report["written"][name] = touched
        else:
            report["unchanged"].append(name)
    return report
