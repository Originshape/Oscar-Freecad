"""Private helpers of ``oscar.tools`` (the registry and the ``run_tool`` pipeline).

PURE PYTHON: no FreeCAD, no Qt, no ``oscar`` imports at import time, so every function here
is unit-testable without a document. ``oscar.tools`` owns the public API (``clip_text``,
``derive_title`` and ``specs`` are re-exported there); this module only keeps ``tools.py``
readable. Nothing in here is public API.

Contents:

* ``clip_text``                     result clipping
* ``derive_title`` / ``clean_title``  row titles for steps
* ``first_sentence``                compact descriptions for the ``lite`` tool set
* ``coerce_value`` / ``check_argument``  argument repair and validation in the fix-it style
* ``view_word`` / ``resolve_view_name``  the cosmetic ``view_name`` is resolved, never rejected
* ``delta_between`` & co.           object deltas computed from two ``StateSnapshot`` objects
* ``health_suffix`` & co.           the exact text ``run_tool`` appends to a geometry call
"""

import difflib
import json
import math
import os
import re
from typing import Any, Callable, Iterable

CLIP_LIMIT = 12000
TITLE_LIMIT = 80

# --------------------------------------------------------------------------
# Clipping


def clip_text(text: Any, limit: int = CLIP_LIMIT) -> str:
    """Clip a tool result to about ``limit`` characters, keeping its head and its tail.

    Contract: text of at most ``limit`` characters is returned unchanged;
    longer text becomes the first ``limit * 2/3`` characters (8000 for the default 12000), the
    marker ``\\n...(N chars omitted; call get_objects / get_object for details)...\\n`` and the
    last ``limit * 7/24`` characters (3500). ``N`` is the number of characters left out. The
    tail matters because ``run_tool`` appends the ``[health]`` line AFTER clipping, and
    handlers put their conclusion (``Next stage: ...``) last.
    """
    text = "" if text is None else str(text)
    limit = max(int(limit), 1)
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit * 7 // 24
    omitted = len(text) - head - tail
    tail_text = text[len(text) - tail:] if tail else ""
    return (f"{text[:head]}\n...({omitted} chars omitted; call get_objects / get_object "
            f"for details)...\n{tail_text}")


# --------------------------------------------------------------------------
# Titles


def clean_title(value: Any, limit: int = TITLE_LIMIT) -> str:
    """One-line, whitespace-collapsed title of at most ``limit`` characters ('' for None).

    The model's own ``step_title`` and every derived title pass through here, so a title never
    carries a newline into a panel row or a ``steps.jsonl`` line.
    """
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[:limit - 1].rstrip() + "…"


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _join(*parts: Any) -> str:
    """Space-joined non-empty parts (a missing argument never leaves a double space or a 'None')."""
    return " ".join(p for p in (_text(p) for p in parts) if p)


def _num(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return _text(value)


def _basename(path: Any) -> str:
    return os.path.basename(_text(path).replace("\\", "/").rstrip("/"))


_CODING_PRAGMA = re.compile(r"coding[:=]", re.IGNORECASE)
_DEF_NAME = re.compile(r"^[ \t]*def[ \t]+([A-Za-z_]\w*)", re.MULTILINE)
_LEADING_DOCSTRING = re.compile(r"\A\s*(?:[rRuUbB]{0,2})(\"\"\"|''')(.*?)\1", re.DOTALL)


def _script_title(code: Any) -> str:
    """Title of a script: its first ``# `` comment, else ``Run <first def>``,
    else the first line of a leading docstring, else ``Run script (N lines)``."""
    if not isinstance(code, str) or not code.strip():
        return "Run script (0 lines)"
    lines = code.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("# "):
            continue
        comment = stripped[2:].strip()
        if comment and re.search(r"[A-Za-z0-9]", comment) and not _CODING_PRAGMA.search(comment):
            return comment
    match = _DEF_NAME.search(code)
    if match:
        return f"Run {match.group(1)}"
    doc = _LEADING_DOCSTRING.match(code)
    if doc:
        for line in doc.group(2).splitlines():
            if line.strip():
                return line.strip()
    return f"Run script ({len(lines)} lines)"


def _host(url: Any) -> str:
    from urllib.parse import urlparse

    try:
        return urlparse(_text(url)).netloc
    except ValueError:
        return ""


def _set_parameters_title(args: dict) -> str:
    rows = args.get("params")
    names = []
    if isinstance(rows, list):
        for row in rows:
            name = _text(row.get("name")) if isinstance(row, dict) else ""
            if name:
                names.append(name)
    return "Set parameters: " + ", ".join(names[:4]) if names else "Set parameters"


def _export_title(args: dict) -> str:
    formats = args.get("formats")
    if isinstance(formats, str):
        formats = [formats]
    if not isinstance(formats, (list, tuple)) or not formats:
        formats = ["step", "stl"]
    return "Export " + "+".join(_text(f) for f in formats if _text(f))


# tool name -> title function(args)
_TITLES: dict[str, Callable[[dict], str]] = {
    "create_document": lambda a: _join("Create document", a.get("name")),
    "create_object": lambda a: _join("Create", a.get("obj_type"), a.get("obj_name")),
    "edit_object": lambda a: _join("Edit", a.get("obj_name")),
    "delete_object": lambda a: _join("Delete", a.get("obj_name")),
    "execute_code": lambda a: _script_title(a.get("code")),
    "execute_code_headless": lambda a: _script_title(a.get("code")),
    "execute_code_async": lambda a: _script_title(a.get("code")),
    "import_file": lambda a: _join("Import", _basename(a.get("path"))),
    "import_reference_part": lambda a: _join(
        "Import reference", _text(a.get("id")) or _basename(a.get("path"))),
    "source_reference_part": lambda a: _join(
        "Source reference", _text(a.get("id")) or _host(a.get("url"))),
    "measure_reference_part": lambda a: _join("Measure reference", a.get("object")),
    "build_enclosure": lambda a: "Build enclosure: " + (_text(a.get("stage")) or "next"),
    "add_cutout": lambda a: (
        f"Cut {_num(a.get('w'))} x {_num(a.get('h'))} opening in "
        + _join(a.get("part"), a.get("face"))),
    "add_vents": lambda a: f"Cut {_text(a.get('count')) or '?'} vent slots in {_text(a.get('part'))}".rstrip(),
    "set_parameters": _set_parameters_title,
    "set_parameter": lambda a: _join("Set", a.get("name"), "=", a.get("value")),
    "bind_expression": lambda a: "Bind " + ".".join(
        p for p in (_text(a.get("object")), _text(a.get("property"))) if p),
    "apply_feature_plan": lambda a: "Build the feature plan",
    "reapply_plan_feature": lambda a: _join("Rebuild plan feature", a.get("feature_id")),
    "insert_part_from_library": lambda a: _join("Insert", a.get("relative_path")),
    "save_document": lambda a: _join("Save", a.get("doc_name")),
    "export_model": _export_title,
}


def derive_title(name: str, args: Any, entry: Any = None) -> str:
    """The row title of a tool call when the model gave no ``step_title``.

    Precedence inside this function: the tool's own ``Tool.title(args)`` callable (``entry``)
    when it returns a non-empty string, then the built-in title table, then the tool name. The caller
    (``run_tool``) applies ``step_title`` and ``ctx.state["title_hint"]`` before it. Never
    raises (a bad argument shape falls back to the tool name); the result is one line of at
    most 80 characters.
    """
    args = args if isinstance(args, dict) else {}
    title_fn = getattr(entry, "title", None)
    if callable(title_fn):
        try:
            custom = clean_title(title_fn(args))
        except Exception:  # noqa: BLE001 - a broken title callable must never break a call
            custom = ""
        if custom:
            return custom
    builder = _TITLES.get(name)
    if builder is not None:
        try:
            derived = clean_title(builder(args))
        except Exception:  # noqa: BLE001
            derived = ""
        if derived:
            return derived
    return clean_title(name)


# --------------------------------------------------------------------------
# Lite descriptions

_ABBREVIATIONS = frozenset({"e.g", "i.e", "etc", "vs", "approx", "incl", "cf"})


def first_sentence(text: str, limit: int = 200) -> str:
    """The first sentence of ``text``, whitespace-collapsed and at most ``limit`` characters.

    Used for the ``lite`` tool descriptions. A period that follows an
    abbreviation (``e.g.``, ``i.e.``, ``etc.``) or sits inside a number does not end the
    sentence; a sentence longer than ``limit`` is cut with an ellipsis (still <= limit).
    """
    flat = " ".join(str(text or "").split())
    sentence = flat
    for match in re.finditer(r"[.!?](?=\s|$)", flat):
        word = re.search(r"([A-Za-z.]+)$", flat[:match.start()])
        if word and word.group(1).lower().rstrip(".") in _ABBREVIATIONS:
            continue
        sentence = flat[:match.end()]
        break
    if len(sentence) > limit:
        sentence = sentence[:limit - 1].rstrip() + "…"
    return sentence


# --------------------------------------------------------------------------
# Argument coercion and validation

_TRUE = ("true", "yes", "1", "on")
_FALSE = ("false", "no", "0", "off")
_ARTICLE = {"string": "a string", "integer": "an integer", "number": "a number",
            "boolean": "true or false", "object": "a JSON object", "array": "a JSON array"}


def coerce_value(value: Any, schema: dict[str, Any]) -> Any:
    """Repair the argument shapes models commonly get wrong, driven by the JSON schema fragment.

    Objects and arrays sent as JSON strings are parsed; booleans sent as ``"true"/"yes"/"1"/
    "on"`` or ``"false"/"no"/"0"/"off"`` (or the integers 0/1) become bools; numbers sent as
    strings become numbers; integral floats become ints for ``integer`` properties. A value
    that cannot be repaired is returned unchanged so the validator can report it.
    """
    kind = schema.get("type")
    if isinstance(value, str):
        text = value.strip()
        if kind == "object" and text.startswith("{"):
            try:
                parsed = json.loads(text)
            except ValueError:
                return value
            return parsed if isinstance(parsed, dict) else value
        if kind == "array" and text.startswith("["):
            try:
                parsed = json.loads(text)
            except ValueError:
                return value
            return parsed if isinstance(parsed, list) else value
        if kind == "boolean":
            lowered = text.lower()
            if lowered in _TRUE:
                return True
            if lowered in _FALSE:
                return False
            return value
        if kind in ("integer", "number"):
            try:
                number = float(text)
            except ValueError:
                return value
            if not math.isfinite(number):
                return value
            return int(number) if kind == "integer" and number.is_integer() else number
        return value
    if kind == "boolean" and isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _describe(value: Any) -> str:
    shown = repr(value)
    if len(shown) > 40:
        shown = shown[:37] + "..."
    return shown if isinstance(value, str) else f"{type(value).__name__} {shown}"


# Words models use for a view, folded to the canonical lower-case name.
_VIEW_ALIASES = {"iso": "isometric", "axonometric": "isometric", "rear": "back"}


def view_word(value: Any) -> str:
    """'Rear view' -> 'back': lower-cased, separators and the word 'view' dropped, aliases folded."""
    words = str(value).lower().replace("_", " ").replace("-", " ").split()
    key = " ".join(w for w in words if w != "view")
    return _VIEW_ALIASES.get(key, key)


def resolve_view_name(value: Any, names: Iterable[str], default: str = "Isometric") -> str:
    """The screenshot orientation is cosmetic: an unknown spelling falls back to ``default`` instead of refusing a
    call that changes the model. ``names`` are the canonical spellings (``tools.VIEW_NAMES``)."""
    word = view_word(value)
    return next((name for name in names if name.lower() == word), default)


def similar_options(value: Any, options: Iterable[Any], limit: int = 3) -> list[str]:
    """Enum members that look like ``value``: prefix matches first, then difflib neighbours."""
    text = _text(value).lower()
    choices = [str(o) for o in options]
    found: list[str] = []
    if len(text) >= 2:
        for option in choices:
            low = option.lower()
            if low.startswith(text) or (len(low) >= 3 and text.startswith(low)):
                found.append(option)
    if not found:
        by_lower = {o.lower(): o for o in choices}
        found = [by_lower[m] for m in difflib.get_close_matches(text, list(by_lower), n=limit, cutoff=0.5)]
    return found[:limit]


def _check_enum(key: str, value: Any, options: list[Any]) -> tuple[Any, str]:
    if value in options:
        return value, ""
    if isinstance(value, str):
        folded = [o for o in options if isinstance(o, str) and o.lower() == value.strip().lower()]
        if len(folded) == 1:
            return folded[0], ""            # 'front' -> 'Front'
    close = similar_options(value, options)
    hint = f" (similar: {', '.join(close)})" if close else ""
    return value, (f"'{key}' must be one of {'|'.join(str(o) for o in options)}, "
                   f"got {_describe(value) if not isinstance(value, str) else repr(value)}{hint}")


def _check_bounds(key: str, value: Any, schema: dict[str, Any]) -> str:
    low, high = schema.get("minimum"), schema.get("maximum")
    if low is not None and value < low:
        pass
    elif high is not None and value > high:
        pass
    else:
        return ""
    if low is not None and high is not None:
        rule = f"between {_num(low)} and {_num(high)}"
    elif low is not None:
        rule = f"at least {_num(low)}"
    else:
        rule = f"at most {_num(high)}"
    return f"'{key}' must be {rule}, got {_num(value)}"


def check_argument(key: str, value: Any, schema: dict[str, Any]) -> tuple[Any, str]:
    """Validate (and lightly normalise) one already-coerced argument; returns ``(value, problem)``.

    ``problem`` is ``""`` or one clause in the fix-it style, without a trailing period, for
    example ``'view_name' must be one of Isometric|Front|..., got 'iso' (similar: Isometric)``.
    Checks the JSON type, ``enum`` (a case-only miss is repaired), ``minimum`` / ``maximum`` and
    the item type of arrays. A number sent for a string property becomes its text (the schema
    of ``set_parameter.value`` says plain JSON numbers are accepted). ``None`` always passes
    (optional arguments the model sent as null); a schema without a ``type`` is not checked.
    """
    if value is None:
        return value, ""
    kind = schema.get("type")
    if not isinstance(kind, str):
        return value, ""
    if kind == "string":
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return value, f"'{key}' must be a string, got {_describe(value)}"
        if not isinstance(value, str):
            value = str(value)
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return value, f"'{key}' must be an integer, got {_describe(value)}"
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return value, f"'{key}' must be a number, got {_describe(value)}"
    elif kind == "boolean":
        if not isinstance(value, bool):
            return value, f"'{key}' must be true or false, got {_describe(value)}"
    elif kind == "object":
        if not isinstance(value, dict):
            return value, f"'{key}' must be a JSON object (property name -> value), got {type(value).__name__}"
    elif kind == "array":
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            return value, f"'{key}' must be a JSON array, got {type(value).__name__}"
        items = schema.get("items")
        if isinstance(items, dict) and isinstance(items.get("type"), str):
            fixed = []
            for index, item in enumerate(value):
                item = coerce_value(item, items)
                item, problem = check_argument(f"{key}[{index}]", item, items)
                if problem:
                    return value, problem
                fixed.append(item)
            value = fixed
    options = schema.get("enum")
    if isinstance(options, list) and options:
        value, problem = _check_enum(key, value, options)
        if problem:
            return value, problem
    if kind in ("integer", "number"):
        problem = _check_bounds(key, value, schema)
        if problem:
            return value, problem
    return value, ""


# --------------------------------------------------------------------------
# Object deltas from two StateSnapshot objects (used when a health report carries no delta)


def _objects(snapshot: Any) -> dict[str, dict]:
    objects = getattr(snapshot, "objects", None)
    return objects if isinstance(objects, dict) else {}


def _is_temp(name: str) -> bool:
    return str(name).startswith("_oscar_tmp")


_CHANGE_FIELDS = ("key", "volume", "bbox", "placement", "expr")


def _changed_props(old: dict, new: dict) -> list[str]:
    """Which kinds of property differ between two snapshot entries of one object."""
    props = []
    if old.get("placement") != new.get("placement"):
        props.append("Placement")
    if old.get("expr") != new.get("expr"):
        props.append("Expression")
    if old.get("volume") != new.get("volume") or old.get("bbox") != new.get("bbox"):
        props.append("Shape")
    return props or ["Properties"]


def _regenerated(was: dict, now: dict) -> bool:
    """True when two snapshot entries of one geometry object differ ONLY by the shape hash.

    A geometry key is ``("g", shape_hash, placement, bbox, faces, edges)``. The hash is the identity of
    the OCC shape, so a recompute (or the recompute that follows a rollback) gives the same solid a new
    hash; equal volume, bbox, placement, expression, face and edge counts mean nothing else moved.
    """
    old_key, new_key = was.get("key"), now.get("key")
    if not (isinstance(old_key, tuple) and isinstance(new_key, tuple)
            and len(old_key) >= 4 and len(old_key) == len(new_key)
            and old_key[0] == new_key[0] == "g"):
        return False
    if old_key[2:] != new_key[2:]:
        return False
    return all(was.get(f) == now.get(f) for f in ("volume", "bbox", "placement", "expr"))


def delta_between(before: Any, after: Any, *, ignore_regenerated: bool = False) -> dict[str, Any]:
    """Object delta between two ``StateSnapshot`` objects, in ``ObjectDelta.to_dict()`` shape.

    ``{"added": [names], "removed": [names], "changed": [{"name", "props", "volume_before",
    "volume_after"}]}``. Visibility alone is not a change; ``_oscar_tmp*`` objects are
    ignored; ``before=None`` means "the document did not exist" (everything is added).
    With ``ignore_regenerated`` an object whose only difference is a new shape hash (the same
    solid regenerated by a recompute or a rollback) is not reported: a failed call that rolled
    itself back must not claim it left changes.
    """
    old = {n: v for n, v in _objects(before).items() if not _is_temp(n)}
    new = {n: v for n, v in _objects(after).items() if not _is_temp(n)}
    added = [n for n in new if n not in old]
    removed = [n for n in old if n not in new]
    changed = []
    for name, now in new.items():
        was = old.get(name)
        if was is None:
            continue
        if any(was.get(f) != now.get(f) for f in _CHANGE_FIELDS):
            if ignore_regenerated and _regenerated(was, now):
                continue
            changed.append({"name": name, "props": _changed_props(was, now),
                            "volume_before": was.get("volume"), "volume_after": now.get("volume")})
    return {"added": added, "removed": removed, "changed": changed}


def _without_regenerated(delta: dict[str, Any], before: Any, after: Any) -> dict[str, Any]:
    """``delta`` without the changed entries whose snapshot entries differ only by a regenerated shape hash."""
    old, new = _objects(before), _objects(after)
    kept = [entry for entry in delta.get("changed") or []
            if not (isinstance(entry, dict) and entry.get("name") in old and entry.get("name") in new
                    and _regenerated(old[entry["name"]], new[entry["name"]]))]
    return {**delta, "changed": kept}


def delta_size(delta: dict[str, Any]) -> int:
    """Number of added + removed + changed objects of an ``ObjectDelta`` dict."""
    return len(delta.get("added") or []) + len(delta.get("removed") or []) + len(delta.get("changed") or [])


def delta_names(delta: dict[str, Any]) -> list[str]:
    """Names of the objects a delta added or changed (what ``ctx.state['touched']`` records)."""
    names = list(delta.get("added") or [])
    for entry in delta.get("changed") or []:
        name = entry.get("name") if isinstance(entry, dict) else entry
        if name:
            names.append(str(name))
    return names


def delta_line(delta: dict[str, Any], max_names: int = 3) -> str:
    """One-line form of a delta dict, the same grammar as ``ObjectDelta.line()``: ``+Lid ~Shell``."""
    parts = [f"+{n}" for n in (delta.get("added") or [])[:max_names]]
    parts += [f"-{n}" for n in (delta.get("removed") or [])[:max_names]]
    for entry in (delta.get("changed") or [])[:max_names]:
        if not isinstance(entry, dict):
            parts.append(f"~{entry}")
            continue
        before, after = entry.get("volume_before"), entry.get("volume_after")
        if before is not None and after is not None and before != after:
            parts.append(f"~{entry.get('name')}(vol {before:.1f}->{after:.1f})"
                         if isinstance(before, (int, float)) and isinstance(after, (int, float))
                         else f"~{entry.get('name')}")
        else:
            parts.append(f"~{entry.get('name')}")
    return " ".join(parts)


def resolve_doc_name(name: Any, candidates: Iterable[str]) -> str:
    """Resolve a model-supplied document name against the open ones, tolerating near-misses.

    Same rules as the engine's tolerant lookup: exact, then case-insensitive, then a unique
    case-insensitive prefix. Returns ``""`` when nothing (or more than one) matches.
    """
    wanted = _text(name)
    names = list(candidates)
    if not wanted:
        return ""
    if wanted in names:
        return wanted
    lowered = [n for n in names if n.lower() == wanted.lower()]
    if len(lowered) == 1:
        return lowered[0]
    prefixed = [n for n in names if n.lower().startswith(wanted.lower())]
    return prefixed[0] if len(prefixed) == 1 else ""


# --------------------------------------------------------------------------
# Health rendering

NOT_OK_BANNER = ("NOT OK: the call ran but the model is in an error state now. "
                 "Fix it before building on it.")


def _call(obj: Any, name: str, default: Any = None) -> Any:
    """``obj.name`` whether it is a property or a zero-argument method; ``default`` when absent."""
    member = getattr(obj, name, default)
    return member() if callable(member) else member


def finding_key(finding: Any) -> str:
    """Stable string key of a finding (``CODE:object``), used by ``ctx.state['problems']``."""
    return f"{getattr(finding, 'code', '')}:{getattr(finding, 'object', '')}"


def finding_line(finding: Any) -> str:
    """``  CODE object: message`` (the line format of the OPEN PROBLEMS / STILL FAILING blocks)."""
    obj = getattr(finding, "object", "")
    label = f"{getattr(finding, 'code', '')} {obj}".rstrip()
    return f"  {label}: {getattr(finding, 'message', '')}"


def summarize_report(report: Any) -> dict[str, Any]:
    """``{"valid", "errors", "warnings", "line", "findings"}`` of a ``HealthReport`` (findings <= 5, errors first)."""
    errors = list(_call(report, "errors", []) or [])
    warnings = list(_call(report, "warnings", []) or [])
    findings = []
    for finding in (errors + warnings)[:5]:
        to_dict = getattr(finding, "to_dict", None)
        findings.append(to_dict() if callable(to_dict) else {
            "severity": getattr(finding, "severity", ""), "code": getattr(finding, "code", ""),
            "object": getattr(finding, "object", ""), "message": getattr(finding, "message", "")})
    return {"valid": bool(_call(report, "valid", True)), "errors": len(errors),
            "warnings": len(warnings), "line": str(_call(report, "line", "") or ""),
            "findings": findings}


def health_suffix(report: Any, new: list, persisting: list, *, handler_ok: bool,
                  delta_text: str, extra_lines: Iterable[str] = ()) -> str:
    """The text ``run_tool`` appends to a geometry call's result.

    * handler ok, no new errors: ``\\n\\n<report.line()>`` plus up to 3 ``OPEN PROBLEMS:`` lines
      for errors and warnings that already existed before the call;
    * handler ok, NEW errors: ``\\n\\nNOT OK: ...`` banner, ``report.render("compact")`` and up
      to 3 ``STILL FAILING (not new):`` lines (persisting errors only);
    * handler failed but the call left changes (``delta_text`` non-empty): a ``PARTIAL STATE``
      line plus the report line (or the compact table when errors exist);
    * handler failed and nothing changed: ``""`` (the error text stands alone).

    ``extra_lines`` are the report lines of OTHER documents the call changed.
    """
    line = str(_call(report, "line", "") or "")
    extras = "".join(f"\n{text}" for text in extra_lines if text)
    errors_present = bool(list(_call(report, "errors", []) or []))
    if not handler_ok:
        if not delta_text:
            return ""
        body = (str(report.render(mode="compact")) if errors_present and hasattr(report, "render") else line)
        return ("\n\nPARTIAL STATE: the failed call left changes: "
                f"{delta_text}. Inspect with get_objects before retrying.\n{body}{extras}")
    if new:
        text = f"\n\n{NOT_OK_BANNER}\n{report.render(mode='compact')}"
        still = [f for f in persisting if str(getattr(f, "severity", "error")) == "error"][:3]
        if still:
            text += "\nSTILL FAILING (not new):\n" + "\n".join(finding_line(f) for f in still)
        return text + extras
    text = f"\n\n{line}"
    if persisting:
        text += "\nOPEN PROBLEMS:\n" + "\n".join(finding_line(f) for f in persisting[:3])
    return text + extras


def is_dispatch_timeout(text: Any, handler_ok: bool) -> bool:
    """True when a FAILED handler result is a GUI-dispatch timeout (the script may still be running).

    The dispatcher reports ``'<op>' did not complete within <t>s (the GUI
    thread stayed busy)``; the stuck-script variant adds ``STILL RUNNING``. A timed-out call
    skips health (and the Project's GUI hooks) so a stuck script costs one timeout, not four.
    """
    if handler_ok:
        return False
    body = str(text or "")
    return "did not complete within" in body or "STILL RUNNING" in body


# --------------------------------------------------------------------------
# The health stage of run_tool


def pick_doc(requested: Any, before: dict, after: dict) -> str:
    """The document a geometry call is judged on.

    The one the call named (tolerant of near-misses), else the one with the largest object
    delta between the two snapshot dicts, else ``""`` (= the active document at dispatch time).
    """
    resolved = resolve_doc_name(requested, after or before)
    if resolved:
        return resolved
    best, best_size = "", 0
    if before:
        for name, snapshot in after.items():
            size = delta_size(delta_between(before.get(name), snapshot))
            if size > best_size:
                best, best_size = name, size
    return best


def _record_problems(doc: str, report: Any, into: dict) -> None:
    """Add the errors and warnings of ``report`` to ``into`` as ``"<doc>|<code>:<object>" -> line``."""
    findings = list(_call(report, "errors", []) or []) + list(_call(report, "warnings", []) or [])
    for finding in findings:
        into[f"{doc}|{finding_key(finding)}"] = finding_line(finding).strip()


def health_stage(state: dict, health: Any, prepared: dict, before: dict,
                 handler_ok: bool) -> dict | None:
    """Judge the model after a geometry call; ``health`` is the ``oscar.cad.health`` module.

    ``before`` is the ``{doc: StateSnapshot}`` taken before the handler ran. Returns ``None``
    when nothing could or needed to be judged (no report because of a dispatch timeout, or the
    call failed without changing any document); else ``{"report", "doc", "new", "persisting",
    "extra_lines", "delta"}``: ``new`` are the error findings introduced by this call,
    ``persisting`` the errors and warnings that were already known, ``extra_lines`` one
    ``report.line()`` per OTHER document the call changed (at most 3), ``delta`` the object
    delta of the judged document as an ``ObjectDelta.to_dict()``. After a FAILED call, objects
    that differ only by a regenerated shape hash (a rollback and its recompute) are not changes.

    Writes the per-turn bookkeeping into ``state``: ``problems`` (replaced by
    this call's findings), ``last_health`` (``{doc: report.to_dict()}``), ``touched``
    (``{doc: set of added / changed object names}``) and ``last_doc``.
    """
    after = health.snapshot_all_via_dispatch() or {}
    doc = pick_doc(prepared.get("doc_name"), before, after)
    if not handler_ok and before and not any(
            delta_size(delta_between(before.get(n), snap, ignore_regenerated=True)) for n, snap in after.items()):
        return None                                  # the call failed and left no trace (regenerated shapes are no trace)
    report = health.health_via_dispatch(doc or None, prev=before.get(doc) if doc else None)
    if report is None:
        from . import turnlog

        turnlog.log_event("hook_skipped", hook="health", why="no report (timeout or document gone)")
        return None
    doc = getattr(report, "doc", "") or doc
    prev = before.get(doc)
    new = list(report.new_errors(prev))
    known = set(state.get("problems") or {})
    known.update(f"{doc}|{code}:{obj}" for code, obj in (getattr(prev, "finding_keys", None) or ()))
    if prev is None and not before:
        # No baseline at all (the snapshot before the call failed or timed out): every error
        # would look new. What this turn already reported is the only baseline left.
        new = [f for f in new if f"{doc}|{finding_key(f)}" not in known]
    new_keys = {f"{doc}|{finding_key(f)}" for f in new}
    everything = list(_call(report, "errors", []) or []) + list(_call(report, "warnings", []) or [])
    persisting = [f for f in everything
                  if (key := f"{doc}|{finding_key(f)}") in known and key not in new_keys]
    report_delta = getattr(report, "delta", None)
    if report_delta is not None and hasattr(report_delta, "to_dict"):
        delta = report_delta.to_dict()
    else:
        delta = delta_between(prev, after.get(doc)) if doc in after else {}
    if not handler_ok and prev is not None and doc in after:
        # After a failure only REAL changes count: a rollback (and the recompute after it) regenerates the
        # same solids under new shape hashes, which the report's hash-keyed delta calls "changed".
        delta = _without_regenerated(delta, prev, after[doc])
    problems: dict[str, str] = {}
    _record_problems(doc, report, problems)
    state.setdefault("last_health", {})[doc] = report.to_dict()
    state.setdefault("touched", {}).setdefault(doc, set()).update(delta_names(delta))
    state["last_doc"] = doc
    extra_lines: list[str] = []
    if before:
        for other in [n for n in after if n != doc][:3]:
            other_delta = delta_between(before.get(other), after[other], ignore_regenerated=not handler_ok)
            if not delta_size(other_delta):
                continue
            other_report = health.health_via_dispatch(other, prev=before.get(other))
            if other_report is None:
                continue
            extra_lines.append(str(other_report.line()))
            state["last_health"][other] = other_report.to_dict()
            state["touched"].setdefault(other, set()).update(delta_names(other_delta))
            _record_problems(other, other_report, problems)
    state["problems"] = problems
    return {"report": report, "doc": doc, "new": new, "persisting": persisting,
            "extra_lines": extra_lines, "delta": delta}


# --------------------------------------------------------------------------
# Build failures


def _fallback_failure_text(result: Any, feature_id: str) -> str:
    """Same output as ``plan.builder.format_build_failure`` (kept in step with it)."""
    if not isinstance(result, dict):
        return f"Plan failed: {_text(result) or 'unknown error'}"
    where = _text(result.get("feature_id")) or feature_id or _text(result.get("stage")) or "plan"
    detail = _text(result.get("error"))
    if not detail:
        errors = result.get("errors")
        if isinstance(errors, (list, tuple)):
            detail = "; ".join(t for t in (_text(e) for e in errors) if t)
        else:
            detail = _text(errors)
    text = f"Plan failed at '{where}': {detail or 'unknown error'}"
    completed = result.get("completed")
    if completed is not None:
        done = list(completed) if isinstance(completed, (list, tuple, set)) else [completed]
        text += f"\nCompleted features: {done}"
    return text


def build_failure_text(result: dict, feature_id: str = "") -> str:
    """Model-facing text of a failed ``apply_feature_plan`` / ``reapply_plan_feature`` result.

    Uses ``plan.builder.format_build_failure(result)`` when it exists, so the wording has
    one owner; otherwise the same output is built here: ``Plan failed at '<feature_id or
    stage>': <error, or the errors joined by '; '>`` then, when the result has a ``completed``
    list, ``Completed features: [...]``. A missing field never prints as ``None`` (the old
    "Plan failed at validation: None" bug) and a result without any reason reads ``unknown
    error``. ``feature_id`` is the id the caller asked for, used when the result names none.
    Never raises.
    """
    if feature_id and isinstance(result, dict) and not _text(result.get("feature_id")):
        result = dict(result, feature_id=feature_id)
    formatter = None
    try:
        from .plan import builder as _builder

        formatter = getattr(_builder, "format_build_failure", None)
    except ImportError:
        formatter = None
    if callable(formatter):
        try:
            text = str(formatter(result))
            if text:
                return text
        except Exception:  # noqa: BLE001 - never lose the failure to a formatter bug
            pass
    return _fallback_failure_text(result, "")
