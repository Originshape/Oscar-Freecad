"""Parameter tools: ``set_parameters``, ``get_parameters``, ``bind_expression``, ``set_parameter``.

The ``Params`` spreadsheet is the single source of truth of every driving dimension; these tools are the model's way to read and
edit it (``oscar.kit.params.Params`` does the cell work). Registered by ``tools_kit`` (imported at its end), so ``_TOOL_MODULES``
needs one entry for the whole enclosure family.

Behaviour shared by the editing tools:

* pure argument validation first, every problem listed in ONE ``ToolError``; then the document on the GUI thread
  (``tools_kit_base.gui_call``; the context is captured before dispatching, no lock is held);
* an edit is ONE undo transaction under ``UndoMode 1`` (``kit.build.atomic``): a refused edit (an alias FreeCAD rejects, a
  spec rule such as the minimum wall, an expression that breaks a recompute) leaves the document exactly as it was;
* a COUNT / shape parameter (``kit.params.STAGE_OF_PARAM``: vent and hook numbers, ``port_margin``) is not expression-bound, so
  after the cell changed the stages that depend on it and were already built are rebuilt (``kit.build.rebuild_built``) from
  the effective spec; when that rebuild fails the cell is restored and the model rebuilt from the old value;
* a STRUCTURAL parameter (``kit.params.STRUCTURAL_STAGE_OF_PARAM``: ``lip_h``, ``lid_gap``, ``lid_fit``, ``pcb_t``, ``tallest_z``) is
  expression-bound but decides which skirt notches / merged openings exist or feeds numbers frozen at build time, so it is rebuilt
  the same way (a plain recompute left a model that differs from a fresh build with the same value);
* the Project keeps ONE spec for the chat: an edit in a document that is not the enclosure's (``base.spec_applies``) is neither
  validated against it nor written into it;
* handlers tell ``run_tool`` which ``Params`` names exist afterwards through ``ctx.state["extra"]["param_names"]``.

``params_factory`` is the ``P`` helper of ``execute_code`` scripts (registered by ``tools_kit``): ``P = P()`` is the ``Params`` of the
active document.
"""

import copy
import difflib
from typing import Any

from . import tools_kit_base as base
from .kit import params as _params
from .kit import spec as _spec
from .tools import ToolError, current_context, tool

_MAX_ROWS = 40
_MAX_ITEMS = 80
_PHASES_SET = ("PARAMS", "BUILD", "VERIFY")
_PHASES_GET = ("PLAN", "PARAMS", "BUILD", "VERIFY", "DELIVER")


class _Refused(Exception):
    """An edit that must be rolled back; the message is model-facing."""


def _clean(exc: Exception) -> str:
    text = str(exc)
    return text[len("Params: "):] if text.startswith("Params: ") else text


def params_factory(doc: Any = None, sheet: str = _params.SHEET_NAME) -> "_params.Params":
    """The ``Params`` of ``doc`` (default: the active document), created with its header when missing (the ``P`` helper of scripts)."""
    if doc is None:
        import FreeCAD as App

        doc = App.ActiveDocument
        if doc is None:
            raise _params.ParamsError("Params: there is no active document; create one first or pass the document: P(doc)")
    return _params.ensure_sheet(doc, sheet)


# --------------------------------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------------------------------

def render_table(rows: "list[dict]", sheet: str, cap: int = _MAX_ROWS) -> str:
    """The parameter table as aligned text (``name value unit source bound``; a formula shows in its own column)."""
    if not rows:
        return f"Sheet '{sheet}' has no parameters yet."
    bound_total = sum(1 for r in rows if r.get("bound"))
    head = f"PARAMETERS (sheet '{sheet}', {len(rows)} rows, {bound_total} bound to objects)"
    width = max(len(str(r["name"])) for r in rows[:cap]) + 2
    lines = [head, f"{'name':<{width}}{'value':<11}{'unit':<6}{'source':<18}bound  formula"]
    for r in rows[:cap]:
        value = "?" if r.get("value") is None else base.fmt(r["value"])
        mark = "" if not r.get("needs_rebuild") or r.get("bound") else " (rebuild: set_parameter)"
        lines.append(f"{r['name']:<{width}}{value:<11}{(r.get('unit') or '-'):<6}{(r.get('source') or '-'):<18}"
                     f"{r.get('bound', 0)!s:<7}{r.get('expr') or ''}{mark}")
    if len(rows) > cap:
        lines.append(f"+{len(rows) - cap} more")
    return "\n".join(lines)


def _row(rows: "list[dict]", name: str) -> "dict | None":
    return next((r for r in rows if r["name"] == name), None)


# --------------------------------------------------------------------------------------------------------------------
# geometry state (what changed after an edit)
# --------------------------------------------------------------------------------------------------------------------

def _geometry_state(doc: Any) -> "dict[str, tuple]":
    """``{object Name: (volume, bbox)}`` of every geometry object of ``doc`` (GUI thread)."""
    from .cad import health

    out: dict[str, tuple] = {}
    for obj in doc.Objects:
        try:
            if not health.is_geometry(obj):
                continue
            shape = health.global_shape(obj)
            if shape is None or shape.isNull():
                continue
            bb = shape.BoundBox
            out[obj.Name] = (round(float(shape.Volume), 3), tuple(round(v, 3) for v in (bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax)))
        except Exception:  # noqa: BLE001, S112 - an object without a usable shape is not part of the comparison
            continue
    return out


def _invalid_names(doc: Any) -> "set[str]":
    return {o.Name for o in doc.Objects if "Invalid" in (getattr(o, "State", None) or [])}


def _changed_text(doc: Any, before: "dict[str, tuple]", after: "dict[str, tuple]", kit_id: str) -> "tuple[str, list[str]]":
    """``("changed: Shell, Lid", [names])``: the kit tips by label when the kit is built, else the object names (at most 6)."""
    names = [n for n in after if before.get(n) != after[n]] + [n for n in before if n not in after]
    if not names:
        return "nothing changed", []
    shown: list[str] = []
    try:
        from .kit import geometry as G

        tips = {role: G.tip(doc, role, kit_id) for role in ("shell", "lid")}
        tip_names = {t.Name: t.Label for t in tips.values() if t is not None}
    except Exception:  # noqa: BLE001
        tip_names = {}
    for n in names:
        if n in tip_names:
            shown.append(tip_names[n])
    if shown:
        return "changed: " + ", ".join(sorted(shown, key=lambda x: (x != "Shell", x))), names
    shown = names[:6]
    return "changed: " + ", ".join(shown) + (f" (+{len(names) - 6} more)" if len(names) > 6 else ""), names


def _tip_sizes(doc: Any, kit_id: str) -> str:
    try:
        from .kit import geometry as G
    except Exception:  # noqa: BLE001
        return ""
    out = []
    for role, label in (("shell", "Shell"), ("lid", "Lid")):
        tip = G.tip(doc, role, kit_id)
        if tip is None:
            continue
        bb = G.shape_facts(tip).get("bbox")
        if bb:
            out.append(f"{label} {base.fmt(bb[3] - bb[0])} x {base.fmt(bb[4] - bb[1])} x {base.fmt(bb[5] - bb[2])} mm")
    return ", ".join(out)


def _effective_errors(doc: Any, spec_dict: "dict | None") -> "list[str]":
    """Validation errors of the spec the sheet currently produces (``[]`` without a spec)."""
    if not spec_dict:
        return []
    from .kit import build

    board = _spec.load_board(str(spec_dict.get("board") or "rpi4b"))
    eff = build.spec_from_sheet(doc, spec_dict)
    return list(base.spec_errors(_spec.to_dict(eff), board)[1])


def _rebuild_dependents(doc: Any, spec_dict: "dict | None", names: "list[str]", ref_status: str) -> "tuple[list, str]":
    """Rebuild the BUILT stages that depend on the parameters in ``names``. ``(results, note)``; ``note`` explains a skip.

    Two kinds of parameter need it: count / shape parameters (``STAGE_OF_PARAM``, not expression-bound) and structural ones
    (``STRUCTURAL_STAGE_OF_PARAM``: bound, but they decide which notches / merged openings exist or feed numbers frozen at build
    time, so only a rebuild equals a fresh build). ``pcb_t`` additionally rebuilds the nominal reference, which nothing is built on.
    """
    wanted = {n: _params.STAGE_OF_PARAM.get(n) or _params.STRUCTURAL_STAGE_OF_PARAM.get(n) for n in names}
    stages = sorted({s for s in wanted.values() if s}, key=_spec.STAGES.index)
    if not stages:
        return [], ""
    if not spec_dict:
        return [], ("the stages that depend on it were not rebuilt because no enclosure spec is stored: run "
                    f"build_enclosure(stage='{stages[0]}', rebuild=true) after setting a spec")
    from .kit import build

    sp = _spec.from_dict(spec_dict)
    built = build.built_stages(doc, sp.kit_id)
    board = _spec.load_board(sp.board)
    results: list = []
    if "pcb_t" in names and "reference" in built:
        results += build.rebuild_built(doc, sp, "reference", board, reference_status=ref_status)
        if results and not results[-1].ok:
            return results, ""
    first = next((s for s in stages if s in built and s not in ("params", "reference")), "")
    if first:
        results += build.rebuild_built(doc, sp, first, board, reference_status=ref_status)
    return results, ""


def _stages_current(doc: Any, spec_dict: "dict | None") -> bool:
    """True when no built kit stage is stale against ``spec_dict`` (so a spec sync may re-stamp the hashes honestly)."""
    if not spec_dict:
        return False
    try:
        from .kit import build

        return not any(r["state"] == "stale" for r in build.stage_states(doc, _spec.from_dict(spec_dict)))
    except Exception:  # noqa: BLE001
        return False


def _sync_spec(doc: Any, spec_dict: "dict | None", names: "list[str]", was_current: bool) -> "dict | None":
    """The declared spec with every design field in ``names`` set from the sheet (None when nothing changed or there is no spec).

    The sheet is the truth after an edit; the stored spec must follow, or the next stale push (``params`` stage) would write the
    old spec value back over the user's edit. When every stage was current before the edit the stage hashes are re-stamped for the
    new spec, so the model does not look stale (the geometry already follows the sheet: bound parameters through expressions,
    count parameters through the rebuild)."""
    design = [n for n in names if n in _spec.DESIGN_PARAMS]
    if not spec_dict or not design:
        return None
    from .kit import build

    values = _params.Params(doc).values()
    new = copy.deepcopy(spec_dict)
    for name in design:
        if name not in values:
            continue
        path = _spec.DESIGN_PARAMS[name][0]
        node = new
        parts = path.split(".")
        for key in parts[:-1]:
            node = node[key]
        x = values[name]
        node[parts[-1]] = int(round(x)) if path in build._INT_PATHS else (int(x) if path.endswith("_deg") and float(x).is_integer() else float(x))
    new = _spec.to_dict(_spec.from_dict(new))
    if new == _spec.to_dict(_spec.from_dict(spec_dict)):
        return None
    if was_current:
        build.restamp_hashes(doc, _spec.from_dict(new))
    return new


def _fit_summary(doc: Any, spec_dict: "dict | None", ref_status: str) -> str:
    """One line (plus the FAIL / WARN rows) of the kit fit rows after an edit; ``""`` without a spec or a built shell."""
    if not spec_dict:
        return ""
    try:
        from .kit import checks

        rows = checks.fit_rows(doc, spec_dict, ref_status) or []
    except Exception:  # noqa: BLE001 - a recheck must never fail the edit
        return ""
    counts = {s: sum(1 for r in rows if r.get("status") == s) for s in ("PASS", "WARN", "FAIL", "SKIP")}
    if not any(counts.values()):
        return ""
    line = "KIT FIT: " + ", ".join(f"{n} {s}" for s, n in counts.items())
    bad = [f"{r['status']} {r['id']}: {r.get('detail') or r.get('metric')}" for r in rows if r.get("status") in ("FAIL", "WARN")]
    return line + ("\n" + "\n".join(bad[:4]) if bad else "")


# --------------------------------------------------------------------------------------------------------------------
# set_parameters
# --------------------------------------------------------------------------------------------------------------------

def _validate_items(items: Any) -> "tuple[list[dict], list[str]]":
    problems: list[str] = []
    out: list[dict] = []
    if not isinstance(items, list) or not items:
        return [], ["'params': must be a non-empty list of {name, value, source}"]
    if len(items) > _MAX_ITEMS:
        problems.append(f"'params': {len(items)} rows is too many in one call; send at most {_MAX_ITEMS}")
    seen: set[str] = set()
    for i, it in enumerate(items[:_MAX_ITEMS]):
        where = f"params[{i}]"
        if not isinstance(it, dict):
            problems.append(f"{where}: must be an object {{name, value, source}}")
            continue
        name = it.get("name")
        if not isinstance(name, str) or not name.strip():
            problems.append(f"{where}: 'name' is missing")
            continue
        name = name.strip()
        ok, reason = _params.valid_alias(name)
        if not ok:
            problems.append(f"{where}: {reason}")
        if name in seen:
            problems.append(f"{where}: '{name}' appears twice")
        seen.add(name)
        value = it.get("value")
        if value is None or isinstance(value, bool) or not isinstance(value, (str, int, float)) or (isinstance(value, str) and not value.strip()):
            problems.append(f"'{name}': 'value' is missing; give a number, a string with a unit like '2.4 mm' or a formula starting with '='")
        source = it.get("source")
        if source not in _params.SOURCES:
            problems.append(f"'{name}': source {source!r} is not valid; use one of {', '.join(_params.SOURCES)}")
        unit = it.get("unit")
        if unit is not None and unit not in _params.UNITS:
            problems.append(f"'{name}': unit {unit!r} is not supported; use 'mm', 'deg' or '' (a count)")
        out.append({"name": name, "value": value, "unit": unit, "source": source, "note": str(it.get("note") or "")})
    return out, problems


@tool(
    "set_parameters",
    "Create or update rows of the Params spreadsheet (the single source of truth for every driving dimension). Each row has a name, a "
    "value (number, a string with a unit like '2.4 mm', or a formula starting with '='), a unit, a source and a note. Returns the table.",
    {"type": "object", "properties": {
        "doc_name": {"type": "string"},
        "params": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string", "description": "lower_snake_case, 3-32 chars, e.g. wall_t"},
            "value": {"type": "string", "description": "A number ('2.4'), a number with unit ('2.4 mm') or a formula ('=wall_t*2'). "
                                                       "Plain JSON numbers are accepted too."},
            "unit": {"type": "string", "description": "mm, deg or empty for counts"},
            "source": {"type": "string", "enum": list(_params.SOURCES)},
            "note": {"type": "string"}}, "required": ["name", "value", "source"]}},
        "sheet": {"type": "string", "default": "Params"}},
     "required": ["doc_name", "params"]},
    category="params", mutating=True, geometry=True, phases=_PHASES_SET,
)
def _set_parameters(doc_name: str, params: list, sheet: str = _params.SHEET_NAME):
    """Upsert the rows (one transaction; refused as a whole on any problem) and rebuild the kit stages that depend on count parameters."""
    ctx = current_context()
    items, problems = _validate_items(params)
    sheet_name = str(sheet or _params.SHEET_NAME).strip() or _params.SHEET_NAME
    if problems:
        raise ToolError("set_parameters: " + "; ".join(problems[:8]) + (f"; and {len(problems) - 8} more" if len(problems) > 8 else "") + ".")
    actual = base.require_doc(doc_name, "set_parameters")
    spec_dict = base.load_spec(ctx, actual)
    owner = base.spec_owner(ctx)
    ref_status = base.reference_status(ctx)

    def task() -> dict:
        from .kit import build

        doc = base.open_doc(actual)
        if doc is None:
            return {"error": f"document '{actual}' is not open"}
        spec_here = spec_dict if base.spec_applies(doc, spec_dict, owner) else None    # the chat's spec may belong to another document
        before_state = _geometry_state(doc)
        invalid_before = _invalid_names(doc)
        was_current = _stages_current(doc, spec_here)
        try:
            with build.atomic(doc, "Oscar set_parameters"):
                table = _params.ensure_sheet(doc, sheet_name)
                for it in items:
                    table.set(it["name"], it["value"], unit=it["unit"], source=it["source"], note=it["note"])
                doc.recompute()
                design = [it["name"] for it in items if it["name"] in _spec.DESIGN_PARAMS]
                if design and sheet_name == _params.SHEET_NAME:
                    errs = _effective_errors(doc, spec_here)
                    if errs:
                        raise _Refused("the values break the enclosure spec rules:\n" + "\n".join(f"- {e}" for e in errs[:6]))
                broke = sorted(_invalid_names(doc) - invalid_before)
                if broke:
                    raise _Refused(f"the values leave {', '.join(broke[:4])} invalid (" + "; ".join(
                        f"{n}: {doc.getObject(n).getStatusString()}" for n in broke[:2]) + "); choose values that rebuild")
        except _Refused as exc:
            return {"error": f"{exc}; nothing was changed"}
        except _params.ParamsError as exc:
            return {"error": f"{_clean(exc)}; nothing was changed"}
        results, skip_note = ([], "")
        if sheet_name == _params.SHEET_NAME:
            results, skip_note = _rebuild_dependents(doc, spec_here, [it["name"] for it in items], ref_status)
        failed = [r for r in results if not r.ok]
        if failed:
            return {"error": f"the dependent stage could not be rebuilt ({failed[0].message}); the sheet now holds the new values, "
                             "fix the values or set them back"}
        table = _params.Params(doc, sheet_name)
        after_state = _geometry_state(doc)
        kit_id = str((spec_here or {}).get("kit_id") or "enc1")
        changed, _names = _changed_text(doc, before_state, after_state, kit_id)
        new_spec = _sync_spec(doc, spec_here, [it["name"] for it in items], was_current) if sheet_name == _params.SHEET_NAME else None
        return {"rows": table.table(), "names": table.names(), "rebuilt": [r.stage for r in results], "skip": skip_note, "changed": changed,
                "spec": new_spec}

    out = base.gui_call("set_parameters", task)
    if out.get("error"):
        raise ToolError(f"set_parameters: {out['error']}")
    base.set_extra(ctx, param_names=out["names"])
    if out.get("spec"):
        base.store_spec(ctx, actual, out["spec"])
    lines = [f"Set {len(items)} parameter{'s' if len(items) != 1 else ''} in sheet '{sheet_name}': {base.limit_list([it['name'] for it in items], 8)}.",
             render_table(out["rows"], sheet_name)]
    if out["rebuilt"]:
        lines.append(f"Rebuilt stage{'s' if len(out['rebuilt']) != 1 else ''} {', '.join(out['rebuilt'])}; {out['changed']}.")
    if out["skip"]:
        lines.append(f"Note: {out['skip']}.")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
# get_parameters
# --------------------------------------------------------------------------------------------------------------------

@tool(
    "get_parameters",
    "Read the Params spreadsheet: every parameter with value, unit, formula, source, and how many object properties are bound to it.",
    {"type": "object", "properties": {"doc_name": {"type": "string"}, "sheet": {"type": "string", "default": "Params"}},
     "required": ["doc_name"]},
    category="params", phases=_PHASES_GET,
)
def _get_parameters(doc_name: str, sheet: str = _params.SHEET_NAME):
    """The table of the sheet (at most 40 rows); a missing sheet is reported as text with the way to create it."""
    ctx = current_context()
    sheet_name = str(sheet or _params.SHEET_NAME).strip() or _params.SHEET_NAME
    actual = base.require_doc(doc_name, "get_parameters")

    def task() -> dict:
        doc = base.open_doc(actual)
        if doc is None:
            return {"error": f"document '{actual}' is not open"}
        try:
            table = _params.Params(doc, sheet_name)
        except _params.ParamsError as exc:
            return {"missing": _clean(exc)}
        return {"rows": table.table(), "names": table.names()}

    out = base.gui_call("get_parameters", task)
    if out.get("error"):
        raise ToolError(f"get_parameters: {out['error']}")
    if out.get("missing"):
        base.set_extra(ctx, param_names=[])
        return (f"Document '{actual}' has no sheet '{sheet_name}' yet. Create it with set_parameters(doc_name='{actual}', params=[...]) "
                "or build_enclosure(stage='params').")
    base.set_extra(ctx, param_names=out["names"])
    return render_table(out["rows"], sheet_name)


# --------------------------------------------------------------------------------------------------------------------
# bind_expression
# --------------------------------------------------------------------------------------------------------------------

def _find_object(doc: Any, name: str) -> Any:
    obj = doc.getObject(name)
    if obj is not None:
        return obj
    by_label = doc.getObjectsByLabel(name)
    return by_label[0] if by_label else None


def _read_property(obj: Any, prop: str) -> str:
    try:
        value: Any = obj
        for part in prop.split("."):
            value = getattr(value, part)
        return str(getattr(value, "UserString", value))
    except Exception:  # noqa: BLE001
        return "?"


@tool(
    "bind_expression",
    "Bind an object property to a spreadsheet expression so it follows the parameter, for example Length = Params.wall_t * 2 or "
    "Placement.Base.z = Params.lid_lift. The expression is validated by a recompute; on error nothing is changed.",
    {"type": "object", "properties": {
        "doc_name": {"type": "string"}, "object": {"type": "string"},
        "property": {"type": "string", "description": "Length, Radius, Placement.Base.z, ..."},
        "expression": {"type": "string", "description": "e.g. Params.wall_t*2"}},
     "required": ["doc_name", "object", "property", "expression"]},
    category="params", mutating=True, geometry=True, phases=_PHASES_SET,
)
def _bind_expression(doc_name: str, object: str, property: str, expression: str):  # noqa: A002
    """Bind ``object.property`` to ``expression`` (bare parameter names get the ``Params.`` prefix); a recompute that fails undoes it."""
    ctx = current_context()
    obj_name, prop, expr = str(object or "").strip(), str(property or "").strip(), str(expression or "").strip()
    problems = []
    if not obj_name:
        problems.append("'object' is empty; name the object (see get_objects)")
    if not prop:
        problems.append("'property' is empty; for example 'Length' or 'Placement.Base.z'")
    if not expr.lstrip("= "):
        problems.append("'expression' is empty; for example 'Params.wall_t*2' or 'wall_t*2'")
    if problems:
        raise ToolError("bind_expression: " + "; ".join(problems) + ".")
    actual = base.require_doc(doc_name, "bind_expression")

    def task() -> dict:
        from .kit import build

        doc = base.open_doc(actual)
        if doc is None:
            return {"error": f"document '{actual}' is not open"}
        obj = _find_object(doc, obj_name)
        if obj is None:
            pool = [o.Name for o in doc.Objects] + [o.Label for o in doc.Objects]
            close = difflib.get_close_matches(obj_name, pool, n=3, cutoff=0.5)
            return {"error": f"'{obj_name}' is not an object of '{actual}'" + (f" (similar: {', '.join(dict.fromkeys(close))})" if close else "")
                    + "; list objects with get_objects"}
        top = prop.split(".")[0]
        props = list(getattr(obj, "PropertiesList", ()) or ())
        if top not in props:
            close = difflib.get_close_matches(top, props, n=4, cutoff=0.5)
            return {"error": f"'{obj.Name}' has no property '{top}'" + (f" (similar: {', '.join(close)})" if close else "")
                    + f"; it has {', '.join(props[:12])}{' ...' if len(props) > 12 else ''}"}
        try:
            table = _params.Params(doc)
        except _params.ParamsError as exc:
            return {"error": f"{_clean(exc)}; create it with set_parameters first"}
        old_value = _read_property(obj, prop)
        invalid_before = _invalid_names(doc)
        try:
            with build.atomic(doc, "Oscar bind_expression"):
                table.bind(obj, prop, expr)
                doc.recompute()
                broke = sorted(_invalid_names(doc) - invalid_before)
                if broke:
                    detail = doc.getObject(broke[0]).getStatusString()
                    raise _Refused(f"the recompute failed on {', '.join(broke[:3])} ({detail})")
        except _Refused as exc:
            return {"error": f"{exc}; the binding was not applied (nothing was changed)"}
        except _params.ParamsError as exc:
            return {"error": f"{_clean(exc)}; nothing was changed"}
        final = next((str(e[1]) for e in (getattr(obj, "ExpressionEngine", None) or []) if str(e[0]) == prop), expr)
        return {"object": obj.Name, "expr": final, "old": old_value, "new": _read_property(obj, prop), "names": table.names()}

    out = base.gui_call("bind_expression", task)
    if out.get("error"):
        raise ToolError(f"bind_expression: {out['error']}")
    base.set_extra(ctx, param_names=out["names"])
    return (f"Bound {out['object']}.{prop} = {out['expr']} (value now {out['new']}, was {out['old']}). "
            "It follows the parameter from now on.")


# --------------------------------------------------------------------------------------------------------------------
# set_parameter
# --------------------------------------------------------------------------------------------------------------------

def _value_text(value: Any, unit: str) -> str:
    return f"{base.fmt(value)} {unit}".strip() if value is not None else "?"


@tool(
    "set_parameter",
    "Change ONE parameter value, recompute the model and report what changed and whether it is still valid. This is how you tune a "
    "dimension (for example wall_t to 2.4) without rebuilding by hand.",
    {"type": "object", "properties": {
        "doc_name": {"type": "string"}, "name": {"type": "string"},
        "value": {"type": "string", "description": "A number ('2.4'), a number with unit ('2.4 mm') or a formula ('=wall_t*2'). "
                                                   "Plain JSON numbers are accepted too."},
        "recompute": {"type": "boolean", "default": True},
        "recheck": {"type": "boolean", "default": True, "description": "Re-run the health check and fit checks after the change"}},
     "required": ["doc_name", "name", "value"]},
    category="params", mutating=True, geometry=True, phases=("BUILD", "VERIFY", "DELIVER"),
)
def _set_parameter(doc_name: str, name: str, value: Any, recompute: bool = True, recheck: bool = True):
    """Edit ONE cell, recompute, rebuild count-parameter stages, validate the effective spec, report what changed (module docstring)."""
    ctx = current_context()
    param = str(name or "").strip()
    problems = []
    if not param:
        problems.append("'name' is empty; use a parameter name from get_parameters")
    if value is None or isinstance(value, bool) or not isinstance(value, (str, int, float)) or (isinstance(value, str) and not value.strip()):
        problems.append("'value' is missing; give a number ('2.4'), a number with unit ('2.4 mm') or a formula starting with '='")
    if problems:
        raise ToolError("set_parameter: " + "; ".join(problems) + ".")
    actual = base.require_doc(doc_name, "set_parameter")
    spec_dict = base.load_spec(ctx, actual)
    owner = base.spec_owner(ctx)
    ref_status = base.reference_status(ctx)
    is_formula = isinstance(value, str) and value.strip().startswith("=")
    do_recompute, do_recheck = bool(recompute), bool(recheck)

    def task() -> dict:
        from .kit import build

        doc = base.open_doc(actual)
        if doc is None:
            return {"error": f"document '{actual}' is not open"}
        try:
            table = _params.Params(doc)
        except _params.ParamsError:
            return {"error": f"document '{actual}' has no 'Params' sheet; create it with build_enclosure(stage='params') or set_parameters"}
        rows = table.table()
        row = _row(rows, param)
        if row is None:
            names = [r["name"] for r in rows]
            close = difflib.get_close_matches(param, names, n=3, cutoff=0.6)
            return {"error": f"'{param}' is not a parameter in sheet 'Params'. Parameters: {base.limit_list(names, 12)}"
                             + (f" (similar: {', '.join(close)})" if close else "") + "."}
        if row["expr"] and not is_formula:
            return {"error": f"'{param}' is derived ({row['expr']}), so a number would be overwritten at the next rebuild; change its "
                             "inputs with set_parameter, or pass a formula starting with '=' to replace the derivation"}
        spec_here = spec_dict if base.spec_applies(doc, spec_dict, owner) else None    # the chat's spec may belong to another document
        old_contents = table.sheet.getContents(row["cell"]) or ""
        old = {"value": row["value"], "unit": row["unit"], "source": row["source"], "note": row["note"], "contents": old_contents}
        before_state = _geometry_state(doc)
        invalid_before = _invalid_names(doc)
        was_current = _stages_current(doc, spec_here)
        recomputed = 0
        try:
            with build.atomic(doc, f"Oscar set_parameter {param}"):
                table.set(param, value, source="user_requirement", note=old["note"])
                if do_recompute:
                    recomputed = int(doc.recompute() or 0)
                    if param in _spec.DESIGN_PARAMS:
                        errs = _effective_errors(doc, spec_here)
                        if errs:
                            raise _Refused("the value breaks the enclosure spec rules:\n" + "\n".join(f"- {e}" for e in errs[:5]))
                    broke = sorted(_invalid_names(doc) - invalid_before)
                    if broke:
                        raise _Refused(f"the value leaves {', '.join(broke[:4])} invalid ({doc.getObject(broke[0]).getStatusString()})")
        except _Refused as exc:
            return {"error": f"{exc}; '{param}' stays {_value_text(old['value'], old['unit'])}"}
        except _params.ParamsError as exc:
            return {"error": f"{_clean(exc)}; '{param}' stays {_value_text(old['value'], old['unit'])}"}
        results, skip_note = ([], "")
        if do_recompute:
            results, skip_note = _rebuild_dependents(doc, spec_here, [param], ref_status)
            failed = [r for r in results if not r.ok]
            if failed:
                table.set(param, old["contents"], unit=old["unit"], source=old["source"] or "assumption", note=old["note"])
                doc.recompute()
                _rebuild_dependents(doc, spec_here, [param], ref_status)
                return {"error": f"rebuilding '{failed[0].stage}' with {param} = {value} failed ({failed[0].message}); "
                                 f"'{param}' was set back to {_value_text(old['value'], old['unit'])}"}
        table = _params.Params(doc)
        new_row = _row(table.table(), param) or row
        after_state = _geometry_state(doc)
        kit_id = str((spec_here or {}).get("kit_id") or "enc1")
        changed, _names = _changed_text(doc, before_state, after_state, kit_id)
        new_spec = _sync_spec(doc, spec_here, [param], was_current) if do_recompute else None
        out = {"old": old, "new": new_row, "spec": new_spec, "recomputed": recomputed, "changed": changed, "rebuilt": [r.stage for r in results],
               "skip": skip_note, "sizes": _tip_sizes(doc, kit_id), "names": table.names(), "recomputed_flag": do_recompute}
        if do_recheck and do_recompute:
            out["fit"] = _fit_summary(doc, spec_here, ref_status)
        return out

    out = base.gui_call("set_parameter", task)
    if out.get("error"):
        raise ToolError(f"set_parameter: {out['error']}")
    base.set_extra(ctx, param_names=out["names"])
    if out.get("spec"):
        base.store_spec(ctx, actual, out["spec"])
    old, new = out["old"], out["new"]
    lines = [f"{param}: {_value_text(old['value'], old['unit'])} -> {_value_text(new['value'], new['unit'])} (source: {new['source'] or 'user_requirement'})."]
    if out["recomputed_flag"]:
        lines[0] += f" Recomputed {out['recomputed']} object{'s' if out['recomputed'] != 1 else ''}; {out['changed']}."
        if out["rebuilt"]:
            why = ("count parameters are not expression-bound" if param in _params.STAGE_OF_PARAM
                   else "this value decides which openings, notches and hook numbers exist, so the model is rebuilt, not only recomputed")
            lines.append(f"Rebuilt stage{'s' if len(out['rebuilt']) != 1 else ''} {', '.join(out['rebuilt'])} ({why}).")
        if out["sizes"]:
            lines.append(f"Now: {out['sizes']}.")
    else:
        lines.append("Not recomputed (recompute=false): the model still shows the old value; set_parameter again with recompute=true to apply it.")
    if out["skip"]:
        lines.append(f"Note: {out['skip']}.")
    if out.get("fit"):
        lines.append(out["fit"])
    return "\n".join(lines)
