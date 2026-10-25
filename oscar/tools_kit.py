"""Enclosure tools: ``enclosure_spec``, ``build_enclosure``, ``add_cutout``, ``add_vents``; the four
parameter tools (``set_parameters``, ``get_parameters``, ``bind_expression``, ``set_parameter``) live in ``tools_kit_params`` and
are registered by importing it at the end of this module (``_TOOL_MODULES`` lists only ``tools_kit``).

The model never improvises geometry for an enclosure: it chooses VALUES with ``enclosure_spec`` and calls ``build_enclosure`` once per
stage (``params, reference, shell, standoffs, ports, lid, vents, hooks, assembly``) so the user watches the part grow. The kit
(``oscar.kit``) does the CAD; this module is the thin, defensive tool layer:

* pure argument validation first (all problems in ONE error), then the document, always on the GUI thread through
  ``tools_kit_base.gui_call`` with the context captured before dispatching and no lock held;
* the spec lives in the bound Project (``spec.json``, ``project.save_spec``); a bare call without a Project keeps it in a
  process cache (``tools_kit_base``);
* ``build_enclosure`` applies the board defaults itself when no spec exists (reported, recorded as an ``assumption`` note) and sets
  ``ctx.state["title_hint"]`` / ``["args_override"]`` (the resolved stage, never ``"next"``) / ``["extra"]`` (``spec_ok``,
  ``param_names``) for ``run_tool``;
* ``enclosure_spec``: after two failing ``set`` / ``validate`` calls in one turn (a private counter keyed by ``ctx.turn_id``) the
  THIRD failing call applies the defaults plus the valid subset of the patch and says ``SPEC DEFAULTS APPLIED``, so a model that cannot
  get a spec right is never blocked;
* ``add_cutout`` / ``add_vents`` write the spec (``cutouts`` / ``vents``) and rebuild exactly the stages that depend on the change and
  were already built, so the spec and the model never disagree; on a failed rebuild the previous spec is restored.

Importing this module also registers the ``kit`` and ``P`` helpers for ``execute_code`` scripts (``scripting.register_helpers``;
this is the ONLY caller, 7.5).
"""

import json
import re
import threading
from typing import Any

from . import tools_kit_base as base
from .kit import spec as _spec
from .tools import ToolError, current_context, tool

_FACES = list(_spec.FACES)
_STAGE_CHOICES = ["next", *_spec.STAGES]
_ACTIONS = ("get", "set", "validate", "defaults")
_VIEWS = ["Isometric", "Front", "Top", "Right", "Back", "Left", "Bottom", "Dimetric", "Trimetric"]
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,23}$")
_SPEC_PHASES = ("PLAN", "PARAMS", "BUILD")
_FAIL_LIMIT = 2            # failing set / validate calls per turn before the next one applies the defaults

_FAILS: "dict[str, int]" = {}
_FAILS_LOCK = threading.Lock()
_MAX_TURNS = 32


# --------------------------------------------------------------------------------------------------------------------
# per-turn failure counter of enclosure_spec
# --------------------------------------------------------------------------------------------------------------------

def _fails(turn: str) -> int:
    with _FAILS_LOCK:
        return _FAILS.get(turn, 0)


def _bump(turn: str) -> int:
    """Count one failing set / validate call of ``turn``. A call outside any turn (replay, the bridge: ``turn_id == ""``) is never
    counted, so unrelated failing calls can never add up to the silent "defaults applied" rescue, which is a per-turn rule."""
    if not turn:
        return 1
    with _FAILS_LOCK:
        if turn not in _FAILS and len(_FAILS) >= _MAX_TURNS:
            _FAILS.pop(next(iter(_FAILS)))
        _FAILS[turn] = _FAILS.get(turn, 0) + 1
        return _FAILS[turn]


def _clear(turn: str) -> None:
    with _FAILS_LOCK:
        _FAILS.pop(turn, None)


def reset_failure_counters() -> None:
    """Forget every per-turn failure count (tests)."""
    with _FAILS_LOCK:
        _FAILS.clear()


# --------------------------------------------------------------------------------------------------------------------
# enclosure_spec
# --------------------------------------------------------------------------------------------------------------------

def _problems_text(errors: "list[str]", cap: int = 8) -> str:
    shown = "\n".join(f"- {e}" for e in errors[:cap])
    return shown + (f"\n- and {len(errors) - cap} more" if len(errors) > cap else "")


def _param_names_for(doc_name: str) -> "list[str] | None":
    """The ``Params`` names of the open document ``doc_name`` (None when no such document is open)."""
    if not doc_name:
        return None

    def task() -> Any:
        doc = base.open_doc(doc_name)
        return None if doc is None else base.param_names(doc)

    try:
        return base.gui_call("enclosure_spec", task, 30.0)
    except ToolError:
        return None


def _spec_title(args: dict) -> str:
    action = str((args or {}).get("action") or "get")
    return {"get": "Read the enclosure spec", "set": "Set the enclosure spec", "validate": "Validate the enclosure spec",
            "defaults": "Apply the enclosure defaults"}.get(action, "Enclosure spec")


def _board_or_error(board_id: str, tool_name: str) -> dict:
    return base.load_board(board_id, tool_name)


def _apply_defaults(ctx: Any, doc: str, board_id: str, patch: dict, why: str) -> str:
    """Store the defaults of ``board_id`` plus the valid subset of ``patch``; return the model-facing text."""
    board = _board_or_error(board_id, "enclosure_spec")
    default = base.default_spec_dict(board_id)
    final, kept, dropped = base.valid_subset(default, patch or {}, board)
    note = base.store_spec(ctx, doc, final)
    summary = base.spec_summary(final, board)
    lines = [f"SPEC DEFAULTS APPLIED: {why}", f"Spec: {summary}."]
    if kept:
        lines.append("Kept from your values: " + base.limit_list(kept, 10) + ".")
    if dropped:
        lines.append("Dropped (invalid): " + "; ".join(dropped[:6]) + ("; ..." if len(dropped) > 6 else ""))
    adapted = _adapted(board_id, final.get("material", "ABS"))
    if adapted:
        lines.append("Sized for this board: " + adapted + ".")
    record = base.record_assumption(ctx, f"enclosure spec: the defaults of board '{board_id}' were applied" + (f" ({why})" if why else "")
                                    + (f"; kept {base.limit_list(kept, 6)}" if kept else ""))
    lines.append("Recorded as an assumption." if record else "(no project is bound, so nothing was recorded)")
    if note:
        lines.append(note)
    lines.append("spec_ok: True. Next: build_enclosure(doc_name, stage='next').")
    base.set_extra(ctx, spec_ok=True)
    return "\n".join(lines)


def _adapted(board_id: str, material: str) -> str:
    try:
        from .kit import _defaults

        changed = _defaults.adapted_fields(board_id, material)
    except Exception:  # noqa: BLE001
        return ""
    return ", ".join(f"{path} {base.fmt(old) if not isinstance(old, (list, dict)) else old} -> {base.fmt(new) if not isinstance(new, (list, dict)) else new}"
                     for path, (old, new) in list(changed.items())[:8])


@tool(
    "enclosure_spec",
    "Read, set or validate the enclosure specification (board, material, wall, clearance, lid style, vents, hooks, ports). Setting "
    "merges the values you give into the spec and validates the whole thing; the result lists errors (fix them) and warnings. The kit "
    "builds from this spec, so choose values here instead of improvising geometry. If two calls in a row come back with errors, the "
    "third call applies the board defaults for you and says so, so the build is never blocked by a spec you cannot get right.",
    {"type": "object", "properties": {
        "action": {"type": "string", "enum": list(_ACTIONS), "default": "get"},
        "spec": {"type": "object", "description": "Partial EnclosureSpec to merge when action=set"},
        "board": {"type": "string", "description": "Board id for action=defaults / set, e.g. 'rpi4b'"},
        "doc_name": {"type": "string"}}},
    category="plan", mutating=True, phases=_SPEC_PHASES, title=_spec_title,
)
def _enclosure_spec(action: str = "get", spec: "dict | None" = None, board: str = "", doc_name: str = ""):
    """The four actions of ``enclosure_spec`` (module docstring). Raises ``ToolError`` for everything the model can correct."""
    ctx = current_context()
    act = str(action or "get").strip().lower()
    problems = []
    if act not in _ACTIONS:
        problems.append(f"'action': {action!r} is not valid; use one of {', '.join(_ACTIONS)}")
    if spec is not None and not isinstance(spec, dict):
        problems.append(f"'spec': must be an object like {{\"wall_t\": 2.4}}, not {type(spec).__name__}")
    if board is not None and not isinstance(board, str):
        problems.append("'board': must be a board id string such as 'rpi4b'")
    if problems:
        raise ToolError("enclosure_spec: " + "; ".join(problems) + ".")
    patch = dict(spec) if isinstance(spec, dict) else {}
    if isinstance(patch.get("board"), str):
        patch["board"] = base.resolve_board(patch["board"])          # 'Raspberry Pi 4 Model B' -> 'rpi4b'
    board_id = base.resolve_board(board)
    doc = str(doc_name or "").strip()
    turn = str(getattr(ctx, "turn_id", "") or "")

    if act == "get":
        return _spec_get(ctx, doc, board_id)
    if act == "defaults":
        return _spec_defaults(ctx, doc, board_id, patch, turn)
    return _spec_set_or_validate(ctx, act, doc, board_id, patch, turn)


def _spec_get(ctx: Any, doc: str, board_id: str) -> str:
    stored = base.load_spec(ctx, doc)
    if not stored:
        guess = board_id or base.default_board_id(ctx, doc)
        return (f"No enclosure spec has been set yet. Board defaults for '{guess}' will be used: call enclosure_spec(action='defaults', "
                f"board='{guess}') to store them, or enclosure_spec(action='set', spec={{...}}) with your own values, or just "
                "build_enclosure (it applies the defaults itself).\nspec_ok: False")
    board = _board_or_error(str(stored.get("board") or "rpi4b"), "enclosure_spec")
    ok, errors, warnings = base.spec_errors(stored, board)
    base.set_extra(ctx, spec_ok=bool(ok))
    names = _param_names_for(doc)
    if names is not None:
        base.set_extra(ctx, param_names=names)
    lines = [f"Enclosure spec ({base.spec_summary(stored, board)}); " + ("valid." if ok else f"INVALID ({len(errors)} errors)."),
             json.dumps(stored, separators=(",", ":"), sort_keys=False)[:6000]]
    if errors:
        lines.append("Errors:\n" + _problems_text(errors))
    if warnings:
        lines.append("Warnings:\n" + _problems_text(warnings))
    return "\n".join(lines)


def _spec_defaults(ctx: Any, doc: str, board_id: str, patch: dict, turn: str) -> str:
    board_name = board_id or str((patch.get("board") if isinstance(patch.get("board"), str) else "") or "").strip().lower() \
        or str((base.load_spec(ctx, doc) or {}).get("board") or "") or base.default_board_id(ctx, doc)
    _board_or_error(board_name, "enclosure_spec")
    patch = {k: v for k, v in patch.items() if k != "board"}
    text = _apply_defaults(ctx, doc, board_name, patch, f"action='defaults' for board '{board_name}'")
    _clear(turn)
    names = _param_names_for(doc)
    if names is not None:
        base.set_extra(ctx, param_names=names)
    return text


def _spec_set_or_validate(ctx: Any, act: str, doc: str, board_id: str, patch: dict, turn: str) -> str:
    stored = base.load_spec(ctx, doc)
    if board_id:
        patch["board"] = board_id
    if stored:
        current, fresh = stored, False
    else:
        current, fresh = base.default_spec_dict(board_id or base.default_board_id(ctx, doc)), True
    candidate, errors, warnings = base.merge_validate(current, patch)
    ok = candidate is not None
    board_name = str((candidate or current).get("board") or "rpi4b")
    board = base.load_board(board_name, "enclosure_spec") if ok else None
    if not ok:
        count = _bump(turn)
        if count > _FAIL_LIMIT:
            fallback_board = board_name if board_name in base.known_boards() else base.default_board_id(ctx, doc)
            return _apply_defaults(ctx, doc, fallback_board, {k: v for k, v in patch.items() if k != "board"},
                                   f"the spec came back with errors {count} times in this turn")
        left = _FAIL_LIMIT + 1 - count
        raise ToolError("enclosure_spec: the spec is not valid; fix these and call again:\n" + _problems_text(errors)
                        + f"\n(after {left} more failing call{'s' if left != 1 else ''} in this turn the board defaults are applied for you)")
    _clear(turn)
    base.set_extra(ctx, spec_ok=True)
    names = _param_names_for(doc)
    if names is not None:
        base.set_extra(ctx, param_names=names)
    changes = base.diff(current, candidate)
    shown = "; ".join(f"{p} {base.fmt(o) if not isinstance(o, (list, dict)) else o} -> {base.fmt(n) if not isinstance(n, (list, dict)) else n}"
                      for p, o, n in changes[:8])
    if act == "validate":
        head = f"Spec valid: {base.spec_summary(candidate, board)}." + (" (nothing stored yet: the board defaults were checked)" if fresh and not patch else "")
        lines = [head]
        if patch and changes:
            lines.append(f"(checked with your values merged: {shown}; nothing was stored: use action='set' to keep them)")
        lines.append("spec_ok: True")
    else:
        note = base.store_spec(ctx, doc, candidate)
        lines = [f"Spec saved ({len(changes)} change{'s' if len(changes) != 1 else ''}"
                 + (f": {shown}{'; ...' if len(changes) > 8 else ''}" if changes else "") + ").",
                 f"Spec: {base.spec_summary(candidate, board)}."]
        if fresh:
            lines.append("(no earlier spec existed: your values were merged onto the defaults of the board)")
        if note:
            lines.append(note)
        lines.append("spec_ok: True. Next: build_enclosure(doc_name, stage='next').")
    if warnings:
        lines.append("Warnings:\n" + _problems_text(warnings))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
# build_enclosure
# --------------------------------------------------------------------------------------------------------------------

def _refresh_params(doc: Any, sp: Any, board: dict, ref_status: str) -> Any:
    """The ``params`` stage when it is built but its recorded hash is stale (a spec edit not yet pushed into the sheet), else None.

    Every stage reads the EFFECTIVE spec (the declared spec overlaid with the sheet), so a spec edit has no effect on geometry until
    the sheet followed; doing it here keeps ``build_enclosure(stage=...)``, ``add_cutout`` and ``add_vents`` honest."""
    from .kit import build

    recorded = build.built_stages(doc, sp.kit_id).get("params")
    if not recorded or recorded == "imported" or recorded == _spec.spec_hash(sp, "params"):
        return None
    return build.build_stage(doc, sp, "params", board=board, reference_status=ref_status)


def _build_title(args: dict) -> str:
    stage = str((args or {}).get("stage") or "next")
    return f"Build enclosure: {stage}"


def _tips_text(result: Any) -> str:
    solids, volumes = dict(getattr(result, "solids", {}) or {}), dict(getattr(result, "volumes", {}) or {})
    if not solids:
        return ""
    parts = []
    for label in solids:
        n = solids[label]
        parts.append(f"{n} solid{'s' if n != 1 else ''}, {base.fmt(round(volumes.get(label, 0.0), 1))} mm3")
    if len(solids) == 1:
        return f" ({parts[0]})"
    return " (" + "; ".join(f"{label} {part}" for label, part in zip(solids, parts)) + ")"


def _bbox_text(result: Any) -> str:
    tips = (getattr(result, "metrics", {}) or {}).get("tips") or {}
    out = []
    for label, info in tips.items():
        bb = (info or {}).get("bbox")
        if bb and len(bb) == 6:
            out.append(f"{label} {base.fmt(bb[3] - bb[0])} x {base.fmt(bb[4] - bb[1])} x {base.fmt(bb[5] - bb[2])} mm")
    return "Size: " + ", ".join(out) + "." if out else ""


@tool(
    "build_enclosure",
    "Build the next stage of the enclosure from the spec: params, reference, shell, standoffs, ports, lid, vents, hooks, assembly. "
    "Each call builds exactly ONE stage so the user watches it appear; call again until all 9 stages are built. If no spec has been "
    "set yet the board defaults are applied (and reported), so you can start building at once. Stages are idempotent: rebuilding a "
    "stage replaces its objects and the later ones (the params stage only adds missing rows).",
    {"type": "object", "properties": {
        "doc_name": {"type": "string"},
        "stage": {"type": "string", "enum": _STAGE_CHOICES, "default": "next"},
        "spec": {"type": "object", "description": "Optional partial spec to merge before building"},
        "rebuild": {"type": "boolean", "default": False, "description": "Rebuild the stage (and the later ones) even when it is up to date"},
        "include_screenshot": {"type": "boolean", "default": False},
        "view_name": {"type": "string", "enum": _VIEWS, "default": "Isometric"}},
     "required": ["doc_name"]},
    category="build", mutating=True, geometry=True, phases=("BUILD",),
)
def _build_enclosure(doc_name: str, stage: str = "next", spec: "dict | None" = None, rebuild: bool = False,
                     include_screenshot: bool = False, view_name: str = "Isometric"):
    """Build ONE stage (``stage='next'`` = the first missing or stale one). Raises ``ToolError`` for a bad argument, an invalid spec,
    a stage out of order or a failed build (the document is then unchanged)."""
    ctx = current_context()
    problems = []
    stage_name = str(stage or "next").strip().lower()
    if stage_name not in _STAGE_CHOICES:
        problems.append(f"'stage': {stage!r} is not a stage; use one of {', '.join(_STAGE_CHOICES)}")
    if spec is not None and not isinstance(spec, dict):
        problems.append(f"'spec': must be an object like {{\"wall_t\": 2.4}}, not {type(spec).__name__}")
    if problems:
        raise ToolError("build_enclosure: " + "; ".join(problems) + ".")
    actual = base.require_doc(doc_name, "build_enclosure")
    ctx.state["title_hint"] = _build_title({"stage": stage_name})

    current, note = base.ensure_spec(ctx, actual, "build_enclosure")
    if note:
        base.set_extra(ctx, spec_ok=True)
    lines = [note] if note else []
    if spec:
        patch = dict(spec)
        if isinstance(patch.get("board"), str):
            patch["board"] = base.resolve_board(patch["board"])
        merged, errors, _w = base.merge_validate(current, patch)
        if merged is None:
            raise ToolError("build_enclosure: the 'spec' patch makes the spec invalid; fix these:\n" + _problems_text(errors))
        if merged != current:
            lines.append(f"Spec updated: {'; '.join(f'{p} -> {base.fmt(n) if not isinstance(n, (list, dict)) else n}' for p, _o, n in base.diff(current, merged)[:6])}.")
            note = base.store_spec(ctx, actual, merged)
            if note:
                lines.append(note)
        current = merged
    board = base.load_board(str(current.get("board") or "rpi4b"), "build_enclosure")
    ok, errors, _w = base.spec_errors(current, board)
    if not ok:
        raise ToolError("build_enclosure: the stored spec is not valid; fix it with enclosure_spec(action='set', spec={...}):\n"
                        + _problems_text(errors))
    ref_status = base.reference_status(ctx)
    want_shot = bool(include_screenshot) and not ctx.state.get("lite")

    def task() -> dict:
        from .kit import build

        doc = base.open_doc(actual)
        if doc is None:
            return {"error": f"document '{actual}' is not open"}
        sp = _spec.from_dict(current)
        chosen = stage_name
        if chosen == "next":
            chosen = build.next_stage(doc, sp)
            if chosen is None:
                if rebuild:
                    return {"error": "stage='next' has nothing left to build; name the stage to rebuild, for example "
                                     "build_enclosure(stage='shell', rebuild=true)"}
                return {"stage": "", "done": True, "names": base.param_names(doc)}
        refreshed = None
        if chosen not in ("params", "reference"):
            refreshed = _refresh_params(doc, sp, board, ref_status)
            if refreshed is not None and not refreshed.ok:
                return {"error": f"the Params sheet could not be refreshed from the spec first: {refreshed.message}"}
        result = build.build_stage(doc, sp, chosen, board=board, rebuild=bool(rebuild), reference_status=ref_status)
        out = {"stage": chosen, "result": result, "names": base.param_names(doc)}
        if refreshed is not None and not refreshed.skipped:
            out["refreshed"] = list((refreshed.metrics or {}).get("updated") or [])
        return out

    out = base.gui_call("build_enclosure", task)
    if out.get("error"):
        raise ToolError(f"build_enclosure: {out['error']}")
    if out.get("names") is not None:
        base.set_extra(ctx, param_names=out["names"])
    base.set_extra(ctx, spec_ok=True)
    if out.get("done"):
        lines.append("All 9 stages are built and up to date. Next: validate_model, then check_fit.")
        return "\n".join(lines)
    chosen, result = out["stage"], out["result"]
    if out.get("refreshed"):
        lines.append("The Params rows followed the spec first: " + base.limit_list(out["refreshed"], 8) + ".")
    ctx.state["args_override"] = {"stage": chosen}
    ctx.state["title_hint"] = _build_title({"stage": chosen})
    if not result.ok:
        raise ToolError(f"build_enclosure: {result.message}")
    idx = _spec.STAGES.index(chosen) + 1
    if result.skipped:
        lines.append(f"Stage '{chosen}' ({idx} of 9): {result.message}.")
    else:
        names = list(result.objects)
        shown = ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")
        head = f"Stage '{chosen}' built ({idx} of 9)."
        lines.append(head + (f" Objects: {shown}{_tips_text(result)}." if names else " Nothing had to be created for this stage."))
        if result.removed:
            lines.append(f"Replaced {len(result.removed)} earlier object{'s' if len(result.removed) != 1 else ''} (the stage and the later ones were rebuilt).")
        size = _bbox_text(result)
        if size:
            lines.append(size)
    for w in list(result.warnings)[:5]:
        lines.append(f"Warning: {w}.")
    ns = (result.metrics or {}).get("next_stage")
    lines.append(f"Next stage: {ns}." if ns else "All 9 stages are built. Next: validate_model, then check_fit.")
    text = "\n".join(lines)
    return _with_shot(text, want_shot, view_name, actual)


def _with_shot(text: str, include: bool, view_name: str, doc_name: str):
    """``(text, image)`` through ``tools._shot`` (the shared screenshot tail)."""
    if not include:
        return text
    from . import tools

    return tools._shot(text, True, view_name if view_name in _VIEWS else "Isometric", doc_name)


# --------------------------------------------------------------------------------------------------------------------
# add_cutout / add_vents: a spec edit plus the rebuild of the stages that depend on it
# --------------------------------------------------------------------------------------------------------------------

def _apply_spec_edit(ctx: Any, doc: str, tool_name: str, before: dict, after: dict, first_stage: str) -> "tuple[list, str]":
    """Store ``after``; rebuild ``first_stage`` and the later BUILT stages. ``([StageResult], note)``; on a failed rebuild the previous
    spec is stored again, the model is rebuilt from it and a ``ToolError`` says what failed."""
    board = base.load_board(str(after.get("board") or "rpi4b"), tool_name)
    ref_status = base.reference_status(ctx)

    def rebuild(d: dict) -> "list | dict":
        from .kit import build

        document = base.open_doc(doc)
        if document is None:
            return {"error": f"document '{doc}' is not open"}
        sp = _spec.from_dict(d)
        refreshed = _refresh_params(document, sp, board, ref_status)
        if refreshed is not None and not refreshed.ok:
            return [refreshed]
        built = build.built_stages(document, sp.kit_id)
        if first_stage not in built:
            return []
        return build.rebuild_built(document, sp, first_stage, board, reference_status=ref_status)

    results = base.gui_call(tool_name, lambda: rebuild(after))
    if isinstance(results, dict):
        raise ToolError(f"{tool_name}: {results['error']}")
    failed = [r for r in results if not r.ok]
    if failed:
        base.gui_call(tool_name, lambda: rebuild(before))
        raise ToolError(f"{tool_name}: the change could not be built ({failed[0].message}); the previous model and spec were kept.")
    note = base.store_spec(ctx, doc, after)
    return results, note


def _face_extent(spec_dict: dict, board: dict, part: str, face: str) -> "tuple[float, float, float, float] | None":
    """``(u0, u1, v0, v1)`` of the outer face an opening must lie inside, or None when the extent is not known (a lid skirt)."""
    dv = _spec.derived(spec_dict, board)
    g = {k: dv[k]["value"] for k in ("cav_x0", "cav_x1", "cav_y0", "cav_y1", "z_floor_bot", "z_cavity_top")}
    wall = float(spec_dict["wall_t"])
    x0, x1, y0, y1 = g["cav_x0"] - wall, g["cav_x1"] + wall, g["cav_y0"] - wall, g["cav_y1"] + wall
    if part == "lid":
        if face != "top":
            return None
        grow = float(spec_dict["lid"]["fit"]) + float(spec_dict["lid"]["lip_t"])
        return x0 - grow, x1 + grow, y0 - grow, y1 + grow
    if face in ("-y", "+y"):
        return x0, x1, g["z_floor_bot"], g["z_cavity_top"]
    if face in ("-x", "+x"):
        return y0, y1, g["z_floor_bot"], g["z_cavity_top"]
    return x0, x1, y0, y1


def _wall_label(face: str) -> str:
    return "top face" if face == "top" else ("floor" if face == "bottom" else f"{face} wall")


def _stage_summary(results: list) -> str:
    if not results:
        return ""
    names = [r.stage for r in results]
    return f"Rebuilt stage{'s' if len(names) != 1 else ''} {', '.join(names)}."


@tool(
    "add_cutout",
    "Cut one rectangular opening (optionally with rounded corners) through a wall, the floor or the lid of the enclosure. Wall faces "
    "use u along the wall and v = height above the PCB bottom; top and bottom faces use u = x and v = y. Frame: origin at the PCB "
    "lower-left corner, +x along the long edge, +z up through the PCB. Idempotent per id.",
    {"type": "object", "properties": {
        "doc_name": {"type": "string"},
        "part": {"type": "string", "enum": list(_spec.PARTS)},
        "face": {"type": "string", "enum": _FACES},
        "u": {"type": "number", "description": "Centre, mm"}, "v": {"type": "number", "description": "Centre, mm"},
        "w": {"type": "number", "minimum": 0.1}, "h": {"type": "number", "minimum": 0.1},
        "r": {"type": "number", "default": 0, "minimum": 0, "description": "Corner radius"},
        "margin": {"type": "number", "default": 0, "minimum": 0, "description": "Added around w x h"},
        "id": {"type": "string", "description": "Tag; re-using it replaces that cutout"},
        "through": {"type": "boolean", "default": True}},
     "required": ["doc_name", "part", "face", "u", "v", "w", "h"]},
    category="build", mutating=True, geometry=True, phases=("BUILD", "VERIFY"),
)
def _add_cutout(doc_name: str, part: str, face: str, u: float, v: float, w: float, h: float, r: float = 0.0,
                margin: float = 0.0, id: str = "", through: bool = True):  # noqa: A002
    """Add or replace the cut-out ``id`` in the spec and rebuild the dependent stages (``ports`` for the shell, ``vents`` for the lid)."""
    ctx = current_context()
    part_name, face_name = str(part or "").strip().lower(), str(face or "").strip().lower()
    problems = []
    if part_name not in _spec.PARTS:
        problems.append(f"'part': {part!r} is not valid; use 'shell' or 'lid'")
    if face_name not in _FACES:
        problems.append(f"'face': {face!r} is not valid; use one of {', '.join(_FACES)}")
    if part_name == "shell" and face_name == "top":
        problems.append("a shell has no top face; cut the lid ('part': 'lid', 'face': 'top') or a wall / the 'bottom' floor")
    if part_name == "lid" and face_name == "bottom":
        problems.append("a lid has no bottom face; use 'top' or a wall face")
    nums = {}
    for key, val in (("u", u), ("v", v), ("w", w), ("h", h), ("r", r), ("margin", margin)):
        try:
            nums[key] = float(val)
        except (TypeError, ValueError):
            problems.append(f"'{key}': {val!r} is not a number (millimetres)")
    if "w" in nums and nums["w"] < 0.1:
        problems.append(f"'w': {nums['w']:g} is below 0.1 mm; use 0.1 or more")
    if "h" in nums and nums["h"] < 0.1:
        problems.append(f"'h': {nums['h']:g} is below 0.1 mm; use 0.1 or more")
    if nums.get("r", 0) < 0 or nums.get("margin", 0) < 0:
        problems.append("'r' and 'margin' must not be negative")
    cid = str(id or "").strip()
    if cid and not _NAME_RE.match(cid):
        problems.append(f"'id': {cid!r} is not a usable tag; use letters, digits and underscores, starting with a letter (at most 24)")
    if problems:
        raise ToolError("add_cutout: " + "; ".join(problems) + ".")
    actual = base.require_doc(doc_name, "add_cutout")
    before, note = base.ensure_spec(ctx, actual, "add_cutout")
    entries = [dict(e) for e in (before.get("cutouts") or []) if isinstance(e, dict)]
    if not cid:
        taken = {str(e.get("id")) for e in entries}
        n = 1
        while f"cutout{n}" in taken:
            n += 1
        cid = f"cutout{n}"
    entry = {"id": cid, "part": part_name, "face": face_name, "u": nums["u"], "v": nums["v"], "w": nums["w"], "h": nums["h"],
             "r": nums.get("r", 0.0), "margin": nums.get("margin", 0.0), "through": bool(through)}
    replaced = any(str(e.get("id")) == cid for e in entries)
    entries = [entry if str(e.get("id")) == cid else e for e in entries] + ([] if replaced else [entry])
    after, errors, warnings = base.merge_validate(before, {"cutouts": entries})
    if after is None:
        raise ToolError(f"add_cutout: the opening '{cid}' cannot be added:\n" + _problems_text(errors))
    board = base.load_board(str(after.get("board") or "rpi4b"), "add_cutout")
    extent = _face_extent(after, board, part_name, face_name)
    if extent is not None:
        u0, u1, v0, v1 = extent
        eu, ev = nums["w"] / 2 + nums.get("margin", 0.0), nums["h"] / 2 + nums.get("margin", 0.0)
        if nums["u"] - eu < u0 - 1e-6 or nums["u"] + eu > u1 + 1e-6 or nums["v"] - ev < v0 - 1e-6 or nums["v"] + ev > v1 + 1e-6:
            raise ToolError(
                f"add_cutout: the opening '{cid}' (u {base.fmt(nums['u'] - eu)}..{base.fmt(nums['u'] + eu)}, v {base.fmt(nums['v'] - ev)}..{base.fmt(nums['v'] + ev)} mm) "
                f"does not lie on the {_wall_label(face_name)} of the {part_name}, which spans u {base.fmt(u0)}..{base.fmt(u1)} and "
                f"v {base.fmt(v0)}..{base.fmt(v1)} mm; move or shrink it (frame: origin at the PCB lower-left corner, v = height above the PCB bottom).")
    first = "ports" if part_name == "shell" else "vents"
    results, store_note = _apply_spec_edit(ctx, actual, "add_cutout", before, after, first)
    base.set_extra(ctx, spec_ok=True)
    label = "Shell" if part_name == "shell" else "Lid"
    lines = [note] if note else []
    lines.append(f"{'Replaced' if replaced else 'Cut'} opening {cid} in {label} ({_wall_label(face_name)}): u={base.fmt(nums['u'])} "
                 f"v={base.fmt(nums['v'])} w={base.fmt(nums['w'])} h={base.fmt(nums['h'])} mm"
                 + (f", corner radius {base.fmt(nums['r'])}" if nums.get("r") else "") + ("" if through else ", blind") + ".")
    if results:
        lines.append(_stage_summary(results))
    else:
        lines.append(f"The {label.lower()} geometry for stage '{first}' is not built yet: the opening is in the spec and is cut when build_enclosure reaches it.")
    for r in results:
        lines.extend(f"Warning: {w}." for w in r.warnings[:3])
    lines.extend(f"Warning: {w}" for w in warnings[:3])
    if store_note:
        lines.append(store_note)
    return "\n".join(lines)


@tool(
    "add_vents",
    "Cut an array of vent slots (or round holes) into the shell or lid with a guaranteed minimum web between them. Idempotent per id.",
    {"type": "object", "properties": {
        "doc_name": {"type": "string"},
        "part": {"type": "string", "enum": list(_spec.PARTS)},
        "face": {"type": "string", "enum": _FACES},
        "pattern": {"type": "string", "enum": list(_spec.VENT_PATTERNS), "default": "slots"},
        "count": {"type": "integer", "minimum": 1, "maximum": 60},
        "slot_w": {"type": "number", "minimum": 0.5}, "slot_l": {"type": "number", "minimum": 0.5},
        "pitch": {"type": "number", "minimum": 1.0, "description": "Centre-to-centre distance"},
        "center": {"type": "array", "items": {"type": "number"}, "description": "[u, v]; default the centre of the face over the board"},
        "angle_deg": {"type": "number", "default": 0},
        "id": {"type": "string"}},
     "required": ["doc_name", "part", "face", "count", "slot_w", "slot_l", "pitch"]},
    category="build", mutating=True, geometry=True, phases=("BUILD", "VERIFY"),
)
def _add_vents(doc_name: str, part: str, face: str, count: int, slot_w: float, slot_l: float, pitch: float, pattern: str = "slots",
               center: "list | None" = None, angle_deg: float = 0.0, id: str = ""):  # noqa: A002
    """Replace the vent array of the spec and rebuild the ``vents`` stage and the later built stages."""
    ctx = current_context()
    part_name, face_name = str(part or "").strip().lower(), str(face or "").strip().lower()
    pattern_name = str(pattern or "slots").strip().lower()
    problems = []
    if part_name not in _spec.PARTS:
        problems.append(f"'part': {part!r} is not valid; use 'shell' or 'lid'")
    if face_name not in _FACES:
        problems.append(f"'face': {face!r} is not valid; use one of {', '.join(_FACES)}")
    if pattern_name not in _spec.VENT_PATTERNS:
        problems.append(f"'pattern': {pattern!r} is not valid; use 'slots' or 'holes'")
    nums = {}
    for key, val in (("slot_w", slot_w), ("slot_l", slot_l), ("pitch", pitch), ("angle_deg", angle_deg)):
        try:
            nums[key] = float(val)
        except (TypeError, ValueError):
            problems.append(f"'{key}': {val!r} is not a number (millimetres)")
    try:
        n_count = int(count)
        if n_count != float(count) or not 1 <= n_count <= 60:
            raise ValueError
    except (TypeError, ValueError):
        n_count = 0
        problems.append(f"'count': {count!r} is not a whole number from 1 to 60")
    ctr = None
    if center is not None:
        if isinstance(center, (list, tuple)) and len(center) == 2 and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in center):
            ctr = [float(center[0]), float(center[1])]
        else:
            problems.append("'center': must be [u, v], two numbers in millimetres")
    if nums.get("slot_w", 1) < 0.5 or nums.get("slot_l", 1) < 0.5:
        problems.append("'slot_w' and 'slot_l' must be at least 0.5 mm")
    if nums.get("pitch", 2) < 1.0:
        problems.append("'pitch' must be at least 1.0 mm")
    if part_name == "shell" and face_name == "top":
        problems.append("a shell has no top face; vent the lid ('part': 'lid', 'face': 'top') or a wall")
    if part_name == "lid" and face_name == "bottom":
        problems.append("a lid has no bottom face; use 'top' or a wall face")
    if problems:
        raise ToolError("add_vents: " + "; ".join(problems) + ".")
    actual = base.require_doc(doc_name, "add_vents")
    before, note = base.ensure_spec(ctx, actual, "add_vents")
    vents = {"enabled": True, "part": part_name, "face": face_name, "pattern": pattern_name, "count": n_count,
             "slot_w": nums["slot_w"], "slot_l": nums["slot_l"], "pitch": nums["pitch"], "center": ctr, "angle_deg": nums["angle_deg"]}
    after, errors, warnings = base.merge_validate(before, {"vents": vents})
    if after is None:
        raise ToolError("add_vents: this vent array does not fit:\n" + _problems_text(errors))
    board = base.load_board(str(after.get("board") or "rpi4b"), "add_vents")
    results, store_note = _apply_spec_edit(ctx, actual, "add_vents", before, after, "vents")
    base.set_extra(ctx, spec_ok=True)
    label = "Shell" if part_name == "shell" else "Lid"
    what = "round holes" if pattern_name == "holes" else "vent slots"
    lines = [note] if note else []
    size = f"slot {base.fmt(nums['slot_w'])} x {base.fmt(nums['slot_l'])} mm" if pattern_name == "slots" else f"hole {base.fmt(nums['slot_w'])} mm"
    lines.append(f"Cut {n_count} {what} in {label} ({_wall_label(face_name)}): {size}, pitch {base.fmt(nums['pitch'])} mm"
                 + (f", at u={base.fmt(ctr[0])} v={base.fmt(ctr[1])}" if ctr else "")
                 + (f", rotated {base.fmt(nums['angle_deg'])} deg" if nums["angle_deg"] else "") + ".")
    if str(id or "").strip():
        lines.append(f"(the enclosure has ONE vent array: '{str(id).strip()}' replaced the previous one)")
    lines.append(_stage_summary(results) if results else
                 "The lid is not built yet: the vents are in the spec and are cut when build_enclosure reaches stage 'vents'.")
    for r in results:
        lines.extend(f"Warning: {w}." for w in r.warnings[:3])
    lines.extend(f"Warning: {w}" for w in warnings[:3])
    if store_note:
        lines.append(store_note)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
# registration seams (imports at the END: tools_kit_params and the script helpers need the definitions above)
# --------------------------------------------------------------------------------------------------------------------

from . import tools_kit_params  # noqa: E402,F401  (registers set_parameters / get_parameters / bind_expression / set_parameter)


def _register_script_helpers() -> None:
    """Make ``kit`` (the kit package) and ``P`` (a ``Params`` factory) available to ``execute_code`` scripts. Idempotent, never raises."""
    try:
        from . import kit
        from .cad import scripting

        scripting.register_helpers("kit", kit)
        scripting.register_helpers("P", tools_kit_params.params_factory)
    except Exception:  # noqa: BLE001 - scripts simply do without the helpers
        pass


_register_script_helpers()
