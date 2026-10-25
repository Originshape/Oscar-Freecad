"""The registry API and the ``run_tool`` pipeline of ``oscar.tools``.

``oscar/tools.py`` owns the ``Tool`` class, the ``@tool`` decorator, ``_HANDLERS`` and the
handlers of the original tools and re-exports everything defined here (``specs``,
``tool_meta``, ``run_tool`` ...) at its bottom, so ``oscar.tools.run_tool`` is the one public
entry point. This module only exists to keep ``tools.py`` readable; nothing here is public
beyond the names that ``oscar.tools`` re-exports. It reads the
registry through ``oscar.tools`` at CALL time, so tests may patch ``oscar.tools._HANDLERS``
members freely; patch the pipeline helpers (``_health_api``, ``_log``) on THIS module.
Importing this module first or ``oscar.tools`` first both work (the import of ``oscar.tools``
sits at the bottom of this file).
"""

import inspect
import os
import time
from typing import Any, Iterable

from . import toolpipe
from . import turnlog as _turnlog
from .providers import ToolSpec
from .toolctx import ToolOutcome, current_context
from .toolpipe import clip_text

# --------------------------------------------------------------------------
# Registry API

# The tools every phase offers: the plan, the notes and the read-only eyes.
CORE_TOOLS = frozenset({
    "plan_set", "plan_update", "record_note", "list_documents", "get_objects", "get_object", "get_view",
})

# What a small-context model is offered (19 names): the CORE tools plus the ones that build.
LITE_TOOLS = CORE_TOOLS | frozenset({
    "create_document", "create_object", "edit_object", "execute_code", "import_file",
    "find_reference_part", "source_reference_part", "build_enclosure", "enclosure_spec",
    "set_parameter", "validate_model", "check_fit",
})

# Same list as ``oscar.tools.CATEGORIES`` (a test pins the equality); spelled out here so this
# module can be imported before ``oscar.tools`` without a circular import.
_ALL_CATEGORIES = frozenset({"session", "read", "build", "params", "source", "verify", "deliver",
                             "plan", "fem", "net"})
_BUILD_CATEGORIES = frozenset({"session", "read", "build", "params", "source", "verify",
                               "deliver", "plan", "net"})

# Which tool categories an intent is offered. Visibility only: run_tool
# executes ANY registered tool (a hidden tool a model calls anyway still runs).
INTENT_CATEGORIES: dict[str, frozenset[str]] = {
    **{intent: _BUILD_CATEGORIES for intent in
       ("enclosure", "fixture", "bracket", "generic_part", "modify", "deliver")},
    "fem": _ALL_CATEGORIES,
    **{intent: _ALL_CATEGORIES for intent in ("inspect", "question", "other")},
}

# Tools the kit intents never see: the kit builds, saves and reloads for the model.
_KIT_HIDDEN = frozenset({
    "execute_code_async", "get_async_status", "execute_code_headless", "apply_feature_plan",
    "reapply_plan_feature", "insert_part_from_library", "get_parts_list", "reload_document",
    "feature_tree_status", "download_file", "save_document",
})
INTENT_HIDDEN: dict[str, frozenset[str]] = {
    intent: _KIT_HIDDEN for intent in ("enclosure", "fixture", "bracket")
}


def get_tools() -> "list[_tools.Tool]":
    """Every registered tool, in registration order."""
    return list(_tools._HANDLERS.values())


def tool_meta(name: str) -> "_tools.Tool | None":
    """The registered ``Tool`` (metadata and handler) named ``name``, or ``None``."""
    return _tools._HANDLERS.get(name)


def is_mutating(name: str) -> bool:
    """True when the tool is recorded as a step record (False for unknown names)."""
    entry = _tools._HANDLERS.get(name)
    return bool(entry and entry.mutating)


def is_geometry(name: str) -> bool:
    """True when the tool changes the model (health, delta, snapshot, checkpoint run for it)."""
    entry = _tools._HANDLERS.get(name)
    return bool(entry and entry.geometry)


def mutating_names() -> frozenset[str]:
    """Names of every mutating tool (the panel derives its ``_MUTATING_TOOLS`` from this)."""
    return frozenset(name for name, entry in list(_tools._HANDLERS.items()) if entry.mutating)


def specs(phases: Iterable[str] | None = None, categories: Iterable[str] | None = None,
          names: Iterable[str] | None = None, *, intent: str | None = None, lite: bool = False,
          exclude: Iterable[str] | None = None) -> list[ToolSpec]:
    """The ToolSpecs to offer the model, in registration order (stable: provider caching).

    With no argument every registered tool is returned. ``phases``: a tool is kept when its
    phase mask intersects them (``None`` or an empty collection = no phase filter, so a ledger
    that reports no phase can never leave the model without tools). ``categories`` and
    ``names`` restrict to those categories / names. ``intent`` applies ``INTENT_CATEGORIES``
    and ``INTENT_HIDDEN`` (an unknown intent = no gating). ``lite=True`` keeps only
    ``LITE_TOOLS`` with first-sentence descriptions and without the optional ``step_title`` /
    ``view_name`` / ``timeout`` / ``include_screenshot`` properties. ``exclude`` removes names.
    """
    phase_set = frozenset(phases) if phases else None
    category_set = frozenset(categories) if categories is not None else None
    name_set = frozenset(names) if names is not None else None
    excluded = frozenset(exclude) if exclude else frozenset()
    intent_key = intent.strip().lower() if isinstance(intent, str) else None
    intent_categories = INTENT_CATEGORIES.get(intent_key) if intent_key else None
    intent_hidden = INTENT_HIDDEN.get(intent_key, frozenset()) if intent_key else frozenset()
    chosen: list[ToolSpec] = []
    for entry in list(_tools._HANDLERS.values()):
        if name_set is not None and entry.name not in name_set:
            continue
        if entry.name in excluded or entry.name in intent_hidden:
            continue
        if lite and entry.name not in LITE_TOOLS:
            continue
        if category_set is not None and entry.category not in category_set:
            continue
        if intent_categories is not None and entry.category not in intent_categories:
            continue
        if phase_set is not None and phase_set.isdisjoint(entry.phases):
            continue
        chosen.append(entry.spec(lite=lite))
    return chosen


def derive_title(name: str, args: dict[str, Any]) -> str:
    """Row title of a call when the model gave no ``step_title``.

    The registered tool's own ``Tool.title(args)`` callable wins, then the built-in title table
    (``Create Part::Box Base``, ``Cut 9.8 x 4.1 opening in shell -y`` ...), then the tool
    name. One line, at most 80 characters; never raises.
    """
    return toolpipe.derive_title(name, args, _tools._HANDLERS.get(name))


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    """Repair the argument shapes models commonly get wrong: objects and arrays sent as JSON
    strings, booleans as "true"/"no"/"0", numbers as strings (see ``toolpipe.coerce_value``)."""
    return toolpipe.coerce_value(value, schema)


def _prepare_arguments(entry: "_tools.Tool", arguments: Any) -> dict[str, Any]:
    """Validate a call against the tool's schema; raise ToolError with a
    message that tells the model exactly how to fix the call.

    Beyond unknown / missing parameters this drops an optional parameter sent as null (or ``""``
    for an enum) so the handler default applies, resolves the cosmetic ``view_name`` (an unknown
    spelling becomes ``Isometric``), repairs shapes (JSON strings for objects and
    arrays, boolean words, numeric strings), validates ``enum`` (with a ``similar:`` hint),
    ``minimum`` / ``maximum`` and array item types with ALL problems in one error, removes the
    universal ``step_title`` before the handler is bound (it lands in ``ctx.step_title``) and,
    in a lite turn (``ctx.state["lite"]``), forces ``include_screenshot=False``.
    """
    if not isinstance(arguments, dict):
        raise _tools.ToolError(f"Bad arguments for {entry.name}: expected a JSON object.")
    ctx = current_context()
    arguments = dict(arguments)
    if "step_title" in arguments:
        ctx.step_title = toolpipe.clean_title(arguments.pop("step_title"))
    properties = entry.parameters.get("properties", {})
    accepted = ", ".join(properties) or "(none)"
    unknown = [key for key in arguments if key not in properties]
    if unknown:
        raise _tools.ToolError(
            f"Bad arguments for {entry.name}: unknown parameter(s) "
            f"{', '.join(repr(k) for k in unknown)}. Accepted parameters: {accepted}."
        )
    missing = [
        key for key in entry.parameters.get("required", [])
        if arguments.get(key) in (None, "")
    ]
    if missing:
        raise _tools.ToolError(
            f"Bad arguments for {entry.name}: missing required parameter(s) "
            f"{', '.join(repr(k) for k in missing)}. Accepted parameters: {accepted}."
        )
    required = set(entry.parameters.get("required", []))
    prepared: dict[str, Any] = {}
    problems: list[str] = []
    for key, value in arguments.items():
        schema = properties[key]
        # An explicit null (or "" for an enum) means "not given": dropping it lets the handler default apply instead of
        # overriding it with None. A required parameter was already checked above.
        if key not in required and (value is None or (isinstance(value, str) and value == "" and "enum" in schema)):
            continue
        options = schema.get("enum")
        if isinstance(options, list) and options and value not in options:
            if options == _tools.VIEW_NAMES:
                # the screenshot orientation is cosmetic: resolve an unknown spelling, never refuse the call
                prepared[key] = toolpipe.resolve_view_name(value, options)
                continue
        value, problem = toolpipe.check_argument(key, _coerce(value, schema), schema)
        if problem:
            hint = schema.get("description")
            if schema.get("type") in ("integer", "number") and isinstance(hint, str) and hint:
                problem += f" ({hint})"     # what the number means: 'a face COUNT (>= 4), not a ratio'
            problems.append(problem)
        prepared[key] = value
    if problems:
        raise _tools.ToolError(f"Bad arguments for {entry.name}: {'; '.join(problems)}.")
    if "include_screenshot" in properties and ctx.state.get("lite"):
        prepared["include_screenshot"] = False
    try:
        inspect.signature(entry.handler).bind(**prepared)
    except TypeError as e:
        raise _tools.ToolError(f"Bad arguments for {entry.name}: {e}. Accepted parameters: {accepted}.") from e
    return prepared


# --------------------------------------------------------------------------
# run_tool: the one pipeline every call goes through


def _flow_enabled(ctx: Any = None) -> bool:
    """False when flow enforcement is off: the kill switch ``OSCAR_FLOW=0`` or the turn's own flag
    (the agent stores ``ctx.state["flow_enforcement"] = False`` when ``AgentWorker(flow=False)`` or the
    ``OscarFlow`` pref switched it off). The health check of the model and the ledger hook are then skipped
    (Project recording stays on). Calls without a context follow the environment alone."""
    if os.environ.get("OSCAR_FLOW", "1").strip() == "0":
        return False
    state = getattr(ctx, "state", None)
    return not (isinstance(state, dict) and state.get("flow_enforcement") is False)


def _health_api():
    """``oscar.cad.health`` (the C2 health API), or ``None`` while that module is not delivered.

    A missing DEPENDENCY inside it is a real bug and is raised (the caller logs it and goes on
    without health, so a broken hook never turns a successful call into an error)."""
    try:
        from .cad import health
    except ModuleNotFoundError as e:
        if e.name not in ("oscar.cad.health", f"{__package__}.cad.health"):
            raise
        return None
    return health


def _log(kind: str, **fields: Any) -> None:
    _turnlog.log_event(kind, **fields)


def _hook_failed(hook: str, error: BaseException) -> None:
    """A hook raised: log it (names and the exception text only) and carry on."""
    _log("project_error" if hook == "project" else f"{hook}_error", hook=hook,
         error=f"{type(error).__name__}: {error}"[:200])


def _active_doc_name() -> str:
    """Name of the active document, or ``""`` (GUI-thread work is done through the engine)."""
    try:
        return _tools._engine()._active_document_name() or ""
    except Exception:  # noqa: BLE001
        return ""


def _unpack_result(result: Any) -> tuple[str, str | None]:
    """A handler returns ``(text, image)``; tolerate a bare string or a short tuple."""
    if isinstance(result, tuple):
        text = result[0] if result else ""
        image = result[1] if len(result) > 1 else None
    else:
        text, image = result, None
    return ("" if text is None else str(text)), (image or None)


def run_tool(name: str, arguments: dict[str, Any]) -> tuple[str, str | None, bool]:
    """Execute a tool by name. Returns ``(text, image_b64, is_error)``.

    The pipeline, in this order: unknown name -> error; strip ``step_title``;
    for a mutating call under a bound context (``current_context().active``) and a geometry
    tool: snapshot the documents and make sure the turn checkpoint exists; repair and validate
    the arguments; run the handler; judge the model after a geometry call (``cad.health``:
    NEW errors turn the result into an error and append ``NOT OK`` plus a table, otherwise a
    ``[health]`` line is appended); derive the step title; fill ``ctx.outcome``; record the
    step through ``ctx.project.on_tool_result``; tick ``ctx.ledger``. Every hook is guarded: a
    hook failure is logged and never changes what the handler returned. ``ToolOutcome.ok`` and
    the returned ``is_error`` carry the health verdict, ``handler_ok`` does not. Unknown tools
    and calls without a bound context run no hook. Any registered tool runs in any phase
    (visibility is not permission); an out-of-phase call is only logged.
    """
    entry = _tools._HANDLERS.get(name)
    ctx = current_context()
    ctx.outcome = None
    if entry is None:
        message = f"Unknown tool '{name}'. Available tools: {', '.join(_tools._HANDLERS)}."
        _log("tool", name=name, ok=False, error="unknown tool")
        ctx.outcome = ToolOutcome(name=str(name), call_id=ctx.call_id, title=toolpipe.clean_title(name),
                                  ok=False, handler_ok=False, text=message)
        return message, None, True
    started = time.monotonic()
    args = dict(arguments) if isinstance(arguments, dict) else (arguments if arguments is not None else {})
    step_title = ""
    if isinstance(args, dict):
        step_title = toolpipe.clean_title(args.pop("step_title", ""))
    ctx.step_title = step_title
    if ctx.phase and ctx.phase not in entry.phases:
        _log("tool_out_of_phase", name=name, phase=ctx.phase)

    hooks = ctx.active and entry.mutating
    flow_on = _flow_enabled(ctx)
    health = None
    before: dict = {}
    if hooks and entry.geometry:
        if flow_on:
            try:
                health = _health_api()
                if health is not None:
                    before = health.snapshot_all_via_dispatch() or {}
            except Exception as e:  # noqa: BLE001
                _hook_failed("health", e)
                health, before = None, {}
        if ctx.project is not None:
            try:
                requested = args.get("doc_name") if isinstance(args, dict) else None
                target = toolpipe.resolve_doc_name(requested, before) or _text_or(requested) or _active_doc_name()
                if target:
                    ctx.project.ensure_turn_checkpoint(target)
            except Exception as e:  # noqa: BLE001
                _hook_failed("project", e)

    text, image, handler_ok, prepared = "", None, True, None
    try:
        prepared = _prepare_arguments(entry, args)
        text, image = _unpack_result(entry.handler(**prepared))
    except _tools.ToolError as e:
        text, handler_ok = str(e), False
    except Exception as e:  # noqa: BLE001 - tool failures are model-visible, not fatal
        import traceback

        text, handler_ok = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}", False
    is_error = not handler_ok
    health_failed = False
    handler_text = clip_text(text)
    final_text = handler_text

    stage: dict | None = None
    if (health is not None and prepared is not None and not ctx.should_stop()
            and not toolpipe.is_dispatch_timeout(text, handler_ok)):
        try:
            stage = toolpipe.health_stage(ctx.state, health, prepared, before, handler_ok)
        except Exception as e:  # noqa: BLE001
            _hook_failed("health", e)
            stage = None
    elif health is not None and prepared is not None:
        _log("hook_skipped", hook="health", why="stopped" if ctx.should_stop() else "dispatch timeout")
    report = stage["report"] if stage else None
    new = stage["new"] if stage else []
    doc = stage["doc"] if stage else _text_or(prepared.get("doc_name") if prepared else "")
    if stage:
        if handler_ok and new:
            is_error = health_failed = True
        try:
            final_text = handler_text + toolpipe.health_suffix(
                report, new, stage["persisting"], handler_ok=handler_ok,
                delta_text=toolpipe.delta_line(stage["delta"]), extra_lines=stage["extra_lines"])
        except Exception as e:  # noqa: BLE001
            _hook_failed("health", e)

    override = ctx.state.pop("args_override", None)
    recorded = dict(args) if isinstance(args, dict) else {}
    if isinstance(override, dict):
        recorded.update(override)
    hint = ctx.state.pop("title_hint", "")
    title = step_title or toolpipe.clean_title(hint) or derive_title(name, recorded)
    summary = toolpipe.summarize_report(report) if report is not None else None
    delta = stage["delta"] if stage else {}
    outcome = ToolOutcome(
        name=name, call_id=ctx.call_id, title=title, ok=not is_error, handler_ok=handler_ok,
        health_failed=health_failed, text=final_text, dur_s=round(time.monotonic() - started, 3),
        doc=doc, mutating=entry.mutating, geometry=entry.geometry, delta=delta, health=summary,
        new_errors=[getattr(f, "message", str(f)) for f in new], image_b64=image or "",
        args=recorded, extra=dict(ctx.state.pop("extra", None) or {}))
    ctx.outcome = outcome

    if ctx.project is not None and entry.mutating:
        try:
            record = ctx.project.on_tool_result(
                turn_id=ctx.turn_id, round=ctx.round, call_id=ctx.call_id, tool=name, title=title,
                args=recorded, text=final_text, ok=handler_ok, dur_s=outcome.dur_s, doc=doc,
                delta=delta, health=summary, mutating=True, geometry=entry.geometry)
            if record is not None:
                outcome.step_index = int(getattr(record, "index", -1))
                snapshot, script = getattr(record, "snapshot", ""), getattr(record, "script", "")
                outcome.snapshot_path = ctx.project.path(snapshot) if snapshot else ""
                outcome.snapshot_note = getattr(record, "snapshot_note", "") or ""
                outcome.checkpoint = getattr(record, "checkpoint", "") or ""
                outcome.script_path = ctx.project.path(script) if script else ""
        except Exception as e:  # noqa: BLE001 - recording must never fail the call
            _hook_failed("project", e)

    if ctx.ledger is not None and flow_on:
        try:
            ctx.ledger.on_tool_result(outcome)
        except Exception as e:  # noqa: BLE001
            _hook_failed("ledger", e)

    fields = {"name": name, "ok": not is_error, "seconds": round(time.monotonic() - started, 2),
              "result_chars": len(final_text), "phase": ctx.phase, "geometry": entry.geometry,
              "step_index": outcome.step_index}
    if is_error:
        fields["error"] = final_text[:200]
    _log("tool", **fields)
    return final_text, image, is_error


def _text_or(value: Any) -> str:
    return "" if value is None else str(value).strip()


# ``oscar.tools`` re-exports this module's API at ITS bottom and this module reads the registry
# through ``oscar.tools`` at call time: whichever of the two is imported first, the other is
# found complete enough (everything above is defined before this line).
from . import tools as _tools  # noqa: E402
