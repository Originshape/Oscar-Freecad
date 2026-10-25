"""The enclosure builder: nine stages, one per call, each idempotent, transactional and tagged.

FreeCAD module, GUI-thread only (``cad.dispatch`` runs it there); the tools wrap it. Headless safe (``freecadcmd``).

Stages, in build order (``STAGES``): ``params``, ``reference``, ``shell``, ``standoffs``, ``ports``, ``lid``, ``vents``, ``hooks``,
``assembly`` (the lid is built BEFORE the vents so there is a lid to cut). The nominal flow needs nine ``next`` calls, eight when
a reference (an imported STEP, an attached file) already provides ``Ref_<ID>``.

Public API
----------
``build_stage(doc, spec, stage, *, board=None, rebuild=False, reference_status="none") -> StageResult``
    Build ONE stage. ``spec`` is an ``EnclosureSpec`` or a dict, ``board`` the board data (default: loaded from
    ``spec.board``). Everything stage-dependent is computed from the EFFECTIVE spec: the declared spec overlaid with the
    current ``Params`` design rows (``spec_from_sheet``), so a hand-edited sheet is honoured; that spec must pass
    ``spec.validate`` (an error fails the call before anything is touched). Removing the stage's own earlier objects and every
    LATER stage's (``remove_stages``: the chains build on each other) and building the stage happen in ONE transaction under
    ``UndoMode 1``: any exception or a failed post-condition (a part tip that is not one valid closed solid, an object in an error
    state) aborts it and leaves the document identical to before the call. A stage that is already built with the current spec hash
    is skipped (``rebuild=False``); ``rebuild=True`` forces it. ``params`` never removes anything: it adopts or creates the sheet and
    upserts rows (see ``_stages.stage_params``; ``rebuild`` or a stale hash lets the spec overwrite differing rows).
``next_stage(doc, spec) -> str | None``  the first stage that is missing or recorded with another spec hash (``reference`` is
    satisfied by an existing ``Ref_<ID>``); ``None`` when every stage is current.
``built_stages(doc, kit_id) -> dict[str, str]``  stage -> recorded spec hash (``'imported'`` for a reference that is not ours).
``tip(doc, role, kit_id=None)``  the last chain object of ``shell`` / ``lid`` (or the ``Ref_<ID>`` container for ``ref``).
``remove_stages(doc, kit_id, from_stage) -> list[str]``  delete the kit objects of ``from_stage`` and every later stage, dependents
    first, then restore the previous tips (visibility, labels). ``Part_*`` containers survive (except ``from_stage == 'params'``,
    which removes every kit object but the sheet itself); objects without kit tags are never touched. ``reference`` removes only the
    reference (nothing is built on it).
``rebuild_all(doc, spec, board=None, *, reference_status="none") -> list[StageResult]``  every stage in order with ``rebuild=True``;
    stops at the first failure. ``rebuild_from(doc, spec, from_stage, board=None, *, reference_status="none")`` does the same from
    ``from_stage`` on (what ``set_parameter`` needs for a count parameter: the stage and every later one; ``params`` and
    ``reference`` never cascade: nothing is built on them).
``rebuild_built(doc, spec, from_stage, board=None, *, reference_status="none")``  like ``rebuild_from`` but only the stages that were already
    built (what ``add_cutout`` / ``add_vents`` / ``set_parameter`` use). ``atomic(doc, label)`` is the transaction context manager.
``restamp_hashes(doc, spec) -> int``  record ``spec``'s stage hashes on the kit objects of every built stage (what ``set_parameter`` does
    after it synced the declared spec with the sheet: the geometry already matches, so the stages must not look stale).
``spec_from_sheet(doc, spec) -> EnclosureSpec``  the effective spec; ``stage_states(doc, spec) -> list[dict]`` one row per stage.

``StageResult`` (``.to_dict()`` gives the same as a plain dict):
    ``stage``     the stage name;
    ``ok``        False when the stage was refused or failed (the document is unchanged);
    ``message``   one or two sentences, error style when not ok;
    ``objects``   Names of the objects created, in creation order (the ``Params`` sheet when this call created it);
    ``warnings``  strings (merged-with-rim openings, kept sheet values, label problems ...), never errors;
    ``solids``    ``{"Shell": 1, "Lid": 1}`` number of solids of each existing tip;
    ``volumes``   ``{"Shell": mm3, "Lid": mm3}`` of each existing tip;
    ``metrics``   per stage facts (see the stage docstrings in ``_stages``), plus ``tips`` ({label: {"object", "bbox"}}) and
                  ``next_stage`` (what ``next_stage`` reports now);
    ``removed``   Names removed before building; ``skipped``  True when the stage was already current.

Object names (these names and tags are what the checks rely on): ``Params``; ``Ref_<ID>`` with ``Ref_<ID>_PCB`` and
``Ref_<ID>_<connector id>``; ``Part_Shell`` / ``Part_Lid`` / ``Assembly`` (``App::Part``); chains ``Shell_Body``, ``Shell_Standoffs``,
``Shell_Ports``, ``Shell_Vents``, ``Shell_Hooks`` and ``Lid_Plate``, ``Lid_Skirt``, ``Lid_Vents``, ``Lid_Hooks`` (role ``shell`` / ``lid``;
the newest is the tip and carries the Label ``Shell`` / ``Lid``); ``Lid_Hook<i>_Beam`` / ``Lid_Hook<i>_Barb`` (role ``hook``),
``Shell_Pocket<i>``, ``Shell_Port_<id>``, ``Lid_Notch_<id>``, ``Shell_StandoffPost<i>`` / ``Shell_StandoffPilot<i>`` and the sketches /
extrusions / unions in between (role ``aux``). ``rebuild`` never leaves ``Shell_Body001``-style drift: removed Names are reused.

Every object a stage creates is tagged ``OscarKit`` / ``OscarStage`` / ``OscarRole`` / ``OscarPart`` / ``OscarSpecHash`` (geometry
module); a stage that creates no geometry (no ports, vents disabled, a screw lid) leaves the marker group ``Kit_<stage>`` so it still
counts as built.
"""

import contextlib
import dataclasses
from dataclasses import dataclass, field
from typing import Any

import FreeCAD as App

from . import _stages as _st
from . import geometry as G
from . import params as _params
from . import spec as _spec
from .geometry import STAGE_INDEX, shape_facts, tags_of, tip  # noqa: F401  (tip: contract name)
from .spec import STAGES

KitBuildError = _st.KitBuildError
CHAIN_ROLES = G.CHAIN_ROLES


@dataclass
class StageResult:
    """The outcome of one ``build_stage`` call (module docstring)."""

    stage: str
    ok: bool
    message: str
    objects: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    solids: dict = field(default_factory=dict)
    volumes: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    removed: list = field(default_factory=list)
    skipped: bool = False

    def to_dict(self) -> dict:
        """A plain, JSON-able copy."""
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------------------------------------
# the effective spec
# --------------------------------------------------------------------------------------------------------------------

_INT_PATHS = ("hooks.count", "vents.count")


def spec_from_sheet(doc: Any, spec: "_spec.EnclosureSpec | dict") -> "_spec.EnclosureSpec":
    """The spec the builders really use: ``spec`` with every design field replaced by its current ``Params`` row.

    A sheet edited by hand (or by ``set_parameter``) therefore wins over a stale spec for geometry. Rows that are missing or do not
    evaluate leave the declared value alone; with no sheet the result equals ``spec``. Pure reads; the declared spec is not modified.
    """
    sp = _spec._as_spec(spec)
    try:
        values = _params.Params(doc).values()
    except _params.ParamsError:
        return sp
    d = _spec.to_dict(sp)
    for name, (path, _unit, _note) in _spec.DESIGN_PARAMS.items():
        if name not in values:
            continue
        node = d
        parts = path.split(".")
        for key in parts[:-1]:
            node = node[key]
        x = values[name]
        node[parts[-1]] = int(round(x)) if path in _INT_PATHS else (int(x) if path.endswith("_deg") and float(x).is_integer() else float(x))
    return _spec.from_dict(d)


# --------------------------------------------------------------------------------------------------------------------
# state queries
# --------------------------------------------------------------------------------------------------------------------

_PRIMARY_ROLES = ("shell", "lid", "assembly", "ref", "params")


def built_stages(doc: Any, kit_id: str) -> "dict[str, str]":
    """``{stage: recorded spec hash}`` of every stage whose product exists, in stage order.

    A stage counts as built when an object of that stage with a chain / assembly / reference / params role (or its ``Kit_<stage>`` marker)
    exists: leftovers of a half-deleted stage do not count. A reference that is not ours (an imported ``Ref_*`` part) is reported as
    ``{"reference": "imported"}`` unless the nominal one is present.
    """
    found: dict[str, str] = {}
    for o in doc.Objects:
        t = tags_of(o)
        if t["OscarKit"] != kit_id or t["OscarStage"] not in STAGE_INDEX or t["OscarRole"] == "container":
            continue
        if t["OscarRole"] in _PRIMARY_ROLES or o.Name.startswith("Kit_"):
            found.setdefault(t["OscarStage"], t["OscarSpecHash"])
    if "reference" not in found and any(o.Name.startswith("Ref_") and o.TypeId == "App::Part" and not tags_of(o)["OscarKit"] for o in doc.Objects):
        found["reference"] = "imported"
    return {s: found[s] for s in STAGES if s in found}


def stage_states(doc: Any, spec: "_spec.EnclosureSpec | dict") -> list[dict]:
    """One row per stage: ``{"stage", "state": "current" | "stale" | "missing" | "imported", "hash"}`` against ``spec``."""
    sp = _spec._as_spec(spec)
    built = built_stages(doc, sp.kit_id)
    rows = []
    for stage in STAGES:
        h = built.get(stage)
        if h is None:
            state = "missing"
        elif h == "imported":
            state = "imported"
        else:
            state = "current" if h == _spec.spec_hash(sp, stage) else "stale"
        rows.append({"stage": stage, "state": state, "hash": h or ""})
    return rows


def next_stage(doc: Any, spec: "_spec.EnclosureSpec | dict") -> "str | None":
    """The first stage that is missing or whose recorded hash differs from the spec's, in build order; ``None`` when all are current.

    ``reference`` is satisfied by an existing ``Ref_<ID>`` (imported STEP / attachment), so the Pi 4 flow with an imported STEP needs
    eight ``next`` calls and the nominal flow nine.
    """
    sp = _spec._as_spec(spec)
    imported = G.ref_container(doc, sp.board)
    imported = imported is not None and not G.is_kit_object(imported, sp.kit_id)
    built = built_stages(doc, sp.kit_id)
    for stage in STAGES:
        if stage == "reference" and imported:
            continue
        h = built.get(stage)
        if h is None or (h != "imported" and h != _spec.spec_hash(sp, stage)):
            return stage
    return None


# --------------------------------------------------------------------------------------------------------------------
# removal (idempotency)
# --------------------------------------------------------------------------------------------------------------------

def _unassemble(doc: Any, kit_id: str) -> None:
    """Take ``Part_Shell`` / ``Part_Lid`` out of every ``Assembly`` of this kit and free the lid placement."""
    for o in list(doc.Objects):
        t = tags_of(o)
        if t["OscarKit"] == kit_id and t["OscarRole"] == "assembly":
            for child in list(getattr(o, "Group", []) or []):
                o.removeObject(child)
    for o in doc.Objects:
        t = tags_of(o)
        if t["OscarKit"] == kit_id and t["OscarRole"] == "container" and t["OscarPart"] == "Lid":
            try:
                o.setExpression("Placement.Base.z", None)
            except Exception:    # noqa: BLE001, S110 - no binding to drop
                pass
            o.Placement = App.Placement()


def _removal_order(targets: list) -> list:
    """``targets`` ordered so that an object comes before everything it depends on (dependents first)."""
    remaining = {o.Name: o for o in targets}
    order: list = []
    while remaining:
        ready = [o for o in remaining.values()
                 if not any(d.Name in remaining and d is not o and getattr(d, "TypeId", "") != "App::Part" for d in o.InList)]
        if not ready:                      # a cycle cannot happen in a DAG; fall back to newest first so the loop always ends
            ready = [list(remaining.values())[-1]]
        for o in ready:
            order.append(o)
            remaining.pop(o.Name, None)
    return order


def _remove(doc: Any, kit_id: str, from_stage: str) -> list[str]:
    """Delete the kit objects of ``from_stage`` and later (no transaction here; callers wrap). Returns the Names removed."""
    idx = STAGE_INDEX[from_stage]
    scope = {"reference"} if from_stage == "reference" else set(STAGES[idx:])
    full = from_stage == "params"
    plain, containers = [], []
    for o in G.kit_objects(doc, kit_id):
        t = tags_of(o)
        if t["OscarRole"] == "params":
            continue                       # the sheet carries the user's parameters: never removed
        if t["OscarRole"] == "container":
            if full:
                containers.append(o)
        elif t["OscarStage"] in scope:
            plain.append(o)
    if any(tags_of(o)["OscarRole"] == "assembly" for o in plain):
        _unassemble(doc, kit_id)
    names: list[str] = []
    for o in _removal_order(plain) + containers:
        name = o.Name
        try:
            doc.removeObject(name)
            names.append(name)
        except Exception as exc:    # noqa: BLE001
            raise KitBuildError(f"cannot remove '{name}' while rebuilding stage '{from_stage}': {exc}; delete it by hand and rebuild") from None
    return names


@contextlib.contextmanager
def _atomic(doc: Any, label: str):
    """One undo transaction under ``UndoMode 1`` that commits on success and aborts on any exception (which propagates).

    ``freecadcmd`` documents run with ``UndoMode 0`` where ``abortTransaction`` undoes nothing (verified), so the mode is raised for
    the duration and restored afterwards. After an abort the document is recomputed so touched flags settle.
    """
    prev = doc.UndoMode
    if prev != 1:
        doc.UndoMode = 1
    doc.openTransaction(label)
    committed = False
    try:
        yield
        committed = True
    finally:
        try:
            if committed:
                doc.commitTransaction()
            else:
                doc.abortTransaction()
        except Exception as exc:    # noqa: BLE001 - a transaction problem must never mask the real error
            App.Console.PrintWarning(f"Oscar kit: closing the transaction '{label}' failed: {exc}\n")
        if not committed:
            try:
                doc.recompute()
            except Exception:    # noqa: BLE001, S110
                pass
        try:
            if doc.UndoMode != prev:
                doc.UndoMode = prev
        except Exception:    # noqa: BLE001, S110
            pass


atomic = _atomic    # public name for the tools (set_parameters, bind_expression ...): one undo transaction that aborts on any exception


def remove_stages(doc: Any, kit_id: str, from_stage: str) -> list[str]:
    """Remove the kit objects of ``from_stage`` and every later stage; restore the previous tips. Returns the Names removed.

    Atomic (one transaction): on a failure nothing is removed. See the module docstring for what survives.
    """
    if from_stage not in STAGE_INDEX:
        raise ValueError(f"stage {from_stage!r} is not a kit stage; use one of {', '.join(STAGES)}")
    with _atomic(doc, f"Oscar remove from {from_stage}"):
        names = _remove(doc, kit_id, from_stage)
        doc.recompute()
        G.refresh_tips(doc, kit_id)
    return names


# --------------------------------------------------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------------------------------------------------

def _fail(stage: str, message: str, **kw: Any) -> StageResult:
    return StageResult(stage=stage, ok=False, message=message, **kw)


def _params_missing(doc: Any) -> list[str]:
    """Rows of the default table that the sheet lacks (``['Params']`` when there is no sheet at all)."""
    try:
        have = set(_params.Params(doc).values())
    except _params.ParamsError:
        return ["Params"]
    return [r["name"] for r in _params.DEFAULT_PARAMS if r["name"] not in have]


def _add_marker(doc: Any, b: "G.Builder") -> None:
    """The tiny group that records a stage that built no geometry (so ``built_stages`` knows it ran)."""
    b.new("App::DocumentObjectGroup", f"Kit_{b.stage}", role="aux", part="", hidden=True)


def _problems(doc: Any, names: list[str]) -> list[str]:
    """Post-condition failures among the objects a stage created (error states; chain tips that are not one valid solid)."""
    out = []
    for name in names:
        obj = doc.getObject(name)
        if obj is None:
            continue
        state = list(getattr(obj, "State", []) or [])
        if "Invalid" in state or "RecomputeError" in state:
            out.append(f"'{name}' did not recompute ({obj.getStatusString()})")
            continue
        if tags_of(obj)["OscarRole"] in CHAIN_ROLES:
            ok, why = G.single_solid(obj)
            if not ok:
                out.append(why)
    return out


def _tip_report(doc: Any, kit_id: str) -> "tuple[dict, dict, dict]":
    solids, volumes, tips = {}, {}, {}
    for role, label in G.TIP_LABEL.items():
        t = tip(doc, role, kit_id)
        if t is None:
            continue
        f = shape_facts(t)
        solids[label], volumes[label] = f["solids"], round(f["volume"], 3)
        tips[label] = {"object": t.Name, "bbox": [round(x, 4) for x in f["bbox"]] if f["bbox"] else None}
    return solids, volumes, tips


def build_stage(doc: Any, spec: "_spec.EnclosureSpec | dict", stage: str, *, board: "dict | None" = None, rebuild: bool = False,
                reference_status: str = "none") -> StageResult:
    """Build ONE stage; see the module docstring for the contract. GUI-thread only. Never raises for bad input or a failed build."""
    if stage not in STAGE_INDEX:
        return _fail(str(stage), f"stage {stage!r} is not a kit stage; use one of {', '.join(STAGES)}")
    try:
        sp = _spec._as_spec(spec)
        board = board if board is not None else _spec.load_board(sp.board)
    except (KeyError, LookupError, ValueError) as exc:
        return _fail(stage, f"stage '{stage}' cannot start: {exc.args[0] if exc.args else exc}")
    kit = sp.kit_id
    shash = _spec.spec_hash(sp, stage)
    built = built_stages(doc, kit)
    idx = STAGE_INDEX[stage]
    if stage not in ("params", "reference"):        # the nominal reference is static board data: no sheet needed
        missing = _params_missing(doc)
        if missing:
            what = "the 'Params' sheet does not exist" if missing == ["Params"] else f"the 'Params' sheet lacks {', '.join(repr(m) for m in missing[:5])}"
            return _fail(stage, f"stage '{stage}' needs the parameter table: {what}; run build_enclosure(stage='params') first")
        for earlier in STAGES[1:idx]:
            if earlier == "reference":
                continue
            if earlier not in built:
                return _fail(stage, f"stage '{stage}' needs stage '{earlier}' first; build the stages in order (build_enclosure(stage='next'))")
    eff = sp if stage == "params" else spec_from_sheet(doc, sp)
    ok, errors, _warns = _spec.validate(sp, board)       # the declared spec must be valid ...
    if ok and eff is not sp:
        ok, errors, _warns = _spec.validate(eff, board)  # ... and so must the one the hand-edited sheet produces
    if not ok:
        return _fail(stage, f"stage '{stage}' cannot be built: " + "; ".join(errors[:4]) + (f" (+{len(errors) - 4} more)" if len(errors) > 4 else "")
                     + "; fix the spec with enclosure_spec or the parameter with set_parameter")
    recorded = built.get(stage)
    stale = bool(recorded) and recorded != shash and recorded != "imported"    # "" = a sheet adopted before any stage ran
    if not rebuild and recorded == shash and not (stage == "params" and _params_missing(doc)):
        ns = next_stage(doc, sp)
        return StageResult(stage=stage, ok=True, message=f"stage '{stage}' is already built and up to date; pass rebuild=True to build it again",
                           skipped=True, metrics={"next_stage": ns})
    if stage == "reference" and recorded == "imported" and not rebuild:
        return StageResult(stage=stage, ok=True, message="the reference part already exists (imported); the nominal board was not built",
                           skipped=True, metrics={"mode": "existing", "next_stage": next_stage(doc, sp)})
    try:
        with _atomic(doc, f"Oscar stage {stage}"):
            removed = _remove(doc, kit, stage) if stage != "params" else []
            values = _params.Params(doc).values() if doc.getObject(G.SHEET) is not None else {}
            builder = G.Builder(doc, kit_id=kit, stage=stage, spec_hash=shash)
            ctx = _st.StageCtx(doc, sp, eff, board, builder, values=values, reference_status=reference_status, rebuild=rebuild, stale=stale)
            _st.STAGE_FUNCS[stage](ctx)
            if stage != "params" and ctx.metrics.get("mode") != "existing" and not G.stage_objects(doc, kit, stage):
                _add_marker(doc, builder)
            doc.recompute()
            bad = _problems(doc, builder.created)
            if bad:
                raise KitBuildError("; ".join(bad[:3]) + (f" (+{len(bad) - 3} more)" if len(bad) > 3 else ""))
            label_warnings = G.refresh_tips(doc, kit, sp.colors)
            doc.recompute()      # the App::Part containers are touched by addObject / refresh_tips; settle them so no stage leaves "needs recompute"
            solids, volumes, tips = _tip_report(doc, kit)
    except _params.ParamsError as exc:
        return _fail(stage, f"stage '{stage}' failed: {exc}; the document is unchanged")
    except (KitBuildError, G.KitGeometryError, ValueError, KeyError, RuntimeError, TypeError, AttributeError, OSError) as exc:
        text = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return _fail(stage, f"stage '{stage}' failed: {text}; the document is unchanged (rolled back)")
    except Exception as exc:    # noqa: BLE001 - FreeCAD raises its own exception types (Base.FreeCADError, Part.OCCError ...)
        return _fail(stage, f"stage '{stage}' failed: {type(exc).__name__}: {exc}; the document is unchanged (rolled back)")
    ctx.metrics["tips"] = tips
    ctx.metrics["next_stage"] = next_stage(doc, sp)
    ctx.metrics["reference_status"] = reference_status
    names = list(builder.created)
    shown = ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")
    msg = f"Stage '{stage}' built ({len(names)} object{'s' if len(names) != 1 else ''}: {shown})" if names else f"Stage '{stage}' built (nothing to create)"
    if removed:
        msg += f"; replaced {len(removed)} earlier object{'s' if len(removed) != 1 else ''}"
    ns = ctx.metrics["next_stage"]
    msg += f"; next stage: {ns}" if ns else "; every stage is built"
    return StageResult(stage=stage, ok=True, message=msg, objects=names, warnings=list(ctx.warnings) + label_warnings, solids=solids,
                       volumes=volumes, metrics=ctx.metrics, removed=removed)


def rebuild_all(doc: Any, spec: "_spec.EnclosureSpec | dict", board: "dict | None" = None, *, reference_status: str = "none") -> list[StageResult]:
    """Every stage in order with ``rebuild=True``; stops after the first failed stage (its result is the last item). Headless safe."""
    sp = _spec._as_spec(spec)
    results = []
    for stage in STAGES:
        r = build_stage(doc, sp, stage, board=board, rebuild=True, reference_status=reference_status)
        results.append(r)
        if not r.ok:
            break
    return results


def rebuild_built(doc: Any, spec: "_spec.EnclosureSpec | dict", from_stage: str, board: "dict | None" = None, *,
                  reference_status: str = "none") -> list[StageResult]:
    """``from_stage`` and every later stage that was BUILT before the call, in order, each with ``rebuild=True`` (what an edit of an
    existing model needs: a new vent array must not build the hooks of a model that has none yet). ``from_stage`` itself is always
    rebuilt. Stops after the first failure (its result is the last item); the stages after it are then missing and ``next_stage``
    reports them. ``params`` and ``reference`` build only themselves."""
    if from_stage not in STAGE_INDEX:
        return [_fail(str(from_stage), f"stage {from_stage!r} is not a kit stage; use one of {', '.join(STAGES)}")]
    sp = _spec._as_spec(spec)
    before = built_stages(doc, sp.kit_id)
    stages = (from_stage,) if from_stage in ("params", "reference") else [s for s in STAGES[STAGE_INDEX[from_stage]:] if s == from_stage or s in before]
    results = []
    for stage in stages:
        r = build_stage(doc, sp, stage, board=board, rebuild=True, reference_status=reference_status)
        results.append(r)
        if not r.ok:
            break
    return results


def restamp_hashes(doc: Any, spec: "_spec.EnclosureSpec | dict") -> int:
    """Set ``OscarSpecHash`` of every kit object to ``spec``'s hash of its stage; returns how many objects changed.

    Only for a model whose geometry already follows ``spec`` (an edit of an expression-bound parameter). Objects of a reference that
    is not ours (``imported``) and containers without a stage are left alone. Never raises for a read-only property.
    """
    sp = _spec._as_spec(spec)
    changed = 0
    for o in doc.Objects:
        t = tags_of(o)
        if t["OscarKit"] != sp.kit_id or t["OscarStage"] not in STAGE_INDEX or t["OscarSpecHash"] in ("", "imported"):
            continue
        new = _spec.spec_hash(sp, t["OscarStage"])
        if new != t["OscarSpecHash"]:
            try:
                G.set_tag(o, "OscarSpecHash", new)
                changed += 1
            except Exception:    # noqa: BLE001, S112 - a tag that cannot be written leaves the stage looking stale, nothing worse
                continue
    return changed


def rebuild_from(doc: Any, spec: "_spec.EnclosureSpec | dict", from_stage: str, board: "dict | None" = None, *,
                 reference_status: str = "none") -> list[StageResult]:
    """``from_stage`` and every later stage in order, each with ``rebuild=True``; stops after the first failure.

    ``params`` and ``reference`` build only themselves (nothing is built on them; ``params`` never cascades, 7.4). An unknown stage
    gives one failed result.
    """
    if from_stage not in STAGE_INDEX:
        return [_fail(str(from_stage), f"stage {from_stage!r} is not a kit stage; use one of {', '.join(STAGES)}")]
    sp = _spec._as_spec(spec)
    stages = (from_stage,) if from_stage in ("params", "reference") else STAGES[STAGE_INDEX[from_stage]:]
    results = []
    for stage in stages:
        r = build_stage(doc, sp, stage, board=board, rebuild=True, reference_status=reference_status)
        results.append(r)
        if not r.ok:
            break
    return results
