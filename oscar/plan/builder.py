"""FeaturePlan → native FreeCAD: Spreadsheet + Sketcher + PartDesign.

The FreeCAD backend for the CAD-neutral FeaturePlan contract (see
schema.py). Runs against an open ``App.Document`` using ONLY App-level
APIs — no GUI — so the same function drives the interactive build (via
the engine, on the GUI thread) and headless builds (tests, ``rebuild.py``).

Structure of the built document:

    Parameters   Spreadsheet::Sheet   one row per plan parameter, alias = name
                                      (derived parameters are ``=`` formulas)
    Body         PartDesign::Body     the native feature tree
      <id>       Sketcher::SketchObject   real sketch geometry
      <id>       PartDesign::Pad/Pocket/Revolution/Hole

Every dimensional constraint and feature length binds by expression to a
spreadsheet alias, so editing a cell + recompute resizes the model. Each
feature is wrapped in its own transaction and checked against post-conditions
(a pad must add volume, a pocket / hole must remove it, the body stays one
valid solid); on failure the feature is rolled back, the document stays at the
last good feature, and the failing feature id + error + metrics come back.

ONE result contract for every entry point: ``{success, error, errors,
feature_id, stage}`` (stages ``validation``, ``document``, ``parameters``,
``build``, ``built``) plus report fields. ``format_build_failure(result)``
turns a failed result into the model-facing text.

Names: the body, the spreadsheet and the feature object names are
parameters (``body_name`` / ``sheet_name`` / ``prefix``). Expressions use the
spreadsheet's REAL ``Name``. Every object the plan builds is tagged with the
plan id, and the sheet and the Body are also stamped ``PLAN_MARK`` (Label2):
only an object carrying BOTH is provably ours. Applying the same plan again
REPLACES its previous build; a user's own ``Parameters`` sheet or ``Body`` (or
a build saved before the stamp existed) is NEVER replaced, adopted or bound
(an explicit name held by a foreign object is refused; a default name is
moved to a free one such as ``Parameters001``).

Undo: the interactive tool runs with an undo stack and Revert-turn pops every
``Oscar:`` step back to where the turn began, so the setup of a whole
apply (clearing the previous build, the new sheet, the Body) is ONE
transaction, each feature its own, and apply_feature / rebuild_from put the
sheet sync and the stale-object removal into the transaction of the feature
they rebuild. A failed call restores the sheet cells it wrote and removes what
it created (console documents have no undo stack, so those are restored by
hand).

Schema v2 (see ``schema``) builds through the same entry points: every op is
dispatched through the registry ``schema.OPS`` (translators in
``_featurebuild`` / ``_dressbuild``), each feature gets the same post-
conditions (volume delta of its kind, one valid solid, a sane bounding box)
and the same marker rules. A plan with ``parts`` builds one Body per part
(``Body_<part id>`` unless ``part.body`` names it), never fused, each stamped
with the plan id, ``PLAN_MARK`` and its part id; placements and the
``assembly`` are expression-bound to the sheet; ``references`` are measured
and reported. Re-applying such a plan replaces ONLY the Bodies and sheet it
provably built and refuses when any of them holds a hand-added feature.

Known v2 limits, reported honestly rather than papered over:
* a sealed hollow (``thickness`` with no open face) is refused: FreeCAD's shell
  tool needs a face to remove; patterns turn about, and ``mirror`` reflects
  across, the Body's own origin axes and planes;
* edges are chosen by rule when the feature is built; FreeCAD 1.1 can turn such
  an edge link Invalid on the first later recompute, so every fillet / chamfer /
  shell is "settled" (touched and recomputed) after its commit, and
  ``refresh_edge_rules(doc)`` repairs one that still went stale from the rule
  stored on the object (``OscarPlanRule``);
* ``references`` compare axis-aligned bounding boxes and only inform.

Known v1 limits, reported honestly rather than papered over:
* polyline profiles bind vertex positions via per-segment DistanceX/Y
  expressions; an arc binds its size only when the plan names a ``radius``
  (or ``diameter``) parameter — without one the arc's size stays free, so
  the sketch is not fully constrained: that is reported as a warning, not
  refused (the reconstruction engine's own arcs are spelled this way); an
  arc that is exactly a semicircle over its chord is singular for a radius
  dimension (name it and the solver reports the redundancy: use a ``slot``
  or two quarter arcs).
"""

from contextlib import contextmanager

from . import _dressbuild, _featurebuild, _partbuild, _sketchbuild, schema  # noqa: F401
from ._dressbuild import refresh_edge_rules
from ._buildcore import (STAGE_BUILD, STAGE_BUILT, STAGE_DOCUMENT,  # noqa: F401
                         STAGE_PARAMETERS, STAGE_VALIDATION, BuildError, Ctx,
                         CAUSE_DEPENDENTS_EXIST, CAUSE_RECOMPUTE_FAILED,
                         RECOVERY_ALTERNATIVES,
                         body_metrics, check_postconditions, doc_errors,
                         failure, find_feature_object,
                         format_build_failure, success, tag_object, tag_of,
                         valid_name)

__all__ = ["BuildError", "apply_plan", "apply_feature", "rebuild_from",
           "sync_parameters", "build_spreadsheet", "format_build_failure",
           "checkpoint", "refresh_edge_rules", "drop_feature"]

_PREFIX_PROP = "OscarPlanPrefix"


# ---------------------------------------------------------------------------
# transaction checkpoint


@contextmanager
def checkpoint(doc, label: str):
    """One feature per transaction; abort rolls the document back to the
    last good state (where the document has undo enabled — console documents
    do not, which is why callers also run ``_remove_created``)."""
    doc.openTransaction(label)
    try:
        yield
    except Exception:
        doc.abortTransaction()
        raise
    doc.commitTransaction()


# Stamped (Label2) on the sheet and Body the builder creates. Name and layout
# alone cannot tell them from a user's own: the standard FreeCAD parameter
# sheet has the very same A = name / B = aliased value rows, and a user's
# Body is called "Body" too. Only an object carrying this was built by us.
PLAN_MARK = "Oscar plan build"


def _marked(obj) -> bool:
    return getattr(obj, "Label2", None) == PLAN_MARK


def _owned(obj, type_id: str, plan_id: str) -> bool:
    """True when ``obj`` is a ``type_id`` that plan ``plan_id`` provably built
    (plan tag AND stamp). Name and layout prove nothing: a user's own
    ``Parameters`` sheet and ``Body`` look exactly like ours."""
    return (obj is not None and obj.TypeId == type_id
            and tag_of(obj) == plan_id and _marked(obj))


# ---------------------------------------------------------------------------
# spreadsheet



def _alias_error(name: str, error: Exception) -> BuildError:
    return BuildError(
        "", f"parameter {name!r}: FreeCAD rejected it as a spreadsheet alias "
            f"({error}); rename it — an alias cannot be a unit symbol, a cell "
            f"address such as 'A1' or start with an underscore",
        stage=STAGE_PARAMETERS)


def _number(value) -> float:
    return float(getattr(value, "Value", value))


def _write_source_metadata(sheet, cell: str, source) -> None:
    """Columns D/E of a parameter's row carry the source kind and ref (Issue 003 Area 6, the kit's
    sheet contract: name | value | unit | source | note). Written only when the plan states a
    source; cells are never cleared, and the value cell (B) is untouched. Never raises."""
    kind = str((source or {}).get("kind") or "").strip()
    if not kind or not cell.startswith("B"):
        return
    row = cell[1:]
    try:
        if sheet.getContents(f"D{row}").lstrip("'").strip() != kind:
            sheet.set(f"D{row}", kind)
        ref = str((source or {}).get("ref") or "").strip()
        if ref:
            sheet.set(f"E{row}", ref[:120])
    except Exception:  # noqa: BLE001 - metadata must never fail the build
        pass


def _upsert_parameters(doc, sheet, plan: dict) -> dict:
    """Write every plan parameter into ``sheet`` by alias.

    A parameter whose alias exists keeps its cell (and is rewritten only when
    the value / formula changed); a new one takes the next free row (column A
    the name, column B the value, aliased by the name). Plain values are
    ``=`` formulas (a text ``-1.5`` would be stored as a string for a unit
    cell); derived parameters are ``=<expr>`` formulas written in dependency
    order after every plain value. Never deletes. Verifies every cell by
    reading it back. Returns ``{"added", "updated", "unchanged"}`` name lists;
    raises BuildError (stage ``parameters``)."""
    params = plan["parameters"]
    values = schema.parameter_values(plan)
    derived = schema.derived_order(plan)
    plain = [n for n in params if not schema.is_derived(params[n])]
    report = {"added": [], "updated": [], "unchanged": []}

    next_row = 1
    while sheet.getContents(f"A{next_row}") != "":
        next_row += 1
    cells: dict[str, str] = {}
    for name in params:                                   # rows in plan order
        cell = sheet.getCellFromAlias(name)
        if cell is None:
            cell = f"B{next_row}"
            sheet.set(f"A{next_row}", name)
            next_row += 1
            report["added"].append(name)
        cells[name] = cell

    def put(name: str, content: str, is_formula: bool) -> None:
        cell = cells[name]
        added = name in report["added"]
        current = sheet.getContents(cell)
        same = (current.replace(" ", "") == content.replace(" ", "") if is_formula
                else _same_number(current, values.get(name)))
        if same and not added:
            _write_source_metadata(sheet, cell, (params.get(name) or {}).get("source"))
            report["unchanged"].append(name)
            return
        sheet.set(cell, content)
        if added:
            try:
                sheet.setAlias(cell, name)
            except Exception as e:  # noqa: BLE001
                raise _alias_error(name, e) from None
        else:
            report["updated"].append(name)
        _write_source_metadata(sheet, cell, (params.get(name) or {}).get("source"))

    for name in plain:
        put(name, f"={float(values[name])!r}", False)
    for name in derived:                                  # sources first
        put(name, "=" + params[name]["expr"].strip(), True)
    doc.recompute()
    for name in plain + derived:
        want = values.get(name)
        try:
            got = _number(sheet.get(name))
        except Exception as e:  # noqa: BLE001
            raise BuildError("", f"parameter {name!r}: the spreadsheet cannot "
                                 f"evaluate it ({e})", stage=STAGE_PARAMETERS) from None
        if want is None or abs(got - want) > 1e-6 * max(1.0, abs(want)):
            raise BuildError(
                "", f"parameter {name!r}: the spreadsheet holds {got:g} but the "
                    f"plan says {want if want is None else format(want, 'g')}; "
                    f"check the value or formula (FreeCAD evaluates '^' and "
                    f"unit-less numbers its own way)", stage=STAGE_PARAMETERS)
    return report


def _same_number(content: str, want) -> bool:
    try:
        return want is not None and abs(float(content.lstrip("=")) - want) <= 1e-12 * max(1.0, abs(want))
    except (TypeError, ValueError):
        return False


def build_spreadsheet(doc, plan: dict, name: str = "Parameters"):
    """Create the parameter sheet: one row per parameter, column A the name,
    column B the value aliased by the name, derived parameters as formulas.

    Returns the sheet. A failure removes the half-written sheet and
    raises BuildError (stage ``parameters``) naming the parameter."""
    sheet = doc.addObject("Spreadsheet::Sheet", name)
    sheet.Label = name
    try:
        tag_object(sheet, plan.get("id", ""))
        sheet.Label2 = PLAN_MARK
        _upsert_parameters(doc, sheet, plan)
    except BuildError:
        _drop(doc, sheet)
        raise
    except Exception as e:  # noqa: BLE001
        _drop(doc, sheet)
        raise BuildError("", f"writing the parameter spreadsheet failed: "
                             f"{type(e).__name__}: {e}", stage=STAGE_PARAMETERS) from None
    return sheet


def _sheet_cells(sheet) -> dict[str, str]:
    """Every used cell of ``sheet`` → its content (a snapshot for
    ``_restore_sheet``)."""
    return {addr: sheet.getContents(addr) for addr in sheet.getUsedCells()}


def _restore_sheet(doc, sheet, before: dict[str, str]) -> None:
    """Put ``sheet`` back to the ``_sheet_cells`` snapshot ``before``. A
    console-mode abortTransaction undoes nothing, so a failed call restores
    the cells its parameter sync wrote by hand; with an undo stack the abort
    already did and this finds nothing to do."""
    changed = False
    for addr in list(sheet.getUsedCells()):
        if addr not in before:
            sheet.clear(addr)           # drops the alias with the cell
            changed = True
    for addr, content in before.items():
        if sheet.getContents(addr) != content:
            sheet.set(addr, content)
            changed = True
    if changed:
        try:
            doc.recompute()
        except Exception:  # noqa: BLE001
            pass


def _changed_names(plan: dict, report: dict) -> list[str]:
    """Parameters a sync wrote (added or updated), in plan order."""
    wrote = set(report.get("added", ())) | set(report.get("updated", ()))
    return [n for n in plan["parameters"] if n in wrote]


def _preflight_parameters(doc, plan: dict) -> None:
    """Dry-run the parameter sheet on a throw-away sheet: raises the
    BuildError (stage ``parameters``) the real one would, BEFORE anything of a
    previous build is removed. A document without an undo stack cannot take
    back the removal of the previous build, and a name FreeCAD refuses as an
    alias only shows when the cell is written."""
    probe = doc.addObject("Spreadsheet::Sheet", "OscarParameterProbe")
    try:
        _upsert_parameters(doc, probe, plan)
    except BuildError:
        raise
    except Exception as e:  # noqa: BLE001
        raise BuildError("", f"writing the parameter spreadsheet failed: "
                             f"{type(e).__name__}: {e}", stage=STAGE_PARAMETERS) from None
    finally:
        _drop(doc, probe)


def sync_parameters(doc, plan: dict, sheet_name: str = "Parameters") -> dict:
    """Upsert the plan's parameters into the existing spreadsheet.

    Used before re-applying a feature of a patched plan: a changed value or
    formula is rewritten in its cell, a new parameter takes the next free row,
    nothing is deleted. Returns the result contract; on success ``added`` /
    ``updated`` / ``unchanged`` list parameter names."""
    sheet = doc.getObject(sheet_name)
    if sheet is None or sheet.TypeId != "Spreadsheet::Sheet":
        return failure(STAGE_DOCUMENT, f"no spreadsheet {sheet_name!r} in the "
                       f"document; apply the whole plan first")
    ok, errors, _warnings = schema.validate(plan)
    if not ok:
        return failure(STAGE_VALIDATION, errors=errors)
    cells = _sheet_cells(sheet)
    try:
        report = _upsert_parameters(doc, sheet, plan)
    except BuildError as e:
        _restore_sheet(doc, sheet, cells)       # a refused name leaves nothing behind
        return failure(e.stage, e.message)
    except Exception as e:  # noqa: BLE001
        _restore_sheet(doc, sheet, cells)
        return failure(STAGE_PARAMETERS, f"{type(e).__name__}: {e}")
    return success(spreadsheet=sheet.Name,
                   parameters_changed=_changed_names(plan, report), **report)


def _drop(doc, obj) -> None:
    try:
        doc.removeObject(obj.Name)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# rollback and tip


def _restore_tip(body) -> None:
    """Point the Body's Tip at its last solid feature (None when it has none)."""
    try:
        feats = [o for o in body.Group if o.isDerivedFrom("PartDesign::Feature")]
        body.Tip = feats[-1] if feats else None
    except Exception:  # noqa: BLE001
        pass


def _remove_created(doc, body, before: set[str]) -> None:
    """Console-mode abortTransaction does not undo object creation — delete
    anything the failed feature added, newest first, restore the body tip
    to the last surviving feature, and recompute."""
    created = [o for o in doc.Objects if o.Name not in before]
    for obj in reversed(created):
        try:
            doc.removeObject(obj.Name)
        except Exception:  # noqa: BLE001
            pass
    _restore_tip(body)
    try:
        doc.recompute()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# one feature


def _build_one(doc, body, plan: dict, ctx: Ctx, feat: dict, built: dict,
               prelude=None):
    """Build ONE plan feature inside its own transaction.

    ``prelude`` (optional) runs first INSIDE that transaction - the sheet sync
    and the removal of the stale objects of a re-applied feature - so one undo
    step takes all of it back. Returns ``(object, info)``; raises BuildError
    naming the feature (a recompute error, a failed post-condition with its
    metrics, or any unexpected exception) after rolling back whatever the
    prelude and the feature created."""
    fid, op = feat["id"], feat["op"]
    solid = op in _featurebuild.SOLID_BUILDERS
    if op != "sketch" and not solid:
        raise BuildError(fid, f"op {op!r} unknown (have: sketch, "
                              f"{', '.join(_featurebuild.SOLID_BUILDERS)})")
    if "sketch" in feat and feat["sketch"] not in built:
        raise BuildError(fid, f"sketch {feat['sketch']!r} is not in the "
                              f"document; re-apply the whole plan")
    names0 = {o.Name for o in doc.Objects}
    try:
        with checkpoint(doc, f"Oscar: plan {fid}"):
            if prelude is not None:
                prelude()
            before_names = {o.Name for o in doc.Objects}
            before = body_metrics(body) if solid else None
            if op == "sketch":
                obj, info = _sketchbuild.build_sketch(doc, body, feat, plan, ctx)
            else:
                obj, info = _featurebuild.SOLID_BUILDERS[op](
                    doc, body, feat, plan, built, ctx)
            doc.recompute()
            made = [o.Name for o in doc.Objects if o.Name not in before_names]
            state = doc_errors(doc, made)
            if state:
                raise BuildError(fid, f"recompute failed: {state}",
                                 cause=CAUSE_RECOMPUTE_FAILED)
            info = dict(info or {})
            if before is not None:
                info["metrics"] = check_postconditions(
                    op, fid, before, body_metrics(body),
                    kind=_post_kind_of(plan, feat),
                    context=_cut_context(op, feat, plan, built, info, doc))
        _dressbuild.settle(doc, body, obj, built)   # after the commit: see _dressbuild
    except BuildError as e:
        _remove_created(doc, body, names0)
        if e.stage != STAGE_PARAMETERS:     # a parameter failure is not this feature's
            e.feature_id = e.feature_id or fid
        raise
    except Exception as e:  # noqa: BLE001 - same contract, raw message
        _remove_created(doc, body, names0)
        raise BuildError(fid, f"{type(e).__name__}: {e}") from None
    return obj, info


def _cut_context(op: str, feat: dict, plan: dict, built: dict, info: dict, doc):
    """The computed facts of a cut-nothing error (Issue 003): the drill direction
    (holes/pockets cut along the plane's NEGATIVE normal) and the consumed profile's
    global box. ``None`` for anything but a subtractive op, or when the pieces are
    unavailable — the error then falls back to the generic hint. Never raises."""
    if op not in ("pocket", "hole"):
        return None
    try:
        basis = schema.sketch_basis(plan, feat.get("plane", {}))
        context = {"drill": [-float(v) for v in basis["normal"]]}
    except Exception:  # noqa: BLE001 - the enrichment must never fail the build
        return None
    sketch = None
    if op == "hole":
        name = str((info or {}).get("sketch") or "")
        sketch = doc.getObject(name) if name else None
    else:
        made = built.get(feat.get("sketch") or "")
        sketch = made[0] if made else None
    try:
        box = sketch.Shape.BoundBox
        context["profile_bbox"] = [round(v, 3) for v in
                                   (box.XMin, box.YMin, box.ZMin, box.XMax, box.YMax, box.ZMax)]
    except Exception:  # noqa: BLE001 - a shapeless sketch just loses its box line
        pass
    return context


def _post_kind_of(plan: dict, feat: dict) -> str | None:
    """The post-condition kind of ``feat``: its registry kind, or - for a
    pattern / mirror (``follow``) - the kind of the feature it repeats."""
    spec = schema.OPS.get(feat.get("op"))
    if spec is None or spec.post != "follow":
        return None
    for other in schema.all_features(plan):
        if other.get("id") == feat.get("of") and other.get("op") in schema.OPS:
            kind = schema.OPS[other["op"]].post
            return kind if kind in ("additive", "subtractive") else "subtractive"
    return "subtractive"


def _run_features(doc, body, plan: dict, ctx: Ctx, features: list, built: dict,
                  out: list, prelude=None) -> BuildError | None:
    """Build ``features`` in order into ``built`` / ``out``; the BuildError of
    the first failure (document rolled back to the last good feature) or None.
    ``prelude`` runs inside the transaction of the first feature."""
    for feat in features:
        try:
            obj, info = _build_one(doc, body, plan, ctx, feat, built, prelude)
        except BuildError as e:
            return e
        prelude = None
        built[feat["id"]] = (obj, info)
        out.append({"id": feat["id"], "op": feat["op"], "object": obj.Name,
                    **({"info": info} if info else {})})
    return None


def _failed(err: BuildError, completed: list[str], **extra) -> dict:
    """The failure result of a BuildError (``completed`` = features built
    before it). A BuildError with a cause code adds it and its recovery
    alternatives to the result."""
    if err.metrics:
        extra["metrics"] = err.metrics
    if err.cause:
        extra["cause"] = err.cause
    if err.alternatives:
        extra["alternatives"] = list(err.alternatives)
    return failure(err.stage, err.message, feature_id=err.feature_id or None,
                   completed=completed, **extra)


# ---------------------------------------------------------------------------
# names and replacement


def _name_problem(label: str, name, allow_empty: bool = False) -> str | None:
    if name is None or (allow_empty and name == ""):
        return None
    if not valid_name(name):
        return (f"{label} {name!r} is not a valid object name (letters, "
                f"digits, underscore; not starting with a digit)")
    return None


def _only_parameter_rows(sheet) -> bool:
    """True while ``sheet`` holds nothing but parameter rows in the sheet
    contract the parameter tools and Issue 003 write: column A the parameter
    name, column B its value aliased by that name, C/D/E the unit, source kind
    and ref. A user who added their own cells beyond that contract owns the
    sheet from then on: the build must not touch it (see ``_pick``)."""
    rows: set[str] = set()
    for addr in sheet.getUsedCells():
        if addr[0] not in "ABCDE" or not addr[1:].isdigit():
            return False
        rows.add(addr[1:])
    for row in rows:
        name = sheet.getContents(f"A{row}").lstrip("'")
        if not name or sheet.getAlias(f"B{row}") != name:
            return False
    return True


def _pick(doc, plan_id: str, explicit, default: str, type_id: str, what: str):
    """Choose the name for the body / sheet of a plan build.

    Returns ``(name, previous_object_or_None, problem_or_None)``. An object
    this plan provably built (plan tag + PLAN_MARK) already under the name is
    the previous build (replaced); one without the stamp, such as a user's own
    ``Parameters`` sheet or ``Body``, is foreign and never touched. An
    EXPLICIT name held by a foreign object is a problem (never overwrite); a
    DEFAULT name held by a foreign object moves to the next free
    ``<default>001``-style name. EXCEPTION (M1 F1): a DEFAULT name held by a
    sheet the parameter tools wrote (``OscarRole == "params"``, the tag
    ``kit.params.ensure_sheet`` sets for ``set_parameters`` /
    ``measure_reference_part`` / the params stage) that still holds nothing
    but parameter rows is ADOPTED: returned with no previous object, so the
    caller updates it in place instead of spawning ``Parameters001`` beside
    it. A user's own hand-made sheet carries no such tag and is never
    adopted."""
    def ours(obj) -> bool:
        # a stamped sheet the user has since added their own cells to is no
        # longer disposable: it is left alone and the new build moves aside
        return _owned(obj, type_id, plan_id) and (
            type_id != "Spreadsheet::Sheet" or _only_parameter_rows(obj))

    if explicit:
        obj = doc.getObject(explicit)
        if obj is None or ours(obj):
            return explicit, obj, None
        return explicit, None, (
            f"{what} name {explicit!r} is already used by a {obj.TypeId} that "
            f"plan {plan_id!r} did not build (or that holds cells of its own "
            f"now); pass another name or remove that object")
    for n in range(0, 1000):
        name = default if n == 0 else f"{default}{n:03d}"
        obj = doc.getObject(name)
        if obj is None or ours(obj):
            return name, obj, None
        if (type_id == "Spreadsheet::Sheet" and obj.TypeId == type_id
                and str(getattr(obj, "OscarRole", "") or "") == "params"
                and _only_parameter_rows(obj)):
            return name, None, None        # adopted: updated in place, never replaced
    return default, None, f"no free {what} name near {default!r}"


def _foreign_features(body, plan_id: str) -> list[str]:
    """Names of the objects inside ``body`` that plan ``plan_id`` did not build (a user's own pad, fillet or
    sketch added by hand). The Origin objects of a Body are not features and are ignored."""
    names = []
    for obj in list(getattr(body, "Group", None) or []):
        if tag_of(obj) == plan_id or str(getattr(obj, "TypeId", "")).startswith("App::"):
            continue
        names.append(obj.Name)
    return names


def _remove_previous(doc, bodies, sheet) -> list[str]:
    """Delete a previous build of the same plan (per Body: features newest
    first, then the Body); returns the removed object names. ``bodies`` is one
    Body, a list of them, or None. The ``sheet`` is deliberately NOT deleted
    (M1 F1): a previous or adopted parameter sheet is updated in place by the
    caller, so rows another tool wrote and user-edited cells survive."""
    removed: list[str] = []
    if bodies is not None and not isinstance(bodies, (list, tuple)):
        bodies = [bodies]
    for body in bodies or []:
        for obj in reversed(list(body.Group)):
            removed.append(obj.Name)
            _drop(doc, obj)
        removed.append(body.Name)
        _drop(doc, body)
    if removed:
        try:
            doc.recompute()
        except Exception:  # noqa: BLE001
            pass
    return removed


def _tag_body(body, plan_id: str, prefix: str, part_id: str | None = None) -> None:
    tag_object(body, plan_id)
    body.Label2 = PLAN_MARK
    if part_id:
        _partbuild.stamp_part(body, part_id)
    if _PREFIX_PROP not in body.PropertiesList:
        body.addProperty("App::PropertyString", _PREFIX_PROP, "Oscar",
                         "Object-name prefix of the plan's features")
    setattr(body, _PREFIX_PROP, prefix)


def _find_tagged(doc, plan_id: str, type_id: str, preferred):
    """The ``type_id`` object plan ``plan_id`` built: ``preferred`` (a name)
    when it is one, else the first one. Only an object carrying the plan tag
    AND PLAN_MARK counts: a user's own ``Body`` / ``Parameters`` is never
    adopted, whatever its name or layout, and neither is a build saved before
    the stamp existed (it cannot be proven ours)."""
    obj = doc.getObject(preferred) if preferred else None
    if _owned(obj, type_id, plan_id):
        return obj
    for cand in doc.Objects:
        if _owned(cand, type_id, plan_id):
            return cand
    return None


# ---------------------------------------------------------------------------
# entry points


def apply_plan(doc, plan: dict, *, body_name: str | None = None,
               sheet_name: str | None = None, prefix: str = "") -> dict:
    """Apply a validated FeaturePlan to an open document.

    ``body_name`` / ``sheet_name`` (default ``Body`` / ``Parameters``, or the
    next free name when a foreign object holds it) name the Body and the
    Parameters spreadsheet; ``prefix`` goes in front of every feature object
    name. Applying the SAME plan id again replaces its previous build (the
    objects it tagged); an explicit name held by a foreign object is refused.

    Returns the result contract. Success adds ``document``, ``spreadsheet``,
    ``body`` (the real object names), ``features`` (per feature ``id``,
    ``op``, ``object``, ``info`` with the solver verdict or the volume
    metrics), ``warnings``, ``feature_count``,
    ``parameter_count`` and ``replaced``. Failure adds ``completed`` (the
    features built before it), the failing ``feature_id`` and, for a failed
    post-condition, ``metrics``; the document stays at the last good feature."""
    ok, errors, warnings = schema.validate(plan)
    if not ok:
        return failure(STAGE_VALIDATION, errors=errors, warnings=warnings)
    problem = (_name_problem("body_name", body_name)
               or _name_problem("sheet_name", sheet_name)
               or _name_problem("prefix", prefix, allow_empty=True))
    if problem:
        return failure(STAGE_DOCUMENT, problem)
    try:
        return _apply_plan(doc, plan, warnings, body_name, sheet_name, prefix or "")
    except BuildError as e:
        return _failed(e, [])
    except Exception as e:  # noqa: BLE001 - never raise into the caller
        return failure(STAGE_DOCUMENT, f"{type(e).__name__}: {e}")


def _pick_bodies(doc, plan: dict, units: list, body_name):
    """``(targets, old_bodies, problem)``: per unit the Body name to build
    under and the previous build to replace. A single-body plan keeps the v1
    rules (``body_name`` / ``Body``); a ``parts`` plan names each Body
    ``part.body`` or ``Body_<part id>``, and every OTHER Body this plan
    provably built (a part the plan no longer has) is replaced too."""
    plan_id = plan["id"]
    multi = len(units) > 1 or units[0]["part"] is not None
    if multi and body_name:
        return None, None, failure(
            STAGE_DOCUMENT, "body_name names the Body of a single-body plan; "
                            "a plan with parts names each Body with part.body")
    targets, olds = [], []
    for unit in units:
        explicit = unit["body"] or (None if multi else body_name)
        default = f"Body_{unit['part']}" if multi else "Body"
        name, old, problem = _pick(doc, plan_id, explicit, default,
                                   "PartDesign::Body", "body")
        if problem:
            return None, None, failure(STAGE_DOCUMENT, problem)
        targets.append(name)
        if old is not None:
            olds.append(old)
    if multi:
        for obj in doc.Objects:
            if (obj not in olds and _owned(obj, "PartDesign::Body", plan_id)):
                olds.append(obj)
    return targets, olds, None


def _apply_plan(doc, plan, warnings, body_name, sheet_name, prefix) -> dict:
    plan_id = plan["id"]
    units = _partbuild.units(plan)
    multi = len(units) > 1 or units[0]["part"] is not None
    body_targets, old_bodies, problem = _pick_bodies(doc, plan, units, body_name)
    if problem:
        return problem
    sheet_target, old_sheet, problem = _pick(
        doc, plan_id, sheet_name, "Parameters", "Spreadsheet::Sheet", "spreadsheet")
    if problem:
        return failure(STAGE_DOCUMENT, problem)
    for old_body in old_bodies:
        foreign = _foreign_features(old_body, plan_id)
        if foreign:
            # re-applying replaces the previous build; features somebody added by hand would be deleted with it
            shown = ", ".join(repr(n) for n in foreign[:5]) + (f" and {len(foreign) - 5} more" if len(foreign) > 5 else "")
            return failure(STAGE_DOCUMENT, (
                f"body {old_body.Name!r} holds {len(foreign)} feature(s) that plan {plan_id!r} did not build ({shown}); "
                "applying the plan again would delete them. Pass another body_name for a fresh build, or remove or "
                "move those features first"))
    # clearing the previous build and creating the new sheet + Body is ONE undo
    # step, so reverting the turn can never restore the old objects next to
    # untracked new ones; the parameters are dry-run first, so a name FreeCAD
    # refuses fails BEFORE the previous build is touched (a document without an
    # undo stack cannot take the removal back)
    removed: list[str] = []
    bodies: list = []
    try:
        with checkpoint(doc, "Oscar: plan setup"):
            _preflight_parameters(doc, plan)
            removed.extend(_remove_previous(doc, old_bodies, old_sheet))
            existing = doc.getObject(sheet_target)
            if existing is not None and existing.TypeId == "Spreadsheet::Sheet":
                # a previous build's sheet or an adopted plain parameter sheet
                # (set_parameters / measure wrote it first): updated in place,
                # never deleted, so foreign rows and edited cells survive
                before = _sheet_cells(existing)
                try:
                    _upsert_parameters(doc, existing, plan)
                except Exception:
                    _restore_sheet(doc, existing, before)
                    raise
                if not _marked(existing):
                    tag_object(existing, plan_id)
                    existing.Label2 = PLAN_MARK
                sheet = existing
            else:
                sheet = build_spreadsheet(doc, plan, sheet_target)
            for unit, target in zip(units, body_targets):
                body = doc.addObject("PartDesign::Body", target)
                body.Label = target
                _tag_body(body, plan_id, prefix, unit["part"])
                _partbuild.place_body(body, unit["placement"], plan, sheet.Name,
                                      unit["part"] or "")
                bodies.append(body)
            doc.recompute()
    except BuildError as e:
        return _failed(e, [], replaced=[n for n in removed if doc.getObject(n) is None])
    replaced = removed

    ctx = Ctx(sheet.Name, prefix, plan_id)
    built: dict[str, tuple] = {}
    report = {"features": [], "warnings": warnings,
              "replaced": replaced}
    for i, (unit, body) in enumerate(zip(units, bodies)):
        err = _run_features(doc, body, plan, ctx, unit["features"], built,
                            report["features"])
        if err is not None:
            for later in bodies[i + 1:]:        # parts never started leave nothing behind
                _drop(doc, later)
            extra = {"part": unit["part"]} if multi else {}
            return _failed(err, list(built), **report, **extra)
    out = success(document=doc.Name, spreadsheet=sheet.Name,
                  body=", ".join(b.Name for b in bodies),
                  feature_count=len(report["features"]),
                  parameter_count=len(plan["parameters"]), **report)
    if multi:
        by_part = {u["part"]: b for u, b in zip(units, bodies)}
        rows, notes = _partbuild.reference_report(plan, by_part)
        out["bodies"] = [b.Name for b in bodies]
        out["parts"] = [{"id": u["part"], "body": b.Name,
                         "features": [f["id"] for f in u["features"]],
                         "placement": u.get("placement_name")}
                        for u, b in zip(units, bodies)]
        out["references"] = rows
        out["warnings"] = list(warnings) + notes
    return out


def _part_body(doc, plan_id: str, part_id: str):
    """The Body plan ``plan_id`` built for part ``part_id`` (stamped with the
    plan tag, PLAN_MARK and the part id), or None."""
    for obj in doc.Objects:
        if _owned(obj, "PartDesign::Body", plan_id) and _partbuild.part_of(obj) == part_id:
            return obj
    return None


def _targets(doc, plan: dict, body_name, sheet_name, part_id=None):
    """``(body, sheet, problem)`` of an existing plan build for the
    re-apply entry points: the objects this plan provably built (an explicit
    name must be one of them, else the tagged ones by default name). A user's
    own ``Body`` / ``Parameters`` is never adopted. ``part_id`` selects the
    Body of that part of a ``parts`` plan."""
    plan_id = plan["id"]

    def pick(explicit, type_id, default, what):
        if explicit:
            obj = doc.getObject(explicit)
            if obj is not None and not _owned(obj, type_id, plan_id):
                return None, failure(
                    STAGE_DOCUMENT,
                    f"{what} {explicit!r} was not built by plan {plan_id!r} "
                    f"(or predates the build stamp); apply the whole plan "
                    f"again for a build that can be repaired")
            return obj, None
        return _find_tagged(doc, plan_id, type_id, default), None

    if part_id is not None:
        body = _part_body(doc, plan_id, part_id)
        problem = None
        if body is not None and body_name and body.Name != body_name:
            body, problem = None, failure(
                STAGE_DOCUMENT, f"body {body_name!r} is not the Body of part "
                                f"{part_id!r} (that is {body.Name!r})")
        if problem:
            return None, None, problem
        if body is None:
            return None, None, failure(
                STAGE_DOCUMENT, f"no Body for part {part_id!r} built by plan "
                                f"{plan_id!r} in the document; apply the "
                                f"whole plan first")
    else:
        body, problem = pick(body_name, "PartDesign::Body", "Body", "body")
        if problem:
            return None, None, problem
    if body is None:
        which = f" {body_name!r}" if body_name else ""
        return None, None, failure(
            STAGE_DOCUMENT, f"no Body{which} built by plan {plan_id!r} in the "
                            f"document; apply the whole plan first")
    sheet, problem = pick(sheet_name, "Spreadsheet::Sheet", "Parameters",
                          "spreadsheet")
    if problem:
        return None, None, problem
    return body, sheet, None


def _replay(doc, plan: dict, feature_id: str, body_name, sheet_name, prefix,
            rebuild_later: bool) -> dict:
    ok, errors, warnings = schema.validate(plan)
    if not ok:
        return failure(STAGE_VALIDATION, errors=errors, warnings=warnings)
    all_ids = [f["id"] for f in schema.all_features(plan)]
    if feature_id not in all_ids:
        return failure(STAGE_VALIDATION,
                       f"feature {feature_id!r} not in plan (features: {all_ids})")
    unit = next(u for u in _partbuild.units(plan)
                if feature_id in [f["id"] for f in u["features"]])
    by_id = {f["id"]: f for f in unit["features"]}
    ids = [f["id"] for f in unit["features"]]       # this part's features, in order
    body, sheet, problem = _targets(doc, plan, body_name, sheet_name, unit["part"])
    if problem:
        return problem
    if prefix is None:
        prefix = getattr(body, _PREFIX_PROP, "") if _PREFIX_PROP in body.PropertiesList else ""
    plan_id = plan["id"]
    pos = ids.index(feature_id)

    def existing(fid: str):
        return find_feature_object(doc, body, plan_id, fid, f"{prefix}{fid}")

    later = [o.Name for fid in ids[pos + 1:] if (o := existing(fid)) is not None]
    if later and not rebuild_later:
        return failure(STAGE_BUILD, f"later features exist ({', '.join(later)}) "
                       f"— re-apply the whole plan, or pass rebuild_later=true "
                       f"to rebuild this feature and every later one in place",
                       feature_id=feature_id, cause=CAUSE_DEPENDENTS_EXIST,
                       alternatives=RECOVERY_ALTERNATIVES[CAUSE_DEPENDENTS_EXIST])
    todo = ids[pos:] if rebuild_later else [feature_id]
    # every check that can reject the call comes before anything is written
    for fid in todo:
        op = by_id[fid]["op"]
        if op != "sketch" and op not in _featurebuild.SOLID_BUILDERS:
            return failure(STAGE_BUILD, f"op {op!r} unknown (have: sketch, "
                           f"{', '.join(_featurebuild.SOLID_BUILDERS)})",
                           feature_id=fid)

    created_sheet = sheet is None
    sheet_before = None if created_sheet else _sheet_cells(sheet)
    if created_sheet:   # the sheet was deleted: rebuild it from the plan
        try:
            with checkpoint(doc, f"Oscar: plan {feature_id} parameters"):
                sheet = build_spreadsheet(doc, plan, "Parameters")
        except BuildError as e:
            return _failed(e, [])
    ctx = Ctx(sheet.Name, prefix, plan_id)
    built: dict[str, tuple] = {}
    for fid in ids[:pos]:
        obj = existing(fid)
        if obj is not None:
            built[fid] = (obj, {})
    prior = [fid for fid in ids[:pos] if existing(fid) is not None]
    features = [by_id[fid] for fid in ids[pos:]] if rebuild_later \
        else [by_id[feature_id]]
    report = {"features": [],
              "warnings": warnings, "parameters": {}}

    def prelude() -> None:
        """Runs inside the first rebuilt feature's transaction: the plan is the
        source of truth, so a patched value or a new parameter reaches the
        sheet before the feature binds to it, and the stale objects of the
        features being rebuilt go (newest first; a hole before its own
        sketch). One undo step takes all of it back."""
        if not created_sheet:
            report["parameters"] = _upsert_parameters(doc, sheet, plan)
        doc.recompute()
        for fid in reversed(todo):
            for key in (fid, fid + "_sketch"):
                stale = existing(key)
                if stale is not None:
                    _drop(doc, stale)
        _restore_tip(body)
        try:
            doc.recompute()
        except Exception:  # noqa: BLE001
            pass

    err = _run_features(doc, body, plan, ctx, features, built, report["features"],
                        prelude)
    if err is not None:
        if not report["features"]:
            # the call changed nothing it can give back: a console document
            # keeps the cells the sync wrote after the abort, so put them back
            if sheet_before is not None:
                _restore_sheet(doc, sheet, sheet_before)
            elif created_sheet:
                _drop(doc, sheet)
        return _failed(err, prior + [f["id"] for f in report["features"]], **report)
    changed = _changed_names(plan, report["parameters"])
    if changed:
        report["parameters_changed"] = changed
    if not rebuild_later:
        obj = built[feature_id][0]
        return success(feature_id=feature_id, object=obj.Name,
                       document=doc.Name, body=body.Name, spreadsheet=sheet.Name,
                       completed=prior + [feature_id], **report)
    return success(feature_id=feature_id, document=doc.Name, body=body.Name,
                   spreadsheet=sheet.Name, completed=prior + [f["id"] for f in report["features"]],
                   feature_count=len(report["features"]),
                   parameter_count=len(plan["parameters"]), **report)


def apply_feature(doc, plan: dict, feature_id: str, *, body_name: str | None = None,
                  sheet_name: str | None = None, prefix: str | None = None) -> dict:
    """(Re-)apply ONE feature from a plan into an existing built document
    (the repair path).

    The plan's parameters are synced into the existing spreadsheet first
    (changed values rewritten, new parameters added), the stale object of the
    feature is replaced, and the result is the contract (success adds
    ``feature_id``, ``object``, ``completed``). Rebuilding a feature that
    later features depend on is rejected with an actionable message — use
    ``rebuild_from`` or re-apply the whole plan. ``body_name`` /
    ``sheet_name`` / ``prefix`` default to the ones the plan was built with."""
    try:
        return _replay(doc, plan, feature_id, body_name, sheet_name, prefix, False)
    except BuildError as e:
        return _failed(e, [])
    except Exception as e:  # noqa: BLE001 - never raise into the caller
        return failure(STAGE_BUILD, f"{type(e).__name__}: {e}", feature_id=feature_id)


def rebuild_from(doc, plan: dict, feature_id: str, *, body_name: str | None = None,
                 sheet_name: str | None = None, prefix: str | None = None) -> dict:
    """Rebuild ``feature_id`` and every feature after it from the plan.

    Parameters are synced first; the objects of those features are removed
    (newest first) and rebuilt in order; the features before it stay. Same
    contract and defaults as ``apply_feature``; success adds ``features`` (the
    rebuilt ones), ``feature_count`` and ``completed`` (all features present
    afterwards)."""
    try:
        return _replay(doc, plan, feature_id, body_name, sheet_name, prefix, True)
    except BuildError as e:
        return _failed(e, [])
    except Exception as e:  # noqa: BLE001 - never raise into the caller
        return failure(STAGE_BUILD, f"{type(e).__name__}: {e}", feature_id=feature_id)


def drop_feature(doc, plan: dict, feature_id: str, *, body_name: str | None = None,
                 sheet_name: str | None = None) -> dict:
    """Remove ONE provably-plan-built feature (and its ``_sketch`` twin) from an
    existing build — the refinement reject path's surgical undo.

    The feature's objects go newest-first (the feature, then its sketch), the
    Body's Tip is restored to the last surviving feature, the ORIGINAL plan's
    parameters are re-synced (any cell the dropped feature's build changed
    returns to the plan's value; a parameter the dropped feature introduced and
    this plan no longer names stays behind as an inert orphan row — nothing is
    ever deleted from the sheet), and the document recomputes. Mirrors the
    ``_replay`` prelude's stale-object sweep as a public, testable entry.
    Fails the document stage when the feature (or its build) is not in the
    document. Same result contract; success adds ``dropped`` (object names) and
    ``parameters``."""
    ok, errors, warnings = schema.validate(plan)
    if not ok:
        return failure(STAGE_VALIDATION, errors=errors, warnings=warnings)
    body, sheet, problem = _targets(doc, plan, body_name, sheet_name)
    if problem:
        return problem
    prefix = getattr(body, _PREFIX_PROP, "") if _PREFIX_PROP in body.PropertiesList else ""
    plan_id = plan["id"]

    def existing(fid: str):
        return find_feature_object(doc, body, plan_id, fid, f"{prefix}{fid}")

    targets = [o for o in (existing(feature_id), existing(feature_id + "_sketch"))
               if o is not None]
    if not targets:
        return failure(STAGE_DOCUMENT,
                       f"no feature {feature_id!r} built by plan {plan_id!r} in "
                       f"the document; nothing to drop")
    dropped: list[str] = []
    try:
        with checkpoint(doc, f"Oscar: drop {feature_id}"):
            for obj in targets:                      # newest-first: feature before sketch
                dropped.append(obj.Name)
                _drop(doc, obj)
            _restore_tip(body)
            if sheet is not None:
                _upsert_parameters(doc, sheet, plan)
            doc.recompute()
    except BuildError as e:
        return _failed(e, [])
    except Exception as e:  # noqa: BLE001 - never raise into the caller
        return failure(STAGE_BUILD, f"{type(e).__name__}: {e}", feature_id=feature_id)
    return success(feature_id=feature_id, document=doc.Name, body=body.Name,
                   spreadsheet=sheet.Name if sheet is not None else "",
                   dropped=dropped, parameters={})


def macro_text(plan: dict) -> str:
    """S8 (B-5): a FreeCAD ``.FCMacro`` that replays ``plan`` through the SAME builder
    (``apply_plan``) — a cheap, dependency-free parametric deliverable: the macro
    embeds the plan as a dict literal, so editing its values and rerunning rebuilds
    the model. Pure text generation; no document access."""
    import os
    import pprint

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return f'''# Oscar reconstruction plan macro (S8) — edit the PLAN values and rerun.
# Replays the SAME FeaturePlan the session built (oscar.plan.builder.apply_plan).
import sys

sys.path.insert(0, {root!r})
import FreeCAD as App  # noqa: E402
from oscar.plan import builder  # noqa: E402

PLAN = {pprint.pformat(plan, width=100, sort_dicts=False)}

doc = App.newDocument("OscarRecon")
report = builder.apply_plan(doc, PLAN)
print("built:" if report.get("success") else "failed:", report)
'''
