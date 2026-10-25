"""Shared pieces of the FeaturePlan → FreeCAD builder (private to oscar.plan).

``builder.py`` is the public face; the sketch and feature translators live in
``_sketchbuild.py`` / ``_featurebuild.py``. What they all share lives here:

* ``BuildError`` — the failing feature id + an actionable message (+ the
  metrics that explain it);
* the ONE result contract every ``builder`` entry point returns
  (``success``, ``error``, ``errors``, ``feature_id``, ``stage``) and
  ``format_build_failure`` that turns a failed result into the model-facing
  text (never the word ``None`` for a missing field);
* expression helpers that bind plan values to the spreadsheet by its REAL
  object name (``sheet.Name``: a second sheet created as "Parameters" is
  ``Parameters001``);
* body metrics + the per-feature post-conditions (volume delta sign, one
  solid, bounding box) that turn a silent no-op into a ``BuildError``.

Everything here runs against an open ``App.Document`` on the owning (GUI)
thread; nothing holds a lock and nothing touches the GUI.
"""

import math
import re
from dataclasses import dataclass
from typing import Any

from . import schema

# dynamic string properties written on the sheet, the Body and every feature
# object a plan built, so re-applying the same plan replaces them, a foreign
# object is never overwritten (idempotent apply_plan) and a feature is found
# again even when FreeCAD had to rename its object (a taken name gets a
# numeric suffix)
TAG_PROP = "OscarPlan"        # id of the plan that built the object
FEATURE_PROP = "OscarPlanFeature"  # plan feature id ("<id>_sketch" for a hole's sketch)

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# stage names of the result contract
STAGE_VALIDATION = "validation"   # the plan failed schema.validate; nothing was touched
STAGE_DOCUMENT = "document"       # document / body / sheet level problem (missing, name taken, foreign)
STAGE_PARAMETERS = "parameters"   # writing the Parameters spreadsheet failed
STAGE_BUILD = "build"             # one feature failed (feature_id says which)
STAGE_BUILT = "built"             # success


class BuildError(Exception):
    """Raised with the failing feature id + actionable message.

    ``metrics`` carries the numbers that explain a post-condition failure
    (volume before / after, solids, bbox) and ``stage`` the contract stage
    the failure belongs to."""

    def __init__(self, feature_id: str, message: str, *,
                 stage: str = STAGE_BUILD, metrics: dict | None = None):
        super().__init__(message)
        self.feature_id = feature_id
        self.message = message
        self.stage = stage
        self.metrics = dict(metrics or {})


@dataclass(frozen=True)
class Ctx:
    """What every translator needs to know about the document it builds in.

    ``sheet`` is the REAL object name of the Parameters spreadsheet (a second
    sheet created as "Parameters" is ``Parameters001``: expressions must use
    ``sheet.Name``, never an assumed literal), ``prefix`` is put in front of
    every feature object name, ``plan_id`` tags what was built."""

    sheet: str = "Parameters"
    prefix: str = ""
    plan_id: str = ""

    def name(self, feature_id: str) -> str:
        """Preferred object name of a plan feature."""
        return f"{self.prefix}{feature_id}"


# ---------------------------------------------------------------------------
# the result contract


def failure(stage: str, error: str | None = None,
            errors: list[str] | tuple[str, ...] | None = None,
            feature_id: str | None = None, **extra) -> dict:
    """A failed builder result in the one contract shape.

    Always carries ``success: False``, ``error`` (a non-empty summary: the
    given text, else the errors joined by '; '), ``errors`` (the list, never
    empty), ``feature_id`` (the failing feature or None) and ``stage``; extra
    keyword fields (``completed``, ``metrics`` …) ride along."""
    errs = [str(e) for e in (errors or []) if str(e).strip()]
    text = (error or "").strip() or "; ".join(errs) or "unknown build error"
    if not errs:
        errs = [text]
    return {"success": False, "error": text, "errors": errs,
            "feature_id": feature_id or None, "stage": stage, **extra}


def success(**extra) -> dict:
    """A successful builder result: the contract keys with empty failure
    fields (``error`` "", ``errors`` [], ``feature_id`` None, ``stage``
    "built"), plus the report fields given."""
    return {"success": True, "error": "", "errors": [], "feature_id": None,
            "stage": STAGE_BUILT, **extra}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def format_build_failure(result: dict) -> str:
    """The model-facing text of a failed builder result.

    ``Plan failed at '<feature_id or stage>': <error, or the errors joined by
    '; '>``, then — when the result has a ``completed`` list — a second line
    ``Completed features: [...]``. A missing field never prints as ``None``:
    no feature id and no stage reads ``plan``, no error text reads ``unknown
    error``. Accepts any dict that follows the contract (old-style results
    with only ``stage`` + ``errors``, or ``feature_id`` + ``error``, work
    too); anything else is reported with its repr."""
    if not isinstance(result, dict):
        return f"Plan failed: {_text(result) or 'unknown error'}"
    where = _text(result.get("feature_id")) or _text(result.get("stage")) or "plan"
    detail = _text(result.get("error"))
    if not detail:
        errs = result.get("errors")
        if isinstance(errs, (list, tuple)):
            detail = "; ".join(t for t in (_text(e) for e in errs) if t)
        else:
            detail = _text(errs)
    text = f"Plan failed at '{where}': {detail or 'unknown error'}"
    completed = result.get("completed")
    if completed is not None:
        done = list(completed) if isinstance(completed, (list, tuple, set)) \
            else [completed]
        text += f"\nCompleted features: {done}"
    return text


# ---------------------------------------------------------------------------
# names and expressions


def valid_name(name: Any) -> bool:
    """True for a string usable as a FreeCAD object name / prefix as is."""
    return isinstance(name, str) and bool(_IDENT.match(name))


def obj_name(prefix: str, feature_id: str) -> str:
    """Object name of a plan feature: the id behind the prefix."""
    return f"{prefix}{feature_id}"


def ref_expr(sheet_name: str, value: Any) -> str | None:
    """Spreadsheet expression for a plan value that names a parameter
    (``'width'`` → ``'Parameters.width'``); None for a literal number."""
    return f"{sheet_name}.{value}" if isinstance(value, str) and value else None


def term(sheet_name: str, value: Any) -> str:
    """An expression operand for a plan value: the parameter reference, or
    the literal number."""
    return ref_expr(sheet_name, value) or repr(float(value))


def center_expr(sheet_name: str, center: Any, dim: Any) -> str:
    """Expression of the lower / left edge of a box centred on ``center``
    with extent ``dim``: ``center - dim/2`` (``-dim/2`` for a centre of 0)."""
    half = f"{term(sheet_name, dim)} / 2"
    if not isinstance(center, str) and float(center) == 0.0:
        return f"-{half}"
    return f"{term(sheet_name, center)} - {half}"


def resolve(plan: dict, value: Any, what: str, feature_id: str = "") -> float:
    """A plan value that is a param name → its number; a number → itself.

    Raises BuildError (naming ``feature_id`` and ``what``) for an unknown
    name, a parameter without a usable number, or anything that is neither."""
    if isinstance(value, str):
        try:
            return schema.resolve(plan, value)
        except KeyError:
            raise BuildError(feature_id, f"{what}: unknown parameter {value!r}"
                             f"{schema.suggest(value, plan.get('parameters') or {})}")
        except ValueError as e:
            raise BuildError(feature_id, f"{what}: {e}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise BuildError(feature_id, f"{what}: expected parameter name or number, got {value!r}")


def orthonormal_placement(basis: dict):
    """The ``App.Placement`` of a sketch plane from ``schema.sketch_basis``
    output (columns x_dir, y_dir, normal; translation origin)."""
    import FreeCAD as App

    x, y, n = basis["x_dir"], basis["y_dir"], basis["normal"]
    m = App.Matrix(x[0], y[0], n[0], basis["origin"][0],
                   x[1], y[1], n[1], basis["origin"][1],
                   x[2], y[2], n[2], basis["origin"][2],
                   0, 0, 0, 1)
    return App.Placement(m)


def doc_errors(doc, built_names: list[str]) -> str:
    """First error state among freshly built objects, '' when clean.

    The text names the object, its State flags and — when FreeCAD has one —
    the status string (``Cut out of base feature failed``)."""
    for name in built_names:
        obj = doc.getObject(name)
        if obj is None:
            continue
        state = list(getattr(obj, "State", []))
        if "Invalid" in state or "Error" in state or "Touched" in state:
            try:
                status = obj.getStatusString()
            except Exception:  # noqa: BLE001 - not every object has one
                status = ""
            detail = f": {status}" if status and status != "Valid" else ""
            return f"{obj.Name} state={state}{detail}"
    return ""


# ---------------------------------------------------------------------------
# tagging (idempotent apply_plan)


def tag_object(obj, plan_id: str, feature_id: str | None = None) -> None:
    """Mark ``obj`` as built by plan ``plan_id`` (persisted string properties).

    ``feature_id`` (when given) names the plan feature the object realises,
    so ``find_feature_object`` can locate it even after FreeCAD renamed it."""
    def put(prop: str, value: str, doc: str) -> None:
        if prop not in obj.PropertiesList:
            obj.addProperty("App::PropertyString", prop, "Oscar", doc)
        setattr(obj, prop, str(value))

    put(TAG_PROP, plan_id, "Id of the feature plan that built this object")
    if feature_id is not None:
        put(FEATURE_PROP, feature_id, "Plan feature this object realises")


def tag_of(obj) -> str | None:
    """The plan id ``obj`` is tagged with, or None when it is not plan-built."""
    if obj is None or TAG_PROP not in getattr(obj, "PropertiesList", []):
        return None
    return getattr(obj, TAG_PROP) or None


def feature_of(obj) -> str | None:
    """The plan feature id ``obj`` realises, or None when untagged."""
    if obj is None or FEATURE_PROP not in getattr(obj, "PropertiesList", []):
        return None
    return getattr(obj, FEATURE_PROP) or None


def find_feature_object(doc, body, plan_id: str, feature_id: str, name: str):
    """The object realising plan feature ``feature_id`` (``<id>_sketch`` for a
    hole's sketch): the one inside ``body`` tagged with that feature id, else
    the member of ``body`` called ``name`` (an object whose tag was lost), else
    None. Only the Body's own members count: a user's object elsewhere in the
    document that happens to share the name is never taken for ours (a
    re-apply deletes what this returns)."""
    members = list(getattr(body, "Group", []) or [])
    for obj in members:
        if tag_of(obj) == plan_id and feature_of(obj) == feature_id:
            return obj
    obj = doc.getObject(name)
    if obj is not None and obj.Name in {m.Name for m in members}:
        return obj
    return None


# ---------------------------------------------------------------------------
# body metrics and per-feature post-conditions

_ADDITIVE = ("pad", "revolution")
_SUBTRACTIVE = ("pocket", "hole")


def body_metrics(body) -> dict:
    """Volume / solid count / validity / bounding box of the Body's current
    shape: ``{"volume", "solids", "valid", "bbox"}`` (``bbox`` is
    ``[xmin, ymin, zmin, xmax, ymax, zmax]`` or None). An empty Body is
    volume 0, no solids."""
    shape = getattr(body, "Shape", None)
    if shape is None or shape.isNull():
        return {"volume": 0.0, "solids": 0, "valid": True, "bbox": None}
    bb = shape.BoundBox
    try:       # the plain box of a fillet / torus surface is inflated; the optimal one is tight
        tight = shape.optimalBoundingBox(True, False)
        if tight.isValid():
            bb = tight
    except Exception:  # noqa: BLE001 - older kernels have no optimalBoundingBox
        pass
    bbox = ([bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax]
            if bb.isValid() else None)
    solids = len(shape.Solids)
    return {"volume": float(shape.Volume) if solids else 0.0,
            "solids": solids, "valid": bool(shape.isValid()), "bbox": bbox}


def _round_metrics(before: dict, after: dict) -> dict:
    def box(b):
        return [round(v, 4) for v in b] if b else None
    return {"volume_before": round(before["volume"], 4),
            "volume_after": round(after["volume"], 4),
            "volume_delta": round(after["volume"] - before["volume"], 4),
            "solids": after["solids"], "bbox": box(after["bbox"])}


def post_kind(op: str, kind: str | None = None) -> str | None:
    """How the Body is judged after ``op``: ``additive`` / ``subtractive`` /
    ``dress`` / ``any`` (the registry's ``post``; a pattern or mirror says
    ``follow`` and the caller passes the kind of its original as ``kind``)."""
    if kind:
        return kind
    spec = schema.OPS.get(op)
    return spec.post if spec is not None else None


BBOX_SLACK_MM = 0.05      # absolute slack of the bounding-box checks (kernel tolerance halo)
BBOX_SLACK_REL = 1e-3     # ... or this fraction of the largest extent, whichever is larger


def check_postconditions(op: str, feature_id: str, before: dict,
                         after: dict, kind: str | None = None) -> dict:
    """Assert what a solid feature must have done to the Body; returns the
    metrics (``volume_before/after/delta``, ``solids``, ``bbox``).

    * an ADDITIVE op (pad, revolution, boss) must ADD volume and a
      SUBTRACTIVE one (pocket, hole) must REMOVE volume (a no-op cut was
      accepted silently before); a DRESS op (fillet, chamfer) must change the
      volume and never grow the box; ``any`` (thickness) must change the
      volume. The kind comes from the op registry; a pattern / mirror is
      judged like its original (``kind``);
    * the Body stays one valid solid;
    * the bounding box never shrinks under an additive feature and never
      grows under a subtractive or dress one, and is never degenerate.

    Raises BuildError (with the metrics) naming what to change."""
    metrics = _round_metrics(before, after)
    kind = post_kind(op, kind)
    additive = kind == "additive"
    subtractive = kind == "subtractive"
    shaping = kind in ("dress", "any")

    def fail(msg: str) -> BuildError:
        return BuildError(feature_id, msg, metrics=metrics)

    if after["solids"] == 0:
        removed = (" — the cut removed all material" if before["solids"] and
                   (subtractive or shaping) else "")
        raise fail(f"{op} produced no solid{removed} (volume "
                   f"{metrics['volume_before']:g} → 0); check the sketch "
                   f"plane, the profile and the dimensions")
    if not after["valid"]:
        raise fail(f"{op} produced an invalid solid; check the profile for "
                   f"self-intersections and the dimensions for degenerate "
                   f"values")
    if after["solids"] != 1:
        raise fail(f"{op} left {after['solids']} separate solids (expected 1): "
                   f"the profile does not touch the existing body; move the "
                   f"sketch onto the body or overlap it")
    tol = max(1e-6 * max(before["volume"], after["volume"], 1.0), 1e-6)
    delta = after["volume"] - before["volume"]
    if additive and delta <= tol:
        what = "added no material" if before["solids"] else "has no volume"
        raise fail(f"{op} {what}: volume {metrics['volume_before']:g} → "
                   f"{metrics['volume_after']:g}; the profile may lie inside "
                   f"the body or the length is too small — check the sketch "
                   f"plane and 'invert'")
    if shaping and before["solids"] and abs(delta) <= tol:
        hint = {"fillet": "the size may be too small to matter, or the edges "
                          "it selected were already round",
                "chamfer": "the size may be too small to matter",
                "thickness": "the shell removed nothing: check the open "
                             "face rule and the wall"}.get(
            op, "the feature changed nothing")
        raise fail(f"{op} changed nothing: volume {metrics['volume_before']:g} "
                   f"→ {metrics['volume_after']:g}; {hint}")
    if subtractive and before["solids"] and delta >= -tol:
        raise fail(f"{op} cut nothing: volume {metrics['volume_before']:g} → "
                   f"{metrics['volume_after']:g}; the profile lies outside "
                   f"the material or cuts the wrong way — check the sketch "
                   f"plane, its position, the depth and 'invert' (holes "
                   f"drill along the plane's negative normal)")
    bb0, bb1 = before["bbox"], after["bbox"]
    if bb1 is None or any(not math.isfinite(v) for v in bb1):
        raise fail(f"{op} left no usable bounding box")
    if bb0 is not None:
        # The kernel's bounding box carries a small tolerance halo after a boolean (BRep tolerances: 0.036 mm on a 100 x 60 x 8
        # plate was seen on a CI runner, exactly 0 on a developer machine). Compare with a slack proportional to the part,
        # far below any real mistake (a 'cut' that ADDS material grows the box by whole millimetres).
        extent = max([b - a for a, b in zip(bb1[:3], bb1[3:])] + [0.0])
        grow = max(1e-6 + 1e-9 * max(abs(v) for v in bb0 + bb1), BBOX_SLACK_MM, BBOX_SLACK_REL * extent)
        if additive and any(a > b + grow for a, b in zip(bb1[:3], bb0[:3])) \
                or additive and any(a < b - grow for a, b in zip(bb1[3:], bb0[3:])):
            raise fail(f"{op} shrank the bounding box "
                       f"({[round(v, 3) for v in bb0]} → "
                       f"{[round(v, 3) for v in bb1]}); an additive feature "
                       f"must not remove material")
        if (subtractive or kind == "dress") and (
                any(a < b - grow for a, b in zip(bb1[:3], bb0[:3]))
                or any(a > b + grow for a, b in zip(bb1[3:], bb0[3:]))):
            raise fail(f"{op} grew the bounding box "
                       f"({[round(v, 3) for v in bb0]} → "
                       f"{[round(v, 3) for v in bb1]}); a cut must not add "
                       f"material")
    return metrics
