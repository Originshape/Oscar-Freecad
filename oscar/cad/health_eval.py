"""FreeCAD-side evaluation behind ``oscar.cad.health``.

EVERYTHING HERE IS GUI-THREAD ONLY (it touches App.Document objects and shapes). Other threads use the
``*_via_dispatch`` wrappers of ``oscar.cad.health``; console runs (tests) dispatch inline.

Cost model (measured on a 300-solid / 7800-face document): ``hashCode`` 6 us, ``BoundBox`` 47 us,
``Volume`` 0.8 ms, ``isValid`` 1.5 ms per object. A per-call check must therefore be keyed on the shape
hash: ``_scan`` fingerprints every object cheaply (``key``) and the expensive facts (volume, validity,
solid volumes) live in a per-document cache that is invalidated only when an object's key changes, so a
call that touched one object pays for one object.

Everything the models read is derived from ``Finding`` s whose codes are in ``FINDING_CODES``.
"""

import math
import os
import re
import threading
import time
import zlib
from typing import Any

import FreeCAD as App

from .health_types import (
    FINDING_CODES,
    TRANSIENT_SUFFIX,
    Finding,
    HealthReport,
    ObjRow,
    StateSnapshot,
    _num,
    delta_between,
)

_ORIGIN_TYPES = frozenset({"App::Origin", "App::Line", "App::Plane", "App::Point"})
_GEOM_SHAPE_TYPES = frozenset({"Solid", "Compound", "CompSolid", "Shell"})
# containers never count as "consumers" of their members and never carry a shape of their own
_CONTAINER_TYPES = frozenset({"App::Part", "App::DocumentObjectGroup", "PartDesign::Body", "App::Origin",
                              "Spreadsheet::Sheet"})
# groups that report a compound ``Shape`` of their members in FreeCAD 1.1 (an ``App::Part`` placed at z=50 reports
# the compound of its children): they are NOT geometry, or every solid would be counted twice
_GROUP_TYPES = frozenset({"App::Part", "App::DocumentObjectGroup", "App::LinkGroup", "App::Link"})
# types that legitimately have no shape of their own (an empty PartDesign::Body is NOT among them: NULL_SHAPE reports it)
_SHAPELESS_OK = frozenset({"App::Part", "App::DocumentObjectGroup", "App::Origin", "Spreadsheet::Sheet"})
_BOOLEAN_TYPES = frozenset({"Part::Cut", "Part::Fuse", "Part::Common"})
_MULTI_TYPES = frozenset({"Part::MultiFuse", "Part::MultiCommon"})
_ROLE_PARTS = frozenset({"shell", "lid"})
_TMP_PREFIX = "_oscar_tmp"
_SKETCH = "Sketcher::SketchObject"

SNAPSHOT_BUDGET_S = 2.0
DEFAULT_HEALTH_BUDGET_S = 4.0
FACE_LIMIT = 5000                # isValid() is skipped above this many faces unless deep=True
FRAGMENT_RATIO = 0.05            # untagged compound: warn when the smallest solid is below 5% of the largest
VOLUME_EQ_REL = 1e-6             # NO_GEOMETRIC_CHANGE tolerance
BBOX_LIMIT = 1e6                 # beyond this a "bounding box" is an infinite Origin-like feature

_PROPERTY_SKIP = frozenset({"Shape", "Proxy", "Visibility", "Label2", "ExpressionEngine", "Placement",
                            "Mesh", "Points", "Constraints", "Geometry", "ViewObject"})


# ---------------------------------------------------------------------------
# small helpers


def default_budget_s() -> float:
    """Wall-clock budget of one ``model_health`` call: env ``OSCAR_HEALTH_BUDGET_S``, else 4.0 s."""
    raw = os.environ.get("OSCAR_HEALTH_BUDGET_S", "")
    try:
        value = float(raw)
        if math.isfinite(value) and value >= 0:
            return value
    except ValueError:
        pass
    return DEFAULT_HEALTH_BUDGET_S


class _Budget:
    """Monotonic wall-clock budget shared by one scan / evaluation."""

    def __init__(self, seconds: float):
        self.limit = max(0.0, float(seconds))
        self.t0 = time.monotonic()

    def exceeded(self) -> bool:
        """True once the budget is used up (a zero budget is exceeded from the start)."""
        return self.limit == 0 or time.monotonic() - self.t0 > self.limit


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """``getattr`` that also swallows the non-AttributeError exceptions FreeCAD properties can raise."""
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _derived(obj: Any, type_id: str) -> bool:
    try:
        return bool(obj.isDerivedFrom(type_id))
    except Exception:
        return False


def _is_group(obj: Any, tid: str) -> bool:
    """A container (``App::Part`` and its descendants, plain groups, links): never geometry, whatever ``Shape`` it reports."""
    return tid in _GROUP_TYPES or _derived(obj, "App::Part") or _derived(obj, "App::DocumentObjectGroup")


def _one_line(text: Any, limit: int = 320) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[:limit - 1] + "…"


def _pl_tuple(pl: Any) -> tuple | None:
    """A hashable, rounded fingerprint of a Placement (quaternion sign-normalised)."""
    if pl is None:
        return None
    try:
        b, q = pl.Base, list(pl.Rotation.Q)
        for c in q:
            if abs(c) > 1e-9:
                if c < 0:
                    q = [-x for x in q]
                break
        return (round(b.x, 4), round(b.y, 4), round(b.z, 4)) + tuple(round(x, 5) for x in q)
    except Exception:
        return None


def _bbox_ok(bb: Any) -> bool:
    """A usable bounding box: valid and finite, and not the +-1e100 box of an Origin-like feature."""
    try:
        if hasattr(bb, "isValid") and not bb.isValid():
            return False
        return max(abs(bb.XMin), abs(bb.XMax), abs(bb.YMin), abs(bb.YMax), abs(bb.ZMin), abs(bb.ZMax)) < BBOX_LIMIT
    except Exception:
        return False


def _bb_list(bb: Any) -> list[float]:
    return [round(float(x), 3) for x in (bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax)]


def _count(shape: Any, kind: str) -> int:
    try:
        return int(shape.countElement(kind))
    except Exception:
        try:
            return len(getattr(shape, kind + "s"))
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# the geometry predicates (one definition for health, serialization, Project, fit checks)


def is_geometry(obj: Any) -> bool:
    """THE predicate for "this object is model geometry". GUI-thread only.

    True for an object that is not an Origin feature (``App::Origin`` / ``App::Line`` / ``App::Plane`` /
    ``App::Point``, whose bounding box is 2e+100 and whose Volume is garbage) and not a group (``App::Part``,
    ``App::DocumentObjectGroup``, links: FreeCAD 1.1 gives an ``App::Part`` a compound ``Shape`` of its members, and
    counting it would double every solid), has a non-null ``Shape`` of type
    Solid / Compound / CompSolid / Shell and a finite bounding box (< 1e6), OR is a mesh object
    (``Mesh::Feature`` with facets: mesh imports). A ``PartDesign::Body`` IS geometry (its Shape is the Tip's);
    sheets and sketches are not. Never raises."""
    try:
        tid = _attr(obj, "TypeId", "")
        if tid in _ORIGIN_TYPES or _is_group(obj, tid):
            return False
        if _derived(obj, "Mesh::Feature"):
            return int(obj.Mesh.CountFacets) > 0
        shape = _attr(obj, "Shape")
        if shape is None or shape.isNull():
            return False
        if shape.ShapeType not in _GEOM_SHAPE_TYPES:
            return False
        return _bbox_ok(shape.BoundBox)
    except Exception:
        return False


def global_shape(obj: Any) -> Any:
    """A COPY of ``obj.Shape`` placed in the GLOBAL frame: ``s = obj.Shape.copy(); s.Placement = obj.getGlobalPlacement()``.
    A shape inside an ``App::Part`` is in the container's local frame (verified: ZMin 0 for an object in a
    container placed at z=50), so every cross-object check, every reported bbox of a tip and the Build Summary
    must go through this. The copy may be mutated by the caller (translated, intersected). GUI-thread only."""
    s = obj.Shape.copy()
    s.Placement = obj.getGlobalPlacement()
    return s


def _global_placement(obj: Any) -> Any:
    try:
        return obj.getGlobalPlacement()
    except Exception:
        return _attr(obj, "Placement")


def _global_bbox(shape: Any, gp: Any) -> list[float] | None:
    """Bounding box of ``shape`` in the global frame; no copy when the shape already carries that placement."""
    try:
        if gp is not None and _pl_tuple(shape.Placement) != _pl_tuple(gp):
            s = shape.copy()
            s.Placement = gp
            shape = s
        return _bb_list(shape.BoundBox)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# per-document cache of the expensive facts


class _Rec:
    """Expensive facts about one object's shape; valid while ``key`` is unchanged."""

    __slots__ = ("key", "shape_type", "faces", "solids", "gbbox", "volume", "valid", "valid_done", "valid_note",
                 "closed", "closed_done", "solid_volumes", "solve_code", "solve_done")

    def __init__(self, key: Any):
        self.key = key
        self.shape_type: str | None = None
        self.faces = 0
        self.solids = 0
        self.gbbox: list[float] | None = None
        self.volume: float | None = None
        self.valid: bool | None = None
        self.valid_done = False
        self.valid_note = ""
        self.closed: bool | None = None
        self.closed_done = False
        self.solid_volumes: list[float] | None = None
        self.solve_code: int | None = None
        self.solve_done = False


_LOCK = threading.RLock()        # guards dict mutation only; no dispatch is ever waited on while held
_RECS: dict[str, dict[str, _Rec]] = {}


def _doc_id(doc: Any) -> str:
    uid = _attr(doc, "Uid")
    return str(uid) if uid else str(_attr(doc, "Name", ""))


def _prune_documents() -> None:
    """Drop the caches of documents that are no longer open."""
    try:
        open_ids = {_doc_id(d) for d in App.listDocuments().values()}
    except Exception:
        return
    with _LOCK:
        for doc_id in [k for k in _RECS if k not in open_ids]:
            del _RECS[doc_id]


def clear_cache() -> None:
    """Forget every cached fact (tests; never needed in normal use)."""
    with _LOCK:
        _RECS.clear()


# ---------------------------------------------------------------------------
# scan: cheap fingerprint of every object


class _Item:
    __slots__ = ("obj", "name", "tid", "role", "ref", "shape", "mesh", "geom", "key", "gp", "placement", "expr",
                 "visible", "rec")

    def __init__(self, obj: Any, name: str, tid: str):
        self.obj = obj
        self.name = name
        self.tid = tid
        self.role = ""
        self.ref = False
        self.shape = None       # a non-null Part shape, else None
        self.mesh = False
        self.geom = False
        self.key: Any = ("x", name)
        self.gp = None
        self.placement = None
        self.expr = 0
        self.visible: bool | None = None
        self.rec: _Rec | None = None


def _crc(text: str) -> int:
    return zlib.crc32(text.encode("utf-8", "replace")) & 0xFFFFFFFF


def _simple(value: Any, depth: int = 0) -> Any:
    """A repr-stable, bounded rendering of a property value for fingerprinting."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if depth > 3:
        return ("t", type(value).__name__)
    try:
        if isinstance(value, App.Units.Quantity):
            return round(float(value.Value), 6)
        if isinstance(value, App.Vector):
            return (round(value.x, 5), round(value.y, 5), round(value.z, 5))
        if isinstance(value, App.Placement):
            return _pl_tuple(value)
        if isinstance(value, App.Rotation):
            return tuple(round(x, 5) for x in value.Q)
        if isinstance(value, App.DocumentObject):
            return ("o", value.Name)
        if isinstance(value, (list, tuple)):
            return [_simple(v, depth + 1) for v in list(value)[:64]]
        if isinstance(value, dict):
            return sorted((str(k), _simple(v, depth + 1)) for k, v in list(value.items())[:64])
    except Exception:
        pass
    return ("t", type(value).__name__)


def _digest(obj: Any, shape_hash: int | None) -> int:
    """Fingerprint of an object's property values (placement excluded: it is reported separately)."""
    parts: list[Any] = []
    for prop in _attr(obj, "PropertiesList", []) or []:
        if prop in _PROPERTY_SKIP:
            continue
        try:
            parts.append((prop, _simple(getattr(obj, prop))))
        except Exception:
            continue
    if shape_hash is not None:
        parts.append(("shape", shape_hash))
    if _attr(obj, "TypeId", "") == _SKETCH:
        try:
            parts.append(("constraints", [(c.Type, c.Name, round(float(c.Value), 6), c.First, c.FirstPos, c.Second,
                                           c.SecondPos, c.Third, c.ThirdPos, bool(c.Driving), bool(c.IsActive))
                                          for c in obj.Constraints]))
            parts.append(("geometry", int(obj.GeometryCount)))
        except Exception:
            pass
    return _crc(repr(parts))


def _sheet_key(sheet: Any) -> int:
    cells: list[Any] = []
    try:
        for addr in sorted(sheet.getUsedCells()):
            cells.append((addr, sheet.getContents(addr), sheet.getAlias(addr)))
    except Exception:
        pass
    return _crc(repr(cells))


def _expr_hash(obj: Any) -> int:
    try:
        ee = obj.ExpressionEngine
    except Exception:
        return 0
    return _crc(repr(list(ee))) if ee else 0


def _is_ref(obj: Any, name: str, role: str) -> bool:
    """Part of a hidden vendor reference: OscarRole 'ref', a ``Ref_*`` name, or a member of a ``Ref_*`` container."""
    if role == "ref" or name.startswith("Ref_"):
        return True
    try:
        node = obj
        for _ in range(8):
            parent = node.getParentGeoFeatureGroup()
            if parent is None:
                break
            if parent.Name.startswith("Ref_"):
                return True
            node = parent
        for owner in obj.InList:
            if owner.Name.startswith("Ref_") and owner.TypeId in _CONTAINER_TYPES:
                return True
    except Exception:
        pass
    return False


def _make_item(obj: Any, name: str, tid: str) -> _Item:
    item = _Item(obj, name, tid)
    try:
        item.role = str(_attr(obj, "OscarRole", "") or "")
        item.ref = _is_ref(obj, name, item.role)
        item.expr = _expr_hash(obj)
        vis = _attr(obj, "Visibility")
        item.visible = bool(vis) if vis is not None else None
        item.gp = _global_placement(obj)
        item.placement = _pl_tuple(item.gp)
        if _derived(obj, "Mesh::Feature"):
            mesh = obj.Mesh
            item.mesh = True
            facets = int(mesh.CountFacets)
            item.geom = facets > 0
            bb = mesh.BoundBox
            item.key = ("m", facets, int(mesh.CountPoints), _bb_list(bb) if _bbox_ok(bb) else None, item.placement)
            return item
        if tid == "Spreadsheet::Sheet":
            item.key = ("s", _sheet_key(obj))
            return item
        group = _is_group(obj, tid)
        shape = None if group else _attr(obj, "Shape")
        shape_hash = None
        if shape is not None and not shape.isNull():
            item.shape = shape
            shape_hash = int(shape.hashCode())
            if shape.ShapeType in _GEOM_SHAPE_TYPES:
                bb = shape.BoundBox
                if _bbox_ok(bb):
                    item.geom = True
                    item.key = ("g", shape_hash, item.placement, tuple(_bb_list(bb)),
                                _count(shape, "Face"), _count(shape, "Edge"))
                    return item
        item.key = ("p", _digest(obj, shape_hash))
    except Exception:
        item.key = ("x", name, time.monotonic())   # unfingerprintable: always "changed", never trusted from cache
    return item


def _resolve_objs(doc: Any, objs: list | None) -> list:
    if objs is None:
        return list(doc.Objects)
    out = []
    for o in objs:
        if isinstance(o, str):
            o = doc.getObject(o)
        if o is not None:
            out.append(o)
    return out


def _scan(doc: Any, objs: list | None = None) -> list[_Item]:
    """Fingerprint every model object (Origin features and ``_oscar_tmp*`` objects are ignored everywhere)."""
    _prune_documents()
    doc_id = _doc_id(doc)
    with _LOCK:
        recs = _RECS.setdefault(doc_id, {})
    items: list[_Item] = []
    for obj in _resolve_objs(doc, objs):
        name = _attr(obj, "Name", "")
        tid = _attr(obj, "TypeId", "")
        if not name or name.startswith(_TMP_PREFIX) or tid in _ORIGIN_TYPES:
            continue
        item = _make_item(obj, name, tid)
        with _LOCK:
            rec = recs.get(name)
            if rec is None or rec.key != item.key:
                rec = recs[name] = _Rec(item.key)
        item.rec = rec
        items.append(item)
    if objs is None:
        present = {i.name for i in items}
        with _LOCK:
            for stale in [n for n in recs if n not in present]:
                del recs[stale]
    return items


# ---------------------------------------------------------------------------
# lazily computed facts


def _fill(item: _Item, deep: bool, budget: _Budget, notes: list[str]) -> bool:
    """Compute (and cache) the expensive facts of a geometry item within the budget.
    Returns False when the budget cut the work short (a note says which)."""
    rec = item.rec
    complete = True
    if item.mesh:
        if rec.shape_type is None:
            mesh = item.obj.Mesh
            rec.shape_type = "Mesh"
            rec.solids = 0
            rec.valid = bool(mesh.isSolid())
            rec.valid_done = True
            if rec.valid:
                rec.volume = _num(mesh.Volume)
            rec.gbbox = _bb_list(mesh.BoundBox) if _bbox_ok(mesh.BoundBox) else None
        return True
    shape = item.shape
    if shape is None:
        return True
    try:
        if rec.shape_type is None:
            rec.shape_type = shape.ShapeType
            rec.faces = _count(shape, "Face")
            rec.solids = _count(shape, "Solid")
        if rec.gbbox is None:
            rec.gbbox = _global_bbox(shape, item.gp)
        if rec.volume is None:
            if budget.exceeded():
                complete = False
            else:
                rec.volume = _num(shape.Volume, 6)
        if rec.solids > 1 and rec.solid_volumes is None:
            if budget.exceeded():
                complete = False
            else:
                rec.solid_volumes = sorted((_num(s.Volume, 6) or 0.0 for s in shape.Solids), reverse=True)
        if not rec.closed_done and rec.solids >= 1:
            if budget.exceeded():
                complete = False
            else:
                rec.closed = bool(all(s.isClosed() for s in shape.Solids)) if rec.solids <= 200 else bool(shape.isClosed())
                rec.closed_done = True
        if not rec.valid_done:
            if rec.faces > FACE_LIMIT and not deep:
                rec.valid_note = "isValid skipped (>5000 faces; deep=true checks it)"
            elif budget.exceeded():
                complete = False
            else:
                try:
                    rec.valid = bool(shape.isValid())
                except Exception:
                    rec.valid = None
                    rec.valid_note = "isValid failed"
                rec.valid_done = True
                rec.valid_note = rec.valid_note if rec.valid is None else ""
    except Exception as e:  # a corrupt shape must not kill the whole report
        notes.append(f"facts failed: {type(e).__name__}")
        return False
    if not complete:
        notes.append("skipped (budget)")
    if rec.valid_note and "isValid" in rec.valid_note:
        notes.append(rec.valid_note)
    return complete


def _entry(item: _Item, budget: _Budget) -> dict:
    """The StateSnapshot entry of one object (volume / bbox only for geometry outside hidden references)."""
    volume = bbox = None
    if item.geom and not item.ref:
        _fill(item, False, budget, [])
        volume = _num(item.rec.volume)
        bbox = list(item.rec.gbbox) if item.rec.gbbox else None
    return {"type": item.tid, "key": item.key, "volume": volume, "bbox": bbox, "placement": item.placement,
            "visible": item.visible, "expr": item.expr, "ref": item.ref}


# ---------------------------------------------------------------------------
# findings


class _Raw:
    """A finding before severity adjustments."""

    __slots__ = ("code", "severity", "message", "transient", "volumes")

    def __init__(self, code: str, message: str, *, severity: str | None = None, transient: bool = False,
                 volumes: list[float] | None = None):
        self.code = code
        self.severity = severity or FINDING_CODES[code][0]
        self.message = _one_line(message)
        self.transient = transient          # may be demoted to a warning while the model is being built
        self.volumes = volumes


def _volumes_text(volumes: list[float], limit: int = 6) -> str:
    shown = ", ".join(f"{v:.1f}" for v in volumes[:limit])
    return shown + (", ..." if len(volumes) > limit else "")


def _is_part_type(obj: Any) -> bool:
    """Part-type object whose Shape must exist: Part::Feature descendants except 2D / sketch objects."""
    return _derived(obj, "Part::Feature") and not _derived(obj, "Part::Part2DObject")


def _boolean_findings(item: _Item) -> list[_Raw]:
    obj, tid = item.obj, item.tid
    out: list[_Raw] = []
    if tid in _BOOLEAN_TYPES:
        missing = [p for p in ("Base", "Tool") if _attr(obj, p) is None]
        if missing:
            out.append(_Raw("BOOLEAN_INPUT_MISSING",
                            f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not set; "
                            "set both Base and Tool (edit_object) before relying on this result", transient=True))
    elif tid in _MULTI_TYPES:
        if not list(_attr(obj, "Shapes") or []):
            out.append(_Raw("BOOLEAN_INPUT_MISSING",
                            "Shapes is empty; add at least two shapes (edit_object) before relying on this result",
                            transient=True))
    return out


def _base_volume(item: _Item, doc_items: dict[str, _Item], budget: _Budget) -> float | None:
    """Volume of the object a boolean starts from (Base, or the first of MultiFuse's Shapes)."""
    obj = item.obj
    base = _attr(obj, "Base") if item.tid in _BOOLEAN_TYPES else None
    if base is None and item.tid == "Part::MultiFuse":
        shapes = list(_attr(obj, "Shapes") or [])
        base = shapes[0] if shapes else None
    if base is None:
        return None
    b_item = doc_items.get(_attr(base, "Name", ""))
    if b_item is not None and b_item.shape is not None and b_item.rec is not None:
        if b_item.rec.volume is None and not budget.exceeded():
            _fill(b_item, False, budget, [])
        return b_item.rec.volume
    try:
        return _num(base.Shape.Volume, 6)
    except Exception:
        return None


def _shape_findings(item: _Item, deep: bool, budget: _Budget, doc_items: dict[str, _Item]) -> tuple[list[_Raw], list[str]]:
    """Checks that need the shape facts (valid / closed / volume / solids / role); returns (findings, row notes)."""
    out: list[_Raw] = []
    notes: list[str] = []
    rec = item.rec
    _fill(item, deep, budget, notes)
    if item.mesh:
        return out, notes
    st, solids = rec.shape_type or "", rec.solids
    if rec.valid is False:
        out.append(_Raw("INVALID_SHAPE", "shape fails the BRep validity check; rebuild it from simpler operations "
                                         "or repair it in a script (shape.fix(), removeSplitter())"))
    if solids >= 1 and rec.closed is False:
        out.append(_Raw("NOT_CLOSED", "solid is not closed (open shell); close the profile or make the faces watertight"))
    if solids >= 1 and rec.volume is not None and rec.volume <= 0:
        out.append(_Raw("NONPOSITIVE_VOLUME", f"solid volume is {rec.volume:.3f} mm3 (zero or inverted); "
                                              "check the profile orientation and the boolean order"))
    tagged = item.role in _ROLE_PARTS
    if solids > 1 and rec.solid_volumes:
        vols = rec.solid_volumes
        if tagged:
            out.append(_Raw("FRAGMENTS", f"{solids} solids ({_volumes_text(vols)} mm3), expected 1; "
                                         "fuse or delete the floating pieces", volumes=vols))
        elif vols[0] > 0 and vols[-1] < FRAGMENT_RATIO * vols[0]:
            out.append(_Raw("FRAGMENTS", f"{solids} solids ({_volumes_text(vols)} mm3), the smallest is under 5% of the "
                                         "largest: a leftover fragment? fuse or delete the floating piece",
                            severity="warning", volumes=vols))
    if tagged and solids <= 1 and not (st == "Solid" or (st == "Compound" and solids == 1)):
        out.append(_Raw("NOT_SOLID", f"'{item.role}' part is a {st or 'shape'} without a solid; "
                                     "make it a closed solid (extrude, thicken or fuse)"))
    elif not tagged and solids == 0 and st in ("Shell", "Compound", "CompSolid"):
        out.append(_Raw("NOT_SOLID", f"{st} without a solid (surface geometry); fine for imports, "
                                     "not for a printable part", severity="info"))
    if item.tid in (_BOOLEAN_TYPES | {"Part::MultiFuse"}) and rec.volume is not None and rec.volume > 0:
        base_vol = _base_volume(item, doc_items, budget)
        if base_vol and abs(rec.volume - base_vol) <= VOLUME_EQ_REL * abs(base_vol):
            how = {"Part::Cut": "cut; the tool does not touch the base",
                   "Part::Fuse": "fuse; the tool lies completely inside the base",
                   "Part::MultiFuse": "fuse; the other shapes lie completely inside the first",
                   "Part::Common": "common; the base lies completely inside the tool"}[item.tid]
            out.append(_Raw("NO_GEOMETRIC_CHANGE", f"volume unchanged by the {how}"))
    return out, notes


def _ids(values: Any) -> str:
    """Constraint numbers as ``[3, 7]`` (at most 8, then ``...``)."""
    try:
        nums = [int(v) for v in values]
    except Exception:
        return "[]"
    return "[" + ", ".join(str(n) for n in nums[:8]) + (", ..." if len(nums) > 8 else "") + "]"


_SOLVER_CODES = {-1: "degenerate geometry or contradictory constraints", -2: "redundant constraints leave it unsolvable",
                 -4: "the sketch is over-constrained", -5: "a constraint is malformed"}


def _sketch_findings(item: _Item, state: list[str]) -> tuple[list[_Raw], int | None]:
    """Solver facts of a sketch. ``solve()`` is only called for an Invalid sketch (once per fingerprint) because
    the lists below already describe the last solve and solving a broken sketch spams the console."""
    obj, rec = item.obj, item.rec
    out: list[_Raw] = []
    conflicts = list(_attr(obj, "ConflictingConstraints", []) or [])
    redundant = list(_attr(obj, "RedundantConstraints", []) or []) + list(_attr(obj, "PartiallyRedundantConstraints", []) or [])
    malformed = list(_attr(obj, "MalformedConstraints", []) or [])
    invalid = "Invalid" in state or "Error" in state
    code = rec.solve_code
    if invalid and not rec.solve_done:
        try:
            rec.solve_code = code = int(obj.solve())
        except Exception:
            code = None
        rec.solve_done = True
    if conflicts:
        out.append(_Raw("SKETCH_CONFLICT", f"conflicting constraints {_ids(conflicts)}; delete one of them "
                                           "(solver code -3)"))
    elif code is not None and code < 0:
        why = _SOLVER_CODES.get(code, "degenerate geometry or contradictory constraints")
        out.append(_Raw("SKETCH_CONFLICT", f"the solver failed (code {code}): {why}; remove the constraint you "
                                           "added last"))
    if redundant:
        out.append(_Raw("SKETCH_REDUNDANT", f"redundant constraints {_ids(redundant)}; delete the extra ones"))
    if malformed:
        out.append(_Raw("SKETCH_MALFORMED", f"malformed constraints {_ids(malformed)}; delete and re-add them"))
    dof = _attr(obj, "DoF")
    if (not invalid and _attr(obj, "FullyConstrained") is False and int(_attr(obj, "GeometryCount", 0) or 0) > 0
            and not conflicts):
        out.append(_Raw("SKETCH_UNDERCONSTRAINED", f"DoF {dof}"))
    return out, (int(dof) if isinstance(dof, (int, float)) and math.isfinite(dof) else None)


_REF_RE = re.compile(r"(?<![\w.$])(<<[^>]+>>|[A-Za-z_]\w*)\.([A-Za-z_]\w*)")


def _sheet_map(doc: Any) -> dict[str, Any]:
    """Name and Label -> spreadsheet object, for resolving ``Params.alias`` references."""
    out: dict[str, Any] = {}
    for obj in _attr(doc, "Objects", []) or []:
        if _attr(obj, "TypeId", "") == "Spreadsheet::Sheet":
            out[obj.Name] = obj
            out["<<" + str(_attr(obj, "Label", obj.Name)) + ">>"] = obj
    return out


def _expression_findings(item: _Item, sheets: dict[str, Any]) -> list[_Raw]:
    """EXPRESSION_ERROR: an expression that fails to evaluate, whose Params alias is missing, or whose Params cell
    is text or not finite. A sheet formula error such as ``=1 mm/0`` leaves the sheet and its dependents 'Valid',
    so State alone never shows it."""
    obj = item.obj
    try:
        entries = list(obj.ExpressionEngine or [])
    except Exception:
        return []
    out: list[_Raw] = []
    for path, expr in entries:
        try:
            value = obj.evalExpression(expr)
        except Exception as e:
            out.append(_Raw("EXPRESSION_ERROR", f"{path} = {expr}: {_one_line(e, 140)}; fix the expression or add "
                                                "the missing Params row (get_parameters lists them)"))
            continue
        raw = getattr(value, "Value", value)
        if isinstance(raw, float) and not math.isfinite(raw):
            out.append(_Raw("EXPRESSION_ERROR", f"{path} = {expr}: evaluates to {raw}; check the Params values it "
                                                "uses for a division by zero"))
            continue
        for sheet_tok, alias in _REF_RE.findall(str(expr)):
            sheet = sheets.get(sheet_tok)
            if sheet is None:
                continue
            try:
                cell = sheet.getCellFromAlias(alias)
                cv = sheet.get(alias) if cell else None
            except Exception:
                continue
            cvr = getattr(cv, "Value", cv)
            if isinstance(cv, str):
                out.append(_Raw("EXPRESSION_ERROR", f"{path} = {expr}: {sheet_tok}.{alias} is text '{_one_line(cv, 40)}', "
                                                    "not a number; write the cell as a formula like '=-1.5 mm'"))
            elif isinstance(cvr, float) and not math.isfinite(cvr):
                out.append(_Raw("EXPRESSION_ERROR", f"{path} = {expr}: {sheet_tok}.{alias} is not finite; "
                                                    "fix the formula in that Params cell"))
    return out


def _iter_linked(value: Any, depth: int = 0):
    """The document objects a link property value points to (Link, LinkList, LinkSub, LinkSubList shapes)."""
    if isinstance(value, App.DocumentObject):
        yield value
    elif depth < 2 and isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_linked(item, depth + 1)


def _links_to(owner: Any, obj: Any) -> bool:
    """``owner`` references ``obj`` through a LINK property (``Base`` / ``Tool`` / ``Shapes`` / ``BaseFeature``), not just
    through an expression (``InList`` also contains objects whose expression reads ``obj``, which do not consume it).
    When the link properties cannot be inspected the answer is True (the conservative, pre-refinement behaviour)."""
    try:
        for prop in owner.PropertiesList:
            try:
                if not owner.getTypeIdOfProperty(prop).startswith("App::PropertyLink"):
                    continue
                if any(o.Name == obj.Name for o in _iter_linked(getattr(owner, prop))):
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return True


def _is_consumed(obj: Any) -> bool:
    """An input of a boolean / body feature (its result is the geometry that counts), or a feature inside a Body."""
    try:
        for owner in obj.InList:
            if owner.TypeId in _CONTAINER_TYPES or _derived(owner, "Part::Part2DObject") or _is_group(owner, owner.TypeId):
                continue
            if hasattr(owner, "Shape") and _links_to(owner, obj):
                return True
        parent = obj.getParentGeoFeatureGroup()
        if parent is not None and parent.TypeId == "PartDesign::Body" and obj.TypeId != "PartDesign::Body":
            return True
    except Exception:
        pass
    return False


def _body_has_features(obj: Any) -> bool:
    try:
        return bool(obj.Group)
    except Exception:
        return False


class _Eval:
    __slots__ = ("rows", "findings", "fragments", "objects", "solids", "budget_exceeded")

    def __init__(self):
        self.rows: list[ObjRow] = []
        self.findings: list[Finding] = []
        self.fragments: list[dict] = []
        self.objects = 0
        self.solids = 0
        self.budget_exceeded = False


def _evaluate(doc: Any, items: list[_Item], *, strict: bool, deep: bool, added: set[str], budget: _Budget) -> _Eval:
    """Evaluate every item. Severity adjustments: findings on hidden reference parts are warnings; in
    non-strict mode the transient build states (BOOLEAN_INPUT_MISSING, NULL_SHAPE, an empty Body) of an object
    ADDED by the current call are warnings with a suffix, and the Invalid state they cause is not reported."""
    ev = _Eval()
    ev.objects = len(items)
    by_name = {i.name: i for i in items}
    sheets = _sheet_map(doc) if any(i.expr for i in items) else {}
    recomputed_errors: set[str] = set()
    for it in items:
        obj = it.obj
        state = [str(s) for s in (_attr(obj, "State", []) or [])]
        try:
            status = str(obj.getStatusString() or "")
        except Exception:
            status = ""
        invalid = "Invalid" in state or "Error" in state
        touched = "Touched" in state
        raws: list[_Raw] = []
        notes: list[str] = []
        dof = None
        if it.tid != "Spreadsheet::Sheet":
            raws.extend(_boolean_findings(it))
        if invalid:
            raws.append(_Raw("RECOMPUTE_ERROR", f"recompute failed ({_one_line(status, 180) or 'Invalid'}); "
                                                "fix the object's inputs, or delete it"))
            recomputed_errors.add(it.name)
        elif touched:
            raws.append(_Raw("TOUCHED", "still marked as changed after a recompute; call doc.recompute() and "
                                        "check the object's inputs"))
        if it.tid == _SKETCH:
            sk, dof = _sketch_findings(it, state)
            raws.extend(sk)
        if it.geom and not invalid:
            sh, notes = _shape_findings(it, deep, budget, by_name)
            raws.extend(sh)
        elif it.geom:
            _fill(it, deep, budget, notes)
        elif (not invalid and it.shape is None and not it.mesh and _is_part_type(obj) and it.tid not in _SHAPELESS_OK
              and not _is_group(obj, it.tid)):
            empty_body = it.tid == "PartDesign::Body" and not _body_has_features(obj)
            if not (it.tid in _BOOLEAN_TYPES | _MULTI_TYPES and any(r.code == "BOOLEAN_INPUT_MISSING" for r in raws)):
                raws.append(_Raw("NULL_SHAPE",
                                 "empty Body (no feature yet); add a sketch-based feature or an additive primitive"
                                 if empty_body else
                                 "has no shape; check that its inputs are set and valid (get_object lists the properties)",
                                 transient=True))
        raws.extend(_expression_findings(it, sheets) if it.expr else [])
        # --- severity adjustments
        transient_here = (not strict and it.name in added and any(r.transient for r in raws))
        for r in raws:
            sev, msg = r.severity, r.message
            if transient_here and r.code == "RECOMPUTE_ERROR":
                continue
            if it.ref and sev == "error":
                sev = "warning"
            elif transient_here and r.transient and sev == "error":
                sev, msg = "warning", msg + TRANSIENT_SUFFIX
            ev.findings.append(Finding(sev, r.code, it.name, msg))
            if r.code == "FRAGMENTS" and r.volumes:
                ev.fragments.append({"object": it.name, "volumes": [round(v, 3) for v in r.volumes]})
        # --- row
        has_finding = any(f.object == it.name for f in ev.findings)
        if it.geom or has_finding:
            ev.rows.append(_row(it, state, status, dof, notes))
            if it.geom and not ev.rows[-1].consumed:
                ev.solids += ev.rows[-1].solids
    ev.budget_exceeded = any("skipped (budget)" in n for r in ev.rows for n in r.notes)
    return ev


def _row(it: _Item, state: list[str], status: str, dof: int | None, notes: list[str]) -> ObjRow:
    rec = it.rec
    geom = it.geom
    shape_type = (rec.shape_type or "") if (geom or it.mesh) else ""
    if not geom and it.shape is not None:
        try:
            shape_type = it.shape.ShapeType
        except Exception:
            shape_type = ""
    return ObjRow(
        name=it.name, label=str(_attr(it.obj, "Label", it.name)), type=it.tid, state=state, status=status,
        shape_type=shape_type, valid=rec.valid if geom else None, solids=rec.solids if geom else 0,
        closed=rec.closed if geom else None, volume=_num(rec.volume) if geom else None,
        bbox=list(rec.gbbox) if (geom and rec.gbbox) else None, visible=it.visible, dof=dof, role=it.role,
        notes=list(dict.fromkeys(notes)), consumed=_is_consumed(it.obj) if geom else False)


# ---------------------------------------------------------------------------
# public (GUI-thread) API


def snapshot_state(doc: Any, prev_findings: list[tuple[str, str]] | None = None, *,
                   budget_s: float | None = None) -> StateSnapshot:
    """Cheap, hash-keyed picture of ``doc`` taken BEFORE a mutating call. GUI-thread only.

    ``objects`` has an entry for every model object; volumes / bboxes are recomputed only for geometry whose key
    changed since the previous snapshot of the document. ``finding_keys`` are ``prev_findings`` when given, else the
    keys of the findings of the document evaluated now (incremental: unchanged objects are free), so that a
    pre-existing problem is never reported as NEW by the call that follows. Budget ``budget_s`` (default 2.0 s):
    objects beyond it have no volume / bbox and no findings."""
    budget = _Budget(SNAPSHOT_BUDGET_S if budget_s is None else budget_s)
    items = _scan(doc)
    entries = {it.name: _entry(it, budget) for it in items}
    if prev_findings is None:
        ev = _evaluate(doc, items, strict=False, deep=False, added=set(), budget=budget)
        keys = list(dict.fromkeys(f.key() for f in ev.findings))
    else:
        keys = [(str(k[0]), str(k[1])) for k in prev_findings]
    return StateSnapshot(doc=str(_attr(doc, "Name", "")), objects=entries, finding_keys=keys, ts=time.time())


def _needs_recompute(doc: Any) -> bool:
    """Some object is Touched and NOT Invalid. An object that already failed to recompute stays
    ['Touched', 'Invalid'] forever, so it must not trigger (and re-fail, re-spam) a recompute on every call."""
    for obj in _attr(doc, "Objects", []) or []:
        state = _attr(obj, "State", []) or []
        if "Touched" in state and "Invalid" not in state:
            return True
    return False


def model_health(doc: Any, objs: list | None = None, prev: StateSnapshot | None = None, *, deep: bool = False,
                 budget_s: float | None = None, strict: bool = False) -> HealthReport:
    """Is the model in a sound state, and what changed?. GUI-thread only.

    ``objs`` limits the evaluation to those objects (names or objects; None = all). ``prev`` is the snapshot taken
    before the call: it provides the delta and, in NON-strict mode, tells which objects were ADDED by this call
    (their transient states are warnings). ``strict=True`` (validate_model / run_validation) keeps every code at its
    full severity. ``doc.recompute()`` runs once, only when some object is Touched and not Invalid. Time budget
    ``budget_s`` (default env ``OSCAR_HEALTH_BUDGET_S``, else 4.0 s): beyond it the remaining objects get a row
    noted ``skipped (budget)`` and ``budget_exceeded`` is set; ``isValid()`` is skipped above 5000 faces unless
    ``deep``."""
    budget = _Budget(default_budget_s() if budget_s is None else budget_s)
    if _needs_recompute(doc):
        try:
            doc.recompute()
        except Exception:
            pass   # the failing objects show up as RECOMPUTE_ERROR below
    items = _scan(doc, objs)
    added = set(n for n in (it.name for it in items) if n not in prev.objects) if isinstance(prev, StateSnapshot) else set()
    ev = _evaluate(doc, items, strict=strict, deep=deep, added=added, budget=budget)
    delta = None
    if isinstance(prev, StateSnapshot):
        now = StateSnapshot(doc=str(_attr(doc, "Name", "")), objects={it.name: _entry(it, budget) for it in items})
        if objs is not None:
            prev = StateSnapshot(doc=prev.doc, objects={n: e for n, e in prev.objects.items() if n in now.objects},
                                 finding_keys=prev.finding_keys, ts=prev.ts)
        delta = delta_between(prev, now)
    errors = sum(1 for f in ev.findings if f.severity == "error")
    warnings = sum(1 for f in ev.findings if f.severity == "warning")
    return HealthReport(doc=str(_attr(doc, "Name", "")), ts=time.time(), rows=ev.rows, findings=ev.findings,
                        fragments=ev.fragments, delta=delta,
                        counts={"objects": ev.objects, "solids": ev.solids, "errors": errors, "warnings": warnings},
                        budget_exceeded=ev.budget_exceeded)


# ---------------------------------------------------------------------------
# the parameter table (the one reader of the Params / Parameters spreadsheet)

_LITERAL_RE = re.compile(r"^=?\s*[-+]?\d+(\.\d+)?([eE][-+]?\d+)?\s*[A-Za-z°%²³/]*\s*$")
_ADDR_RE = re.compile(r"^([A-Z]+)(\d+)$")


def _cell_text(sheet: Any, addr: str) -> str:
    """Text of a cell without the leading apostrophe FreeCAD stores on text cells."""
    try:
        text = sheet.getContents(addr) or ""
    except Exception:
        return ""
    return text[1:] if text.startswith("'") else text


def _find_sheet(doc: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        obj = doc.getObject(name)
        if obj is not None and _attr(obj, "TypeId", "") == "Spreadsheet::Sheet":
            return obj
        try:
            for cand in doc.getObjectsByLabel(name):
                if _attr(cand, "TypeId", "") == "Spreadsheet::Sheet":
                    return cand
        except Exception:
            pass
    return None


def _stage_params() -> set[str]:
    """Names of the kit parameters that rebuild a stage when edited (guarded: the kit may not exist yet)."""
    try:
        from ..kit import params as kit_params
    except ModuleNotFoundError as e:
        if e.name not in ("oscar.kit", "oscar.kit.params"):
            raise
        return set()
    except Exception:
        return set()
    return set(getattr(kit_params, "STAGE_OF_PARAM", {}) or {})


def read_param_table(doc: Any, sheet_names: tuple[str, ...] = ("Params", "Parameters")) -> dict:
    """Rows of the parameter spreadsheet: ``{"sheet": "Params" | "Parameters" | "", "rows": [ {"name", "value",
    "unit", "expr", "source", "note", "cell", "bound", "needs_rebuild"} ]}``. GUI-thread only.

    A row is every ALIASED cell (column B in both layouts); the name is the alias; the unit / source / note come
    from columns C / D / E of the same row (the unit falls back to the quantity's own unit); ``value`` is read
    through ``getattr(v, "Value", v)`` (a Quantity for unit cells, a plain number for unitless ones, the raw text
    when the cell holds text); ``expr`` is the formula text (with its ``=``) when it is a real formula, ``""`` for a
    literal. ``bound`` counts the ExpressionEngine entries of the document that reference ``<sheet>.<alias>``;
    ``needs_rebuild`` marks a kit stage / count parameter that is not expression-bound (editing its cell changes
    nothing until the stage is rebuilt)."""
    sheet = _find_sheet(doc, tuple(sheet_names))
    if sheet is None:
        return {"sheet": "", "rows": []}
    refs: dict[str, int] = {}
    ref_re = re.compile(r"(?<![\w.$])(?:%s|<<%s>>)\.([A-Za-z_]\w*)" % (re.escape(sheet.Name),
                                                                       re.escape(str(_attr(sheet, "Label", sheet.Name)))))
    for obj in _attr(doc, "Objects", []) or []:
        if obj is sheet:
            continue
        try:
            entries = list(obj.ExpressionEngine or [])
        except Exception:
            continue
        for _path, expr in entries:
            for alias in set(ref_re.findall(str(expr))):
                refs[alias] = refs.get(alias, 0) + 1
    stage_params = _stage_params()
    cells: list[tuple[int, str, str]] = []
    try:
        for addr in sheet.getUsedCells():
            m = _ADDR_RE.match(addr)
            if not m:
                continue
            alias = sheet.getAlias(addr)
            if alias:
                cells.append((int(m.group(2)), m.group(1), addr))
    except Exception:
        return {"sheet": sheet.Name, "rows": []}
    rows = []
    for row_no, _col, addr in sorted(cells):
        alias = sheet.getAlias(addr)
        try:
            raw = sheet.get(alias)
        except Exception:
            raw = None
        qty = hasattr(raw, "Value")
        value: Any = getattr(raw, "Value", raw)
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        unit = _cell_text(sheet, f"C{row_no}").strip()
        if not unit and qty:
            try:
                unit = str(raw).split(" ", 1)[1].strip()
            except Exception:
                unit = ""
        content = str(sheet.getContents(addr) or "").strip()
        expr = content if content.startswith("=") and not _LITERAL_RE.match(content) else ""
        bound = refs.get(alias, 0)
        rows.append({"name": alias, "value": value, "unit": unit, "expr": expr,
                     "source": _cell_text(sheet, f"D{row_no}").strip(), "note": _cell_text(sheet, f"E{row_no}").strip(),
                     "cell": addr, "bound": bound,
                     "needs_rebuild": bool(bound == 0 and (alias in stage_params or alias.endswith("_count")))})
    return {"sheet": sheet.Name, "rows": rows}
