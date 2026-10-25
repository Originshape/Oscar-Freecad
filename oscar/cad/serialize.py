"""Objects to JSON-safe dictionaries, and the compact object table of ``get_objects``.

One stable shape feeds the chat context, the bridge clients and the object
browsers: identity, state, placement, the small set of dimensional
properties models actually reason about, and — when geometry exists — the
shape statistics that let an assistant verify its own work (volume, area,
bounding box, solid count, validity).

Origin helpers (``App::Origin`` / ``App::Line`` / ``App::Plane`` / ``App::Point``) report
an infinite bounding box and a garbage volume, so they never get shape statistics and the
object table hides them unless a filter asks for them. Geometry is recognised through the
one predicate of the health API (``cad.health.is_geometry``); a private equivalent keeps
this module working while that module is absent.
"""

import time
from typing import Any

import FreeCAD as App

_DIMENSION_KEYS = (
    "Length", "Width", "Height", "Radius", "InnerRadius", "OuterRadius",
    "Diameter", "Angle", "Thickness", "Pitch", "Segments",
)

ORIGIN_TYPES = frozenset({"App::Origin", "App::Line", "App::Plane", "App::Point"})

# isValid() on a huge shape is slow (1.1 s for a 300-solid document, measured): the table
# spends at most this long on it and shows "?" afterwards; one object (detail) is never capped.
_VALID_BUDGET_S = 2.0
_VALID_FACE_LIMIT = 5000
_SHAPE_TYPES = ("Solid", "Compound", "CompSolid", "Shell")


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, App.Units.Quantity):
        return round(value.Value, 6)  # numeric value in internal units
    if isinstance(value, App.Vector):
        return {"x": round(value.x, 6), "y": round(value.y, 6), "z": round(value.z, 6)}
    if isinstance(value, App.Placement):
        rotation = value.Rotation
        axis = rotation.Axis
        return {
            "Base": {"x": value.Base.x, "y": value.Base.y, "z": value.Base.z},
            "Rotation": {
                "Axis": {"x": axis.x, "y": axis.y, "z": axis.z},
                "Angle": round(rotation.Angle * 180.0 / 3.141592653589793, 4),
            },
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value][:64]
    if isinstance(value, App.DocumentObject):
        return value.Name
    return str(value)


def _health():
    """``oscar.cad.health`` when it exists (the single ``is_geometry`` / ``global_shape``), else None."""
    try:
        from . import health
    except ModuleNotFoundError as e:
        if e.name != f"{__package__}.health":
            raise
        return None
    return health


def is_geometry(obj) -> bool:
    """True for model geometry: delegates to ``cad.health.is_geometry``.

    The private fallback applies the same rule: not an Origin feature, a non-null Shape of
    type Solid / Compound / CompSolid / Shell with a sane bounding box, or a non-empty mesh.
    GUI-thread only.
    """
    module = _health()
    if module is not None and hasattr(module, "is_geometry"):
        return bool(module.is_geometry(obj))
    if getattr(obj, "TypeId", "") in ORIGIN_TYPES:
        return False
    if getattr(obj, "TypeId", "") == "Mesh::Feature":
        try:
            return obj.Mesh.CountFacets > 0
        except Exception:  # noqa: BLE001
            return False
    shape = getattr(obj, "Shape", None)
    if shape is None:
        return False
    try:
        if shape.isNull() or shape.ShapeType not in _SHAPE_TYPES:
            return False
        box = shape.BoundBox
        return max(abs(v) for v in (box.XMin, box.XMax, box.YMin, box.YMax, box.ZMin, box.ZMax)) < 1e6
    except Exception:  # noqa: BLE001
        return False


def global_shape(obj):
    """Copy of ``obj.Shape`` placed in global space (a shape inside an ``App::Part`` is in the
    container's frame). Delegates to ``cad.health.global_shape``. GUI-thread only."""
    module = _health()
    if module is not None and hasattr(module, "global_shape"):
        return module.global_shape(obj)
    shape = obj.Shape.copy()
    shape.Placement = obj.getGlobalPlacement()
    return shape


class _Budget:
    """A wall-clock allowance shared by the objects of one listing."""

    def __init__(self, seconds: float):
        self._until = time.monotonic() + seconds

    def expired(self) -> bool:
        return time.monotonic() >= self._until


def _shape_info(obj, budget: "_Budget | None", detail: bool) -> dict[str, Any] | None:
    """Statistics of ``obj.Shape`` (None when there is no usable shape).

    Keys: ``ShapeType``, ``Solids`` (count, only for solid-bearing types), ``Valid`` (bool, or
    None when the budget skipped the check), ``Volume``, ``Area`` and ``BoundingBox`` in the
    global frame. An object inside a transformed ``App::Part`` is measured on its global shape.
    """
    shape = getattr(obj, "Shape", None)
    if shape is None or getattr(shape, "isNull", lambda: True)():
        return None
    info: dict[str, Any] = {"ShapeType": shape.ShapeType}
    try:
        info["Volume"] = round(shape.Volume, 6)
        info["Area"] = round(shape.Area, 6)
        measured = shape
        if getattr(obj, "getParentGeoFeatureGroup", lambda: None)() is not None:
            try:
                measured = global_shape(obj)
            except Exception:  # noqa: BLE001 - fall back to the container-frame shape
                measured = shape
        box = measured.BoundBox
        info["BoundingBox"] = {
            "Min": {"x": box.XMin, "y": box.YMin, "z": box.ZMin},
            "Max": {"x": box.XMax, "y": box.YMax, "z": box.ZMax},
        }
        if shape.ShapeType in ("Solid", "Compound", "CompSolid"):
            info["Solids"] = len(shape.Solids)
    except Exception:  # noqa: BLE001 - keep whatever was measured
        pass
    valid = None
    if budget is None or not budget.expired():
        try:
            if detail or len(shape.Faces) <= _VALID_FACE_LIMIT:
                valid = bool(shape.isValid())
        except Exception:  # noqa: BLE001
            valid = None
    info["Valid"] = valid
    return info


def _expressions(obj) -> dict[str, str]:
    """``{property path: expression}`` of the expression-bound properties of ``obj``."""
    try:
        return {str(path): str(expression) for path, expression in (obj.ExpressionEngine or [])}
    except Exception:  # noqa: BLE001
        return {}


def object_to_dict(obj, detail: bool = False, budget: "_Budget | None" = None) -> dict[str, Any]:
    """Serialize one document object.

    Always: ``Name``, ``Label``, ``Type``, ``State``, the dimensional properties, ``Placement``,
    ``Visible``, ``Shape`` statistics (not for Origin helpers), ``Expressions`` (bound
    properties), sketch ``DoF`` / ``FullyConstrained`` and a container's ``Group``. A recompute
    failure adds ``Status`` (``getStatusString()``). ``detail=True`` (the ``get_object`` tool)
    adds ``PropertiesList`` (every property name, so the model knows what it can edit), the
    links ``InList`` / ``OutList`` and the parent ``Body`` / ``Part``.
    """
    data: dict[str, Any] = {
        "Name": obj.Name,
        "Label": obj.Label,
        "Type": obj.TypeId,
        "State": list(getattr(obj, "State", []) or []),
    }
    try:
        status = obj.getStatusString()
        if status and status != "Valid":
            data["Status"] = status
    except Exception:  # noqa: BLE001
        pass
    props = obj.PropertiesList if hasattr(obj, "PropertiesList") else []
    for key in _DIMENSION_KEYS:
        if key in props:
            try:
                data[key] = _plain(getattr(obj, key))
            except Exception:  # noqa: BLE001
                pass
    try:
        data["Placement"] = _plain(obj.Placement)
    except Exception:  # noqa: BLE001
        pass
    try:
        data["Visible"] = bool(obj.Visibility)
    except Exception:  # noqa: BLE001
        pass
    if obj.TypeId not in ORIGIN_TYPES:
        info = _shape_info(obj, budget, detail)
        if info is not None:
            data["Shape"] = info
    expressions = _expressions(obj)
    if expressions:
        data["Expressions"] = expressions
    if obj.TypeId == "Sketcher::SketchObject":
        for key in ("DoF", "FullyConstrained", "GeometryCount"):
            try:
                data[key] = _plain(getattr(obj, key))
            except Exception:  # noqa: BLE001
                pass
    group = getattr(obj, "Group", None)
    if group:
        data["Group"] = [child.Name for child in group][:64]
    if detail:
        data["PropertiesList"] = list(props)
        data["InList"] = [o.Name for o in (getattr(obj, "InList", None) or [])][:30]
        data["OutList"] = [o.Name for o in (getattr(obj, "OutList", None) or [])][:30]
        try:
            parent = obj.getParentGeoFeatureGroup()
            if parent is not None:
                data["Parent"] = parent.Name
        except Exception:  # noqa: BLE001
            pass
    return data


def _matches(obj, needle: str) -> bool:
    return (needle in obj.Name.lower() or needle in (obj.Label or "").lower()
            or needle in obj.TypeId.lower())


def collect_objects(doc, *, filter: str = "", limit: int | None = None,
                    offset: int = 0) -> dict[str, Any]:
    """One page of a document's objects for the ``get_objects`` table. GUI-thread only.

    ``filter`` keeps objects whose Name, Label or TypeId contains the text (case-insensitive).
    Origin helpers are hidden unless the filter matches them. Statistics are computed only for
    the objects on the returned page. Returns ``{"objects": [dicts], "total": N (every
    object of the document), "matched": M (rows that pass the filter), "hidden_origin": K}``.
    """
    needle = (filter or "").strip().lower()
    total = len(doc.Objects)
    rows, hidden = [], 0
    for obj in doc.Objects:
        if needle and not _matches(obj, needle):
            continue
        if obj.TypeId in ORIGIN_TYPES and not needle:
            hidden += 1
            continue
        rows.append(obj)
    start = max(int(offset or 0), 0)
    page = rows[start:] if limit is None else rows[start:start + max(int(limit), 0)]
    budget = _Budget(_VALID_BUDGET_S)
    return {"objects": [object_to_dict(o, budget=budget) for o in page], "total": total,
            "matched": len(rows), "hidden_origin": hidden}


# --------------------------------------------------------------------------
# The compact table (pure text: no FreeCAD needed)


def _fmt(value: Any, digits: int = 1) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def _size_text(info: dict[str, Any]) -> str:
    box = info.get("BoundingBox") if info else None
    if not box:
        return "-"
    lo, hi = box["Min"], box["Max"]
    return " x ".join(_fmt(hi[a] - lo[a]) for a in ("x", "y", "z"))


def _valid_text(row: dict[str, Any]) -> str:
    states = [s for s in row.get("State", []) if s in ("Invalid", "Error")]
    if states:
        return states[0]
    valid = (row.get("Shape") or {}).get("Valid")
    if valid is False:
        return "INVALID"
    return "ok" if valid is True or "Shape" not in row else "?"


def _notes(row: dict[str, Any]) -> str:
    notes = []
    if row.get("Status"):
        notes.append(str(row["Status"])[:60])
    elif "Touched" in row.get("State", []):
        notes.append("touched")
    if "DoF" in row:
        notes.append(f"DoF {row['DoF']}")
    if row.get("Expressions"):
        notes.append(f"expr:{len(row['Expressions'])}")
    if row.get("Group"):
        notes.append(f"{len(row['Group'])} children")
    return ", ".join(notes)


def render_objects_table(doc_name: str, page: dict[str, Any], *, filter: str = "",
                         limit: int = 50, offset: int = 0) -> str:
    """The text of ``get_objects``: a header, a fixed-width table and a paging footer.

    ``page`` is ``collect_objects()`` output (or ``{"objects": [], "total": 0, ...}``). Columns:
    Name, Label, Type, Valid (ok / INVALID / Invalid / Error / ?), Solids, Shape (ShapeType),
    Volume mm3, BBox mm (x y z sizes), Vis (yes / no / -) and Notes (status text, DoF,
    expression count, child count). An empty document and an empty match each get one sentence.
    """
    rows = page.get("objects") or []
    total = int(page.get("total", len(rows)))
    matched = int(page.get("matched", len(rows)))
    hidden = int(page.get("hidden_origin", 0))
    needle = (filter or "").strip()
    if not total:
        return f"Document '{doc_name}' is open and empty."
    if not matched:
        if needle:
            return (f"No object of document '{doc_name}' matches filter '{needle}' "
                    f"({total} objects in total).")
        return f"Document '{doc_name}' has {total} objects, all of them origin helpers (hidden)."
    head = f"Document '{doc_name}': {total} objects"
    if needle:
        head += f", {matched} match filter '{needle}'"
    if hidden:
        head += f" ({hidden} origin helpers hidden)"
    cells = [["Name", "Label", "Type", "Valid", "Solids", "Shape", "Volume mm3",
              "BBox mm (x y z)", "Vis", "Notes"]]
    for row in rows:
        info = row.get("Shape") or {}
        visible = row.get("Visible")
        cells.append([
            row.get("Name", ""), row.get("Label", ""), row.get("Type", ""), _valid_text(row),
            str(info["Solids"]) if "Solids" in info else "-", info.get("ShapeType", "-"),
            _fmt(info.get("Volume")) if info.get("Solids") or info.get("ShapeType") == "Solid" else "-",
            _size_text(info), "-" if visible is None else ("yes" if visible else "no"), _notes(row),
        ])
    widths = [min(max(len(c[i]) for c in cells), cap)
              for i, cap in enumerate((28, 28, 28, 8, 6, 9, 12, 24, 3, 40))]
    lines = [head]
    for cell_row in cells:
        lines.append("  ".join(text[:widths[i]].ljust(widths[i]) for i, text in enumerate(cell_row)).rstrip())
    start = int(offset or 0)
    end = start + len(rows)
    if end < matched:
        lines.append(f"(showing rows {start + 1}-{end} of {matched}; call get_objects with "
                     f"offset={end} for more, or narrow it with filter)")
    elif start:
        lines.append(f"(showing rows {start + 1}-{end} of {matched})")
    return "\n".join(lines)
