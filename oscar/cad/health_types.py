"""Health report types and their renderings.

PURE PYTHON: nothing here imports FreeCAD, so the types can be built from a stored
``to_dict()`` (Project step records, ``validation.json``, the agent's ledger) and rendered
anywhere. The FreeCAD-dependent evaluation lives in ``health_eval`` and the public facade
(re-exports plus the ``*_via_dispatch`` wrappers) in ``health``: import from ``oscar.cad.health``.

A ``HealthReport`` answers "is the model in a sound state, and what changed?". It has two
renderings for the model: ``line()`` (one line, appended to every geometry result) and
``render()`` (a compact table plus findings).
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any

SEVERITIES = ("error", "warning", "info")

# code -> (severity, trigger). Codes are API: tests, the FinishGate and NOTES match on them.
# The severity is the DEFAULT one; ``health_eval`` applies the severity adjustments
# (hidden reference parts and transient build states become warnings) without changing codes.
FINDING_CODES = {
    "RECOMPUTE_ERROR": ("error", "obj.State contains 'Invalid' or 'Error' after recompute; message carries obj.getStatusString()"),
    "TOUCHED": ("warning", "obj.State still 'Touched' after one recompute attempt AND not 'Invalid' (an Invalid object stays Touched forever; it is reported once, as RECOMPUTE_ERROR)"),
    "NULL_SHAPE": ("error", "a Part-type object (not a container / sheet / sketch / Origin feature) whose Shape is null"),
    "INVALID_SHAPE": ("error", "shape.isValid() is False"),
    "NOT_CLOSED": ("error", "a solid with isClosed() False"),
    "NONPOSITIVE_VOLUME": ("error", "solid shape with Volume <= 0"),
    "FRAGMENTS": ("error", "role-tagged shell/lid (OscarRole) with more than one solid; for an untagged compound: warning when the smallest solid is < 5% of the largest"),
    "NOT_SOLID": ("error", "role-tagged shell/lid whose ShapeType is not 'Solid' (Compound of one solid is accepted); untagged: info"),
    "NO_GEOMETRIC_CHANGE": ("warning", "Part::Cut / Part::Fuse / Part::Common / MultiFuse result volume equals its Base volume within 1e-6 relative"),
    "BOOLEAN_INPUT_MISSING": ("error", "Part::Cut/Fuse/Common with Base or Tool None, MultiFuse/MultiCommon with an empty Shapes"),
    "SKETCH_CONFLICT": ("error", "Sketcher::SketchObject with ConflictingConstraints non-empty (solve() returned -3) OR solve() < 0 (a degenerate sketch returns -1 with an EMPTY conflict list; only obj.State shows it)"),
    "SKETCH_REDUNDANT": ("warning", "RedundantConstraints or PartiallyRedundantConstraints non-empty"),
    "SKETCH_MALFORMED": ("error", "MalformedConstraints non-empty"),
    "SKETCH_UNDERCONSTRAINED": ("info", "FullyConstrained is False (the row carries DoF)"),
    "EXPRESSION_ERROR": ("error", "an ExpressionEngine entry whose evalExpression raises, or Params alias missing, or a referenced Params alias whose sheet.get(alias) is a str / non-finite number"),
}

TRANSIENT_SUFFIX = " (transient while building; checked again at finish)"


def _num(value: Any, ndigits: int = 3) -> float | None:
    """A finite float rounded to ``ndigits``, else None (JSON-safe: no NaN / inf)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    r = round(f, ndigits)
    return 0.0 if r == 0 else r   # no "-0.0" for a rounding-noise volume


@dataclass
class Finding:
    """One problem (or note) found on one object. ``message`` is ONE line, ``<what>; <what to do>``,
    WITHOUT the object name (renderers prefix it: ``<object>: <message>``)."""

    severity: str            # SEVERITIES
    code: str                # key of FINDING_CODES
    object: str              # object Name, "" when document-level
    message: str

    def key(self) -> tuple[str, str]:
        """``(code, object)``: the identity used to tell a NEW problem from a persisting one."""
        return (self.code, self.object)

    def to_dict(self) -> dict:
        """JSON-safe form (stored in step records and ``validation.json``)."""
        return {"severity": self.severity, "code": self.code, "object": self.object, "message": self.message}

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        """Inverse of ``to_dict``; missing fields get neutral defaults."""
        return cls(severity=str(d.get("severity", "error")), code=str(d.get("code", "")),
                   object=str(d.get("object", "")), message=str(d.get("message", "")))


@dataclass
class ObjRow:
    """One geometry object (or one object that carries a finding)."""

    name: str
    label: str
    type: str                                   # TypeId
    state: list[str]                            # obj.State
    status: str                                 # obj.getStatusString()
    shape_type: str                             # "Solid" | "Compound" | "Shell" | "Face" | "Mesh" | "" (no shape)
    valid: bool | None
    solids: int
    closed: bool | None
    volume: float | None                        # mm3, 3 decimals
    bbox: list[float] | None                    # [xmin, ymin, zmin, xmax, ymax, zmax] in the GLOBAL frame, mm, 3 decimals
    visible: bool | None
    dof: int | None                             # sketches: obj.DoF, else None
    role: str = ""                              # OscarRole tag or ""
    notes: list[str] = field(default_factory=list)   # e.g. "isValid skipped (budget)"
    consumed: bool = False                      # an input of a boolean / body feature (omitted from the compact table unless it has a finding)

    def to_dict(self) -> dict:
        """JSON-safe form of the row."""
        return {"name": self.name, "label": self.label, "type": self.type, "state": list(self.state),
                "status": self.status, "shape_type": self.shape_type, "valid": self.valid,
                "solids": self.solids, "closed": self.closed, "volume": self.volume,
                "bbox": list(self.bbox) if self.bbox is not None else None, "visible": self.visible,
                "dof": self.dof, "role": self.role, "notes": list(self.notes), "consumed": self.consumed}

    @classmethod
    def from_dict(cls, d: dict) -> "ObjRow":
        """Inverse of ``to_dict``; tolerant of missing fields."""
        bbox = d.get("bbox")
        return cls(name=str(d.get("name", "")), label=str(d.get("label", "")), type=str(d.get("type", "")),
                   state=[str(s) for s in (d.get("state") or [])], status=str(d.get("status", "")),
                   shape_type=str(d.get("shape_type", "")), valid=d.get("valid"),
                   solids=int(d.get("solids") or 0), closed=d.get("closed"), volume=d.get("volume"),
                   bbox=[float(x) for x in bbox] if bbox else None, visible=d.get("visible"),
                   dof=d.get("dof"), role=str(d.get("role", "")),
                   notes=[str(n) for n in (d.get("notes") or [])], consumed=bool(d.get("consumed", False)))


@dataclass
class ObjectDelta:
    """What a call changed in one document (computed from two ``StateSnapshot`` s)."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[dict] = field(default_factory=list)   # {"name", "props": ["Shape","Placement","Expression","Properties"], "volume_before", "volume_after"}

    def is_empty(self) -> bool:
        """True when nothing was added, removed or changed."""
        return not (self.added or self.removed or self.changed)

    def line(self, max_names: int = 3) -> str:
        """``+Lid +Ref_RPI4B ~Shell(vol 17020.0->18432.1)``; at most ``max_names`` entries, then ``(+N more)``.
        ``""`` when empty. Order: added, removed, changed."""
        parts = [f"+{n}" for n in self.added] + [f"-{n}" for n in self.removed]
        for c in self.changed:
            before, after = c.get("volume_before"), c.get("volume_after")
            if before is not None and after is not None and f"{before:.1f}" != f"{after:.1f}":
                parts.append(f"~{c.get('name', '')}(vol {before:.1f}->{after:.1f})")
            else:
                parts.append(f"~{c.get('name', '')}")
        if not parts:
            return ""
        limit = max(1, int(max_names))
        text = " ".join(parts[:limit])
        if len(parts) > limit:
            text += f" (+{len(parts) - limit} more)"
        return text

    def to_dict(self) -> dict:
        """JSON-safe form of the delta."""
        return {"added": list(self.added), "removed": list(self.removed),
                "changed": [dict(c) for c in self.changed]}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ObjectDelta":
        """Inverse of ``to_dict``; ``None`` gives an empty delta."""
        d = d or {}
        return cls(added=[str(n) for n in (d.get("added") or [])],
                   removed=[str(n) for n in (d.get("removed") or [])],
                   changed=[dict(c) for c in (d.get("changed") or [])])


@dataclass
class StateSnapshot:
    """Cheap picture of one document, taken BEFORE a mutating call so the call's delta can be computed.

    ``objects`` has an entry for EVERY model object (sheets, sketches, containers too; Origin features and
    ``_oscar_tmp*`` objects are left out): name -> {"type", "key", "volume", "bbox", "placement", "visible",
    "expr", "ref"}. ``key`` is a hashable fingerprint (shape hash + global placement + bbox for geometry,
    cell contents for spreadsheets, property values otherwise); ``volume`` / ``bbox`` are filled only for
    geometry objects outside hidden ``Ref_*`` parts. ``finding_keys`` are the ``(code, object)`` keys of the
    findings known at that time (any severity)."""

    doc: str
    objects: dict[str, dict] = field(default_factory=dict)
    finding_keys: list[tuple[str, str]] = field(default_factory=list)
    ts: float = 0.0

    def finding_key_set(self) -> set[tuple[str, str]]:
        """``finding_keys`` as a set of ``(code, object)`` string pairs."""
        return {(str(k[0]), str(k[1])) for k in self.finding_keys}


def delta_between(prev: "StateSnapshot | None", now: "StateSnapshot") -> ObjectDelta:
    """Delta of ``now`` against ``prev``: added = names not in prev, removed = names no longer present, changed =
    names whose key, rounded volume, bbox, placement or expression hash differ (visibility alone is not a change).
    Each changed entry lists WHAT changed in ``props``: ``Shape`` (geometry), ``Placement``, ``Expression``,
    ``Properties`` (spreadsheet cells and every other property)."""
    if prev is None:
        return ObjectDelta()
    added = [n for n in now.objects if n not in prev.objects]
    removed = [n for n in prev.objects if n not in now.objects]
    changed: list[dict] = []
    for name, cur in now.objects.items():
        old = prev.objects.get(name)
        if old is None:
            continue
        props: list[str] = []
        ck, ok = cur.get("key"), old.get("key")
        if ck != ok:
            if _is_shape_key(ck) and _is_shape_key(ok):
                if ck[:2] != ok[:2] or ck[3:] != ok[3:]:
                    props.append("Shape")
                if ck[2] != ok[2]:
                    props.append("Placement")
            else:
                props.append("Properties")
        if cur.get("placement") != old.get("placement") and "Placement" not in props:
            props.append("Placement")
        if "Shape" not in props and (_differs(cur.get("volume"), old.get("volume"))
                                     or _differs(cur.get("bbox"), old.get("bbox"))):
            props.insert(0, "Shape")
        if cur.get("expr") != old.get("expr"):
            props.append("Expression")
        if props:
            changed.append({"name": name, "props": props,
                            "volume_before": old.get("volume"), "volume_after": cur.get("volume")})
    return ObjectDelta(added=added, removed=removed, changed=changed)


def _differs(a: Any, b: Any) -> bool:
    """Two measured values differ; a value that was not measured (None, over budget) never counts as a change."""
    return a is not None and b is not None and a != b


def _is_shape_key(key: Any) -> bool:
    """Shape keys are ``("g", hash, placement, bbox, faces, edges)`` (geometry) or ``("m", ...)`` (mesh)."""
    return isinstance(key, tuple) and len(key) >= 4 and key[0] in ("g", "m")


@dataclass
class HealthReport:
    """The result of ``model_health`` for one document."""

    doc: str
    ts: float
    rows: list[ObjRow] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    fragments: list[dict] = field(default_factory=list)     # [{"object": "Shell", "volumes": [18000.0, 41.2, 12.9]}]
    delta: ObjectDelta | None = None
    counts: dict = field(default_factory=lambda: {"objects": 0, "solids": 0, "errors": 0, "warnings": 0})
    budget_exceeded: bool = False

    @property
    def errors(self) -> list[Finding]:
        """The findings of severity ``error``."""
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        """The findings of severity ``warning``."""
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def infos(self) -> list[Finding]:
        """The findings of severity ``info``."""
        return [f for f in self.findings if f.severity == "info"]

    @property
    def valid(self) -> bool:
        """True when there is no error finding."""
        return not self.errors

    def new_errors(self, prev: "HealthReport | StateSnapshot | None") -> list[Finding]:
        """Error findings whose ``key()`` is not among the findings ``prev`` knew (any severity); ``prev`` None -> all errors."""
        if prev is None:
            return list(self.errors)
        if isinstance(prev, HealthReport):
            known = {f.key() for f in prev.findings}
        else:
            known = prev.finding_key_set()
        return [f for f in self.errors if f.key() not in known]

    # ------------------------------------------------------------------ rendering

    def _status(self) -> str:
        e, w = len(self.errors), len(self.warnings)
        text = "VALID" if e == 0 else f"{e} ERROR{'S' if e > 1 else ''}"
        if w:
            text += f", {w} warning{'s' if w != 1 else ''}"
        return text

    def _header(self) -> str:
        c = self.counts or {}
        return (f"[health] doc '{self.doc}': {int(c.get('objects', 0))} objects, "
                f"{int(c.get('solids', 0))} solids, {self._status()}")

    def line(self) -> str:
        """One line, no newline: ``[health] doc 'PiCase': 7 objects, 3 solids, VALID · Δ +Lid ~Shell(vol 17020.0->18432.1)``."""
        text = self._header()
        if self.delta is not None and not self.delta.is_empty():
            text += f" · Δ {self.delta.line()}"
        return text

    def render(self, mode: str = "compact", max_rows: int = 12) -> str:
        """Table form: header, rows (findings first, then the delta, then by descending volume), delta, findings.
        ``mode="full"`` has no row limit, lists consumed inputs and appends the ``info`` findings."""
        full = mode == "full"
        out = [self._header()]
        with_finding = {f.object for f in self.findings if f.object and f.severity != "info"}
        in_delta: set[str] = set()
        if self.delta is not None:
            in_delta = set(self.delta.added) | {c.get("name", "") for c in self.delta.changed}
        rows = [r for r in self.rows if full or r.name in with_finding or
                (not r.consumed and (r.shape_type or r.name in in_delta))]
        order = {id(r): i for i, r in enumerate(rows)}
        rows.sort(key=lambda r: (0 if r.name in with_finding else (1 if r.name in in_delta else 2),
                                 -(r.volume or 0.0), order[id(r)]))
        shown = rows if full else rows[:max(1, int(max_rows))]
        if shown:
            out.extend(_table(shown))
        if len(shown) < len(rows):
            out.append(f"(+{len(rows) - len(shown)} more rows; use get_objects)")
        if self.delta is not None and not self.delta.is_empty():
            out.append(_delta_line(self.delta))
        order_sev = {"error": 0, "warning": 1, "info": 2}
        listed = sorted((f for f in self.findings if full or f.severity != "info"),
                        key=lambda f: order_sev.get(f.severity, 3))
        for f in listed:
            label = {"error": "ERROR ", "warning": "WARN  ", "info": "INFO "}.get(f.severity, f.severity.upper() + " ")
            where = f"{f.object}: " if f.object else ""
            out.append(f"{label}[{f.code}] {where}{f.message}")
        if self.budget_exceeded:
            out.append("(health budget exceeded: some objects were skipped; rerun validate_model)")
        return "\n".join(out)

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict:
        """JSON-safe form: ``{"doc", "ts", "valid", "counts", "findings", "rows", "delta", "fragments", "budget_exceeded"}``."""
        return {"doc": self.doc, "ts": self.ts, "valid": self.valid, "counts": dict(self.counts),
                "findings": [f.to_dict() for f in self.findings], "rows": [r.to_dict() for r in self.rows],
                "delta": self.delta.to_dict() if self.delta is not None else None,
                "fragments": [dict(x) for x in self.fragments], "budget_exceeded": self.budget_exceeded}

    @classmethod
    def from_dict(cls, d: dict) -> "HealthReport":
        """Inverse of ``to_dict``; the stored ``valid`` is ignored and recomputed from the findings."""
        delta = d.get("delta")
        return cls(doc=str(d.get("doc", "")), ts=float(d.get("ts") or time.time()),
                   rows=[ObjRow.from_dict(r) for r in (d.get("rows") or [])],
                   findings=[Finding.from_dict(f) for f in (d.get("findings") or [])],
                   fragments=[dict(x) for x in (d.get("fragments") or [])],
                   delta=ObjectDelta.from_dict(delta) if delta is not None else None,
                   counts=dict(d.get("counts") or {"objects": 0, "solids": 0, "errors": 0, "warnings": 0}),
                   budget_exceeded=bool(d.get("budget_exceeded", False)))


def _state_cell(row: ObjRow) -> str:
    for token in ("Invalid", "Error", "Touched"):
        if token in row.state:
            return token
    if row.status and row.status not in ("Valid", ""):
        return row.status.split(":")[0][:14]
    return "ok"


def _size_cell(bbox: list[float] | None) -> str:
    if not bbox or len(bbox) != 6:
        return "-"
    return " x ".join(f"{bbox[i + 3] - bbox[i]:.1f}" for i in range(3))


def _table(rows: list[ObjRow]) -> list[str]:
    w_name = max(12, *(len(r.name) for r in rows))
    w_type = max(15, *(len(r.type) for r in rows))
    w_bbox = max(22, *(len(_size_cell(r.bbox)) for r in rows))
    out = [f"{'Name':<{w_name}} {'Type':<{w_type}} {'Solids':>6}   {'Volume mm3':>10}   {'BBox mm (x y z)':<{w_bbox}} State"]
    for r in rows:
        solids = "-" if r.shape_type in ("", "Mesh") else str(r.solids)
        volume = "-" if r.volume is None else f"{r.volume:.1f}"
        out.append(f"{r.name:<{w_name}} {r.type:<{w_type}} {solids:>6}   {volume:>10}   "
                   f"{_size_cell(r.bbox):<{w_bbox}} {_state_cell(r)}")
    return out


def _names(names: list[str], limit: int = 8) -> str:
    if not names:
        return "-"
    text = ", ".join(names[:limit])
    return text + (f" (+{len(names) - limit} more)" if len(names) > limit else "")


def _delta_line(delta: ObjectDelta) -> str:
    changed = []
    for c in delta.changed[:8]:
        b, a = c.get("volume_before"), c.get("volume_after")
        if b is not None and a is not None and f"{b:.1f}" != f"{a:.1f}":
            changed.append(f"{c.get('name', '')}(volume {b:.1f}->{a:.1f})")
        else:
            changed.append(str(c.get("name", "")))
    if len(delta.changed) > 8:
        changed.append(f"(+{len(delta.changed) - 8} more)")
    return f"Δ added: {_names(delta.added)} · removed: {_names(delta.removed)} · changed: {', '.join(changed) or '-'}"
