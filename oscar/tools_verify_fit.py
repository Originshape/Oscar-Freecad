"""The fit checks of ``check_fit``: the generic rows, plus the guarded call of the kit rows.

Every row is the dict ``{"id", "kind", "a", "b", "metric", "value", "limit", "status", "detail"}`` with ``status`` in
``PASS | WARN | FAIL | SKIP`` (SKIP carries its reason in ``detail``) and ``kind`` in ``interference | clearance |
retention | containment | alignment | coverage``. The check catalogue and its limits are normative:

- ``interference.shell_lid`` / ``interference.reference_shell`` / ``interference.reference_lid``: ``Part.common``
  volume in the ASSEMBLED pose, ``<= 0.05`` mm3 (``lid_lift`` from ``Params`` is subtracted from the lid);
- ``clearance.pcb_to_wall_min``: the minimum over the cavity's vertical inner wall faces of the distance from the PCB
  solid, ``>= clearance_eff - 0.05`` (floor, standoffs and the faces of port cut-outs are excluded: a plain
  ``distToShape(pcb, shell)`` is the standoff contact, always 0);
- ``clearance.lid_to_tallest``: ``(lid plate underside z - lid_lift) - reference global ZMax >= lid_gap - 0.05``;
- ``retention.lid_lift``: lift the lid straight up ``lift_mm`` on a global copy; the lid must still interfere with
  the shell (``> 0``), otherwise the hooks do not retain;
- ``reference.status``: WARN when the reference is only nominal board data.

All geometry goes through ``health.global_shape`` (a shape inside an ``App::Part`` is in the container's frame).
``run_generic`` and ``kit_rows`` are GUI-THREAD ONLY; ``select_ids`` and the row helpers are pure. FreeCAD is
imported lazily so this module can be imported anywhere.
"""

import math
import re
import time
from typing import Any, Callable, Iterable

INTERFERENCE_LIMIT = 0.05          # mm3
CLEARANCE_SLACK = 0.05             # mm
RETENTION_MIN_VOLUME = 1e-3        # mm3: below this the lifted lid does not engage anything
MAX_PARTS = 10

GENERIC_IDS = ("interference.shell_lid", "interference.reference_shell", "interference.reference_lid",
               "clearance.pcb_to_wall_min", "clearance.lid_to_tallest", "retention.lid_lift", "reference.status")
KIT_IDS = ("containment.ports", "alignment.standoffs", "hooks.engagement", "hooks.clear_of_cutouts")
ALL_IDS = GENERIC_IDS + KIT_IDS
KINDS = ("interference", "clearance", "retention", "containment", "alignment", "coverage")
_PAIR_ID = re.compile(r"^interference\.[a-z0-9_]+$")
_NOMINAL = ("none", "catalog-nominal")
_ROW_KEYS = ("id", "kind", "a", "b", "metric", "value", "limit", "status", "detail")


# ---------------------------------------------------------------------------------------------- rows (pure)


def make_row(rid: str, kind: str, a: str, b: str, metric: str, value: Any, limit: str, status: str,
             detail: str = "") -> dict:
    """One fit row; ``value`` is rounded to 3 decimals (``None`` stays 0.0 for the JSON shape)."""
    try:
        v = round(float(value), 3)
        v = 0.0 if (v == 0 or not math.isfinite(v)) else v
    except (TypeError, ValueError):
        v = 0.0
    return {"id": rid, "kind": kind, "a": str(a or ""), "b": str(b or ""), "metric": metric, "value": v,
            "limit": limit, "status": status, "detail": detail}


def skip_row(rid: str, kind: str, why: str, a: str = "", b: str = "", metric: str = "") -> dict:
    """A SKIP row: the check was not evaluated and ``why`` says so."""
    return make_row(rid, kind, a, b, metric, 0.0, "", "SKIP", why)


def normalize_row(row: Any) -> dict:
    """A row from the kit (or a test fake) with every documented key present, the status upper-cased."""
    src = row if isinstance(row, dict) else {}
    out = {k: src.get(k, "" if k not in ("value",) else 0.0) for k in _ROW_KEYS}
    out["status"] = str(out["status"]).upper() or "SKIP"
    if out["status"] not in ("PASS", "WARN", "FAIL", "SKIP"):
        out["status"] = "SKIP"
    return out


def slug(text: Any) -> str:
    """``Lid_Plate`` -> ``lid_plate``: the part of an id built from an object name."""
    s = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    return s or "part"


def select_ids(checks: Iterable[str] | None) -> tuple[set[str] | None, list[str]]:
    """The ``checks`` argument as a set of row ids (``None`` = every check) and the problems with it.

    An entry is an exact id (``interference.shell_lid``) or a family (``interference`` = every row of that kind
    prefix). A pair id of explicit ``parts`` (``interference.body_cover``) is accepted too. Unknown entries are
    all reported in one message list."""
    if checks is None:
        return None, []
    wanted: set[str] = set()
    problems: list[str] = []
    for raw in checks:
        item = str(raw or "").strip()
        if not item:
            continue
        if item in ALL_IDS:
            wanted.add(item)
        elif any(i.split(".")[0] == item for i in ALL_IDS):
            wanted.update(i for i in ALL_IDS if i.split(".")[0] == item)
            wanted.add(item)          # a family also selects the pair rows of explicit parts
        elif _PAIR_ID.match(item):
            wanted.add(item)
        else:
            problems.append(f"'{item}' is not a check id")
    if problems:
        problems.append("use one of: " + ", ".join(ALL_IDS) + " (or a family such as 'interference')")
    return wanted, problems


def wants(ids: set[str] | None, rid: str) -> bool:
    """True when row ``rid`` was selected (family prefixes included); ``ids=None`` selects everything."""
    return ids is None or rid in ids or rid.split(".")[0] in ids


# ---------------------------------------------------------------------------------------------- objects (GUI)


def _role(obj: Any) -> str:
    try:
        return str(getattr(obj, "OscarRole", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _is_tmp(obj: Any) -> bool:
    return str(getattr(obj, "Name", "")).startswith("_oscar_tmp")


def is_reference(obj: Any) -> bool:
    """A member of the reference part: ``OscarRole == "ref"``, a ``Ref_*`` name, or inside a ``Ref_*`` container
    (up to four container levels)."""
    node = obj
    for _ in range(5):
        if node is None:
            return False
        if _role(node) == "ref" or str(getattr(node, "Name", "")).startswith("Ref_"):
            return True
        try:
            node = node.getParentGeoFeatureGroup()
        except Exception:  # noqa: BLE001
            return False
    return False


def _geometry(doc: Any, health_eval: Any) -> list:
    """Part geometry of the document in object order: ``is_geometry`` objects with a real shape (no meshes, no
    temporary ``_oscar_tmp*`` objects)."""
    out = []
    for obj in doc.Objects:
        if _is_tmp(obj) or not health_eval.is_geometry(obj) or not hasattr(obj, "Shape"):
            continue
        out.append(obj)
    return out


def _consumed_by(obj: Any, others: Iterable[Any]) -> bool:
    """True when another object of ``others`` takes ``obj`` as an input (so ``obj`` is not the tip of its chain)."""
    for other in others:
        if other.Name == obj.Name:
            continue
        try:
            if any(o.Name == obj.Name for o in other.OutList):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def role_tip(doc: Any, role: str, health_eval: Any) -> Any:
    """The tip of the part tagged ``OscarRole == role``: the geometry object of that role that no other object of the
    role consumes (the visible one when several remain, else the last in object order). When nothing carries the tag an
    object labelled / named like the role (``Shell`` / ``Lid``) is used. ``None`` when there is none."""
    geo = _geometry(doc, health_eval)
    tagged = [o for o in geo if _role(o) == role and not is_reference(o)]
    if not tagged:
        wanted = role.capitalize()
        tagged = [o for o in geo if (o.Label == wanted or o.Name == wanted) and not is_reference(o)]
    if not tagged:
        return None
    tips = [o for o in tagged if not _consumed_by(o, tagged)] or tagged
    visible = [o for o in tips if _visible(o)]
    return (visible or tips)[-1]


def _visible(obj: Any) -> bool:
    try:
        return bool(obj.Visibility)
    except Exception:  # noqa: BLE001
        return True


def reference_objects(doc: Any, health_eval: Any, name: str | None = None) -> tuple[list, str]:
    """``(geometry leaves of the reference, error)``. With ``name``: that object, or the geometry inside that
    container; else every ``is_reference`` geometry object that no other reference object consumes."""
    if name:
        obj = _find_object(doc, name)
        if obj is None:
            return [], f"reference '{name}' is not an object of document '{doc.Name}'"
        found = _leaves(obj, health_eval)
        return found, "" if found else f"reference '{name}' contains no solid geometry"
    ref = [o for o in _geometry(doc, health_eval) if is_reference(o)]
    return [o for o in ref if not _consumed_by(o, ref)], ""


def _leaves(obj: Any, health_eval: Any, depth: int = 0) -> list:
    if health_eval.is_geometry(obj) and hasattr(obj, "Shape"):
        return [obj]
    out: list = []
    if depth < 6:
        for child in getattr(obj, "Group", None) or []:
            out.extend(_leaves(child, health_eval, depth + 1))
    return out


def _find_object(doc: Any, name: str) -> Any:
    obj = doc.getObject(name)
    if obj is not None:
        return obj
    try:
        found = doc.getObjectsByLabel(name)
    except Exception:  # noqa: BLE001
        found = []
    return found[0] if len(found) == 1 else None


def resolve_parts(doc: Any, names: Iterable[str], health_eval: Any) -> tuple[list, list[str]]:
    """The objects called ``names`` (Name, then unique Label); every problem is collected, none raised."""
    objs, problems = [], []
    for n in names:
        obj = _find_object(doc, str(n))
        if obj is None:
            problems.append(f"'{n}' is not an object of document '{doc.Name}'")
        elif not health_eval.is_geometry(obj) or not hasattr(obj, "Shape"):
            problems.append(f"'{n}' is not solid geometry (a sheet, sketch, container or mesh cannot be fit-checked)")
        else:
            objs.append(obj)
    return objs, problems


# ---------------------------------------------------------------------------------------------- geometry (GUI)


class _Geo:
    """Global shapes of the objects of one check run, computed once each."""

    def __init__(self, health_eval: Any):
        self._he = health_eval
        self._cache: dict[str, Any] = {}

    def shape(self, obj: Any) -> Any:
        key = obj.Name
        if key not in self._cache:
            self._cache[key] = self._he.global_shape(obj)
        return self._cache[key]


def _overlap(b1: Any, b2: Any, tol: float = 1e-6) -> bool:
    return (b1.XMin <= b2.XMax + tol and b2.XMin <= b1.XMax + tol and b1.YMin <= b2.YMax + tol
            and b2.YMin <= b1.YMax + tol and b1.ZMin <= b2.ZMax + tol and b2.ZMin <= b1.ZMax + tol)


def common_volume(a: Any, b: Any) -> float:
    """Volume of ``a.common(b)`` (0.0 when the bounding boxes do not even touch). Raises on an OCC failure."""
    if not _overlap(a.BoundBox, b.BoundBox):
        return 0.0
    c = a.common(b)
    if c is None or c.isNull():
        return 0.0
    return float(c.Volume)


def _lifted(shape: Any, dz: float) -> Any:
    """A copy of ``shape`` moved by ``dz`` along z."""
    import FreeCAD as App

    moved = shape.copy()
    moved.translate(App.Vector(0, 0, dz))
    return moved


def _num(values: dict, *names: str) -> float | None:
    for n in names:
        v = values.get(n)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            return float(v)
    return None


def _spec_num(spec: dict | None, *path: str) -> float | None:
    node: Any = spec
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
    return float(node) if isinstance(node, (int, float)) and not isinstance(node, bool) else None


def _pcb_object(ref_objs: list) -> Any:
    """The PCB solid of the reference: an object labelled pcb / board, else the largest flat (thin) solid."""
    for needle in ("pcb", "board"):
        named = [o for o in ref_objs if needle in str(o.Label).lower() or needle in o.Name.lower()]
        if named:
            return max(named, key=lambda o: o.Shape.BoundBox.XLength * o.Shape.BoundBox.YLength)
    flat = [o for o in ref_objs if o.Shape.BoundBox.ZLength <= 6.0]
    if flat:
        return max(flat, key=lambda o: o.Shape.BoundBox.XLength * o.Shape.BoundBox.YLength)
    return None


def _wall_faces(shell: Any, wall_t: float) -> list:
    """The cavity's vertical inner wall faces of the shell shape: faces with a horizontal normal that lie inside the
    shell's outer bounding box shrunk by ``wall_t`` and touch that shrunk box (so the floor, the standoffs and the
    faces of port cut-outs, which live in the wall band, are all excluded)."""
    bb = shell.BoundBox
    x0, y0, x1, y1 = bb.XMin + wall_t, bb.YMin + wall_t, bb.XMax - wall_t, bb.YMax - wall_t
    tol = 1e-3
    out = []
    for face in shell.Faces:
        fb = face.BoundBox
        if not (fb.XMin >= x0 - tol and fb.XMax <= x1 + tol and fb.YMin >= y0 - tol and fb.YMax <= y1 + tol):
            continue
        if not (abs(fb.XMin - x0) < tol or abs(fb.XMax - x1) < tol or abs(fb.YMin - y0) < tol
                or abs(fb.YMax - y1) < tol):
            continue
        try:
            u0, u1, v0, v1 = face.ParameterRange
            normal = face.normalAt((u0 + u1) / 2, (v0 + v1) / 2)
        except Exception:  # noqa: BLE001
            continue
        if abs(normal.z) < 1e-6:
            out.append(face)
    return out


class _Run:
    """Everything one ``run_generic`` call shares: the document, the parameter values, the spec, the clock."""

    def __init__(self, doc, health_eval, values, spec, reference_status, lift_mm, deadline, clock):
        self.doc, self.he, self.values, self.spec = doc, health_eval, values, spec
        self.reference_status, self.lift_mm, self.deadline, self.clock = reference_status, float(lift_mm), deadline, clock
        self.geo = _Geo(health_eval)
        self.lid_lift = _num(values, "lid_lift") or 0.0

    def out_of_time(self) -> bool:
        return self.clock() > self.deadline

    def assembled(self, obj: Any, *, lid: bool) -> Any:
        """The global shape in the assembled pose: a lid is moved back by the ``lid_lift`` parameter."""
        shape = self.geo.shape(obj)
        return _lifted(shape, -self.lid_lift) if (lid and self.lid_lift) else shape


# ------------------------------------------------------------------------------------------------ the rows


def _interference(run: _Run, rid: str, a_obj: Any, a_lid: bool, b_obj: Any, b_lid: bool) -> dict:
    kind, metric, limit = "interference", "common_volume_mm3", f"<= {INTERFERENCE_LIMIT:g}"
    a, b = a_obj.Label, b_obj.Label
    try:
        value = common_volume(run.assembled(a_obj, lid=a_lid), run.assembled(b_obj, lid=b_lid))
    except Exception as e:  # noqa: BLE001
        return skip_row(rid, kind, f"the boolean failed ({type(e).__name__}: {str(e)[:80]})", a, b, metric)
    status = "PASS" if value <= INTERFERENCE_LIMIT else "FAIL"
    detail = "" if status == "PASS" else (f"{a} and {b} overlap by {value:.3f} mm3 in the assembled pose; "
                                          "move or trim one of them (a lid_lift parameter is subtracted first)")
    return make_row(rid, kind, a, b, metric, value, limit, status, detail)


def _reference_interference(run: _Run, rid: str, part: Any, part_lid: bool, refs: list) -> dict:
    kind, metric, limit = "interference", "common_volume_mm3", f"<= {INTERFERENCE_LIMIT:g}"
    if part is None:
        role = "lid" if part_lid else "shell"
        return skip_row(rid, kind, f"no {role} part found (tag it OscarRole={role} or name it in 'parts')", "", "", metric)
    if not refs:
        return skip_row(rid, kind, "no Ref_* object in the document", part.Label, "", metric)
    shape = run.assembled(part, lid=part_lid)
    total, worst, worst_v, done = 0.0, "", 0.0, 0
    for ref in refs:
        if run.out_of_time():
            break
        try:
            v = common_volume(run.geo.shape(ref), shape)
        except Exception:  # noqa: BLE001 - one bad vendor solid must not hide the others
            continue
        done += 1
        total += v
        if v > worst_v:
            worst, worst_v = ref.Label, v
    if done < len(refs):
        if done == 0:
            return skip_row(rid, kind, "time budget exhausted before any reference solid was checked",
                            part.Label, "reference", metric)
        status = "FAIL" if total > INTERFERENCE_LIMIT else "WARN"
        return make_row(rid, kind, part.Label, "reference", metric, total, f"<= {INTERFERENCE_LIMIT:g}", status,
                        f"only {done} of {len(refs)} reference solids were checked (time budget)")
    status = "PASS" if total <= INTERFERENCE_LIMIT else "FAIL"
    detail = "" if status == "PASS" else (f"the reference overlaps {part.Label} by {total:.3f} mm3 (worst: {worst}, "
                                          f"{worst_v:.3f} mm3); enlarge the opening or the clearance there")
    return make_row(rid, kind, part.Label, "reference", metric, total, limit, status, detail)


def _pcb_to_wall(run: _Run, shell: Any, refs: list) -> dict:
    rid, kind, metric = "clearance.pcb_to_wall_min", "clearance", "min_wall_distance_mm"
    if shell is None:
        return skip_row(rid, kind, "no shell part found", "", "", metric)
    if not refs:
        return skip_row(rid, kind, "no Ref_* object in the document", shell.Label, "", metric)
    clearance = _num(run.values, "clearance_eff") or _spec_num(run.spec, "clearance") or _num(run.values, "clearance")
    wall_t = _num(run.values, "wall_t") or _spec_num(run.spec, "wall_t")
    if clearance is None or wall_t is None:
        missing = "clearance_eff (or spec.clearance)" if clearance is None else "wall_t"
        return skip_row(rid, kind, f"{missing} is unknown (no Params row and no spec)", shell.Label, "", metric)
    pcb = _pcb_object(refs)
    if pcb is None:
        return skip_row(rid, kind, "no PCB solid found in the reference", shell.Label, "", metric)
    try:
        faces = _wall_faces(run.geo.shape(shell), wall_t)
        if not faces:
            return skip_row(rid, kind, "no vertical cavity wall face found on the shell", shell.Label, pcb.Label, metric)
        pcb_shape = run.geo.shape(pcb)
        value = min(float(pcb_shape.distToShape(f)[0]) for f in faces)
    except Exception as e:  # noqa: BLE001
        return skip_row(rid, kind, f"the distance computation failed ({type(e).__name__}: {str(e)[:80]})",
                        shell.Label, pcb.Label, metric)
    threshold = clearance - CLEARANCE_SLACK
    status = "PASS" if value >= threshold else "FAIL"
    detail = "" if status == "PASS" else (f"the PCB is {value:.2f} mm from the nearest inner wall, less than the "
                                          f"{clearance:g} mm clearance; increase the cavity (clearance) or move the board")
    return make_row(rid, kind, shell.Label, pcb.Label, metric, value, f">= {threshold:.2f}", status, detail)


def _lid_to_tallest(run: _Run, lid: Any, refs: list) -> dict:
    rid, kind, metric = "clearance.lid_to_tallest", "clearance", "lid_gap_mm"
    if lid is None:
        return skip_row(rid, kind, "no lid part found", "", "", metric)
    if not refs:
        return skip_row(rid, kind, "no Ref_* object in the document", lid.Label, "", metric)
    gap = _num(run.values, "lid_gap") or _spec_num(run.spec, "lid", "gap")
    if gap is None:
        return skip_row(rid, kind, "lid_gap is unknown (no Params row and no spec)", lid.Label, "", metric)
    plate = next((o for o in run.doc.Objects if str(getattr(o, "OscarStage", "") or "") == "lid"
                  and o.Label == "Lid_Plate" and run.he.is_geometry(o)), None)
    try:
        if plate is not None:
            underside = run.geo.shape(plate).BoundBox.ZMin - run.lid_lift
        else:
            underside = _num(run.values, "z_cavity_top")
            if underside is None:
                return skip_row(rid, kind, "no Lid_Plate object and no z_cavity_top parameter", lid.Label, "", metric)
        top = max(run.geo.shape(r).BoundBox.ZMax for r in refs)
    except Exception as e:  # noqa: BLE001
        return skip_row(rid, kind, f"the measurement failed ({type(e).__name__}: {str(e)[:80]})", lid.Label, "", metric)
    value = underside - top
    threshold = gap - CLEARANCE_SLACK
    status = "PASS" if value >= threshold else "FAIL"
    detail = "" if status == "PASS" else (f"the lid plate is only {value:.2f} mm above the tallest reference point; "
                                          f"it needs at least {gap:g} mm (lid_gap)")
    return make_row(rid, kind, lid.Label, "reference", metric, value, f">= {threshold:.2f}", status, detail)


def _hooks_expected(run: _Run, lid: Any) -> tuple[bool, str]:
    """``(retention applies, why not)``: a spec decides (snap-fit lids only); without one, tagged hook objects."""
    if run.spec is not None:
        style = (run.spec.get("lid") or {}).get("style") if isinstance(run.spec.get("lid"), dict) else None
        if style and style != "snap_fit":
            return False, f"lid style '{style}' has no snap hooks"
        return True, ""
    if any(_role(o) == "hook" or str(getattr(o, "OscarStage", "") or "") == "hooks" for o in run.doc.Objects):
        return True, ""
    return False, "no snap hooks are tagged in the model (OscarRole=hook / OscarStage=hooks) and there is no spec"


def _retention(run: _Run, shell: Any, lid: Any, explicit: bool) -> dict:
    rid, kind, metric = "retention.lid_lift", "retention", "lifted_common_volume_mm3"
    if shell is None or lid is None:
        return skip_row(rid, kind, "needs a shell and a lid part", getattr(shell, "Label", ""), getattr(lid, "Label", ""),
                        metric)
    applies, why = _hooks_expected(run, lid)
    if not applies and not explicit:
        return skip_row(rid, kind, why, shell.Label, lid.Label, metric)
    try:
        lifted = _lifted(run.assembled(lid, lid=True), run.lift_mm)
        value = common_volume(lifted, run.geo.shape(shell))
    except Exception as e:  # noqa: BLE001
        return skip_row(rid, kind, f"the boolean failed ({type(e).__name__}: {str(e)[:80]})", shell.Label, lid.Label,
                        metric)
    status = "PASS" if value > RETENTION_MIN_VOLUME else "FAIL"
    detail = "" if status == "PASS" else (
        f"lifting the lid {run.lift_mm:g} mm leaves zero interference: nothing retains it; check that "
        "hooks.window_margin is below the lift and that the catch pockets exist")
    return make_row(rid, kind, shell.Label, lid.Label, metric, value, "> 0", status, detail)


def _reference_status(run: _Run) -> dict:
    status = run.reference_status or "none"
    nominal = status in _NOMINAL
    return make_row("reference.status", "coverage", "", "", "", 0.0, "",
                    "WARN" if nominal else "PASS",
                    "reference is nominal board data; no STEP-based collision check" if nominal
                    else f"reference_status={status}")


def run_generic(doc: Any, *, parts: list[str] | None = None, reference: str | None = None, spec: dict | None = None,
                reference_status: str = "none", lift_mm: float = 0.5, ids: set[str] | None = None,
                budget_s: float = 100.0, clock: Callable[[], float] = time.monotonic) -> tuple[list[dict], list[str]]:
    """The generic rows for ``doc``: ``(rows, problems)``. GUI-thread only.

    ``problems`` are argument problems found while resolving ``parts`` / ``reference`` (all of them, no rows are
    computed then). Without ``parts`` the shell, lid and reference are found by their ``OscarRole`` tags; with
    ``parts`` the objects named are used: a tagged one keeps its role, and when none is tagged and exactly two are
    named they are the shell (first) and the lid (second); every pair gets an interference row. Rows whose
    prerequisites are missing come back as SKIP with the reason, so the row set never silently shrinks. ``ids``
    restricts the rows (see ``select_ids``). ``budget_s`` bounds the geometry work (rows after it are SKIP)."""
    from .cad import health_eval

    ref_objs, ref_problem = reference_objects(doc, health_eval, reference)
    problems = [ref_problem] if ref_problem else []
    explicit: list = []
    if parts:
        explicit, part_problems = resolve_parts(doc, parts, health_eval)
        problems.extend(part_problems)
    if problems:
        return [], problems
    table = health_eval.read_param_table(doc)
    values = {r["name"]: r.get("value") for r in table.get("rows", [])}
    run = _Run(doc, health_eval, values, spec, reference_status, lift_mm, clock() + float(budget_s), clock)
    roles: dict[str, Any] = {}
    if explicit:
        for obj in explicit:
            if _role(obj) in ("shell", "lid") and _role(obj) not in roles:
                roles[_role(obj)] = obj
        if not roles and len(explicit) == 2:
            roles = {"shell": explicit[0], "lid": explicit[1]}
    else:
        roles = {"shell": role_tip(doc, "shell", health_eval), "lid": role_tip(doc, "lid", health_eval)}
    shell, lid = roles.get("shell"), roles.get("lid")
    rows: list[dict] = []

    def add(rid: str, make: Callable[[], dict]) -> None:
        if not wants(ids, rid):
            return
        if run.out_of_time():
            rows.append(skip_row(rid, rid.split(".")[0], "time budget exhausted"))
        else:
            rows.append(make())

    if shell is not None and lid is not None:
        add("interference.shell_lid", lambda: _interference(run, "interference.shell_lid", shell, False, lid, True))
    else:
        add("interference.shell_lid", lambda: skip_row("interference.shell_lid", "interference",
                                                       "needs a shell and a lid part (tag them OscarRole=shell / lid)"))
    add("interference.reference_shell", lambda: _reference_interference(run, "interference.reference_shell", shell,
                                                                         False, ref_objs))
    add("interference.reference_lid", lambda: _reference_interference(run, "interference.reference_lid", lid, True,
                                                                      ref_objs))
    add("clearance.pcb_to_wall_min", lambda: _pcb_to_wall(run, shell, ref_objs))
    add("clearance.lid_to_tallest", lambda: _lid_to_tallest(run, lid, ref_objs))
    add("retention.lid_lift", lambda: _retention(run, shell, lid, bool(explicit)))
    add("reference.status", lambda: _reference_status(run))
    for i, a in enumerate(explicit):                       # every other pair of explicit parts
        for b in explicit[i + 1:]:
            if shell is not None and lid is not None and {a.Name, b.Name} == {shell.Name, lid.Name}:
                continue                                  # already reported as interference.shell_lid
            rid = f"interference.{slug(a.Label)}_{slug(b.Label)}"
            add(rid, lambda a=a, b=b, rid=rid: _interference(run, rid, a, _role(a) == "lid", b, _role(b) == "lid"))
    return rows, []


def kit_rows(doc: Any, spec: dict | None, reference_status: str, ids: set[str] | None = None,
             coverage: bool = False) -> tuple[list[dict], list[dict]]:
    """``(kit fit rows, spec coverage rows)`` through the guarded ``kit.checks`` import. GUI-thread only.

    Without a spec nothing kit-specific applies and both lists are empty. When ``oscar.kit.checks`` is missing a
    single SKIP row ``kit.checks unavailable`` stands in for the kit rows; an exception inside it becomes one SKIP
    row too (a check must never crash ``check_fit``). ``coverage`` also asks for ``spec_coverage`` (``run_validation``)."""
    if spec is None or not (ids is None or any(wants(ids, k) for k in KIT_IDS)):
        return [], []
    try:
        from .kit import checks as kit_checks
    except ModuleNotFoundError as e:
        if e.name not in ("oscar.kit", "oscar.kit.checks"):
            raise
        return [skip_row("kit.checks", "coverage", "kit.checks unavailable")], []
    rows: list[dict] = []
    try:
        rows = [normalize_row(r) for r in (kit_checks.fit_rows(doc, spec, reference_status) or [])]
    except Exception as e:  # noqa: BLE001
        rows = [skip_row("kit.checks", "coverage", f"kit.checks failed ({type(e).__name__}: {str(e)[:80]})")]
    rows = [r for r in rows if ids is None or r["id"].startswith("kit.") or wants(ids, r["id"])]
    cover: list[dict] = []
    if coverage:
        try:
            from .kit import spec as kit_spec

            board = kit_spec.load_board(str((spec or {}).get("board") or "rpi4b"))
            cover = [dict(r) for r in (kit_checks.spec_coverage(spec, board, doc) or []) if isinstance(r, dict)]
        except Exception:  # noqa: BLE001 - coverage is a nicety of the Build Summary
            cover = []
    return rows, cover
