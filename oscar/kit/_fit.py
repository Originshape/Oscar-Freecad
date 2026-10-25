"""Kit fit rows and spec coverage measured on BUILT geometry (private module of ``oscar.kit``; FreeCAD, GUI-thread only).

``kit.checks.fit_rows`` / ``kit.checks.spec_coverage`` delegate here (``checks`` itself stays importable without FreeCAD).

The four kit rows (the other seven fit rows are the generic ones of ``tools_verify_fit``):

* ``containment.ports``      every connector that opens through a wall: a probe of the connector's cross-section plus
  ``ports.margin`` is pushed through the wall slab and must not touch shell material (``common`` volume <= 0.05 mm3). A real
  geometric test of the built window, not of the numbers the stage used;
* ``alignment.standoffs``    every board hole: the standoff top face lies at the PCB bottom (z = 0) within 0.05 mm and a pilot
  hole axis lies within 0.1 mm of the hole centre;
* ``hooks.engagement``       every barb reaches past the shell's outer wall by at least ``overhang - lid.fit - 0.05``;
* ``hooks.clear_of_cutouts`` every hook beam, widened by 1 mm each side, stays out of every opening on its wall.

Every row has the shape of ``tools_verify_fit.make_row``: ``{"id","kind","a","b","metric","value","limit","status","detail"}``.
A check that cannot run (stage not built, no shell, a geometry failure) is a SKIP row that says why; nothing here raises.
Shapes are always taken in the GLOBAL frame (``geometry.global_shape``): the lid may sit anywhere (``lid_lift``).
"""

from typing import Any

import Part
from FreeCAD import Vector as V

from . import _stages
from . import geometry as G
from . import spec as _spec

INTERFERENCE_TOL = 0.05        # mm3: a window may be touched by at most this much material
TOP_TOL = 0.05                 # mm: standoff top versus PCB bottom
AXIS_TOL = 0.1                 # mm: pilot axis versus board hole
ENGAGE_SLACK = 0.05            # mm: contract slack of the engagement rule
HOOK_WIDEN = 1.0               # mm kept clear beside a hook (hooks.clear_of_cutouts)
PROBE_REACH = 0.3              # mm the probe pokes out of both faces of the wall slab (touching faces alone are fragile)

IDS = ("containment.ports", "alignment.standoffs", "hooks.engagement", "hooks.clear_of_cutouts")


def _row(rid: str, kind: str, a: str, b: str, metric: str, value: float, limit: str, status: str, detail: str = "") -> dict:
    return {"id": rid, "kind": kind, "a": a, "b": b, "metric": metric, "value": round(float(value), 3), "limit": limit,
            "status": status, "detail": detail}


def _skip(rid: str, kind: str, why: str, a: str = "", metric: str = "") -> dict:
    return _row(rid, kind, a, "", metric, 0.0, "", "SKIP", why)


class _Kit:
    """What the rows share: the effective spec (declared spec overlaid with the sheet), the board and the built parts."""

    def __init__(self, doc: Any, spec: "_spec.EnclosureSpec | dict", board: "dict | None" = None) -> None:
        from . import build

        self.doc = doc
        self.declared = _spec._as_spec(spec)
        self.kit = self.declared.kit_id
        self.board = board if board is not None else _spec.load_board(self.declared.board)
        self.spec = build.spec_from_sheet(doc, self.declared)
        self.built = build.built_stages(doc, self.kit)
        self.shell = G.tip(doc, "shell", self.kit)
        self.lid = G.tip(doc, "lid", self.kit)
        self.dv = _spec.derived(self.spec, self.board)
        self.pcb_t = self._pcb_t()

    def _pcb_t(self) -> float:
        from . import params as _params

        try:
            return float(_params.Params(self.doc).values().get("pcb_t", self.board["pcb"]["thickness"]))
        except _params.ParamsError:
            return float(self.board["pcb"]["thickness"])

    def v(self, name: str) -> float:
        return float(self.dv[name]["value"])


# --------------------------------------------------------------------------------------------------------------------
# containment.ports
# --------------------------------------------------------------------------------------------------------------------

def _wall_probe(k: _Kit, c: dict, margin: float) -> Any:
    """The probe box of connector ``c``: its cross-section in the wall plane plus ``margin``, through the wall slab."""
    t_board = float(k.board["pcb"]["thickness"])
    dz = k.pcb_t - t_board
    box = c["box"]
    z0, z1 = (z + dz if z >= t_board else z for z in box["z"])
    z0, z1 = z0 - margin, z1 + margin
    wt = float(k.spec.wall_t)
    face = c["face"]
    reach = PROBE_REACH
    if face in ("-y", "+y"):
        a0, a1 = box["x"][0] - margin, box["x"][1] + margin
        y0 = k.v("cav_y0") - wt - reach if face == "-y" else k.v("cav_y1") - reach
        return Part.makeBox(a1 - a0, wt + 2 * reach, z1 - z0, V(a0, y0, z0))
    a0, a1 = box["y"][0] - margin, box["y"][1] + margin
    x0 = k.v("cav_x0") - wt - reach if face == "-x" else k.v("cav_x1") - reach
    return Part.makeBox(wt + 2 * reach, a1 - a0, z1 - z0, V(x0, a0, z0))


def containment_ports(k: _Kit) -> dict:
    rid, kind, metric = "containment.ports", "containment", "blocked_volume_mm3"
    if k.shell is None:
        return _skip(rid, kind, "the shell is not built (stage 'shell')", metric=metric)
    if "ports" not in k.built:
        return _skip(rid, kind, "the ports stage is not built, so no opening exists to measure", k.shell.Label, metric)
    conns = [c for c in _spec.active_connectors(k.spec, k.board) if c.get("face") in _spec.WALL_FACES and c.get("box")]
    if not conns:
        return _skip(rid, kind, "no connector opens through a wall in this spec", k.shell.Label, metric)
    shape = G.global_shape(k.shell)
    margin = float(k.spec.ports.margin)
    worst, bad = 0.0, []
    for c in conns:
        try:
            blocked = float(shape.common(_wall_probe(k, c, margin)).Volume)
        except Exception as exc:    # noqa: BLE001 - one broken connector must not hide the others
            return _skip(rid, kind, f"the window test failed for '{c.get('id')}' ({type(exc).__name__}: {str(exc)[:80]})", k.shell.Label, metric)
        worst = max(worst, blocked)
        if blocked > INTERFERENCE_TOL:
            bad.append(f"'{c.get('id')}' ({c['face']} wall): {blocked:.2f} mm3 of wall blocks the window")
    if bad:
        return _row(rid, kind, k.shell.Label, "", metric, worst, f"<= {INTERFERENCE_TOL}", "FAIL",
                    "; ".join(bad) + "; rebuild the ports stage (build_enclosure(stage='ports', rebuild=True)) or enlarge ports.margin")
    return _row(rid, kind, k.shell.Label, "", metric, worst, f"<= {INTERFERENCE_TOL}", "PASS",
                f"{len(conns)} connector window{'s' if len(conns) != 1 else ''} contain their connector plus {margin:g} mm margin")


# --------------------------------------------------------------------------------------------------------------------
# alignment.standoffs
# --------------------------------------------------------------------------------------------------------------------

def _top_z(shape: Any, x: float, y: float, post_d: float) -> "float | None":
    """z of the standoff top face around the hole centre (x, y): the highest upward planar face whose footprint is post-sized."""
    best = None
    for f in shape.Faces:
        try:
            if not isinstance(f.Surface, Part.Plane):
                continue
            bb = f.BoundBox
            if bb.ZLength > 1e-6 or bb.XLength > post_d + 0.2 or bb.YLength > post_d + 0.2:
                continue
            if not (bb.XMin - 1e-6 <= x <= bb.XMax + 1e-6 and bb.YMin - 1e-6 <= y <= bb.YMax + 1e-6):
                continue
            if f.normalAt(0, 0).z < 0.99:
                continue
        except Exception:    # noqa: BLE001
            continue
        if best is None or bb.ZMax > best:
            best = bb.ZMax
    return best


def _axis_offset(shape: Any, x: float, y: float, pilot_r: float) -> "float | None":
    """Smallest XY distance between (x, y) and the axis of a vertical pilot cylinder of radius ``pilot_r``."""
    best = None
    for f in shape.Faces:
        try:
            s = f.Surface
            if not isinstance(s, Part.Cylinder) or abs(s.Radius - pilot_r) > 0.05 or abs(s.Axis.z) < 0.99:
                continue
            d = ((s.Center.x - x) ** 2 + (s.Center.y - y) ** 2) ** 0.5
        except Exception:    # noqa: BLE001
            continue
        if best is None or d < best:
            best = d
    return best


def alignment_standoffs(k: _Kit) -> dict:
    rid, kind, metric = "alignment.standoffs", "alignment", "max_deviation_mm"
    if k.shell is None:
        return _skip(rid, kind, "the shell is not built (stage 'shell')", metric=metric)
    if "standoffs" not in k.built:
        return _skip(rid, kind, "the standoffs stage is not built", k.shell.Label, metric)
    shape = G.global_shape(k.shell)
    post_d, pilot_d = float(k.spec.standoff.d), float(k.spec.standoff.pilot_d)
    holes = _stages._hole_positions(k.board)
    worst, bad = 0.0, []
    for i, (x, y) in enumerate(holes, 1):
        top = _top_z(shape, x, y, post_d)
        off = _axis_offset(shape, x, y, pilot_d / 2)
        if top is None:
            bad.append(f"hole {i} at ({x:g}, {y:g}): no standoff top face")
            worst = max(worst, 99.0)
            continue
        if off is None:
            bad.append(f"hole {i} at ({x:g}, {y:g}): no pilot hole")
            worst = max(worst, 99.0)
            continue
        worst = max(worst, abs(top), off)
        if abs(top) > TOP_TOL:
            bad.append(f"hole {i} at ({x:g}, {y:g}): standoff top at z {top:.2f}, the PCB bottom is z 0")
        if off > AXIS_TOL:
            bad.append(f"hole {i} at ({x:g}, {y:g}): pilot axis {off:.2f} mm off the hole")
    limit = f"top <= {TOP_TOL}, axis <= {AXIS_TOL}"
    if bad:
        return _row(rid, kind, k.shell.Label, "", metric, worst, limit, "FAIL",
                    "; ".join(bad[:4]) + "; rebuild the standoffs stage (build_enclosure(stage='standoffs', rebuild=True))")
    return _row(rid, kind, k.shell.Label, "", metric, worst, limit, "PASS",
                f"{len(holes)} standoffs reach the PCB bottom with their pilot holes on the board holes")


# --------------------------------------------------------------------------------------------------------------------
# hooks
# --------------------------------------------------------------------------------------------------------------------

def _hook_objects(k: _Kit, suffix: str) -> list:
    return [o for o in G.stage_objects(k.doc, k.kit, "hooks") if G.tags_of(o)["OscarRole"] == "hook" and o.Name.endswith(suffix)]


def _hooks_applicable(k: _Kit, rid: str, metric: str) -> "dict | None":
    if k.spec.lid.style != "snap_fit":
        return _skip(rid, "alignment", f"lid style '{k.spec.lid.style}' has no snap hooks", metric=metric)
    if k.shell is None or k.lid is None:
        return _skip(rid, "alignment", "the shell and the lid are needed (stages 'shell' and 'lid')", metric=metric)
    if "hooks" not in k.built:
        return _skip(rid, "alignment", "the hooks stage is not built", k.lid.Label, metric)
    return None


def hooks_engagement(k: _Kit) -> dict:
    rid, metric = "hooks.engagement", "min_engagement_mm"
    early = _hooks_applicable(k, rid, metric)
    if early:
        return early
    barbs = _hook_objects(k, "_Barb")
    if not barbs:
        return _skip(rid, "alignment", "no hook barb objects exist although the hooks stage is built", k.lid.Label, metric)
    need = float(k.spec.hooks.overhang) - float(k.spec.lid.fit) - ENGAGE_SLACK
    sb = G.global_shape(k.shell).BoundBox
    mid_y = (sb.YMin + sb.YMax) / 2
    worst, bad = None, []
    for o in barbs:
        bb = G.global_shape(o).BoundBox
        plus = (bb.YMin + bb.YMax) / 2 > mid_y
        pen = (sb.YMax - bb.YMin) if plus else (bb.YMax - sb.YMin)
        worst = pen if worst is None else min(worst, pen)
        if pen < need:
            bad.append(f"{o.Name} ({'+y' if plus else '-y'} wall) engages {pen:.2f} mm, needs {need:.2f}")
    limit = f">= {need:.2f}"
    if bad:
        return _row(rid, "alignment", k.lid.Label, k.shell.Label, metric, worst or 0.0, limit, "FAIL",
                    "; ".join(bad[:4]) + "; rebuild the hooks stage (build_enclosure(stage='hooks', rebuild=True))")
    return _row(rid, "alignment", k.lid.Label, k.shell.Label, metric, worst or 0.0, limit, "PASS",
                f"{len(barbs)} barbs reach {worst:.2f} mm past the outer wall (needs {need:.2f})")


def hooks_clear_of_cutouts(k: _Kit) -> dict:
    rid, metric = "hooks.clear_of_cutouts", "min_gap_mm"
    early = _hooks_applicable(k, rid, metric)
    if early:
        return early
    beams = _hook_objects(k, "_Beam")
    if not beams:
        return _skip(rid, "alignment", "no hook beam objects exist although the hooks stage is built", k.lid.Label, metric)
    sb = G.global_shape(k.shell).BoundBox
    mid_y = (sb.YMin + sb.YMax) / 2
    openings = {w: _spec.wall_openings(k.spec, k.board, w, pcb_t=k.pcb_t) for w in ("+y", "-y")}
    worst, bad = None, []
    for o in beams:
        bb = G.global_shape(o).BoundBox
        wall = "+y" if (bb.YMin + bb.YMax) / 2 > mid_y else "-y"
        lo, hi = bb.XMin - HOOK_WIDEN, bb.XMax + HOOK_WIDEN
        for op in openings[wall]:
            o_lo, o_hi = op["u"] - op["w"] / 2, op["u"] + op["w"] / 2
            gap = max(o_lo - hi, lo - o_hi)          # positive: clear; negative: the widened beam overlaps the opening
            worst = gap if worst is None else min(worst, gap)
            if gap < 0:
                bad.append(f"{o.Name} ({wall} wall) is within {HOOK_WIDEN:g} mm of opening '{op['id']}'")
    if bad:
        return _row(rid, "alignment", k.lid.Label, k.shell.Label, metric, worst or 0.0, ">= 0", "FAIL",
                    "; ".join(bad[:4]) + "; move the opening or use fewer hooks")
    if worst is None:
        return _row(rid, "alignment", k.lid.Label, k.shell.Label, metric, 0.0, ">= 0", "PASS",
                    f"{len(beams)} hooks, no opening on their walls")
    return _row(rid, "alignment", k.lid.Label, k.shell.Label, metric, worst, ">= 0", "PASS",
                f"{len(beams)} hooks keep at least {worst:.1f} mm (after a {HOOK_WIDEN:g} mm widening) from every opening on their walls")


_ROWS = (containment_ports, alignment_standoffs, hooks_engagement, hooks_clear_of_cutouts)


def fit_rows(doc: Any, spec: "_spec.EnclosureSpec | dict", reference_status: str = "none") -> list[dict]:
    """The four kit fit rows of ``doc`` (module docstring). ``reference_status`` is accepted (and ignored) to keep the common fit-row signature."""
    del reference_status
    try:
        k = _Kit(doc, spec)
    except (KeyError, LookupError, ValueError, TypeError) as exc:
        why = exc.args[0] if exc.args else exc
        return [_skip(rid, "alignment", f"the kit rows could not start: {why}") for rid in IDS]
    rows = []
    for fn, rid in zip(_ROWS, IDS):
        try:
            rows.append(fn(k))
        except Exception as exc:    # noqa: BLE001 - a check must never crash check_fit
            rows.append(_skip(rid, "alignment", f"the check failed ({type(exc).__name__}: {str(exc)[:80]})"))
    return rows


# --------------------------------------------------------------------------------------------------------------------
# spec coverage
# --------------------------------------------------------------------------------------------------------------------

# kit requirement key -> prefix of the ledger requirement text that carries it (flow_intent.ENCLOSURE_RULES / _ENCLOSURE_ALWAYS)
REQUIREMENT_TEXT = {"shell": "A bottom shell", "ports": "Cutouts for every", "lid": "A lid", "vents": "Air vent", "snap": "Snap-fit",
                    "material": "Material", "source": "Source the reference", "parameters": "Every driving dimension"}


def _cov(key: str, status: str, by: str, detail: str) -> dict:
    return {"requirement": key, "key": key, "status": status, "by": by, "detail": detail}


def _tip_text(obj: Any) -> str:
    f = G.shape_facts(obj)
    bb = f["bbox"]
    size = " x ".join(f"{bb[i + 3] - bb[i]:.2f}".rstrip("0").rstrip(".") for i in range(3)) if bb else "?"
    return f"{obj.Label}: {f['solids']} solid{'s' if f['solids'] != 1 else ''}, {f['volume']:.0f} mm3, {size} mm"


def coverage_rows(spec: "_spec.EnclosureSpec | dict", board: dict, doc: Any) -> list[dict]:
    """The requirement coverage table: one row per kit requirement, ``covered`` only when the geometry exists in ``doc``.

    ``status`` is ``covered``, ``partial``, ``missing`` or ``n/a`` (the spec does not ask for it); ``by`` names the objects.
    Rows use the kit key as ``requirement`` (``shell``, ``ports`` ...); ``spec_coverage`` swaps in the ledger's ``R<n>`` id when a
    ledger is bound to the calling context.
    """
    from . import build
    from . import params as _params

    s = _spec._as_spec(spec)
    k = _Kit(doc, s, board)
    rows: list[dict] = []
    built = k.built
    # shell and standoffs
    if k.shell is None:
        rows.append(_cov("shell", "missing", "", "the shell is not built (build_enclosure stage 'shell')"))
    else:
        f = G.shape_facts(k.shell)
        ok = f["solids"] == 1 and f["valid"] and f["closed"]
        stand = "standoffs" in built
        rows.append(_cov("shell", "covered" if ok else "partial", k.shell.Label,
                         f"{_tip_text(k.shell)}; standoffs {'built' if stand else 'NOT built'}"))
    # ports
    conns = [c for c in _spec.active_connectors(k.spec, board) if c.get("face") in _spec.WALL_FACES]
    if "ports" not in built:
        rows.append(_cov("ports", "missing", "", f"the ports stage is not built ({len(conns)} openings expected)"))
    else:
        cut = len([o for o in G.stage_objects(doc, s.kit_id, "ports") if o.Name.startswith("Shell_Port_")])
        want = len(conns) + len([c for c in k.spec.cutouts if isinstance(c, dict) and c.get("part") == "shell"])
        rows.append(_cov("ports", "covered" if cut >= want else "partial", "Shell",
                         f"{cut}/{want} openings cut"))
    # lid
    if k.lid is None:
        rows.append(_cov("lid", "missing", "", "the lid is not built (build_enclosure stage 'lid')"))
    else:
        f = G.shape_facts(k.lid)
        rows.append(_cov("lid", "covered" if f["solids"] == 1 and f["valid"] and f["closed"] else "partial", k.lid.Label, _tip_text(k.lid)))
    # vents
    if not k.spec.vents.enabled:
        rows.append(_cov("vents", "n/a", "", "vents are disabled in the spec"))
    elif "vents" not in built:
        rows.append(_cov("vents", "missing", "", "the vents stage is not built"))
    else:
        cutters = [o for o in doc.Objects if G.tags_of(o)["OscarKit"] == s.kit_id and "_Vent" in o.Name and not o.Name.endswith(("Cutters", "Vents"))]
        v = k.spec.vents
        rows.append(_cov("vents", "covered" if cutters else "partial", v.part.capitalize(),
                         f"{v.count} {v.pattern} cut in the {v.part} {v.face}"))
    # snap hooks
    if k.spec.lid.style != "snap_fit":
        rows.append(_cov("snap", "n/a", "", f"lid style '{k.spec.lid.style}' has no snap hooks"))
    elif "hooks" not in built:
        rows.append(_cov("snap", "missing", "", "the hooks stage is not built"))
    else:
        n = len(_hook_objects(k, "_Barb"))
        want = int(k.spec.hooks.count)
        rows.append(_cov("snap", "covered" if n >= want else "partial", "Lid", f"{n}/{want} hooks with catch pockets"))
    # material
    mat = k.spec.material
    rows.append(_cov("material", "covered", "spec", f"{mat}; shrinkage compensated by '{k.spec.shrink_comp}'; not validated on a printer"))
    # reference
    ref = G.tip(doc, "ref", s.kit_id)
    if ref is None:
        rows.append(_cov("source", "missing", "", "no Ref_* object exists in the document"))
    else:
        nominal = bool(G.tags_of(ref)["OscarKit"])
        rows.append(_cov("source", "covered", ref.Name, "nominal board data (no STEP file)" if nominal else "imported or attached reference geometry"))
    # parameters
    try:
        table = _params.Params(doc).table()
        bound = sum(1 for r in table if r["bound"])
        rows.append(_cov("parameters", "covered" if bound else "partial", "Params", f"{len(table)} parameters, {bound} bound to geometry"))
    except _params.ParamsError:
        rows.append(_cov("parameters", "missing", "", "the Params sheet does not exist"))
    return rows


def _ledger_ids() -> "dict[str, str]":
    """``{key: R-id}`` of the requirements of the ledger bound to this thread's tool context ({} when none)."""
    try:
        from ..toolctx import current_context

        ledger = current_context().ledger
        reqs = list(getattr(ledger, "requirements", None) or [])
    except Exception:    # noqa: BLE001
        return {}
    out: dict[str, str] = {}
    for r in reqs:
        text = str(getattr(r, "text", "") or (r.get("text") if isinstance(r, dict) else ""))
        rid = str(getattr(r, "id", "") or (r.get("id") if isinstance(r, dict) else ""))
        for key, prefix in REQUIREMENT_TEXT.items():
            if rid and text.startswith(prefix) and key not in out:
                out[key] = rid
    return out


def spec_coverage(spec: "_spec.EnclosureSpec | dict", board: dict, doc: Any = None) -> list[dict]:
    """``coverage_rows`` with the bound ledger's requirement ids as ``requirement`` (the Build Summary joins on them)."""
    if doc is None:
        return []
    rows = coverage_rows(spec, board, doc)
    ids = _ledger_ids()
    for r in rows:
        r["requirement"] = ids.get(r["key"], r["key"])
    return rows


__all__ = ["IDS", "REQUIREMENT_TEXT", "coverage_rows", "fit_rows", "spec_coverage"]
