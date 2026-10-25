"""Kit checks: design-for-manufacture rows, fit rows and spec coverage.

``dfm_rows`` is PURE (spec + board in, rows out, no FreeCAD): it judges the DESIGN, so it runs before any geometry
exists, after every spec edit, and again in the Build Summary. Every row is ``{"id", "status", "detail"}`` with
``status`` in ``PASS | WARN | FAIL | INFO`` and ids from ``DFM_IDS`` (each id appears exactly once, always in this
order; a row that does not apply to the spec, such as the hook rows for a screw lid, is INFO).

``fit_rows`` and ``spec_coverage`` measure a built document (FreeCAD, GUI thread) and live in the private ``_fit``; they are
imported lazily so this module stays importable without FreeCAD.
"""

import math
from typing import Any

from . import materials as _materials
from . import spec as _spec

DFM_IDS = ("dfm.min_wall", "dfm.min_web", "dfm.hook_strain", "dfm.hook_strain_molded", "dfm.pocket_wall",
           "dfm.retention_angle", "dfm.port_web", "dfm.rim_web", "dfm.standoff_h", "dfm.overhang", "dfm.shrink")
FIT_IDS = ("containment.ports", "alignment.standoffs", "hooks.engagement", "hooks.clear_of_cutouts")
STATUSES = ("PASS", "WARN", "FAIL", "INFO")
THIN_WALL = 1.6          # mm: below this a wall still passes min_wall but prints less reliably (spec warns too)
OVERHANG_LIMIT_DEG = 45  # FDM prints an overhang up to 45 degrees from vertical without support


def _row(rid: str, status: str, detail: str) -> dict:
    return {"id": rid, "status": status, "detail": detail}


class _Ctx:
    """Everything a row needs, computed once."""

    def __init__(self, s: "_spec.EnclosureSpec", board: dict) -> None:
        self.s = s
        self.board = board
        self.mat = _materials.get(s.material)       # KeyError for an unknown material
        self.dv = _spec.derived(s, board)           # ValueError for incomplete data
        self.snap = s.lid.style == "snap_fit"
        self.min_web = self.mat["min_web"]

    @property
    def wall_openings(self) -> list[dict]:
        return [o for o in _spec.wall_openings(self.s, self.board) if o["face"] in _spec.WALL_FACES]


def _na(rid: str, why: str) -> dict:
    return _row(rid, "INFO", f"not applicable: {why}")


def _min_wall(c: _Ctx) -> dict:
    s, mw = c.s, c.mat["min_wall"]
    parts = (("wall_t", s.wall_t), ("floor_t", s.floor_t), ("lid_t", s.lid_t))
    low = [f"{k} {_spec._n(v)}" for k, v in parts if v < mw - _spec.TOL]
    if low:
        return _row("dfm.min_wall", "FAIL", f"{', '.join(low)} mm is below the {s.material} minimum {_spec._n(mw)} mm")
    if s.wall_t < THIN_WALL:
        return _row("dfm.min_wall", "WARN", f"wall_t {_spec._n(s.wall_t)} mm meets the {s.material} minimum {_spec._n(mw)} mm but is thin for FDM; {_spec._n(THIN_WALL)} mm or more prints more reliably")
    return _row("dfm.min_wall", "PASS", f"wall {_spec._n(s.wall_t)}, floor {_spec._n(s.floor_t)}, lid {_spec._n(s.lid_t)} mm against the {s.material} minimum {_spec._n(mw)} mm")


def _min_web(c: _Ctx) -> dict:
    v = c.s.vents
    if not v.enabled:
        return _na("dfm.min_web", "vents are disabled, so there is no vent web")
    web = v.pitch - v.slot_w
    detail = f"vent web {_spec._n(web)} mm (pitch {_spec._n(v.pitch)} - slot_w {_spec._n(v.slot_w)}) against the {c.s.material} minimum {_spec._n(c.min_web)} mm"
    return _row("dfm.min_web", "FAIL" if web < c.min_web - _spec.TOL else "PASS", detail)


def _hook_strain(c: _Ctx) -> dict:
    s = c.s
    if not c.snap:
        return _na("dfm.hook_strain", f"lid style {s.lid.style!r} has no snap hooks")
    y = s.hooks.overhang - s.lid.fit
    if s.hooks.length <= 0 or y <= 0:
        return _row("dfm.hook_strain", "FAIL", f"the barb does not reach past the lid fit gap (overhang {_spec._n(s.hooks.overhang)} <= fit {_spec._n(s.lid.fit)}) or the beam has no length")
    strain, limit = _spec.hook_strain(s), _spec.hook_strain_limit(s)
    ratio = strain / limit
    detail = (f"strain {strain:.4f} against the FDM limit {limit:.4f} ({ratio:.0%}); beam t {_spec._n(s.hooks.t)} mm, "
              f"length {_spec._n(s.hooks.length)} mm, deflection {_spec._n(y)} mm")
    status = "FAIL" if ratio > 1.0 else ("WARN" if ratio > _spec.STRAIN_WARN_FRACTION else "PASS")
    return _row("dfm.hook_strain", status, detail)


def _hook_strain_molded(c: _Ctx) -> dict:
    s = c.s
    if not c.snap:
        return _na("dfm.hook_strain_molded", f"lid style {s.lid.style!r} has no snap hooks")
    limit = _spec.hook_strain_limit(s, molded=True)
    strain = _spec.hook_strain(s)
    share = "" if not math.isfinite(strain) or limit <= 0 else f" (this beam would use {strain / limit:.0%} of it)"
    return _row("dfm.hook_strain_molded", "INFO", f"injection-moulded limit for {s.material} is {limit:.4f}{share}; the FDM limit above is the one that applies to a printed lid")


def _pocket_wall(c: _Ctx) -> dict:
    s = c.s
    if not c.snap:
        return _na("dfm.pocket_wall", f"lid style {s.lid.style!r} has no catch pockets")
    depth = _spec.catch_pocket_depth(s)
    behind = s.wall_t - depth
    mw = c.mat["min_wall"]
    detail = f"{_spec._n(behind)} mm of wall remain behind the {_spec._n(depth)} mm catch pocket (minimum {_spec._n(mw)} mm)"
    return _row("dfm.pocket_wall", "FAIL" if behind < mw - _spec.TOL else "PASS", detail)


def _retention_angle(c: _Ctx) -> dict:
    s = c.s
    if not c.snap:
        return _na("dfm.retention_angle", f"lid style {s.lid.style!r} has no snap hooks")
    ret = s.hooks.retention_deg
    if ret > 60:
        return _row("dfm.retention_angle", "WARN", f"retention face {ret:g} degrees makes the lid hard to open again; a re-openable lid needs 60 degrees or less")
    return _row("dfm.retention_angle", "PASS", f"retention face {ret:g} degrees; the lid can be opened again")


def _port_web(c: _Ctx) -> dict:
    """Material between neighbouring wall openings: the smaller of the horizontal and the vertical gap (negative = overlap)."""
    narrowest: "tuple[float, str, str, str] | None" = None
    bad: list[str] = []
    for wall in _spec.WALL_FACES:
        ops = sorted((o for o in c.wall_openings if o["face"] == wall), key=lambda o: o["u"])
        for i, a in enumerate(ops):
            for b in ops[i + 1:]:
                du = (b["u"] - b["w"] / 2) - (a["u"] + a["w"] / 2)
                dv = max(b["v"] - b["h"] / 2 - (a["v"] + a["h"] / 2), a["v"] - a["h"] / 2 - (b["v"] + b["h"] / 2))
                web = max(du, dv)
                if narrowest is None or web < narrowest[0]:
                    narrowest = (web, a["id"], b["id"], wall)
                if web < c.min_web - _spec.TOL:
                    bad.append(f"'{a['id']}' and '{b['id']}' on the {wall} wall: web {_spec._n(web)} mm" if web >= 0
                               else f"'{a['id']}' and '{b['id']}' on the {wall} wall overlap")
    if narrowest is None:
        return _na("dfm.port_web", "no wall has two openings")
    web, a_id, b_id, wall = narrowest
    if bad:
        return _row("dfm.port_web", "FAIL", f"{'; '.join(bad)} (minimum {_spec._n(c.min_web)} mm for {c.s.material})")
    return _row("dfm.port_web", "PASS", f"narrowest web {_spec._n(web)} mm between '{a_id}' and '{b_id}' on the {wall} wall (minimum {_spec._n(c.min_web)} mm)")


def _rim_web(c: _Ctx) -> dict:
    rim = c.dv["z_cavity_top"]["value"]
    ops = c.wall_openings
    if not ops:
        return _na("dfm.rim_web", "there are no wall openings")
    webs = [(rim - (o["v"] + o["h"] / 2), o["id"]) for o in ops]
    merged = [f"opening '{oid}' merged with the rim (web {_spec._n(w)} < {_spec._n(c.min_web)})" for w, oid in webs if w < c.min_web - _spec.TOL]
    if merged:
        return _row("dfm.rim_web", "WARN", "; ".join(merged) + "; the kit extends it up through the rim as a notch that the lid plate closes from above")
    low = min(webs)
    return _row("dfm.rim_web", "PASS", f"every wall opening keeps at least {_spec._n(low[0])} mm below the rim ('{low[1]}'; minimum {_spec._n(c.min_web)} mm)")


def _standoff_h(c: _Ctx) -> dict:
    s = c.s
    bottom = float((c.board.get("height") or {}).get("min_bottom_z", 0.0))
    required = max(2.0, 1.0 - bottom) if bottom < 0 else 2.0
    under = s.standoff.h + bottom if bottom < 0 else s.standoff.h
    where = f"under the lowest underside part (z {bottom:g})" if bottom < 0 else "under the board"
    if s.standoff.h < required - _spec.TOL:
        return _row("dfm.standoff_h", "FAIL", f"standoff {_spec._n(s.standoff.h)} mm leaves {_spec._n(under)} mm {where}; needs {_spec._n(required)} mm or more")
    return _row("dfm.standoff_h", "PASS", f"standoff {_spec._n(s.standoff.h)} mm leaves {_spec._n(under)} mm {where}; minimum standoff {_spec._n(required)} mm")


def _overhang(c: _Ctx) -> dict:
    """WARN only: lid features that print steeper than 45 degrees from vertical without support."""
    s = c.s
    issues: list[str] = []
    if s.print_orientation.get("lid") == "as_modeled":
        issues.append(f"the lid prints as modeled, so its {_spec._n(c.dv['lid_len']['value'])} x {_spec._n(c.dv['lid_wid']['value'])} mm plate bridges over the skirt; use print_orientation.lid 'top_down' or add supports")
        if c.snap and s.hooks.ramp_deg > OVERHANG_LIMIT_DEG:
            issues.append(f"the barb lead-in ramp ({s.hooks.ramp_deg:g} degrees) overhangs more than {OVERHANG_LIMIT_DEG} degrees")
    elif c.snap and s.hooks.retention_deg > OVERHANG_LIMIT_DEG:
        issues.append(f"the barb retention face ({s.hooks.retention_deg:g} degrees) overhangs more than {OVERHANG_LIMIT_DEG} degrees when the lid prints top_down; add supports under the barbs or use {OVERHANG_LIMIT_DEG} degrees or less")
    if issues:
        return _row("dfm.overhang", "WARN", "; ".join(issues))
    return _row("dfm.overhang", "PASS", f"no overhang beyond {OVERHANG_LIMIT_DEG} degrees: the lid prints {s.print_orientation.get('lid')}, the shell {s.print_orientation.get('shell')}")


def _shrink(c: _Ctx) -> dict:
    s = c.s
    pct = c.mat["shrink"] * 100.0
    if s.shrink_comp == "clearance":
        return _row("dfm.shrink", "INFO", f"{s.material} shrinkage {pct:g}% compensated by clearance only")
    if s.shrink_comp == "none":
        return _row("dfm.shrink", "INFO", f"{s.material} shrinkage {pct:g}% is not compensated (shrink_comp 'none')")
    return _row("dfm.shrink", "INFO", f"shrink_comp {s.shrink_comp!r} is not supported yet; {s.material} shrinkage {pct:g}% is not compensated")


_BUILDERS = (_min_wall, _min_web, _hook_strain, _hook_strain_molded, _pocket_wall, _retention_angle, _port_web,
             _rim_web, _standoff_h, _overhang, _shrink)


def dfm_rows(spec: "_spec.EnclosureSpec | dict", board: dict) -> list[dict]:
    """Design-for-manufacture rows for ``spec`` on ``board``: ``[{"id", "status", "detail"}]``, ids as ``DFM_IDS``.

    PURE: no FreeCAD, no document. One row per id, in ``DFM_IDS`` order; ``status`` is PASS, WARN, FAIL or INFO (INFO:
    a fact that is shown but never judged, or a row that does not apply to this spec). Thresholds:
    wall / floor / lid at least the material's ``min_wall`` (WARN under 1.6 mm), vent web ``pitch - slot_w`` at least
    ``min_web``, hook strain ``1.5 * t * (overhang - fit) / length^2`` against ``max_strain * FDM_DERATE`` (WARN above
    80 %, FAIL above 100 %; the injection-moulded limit is an INFO row), wall behind the catch pocket at least
    ``min_wall``, retention angle WARN above 60 degrees, web between adjacent wall openings at least ``min_web``, rim
    web above every wall opening (``dfm.rim_web`` is a WARN naming each opening the kit merges with the rim, for
    example ``opening 'usb3_pair' merged with the rim (web 0.5 < 1.2)``), standoff height, overhangs (WARN only) and the
    shrinkage note. If the spec or board cannot be evaluated (unknown material, incomplete board data) every row is a
    WARN saying so; the function never raises for bad data.
    """
    try:
        c = _Ctx(_spec._as_spec(spec), board)
    except (KeyError, ValueError, TypeError) as exc:
        why = exc.args[0] if exc.args else exc
        return [_row(rid, "WARN", f"could not be evaluated: {why}") for rid in DFM_IDS]
    rows = []
    for rid, build in zip(DFM_IDS, _BUILDERS):
        try:
            rows.append(build(c))
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
            rows.append(_row(rid, "WARN", f"could not be evaluated: {exc}"))
    return rows


def fit_rows(doc: Any, spec: "_spec.EnclosureSpec | dict", reference_status: str = "none") -> list[dict]:
    """The kit-specific ``check_fit`` rows: ``containment.ports``, ``alignment.standoffs``, ``hooks.engagement`` and
    ``hooks.clear_of_cutouts``, measured on the built geometry of ``doc`` (``_fit`` has the rules).

    GUI-thread only. Rows have the ``check_fit`` shape (``id, kind, a, b, metric, value, limit, status, detail``); a check that
    cannot run (stage not built, no shell, no FreeCAD) is a SKIP row with the reason, and the function never raises. The seven
    generic rows (interference, ``clearance.pcb_to_wall_min``, ``clearance.lid_to_tallest``, ``retention.lid_lift``,
    ``reference.status``) belong to ``tools_verify_fit``; both lists are merged by ``check_fit``.
    """
    try:
        from . import _fit
    except ImportError as exc:       # no FreeCAD in this interpreter (pure unit tests)
        return [{"id": rid, "kind": "alignment", "a": "", "b": "", "metric": "", "value": 0.0, "limit": "", "status": "SKIP",
                 "detail": f"the kit fit rows need FreeCAD ({exc})"} for rid in FIT_IDS]
    return _fit.fit_rows(doc, spec, reference_status)


def spec_coverage(spec: "_spec.EnclosureSpec | dict", board: dict, doc: Any = None) -> list[dict]:
    """``[{"requirement", "status", "by", "detail", "key"}]`` seeding the Build Summary's requirement table.

    One row per kit requirement (shell, ports, lid, vents, snap, material, source, parameters); ``status`` is ``covered`` only
    when the geometry exists in ``doc`` (``partial``, ``missing``, ``n/a`` otherwise). ``requirement`` is the ledger's ``R<n>`` id
    when a ledger is bound to the calling context, else the key. ``[]`` without a document (no coverage information).
    """
    if doc is None:
        return []
    try:
        from . import _fit
    except ImportError:
        return []
    return _fit.spec_coverage(spec, board, doc)
