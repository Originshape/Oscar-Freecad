"""The nine stage builders of the enclosure kit (private module of ``oscar.kit.build``; FreeCAD, GUI-thread only).

Each ``stage_<name>(ctx)`` creates the objects of ONE stage through ``ctx.b`` (a ``geometry.Builder``: tagged, hidden, bound to
``Params``) and records what it did in ``ctx.metrics`` / ``ctx.warnings``. They never remove objects, never open a transaction and
never recompute the whole document: ``build.build_stage`` does the removal (idempotency), the transaction (atomicity), the
recompute and the post-conditions. A stage may rely on the earlier stages being built (``build_stage`` checks that).

Chains (one valid single solid per step; hidden except the tip):

* shell : ``Shell_Body`` -> ``Shell_Standoffs`` -> ``Shell_Ports`` -> ``Shell_Vents`` -> ``Shell_Hooks``
* lid   : ``Lid_Plate`` -> ``Lid_Skirt`` -> ``Lid_Vents`` -> ``Lid_Hooks``
* ref   : ``Ref_<ID>`` (``App::Part``) holding the nominal PCB and connector boxes
* assembly : ``Assembly`` (``App::Part``) holding ``Part_Shell`` and ``Part_Lid``; ``Part_Lid.Placement.Base.z = Params.lid_lift``.

A stage with nothing to build (no ports, vents disabled, a screw lid without hooks) creates no geometry; ``build_stage`` then
adds the tiny marker group ``Kit_<stage>`` so the stage still counts as built.
"""

from typing import Any

import Part
from FreeCAD import Vector as V

from . import _cutters as C
from . import geometry as G
from . import materials as _materials
from . import params as _params
from . import spec as _spec
from .geometry import EXTRA, FUSE_OVERLAP, MIN_RADIUS, PLATE_TOL, mm, signed

WALL_FACES = _spec.WALL_FACES
PROTECTED_SOURCES = ("user_requirement", "step_measurement", "datasheet")
RC_OUTER = f"max(Params.corner_r; {mm(MIN_RADIUS)})"
RC_CAVITY = f"max(Params.corner_r - Params.wall_t; {mm(MIN_RADIUS)})"
RC_LID = f"max(Params.corner_r + Params.lid_fit + Params.lip_t; {mm(MIN_RADIUS)})"
RC_SKIRT_IN = f"max(Params.corner_r + Params.lid_fit; {mm(MIN_RADIUS)})"
FLOOR_OVERLAP_EXPR = "Params.floor_t/2"      # standoff posts reach this far into the floor so they fuse with it


class KitBuildError(ValueError):
    """A stage cannot be built; the message follows the model-facing error style."""


class StageCtx:
    """Everything one stage needs: the document, the declared and the effective spec, the board, the Params values, a Builder.

    ``spec`` is the EFFECTIVE spec (the declared one overlaid with the current ``Params`` design rows, so a hand-edited
    sheet is honoured), ``declared`` the one the caller passed. ``v`` is ``{alias: float}`` of the sheet (empty during the
    ``params`` stage before the sheet exists). ``stale`` is True when the stage was recorded with another spec hash.
    """

    def __init__(self, doc: Any, declared: "_spec.EnclosureSpec", spec: "_spec.EnclosureSpec", board: dict, builder: "G.Builder", *,
                 values: "dict[str, float]", reference_status: str, rebuild: bool, stale: bool) -> None:
        self.doc, self.declared, self.spec, self.board, self.b = doc, declared, spec, board, builder
        self.v = values
        self.reference_status, self.rebuild, self.stale = reference_status, rebuild, stale
        self.warnings: list[str] = []
        self.metrics: dict = {}

    @property
    def kit_id(self) -> str:
        return self.b.kit_id

    @property
    def stage(self) -> str:
        return self.b.stage

    @property
    def pcb_t(self) -> float:
        """The PCB thickness: the sheet's ``pcb_t`` (a STEP measurement may differ from the board file), else the board's."""
        return self.v.get("pcb_t", float(self.board["pcb"]["thickness"]))

    @property
    def min_web(self) -> float:
        return _materials.get(self.spec.material)["min_web"]

    def need(self, *names: str) -> None:
        """Raise ``KitBuildError`` unless every ``Params`` row in ``names`` exists and evaluates."""
        missing = [n for n in names if n not in self.v]
        if missing:
            raise KitBuildError(f"Params: {', '.join(repr(n) for n in missing[:6])} {'is' if len(missing) == 1 else 'are'} missing or not a number; "
                                "run build_enclosure(stage='params') to restore the table")

    def container(self, part: str) -> Any:
        """The ``Part_<part>`` ``App::Part`` of this kit (created on first use)."""
        return self.b.container(f"Part_{part}", part)

    def tip(self, role: str) -> Any:
        t = G.tip(self.doc, role, self.kit_id)
        if t is None:
            raise KitBuildError(f"the {role} part has not been built yet; build the earlier stages first (build_enclosure(stage='next'))")
        return t


# --------------------------------------------------------------------------------------------------------------------
# stage 1: params
# --------------------------------------------------------------------------------------------------------------------

def _squash(text: str) -> str:
    return text.replace(" ", "").lower()


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 + 1e-9 * abs(b)


def stage_params(ctx: StageCtx) -> None:
    """Adopt (or create) the ``Params`` sheet and upsert the rows of the spec; never builds geometry.

    Missing rows are added from ``spec.param_rows``. An existing row is touched only when it differs from the spec:
    derived rows get their formula back; input rows are overwritten when the spec really changed (the stage hash is stale or
    ``rebuild``), except board rows whose source is ``user_requirement`` / ``step_measurement`` (a measured STEP value wins over
    the catalog), and design rows keep their ``user_requirement`` / ``standard`` source; otherwise the sheet value is kept and a
    warning names the difference. ``metrics``: ``{"sheet", "rows", "added", "updated", "kept"}``.
    """
    doc, board = ctx.doc, ctx.board
    rows = _spec.param_rows(ctx.declared, board)
    existed = doc.getObject(G.SHEET) is not None
    P = _params.ensure_sheet(doc, G.SHEET, kit_id=ctx.kit_id)
    if not existed:
        ctx.b.created.append(P.sheet.Name)
    push = ctx.rebuild or ctx.stale
    current = {r["name"]: r for r in P.table()}
    added, updated, kept = [], [], []
    for row in rows:
        name, cur = row["name"], current.get(row["name"])
        if cur is None:
            P.set(name, row["value"], unit=row["unit"], source=row["source"], note=row["note"], formula=row["formula"])
            added.append(name)
            continue
        if row["formula"]:
            if _squash(cur["expr"]) != _squash(row["formula"]):
                P.set(name, row["value"], unit=row["unit"], source="derived", note=cur["note"], formula=row["formula"])
                updated.append(name)
            continue
        if cur["value"] is not None and _close(cur["value"], row["value"]):
            continue
        design = name in _spec.DESIGN_PARAMS or name == "shrink_pct"
        source = cur["source"]
        if push and (design or source not in ("user_requirement", "step_measurement")):
            keep_source = source if design and source in ("user_requirement", "standard") else row["source"]
            P.set(name, row["value"], unit=row["unit"], source=keep_source, note=cur["note"] or row["note"])
            updated.append(name)
        else:
            kept.append(name)
            shown = "not a number" if cur["value"] is None else f"{cur['value']:g}"
            ctx.warnings.append(f"Params: '{name}' is {shown} in the sheet but {row['value']:g} in the spec (source {source or 'none'}); the sheet value is kept"
                                + ("" if source in PROTECTED_SOURCES else "; rebuild the params stage to apply the spec"))
    P.sheet.recompute()
    G.tag(P.sheet, kit_id=ctx.kit_id, stage="params", role="params", part="", spec_hash=ctx.b.spec_hash)
    ctx.metrics.update(sheet=P.sheet.Name, rows=len(rows), added=added, updated=updated, kept=kept)


# --------------------------------------------------------------------------------------------------------------------
# stage 2: reference
# --------------------------------------------------------------------------------------------------------------------

def _connector_shape(c: dict, dz: float, t_board: float) -> Any:
    box = c["box"]
    z0, z1 = (z + dz if z >= t_board else z for z in box["z"])
    return Part.makeBox(box["x"][1] - box["x"][0], box["y"][1] - box["y"][0], z1 - z0, V(box["x"][0], box["y"][0], z0))


def _pcb_shape(board: dict, thickness: float) -> Any:
    """The PCB: the outline extruded ``thickness`` with the mounting holes drilled through (static, board frame)."""
    o = board["outline"]
    pcb = Part.makeBox(float(o["length"]), float(o["width"]), thickness)
    r = float(o.get("corner_radius") or 0.0)
    if r > 1e-6:
        pcb = pcb.makeFillet(r, G.box_edges_along(pcb, 2))
    d = float(board["mounting_holes"]["diameter"])
    for x, y in _hole_positions(board):
        pcb = pcb.cut(Part.makeCylinder(d / 2, thickness + 2, V(x, y, -1)))
    return pcb


def stage_reference(ctx: StageCtx) -> None:
    """Build the nominal ``Ref_<ID>`` (PCB outline with holes plus one box per board connector) unless a reference exists.

    An existing ``Ref_<ID>`` that is not ours (an imported STEP, an attached file) satisfies the stage untouched. The nominal
    reference is static geometry from the board data (hidden), in the kit frame: nothing in the shell or lid depends on it.
    ``metrics``: ``{"mode": "nominal" | "existing", "object", "solids"}``.
    """
    board = ctx.board
    name = G.ref_name(board["id"])
    existing = G.ref_container(ctx.doc, board["id"])
    if existing is not None and not G.is_kit_object(existing, ctx.kit_id):
        ctx.metrics.update(mode="existing", object=existing.Name, solids=0)
        return
    if ctx.reference_status in ("step-imported", "step-measured", "user-attached"):
        ctx.warnings.append(f"reference_status is '{ctx.reference_status}' but no '{name}' object exists; the nominal board was built instead")
    b = ctx.b
    cont = b.new("App::Part", name, role="ref", part="Ref", hidden=True)
    t = ctx.pcb_t
    t_board = float(board["pcb"]["thickness"])
    dz = t - t_board
    b.static(f"{cont.Name}_PCB", _pcb_shape(board, t), role="aux", part="Ref", container=cont)
    count = 1
    for c in board.get("connectors", []):
        try:
            shape = _connector_shape(c, dz, t_board)
        except (KeyError, TypeError, ValueError) as exc:
            ctx.warnings.append(f"connector '{c.get('id')}' has no usable box ({exc}) and was left out of the reference")
            continue
        b.static(f"{cont.Name}_{C.sanitize(c.get('id'))}", shape, role="aux", part="Ref", container=cont)
        count += 1
    ctx.metrics.update(mode="nominal", object=cont.Name, solids=count, pcb_t=t)


# --------------------------------------------------------------------------------------------------------------------
# stage 3: shell
# --------------------------------------------------------------------------------------------------------------------

def stage_shell(ctx: StageCtx) -> None:
    """The shell body: a rounded outer prism minus the cavity, both Sketcher rounded rectangles bound to ``Params``.

    Outer outline ``shell_len`` x ``shell_wid`` with radius ``corner_r`` from ``z_floor_bot`` up ``shell_h``; the cavity
    ``cav_len`` x ``cav_wid`` with radius ``corner_r - wall_t`` from ``z_floor_top`` up through the rim. One ``Part::Cut``
    (``Shell_Body``). ``metrics``: ``{"outer": [len, wid, h], "cavity": [len, wid], "z": [bottom, rim]}``.
    """
    v, b = ctx.v, ctx.b
    ctx.need("cav_x0", "cav_y0", "cav_len", "cav_wid", "shell_len", "shell_wid", "shell_h", "wall_t", "corner_r", "z_floor_bot",
             "z_floor_top", "z_cavity_top")
    cs = ctx.container("Shell")
    sk_o = b.rrect_sketch(
        "Shell_OuterSketch", width=v["shell_len"], height=v["shell_wid"], radius=v["corner_r"],
        origin=(v["cav_x0"] - v["wall_t"], v["cav_y0"] - v["wall_t"], v["z_floor_bot"]),
        exprs={"size_x": "Params.shell_len", "size_y": "Params.shell_wid", "rc": RC_OUTER,
               "Placement.Base.x": "Params.cav_x0 - Params.wall_t", "Placement.Base.y": "Params.cav_y0 - Params.wall_t",
               "Placement.Base.z": "Params.z_floor_bot"}, part="Shell", container=cs)
    outer = b.extrude("Shell_Outer", sk_o, length=v["shell_h"], length_expr="Params.shell_h", part="Shell", container=cs)
    sk_c = b.rrect_sketch(
        "Shell_CavitySketch", width=v["cav_len"], height=v["cav_wid"], radius=max(v["corner_r"] - v["wall_t"], MIN_RADIUS),
        origin=(v["cav_x0"], v["cav_y0"], v["z_floor_top"]),
        exprs={"size_x": "Params.cav_len", "size_y": "Params.cav_wid", "rc": RC_CAVITY, "Placement.Base.x": "Params.cav_x0",
               "Placement.Base.y": "Params.cav_y0", "Placement.Base.z": "Params.z_floor_top"}, part="Shell", container=cs)
    cavity = b.extrude("Shell_Cavity", sk_c, length=v["z_cavity_top"] - v["z_floor_top"] + EXTRA,
                       length_expr=f"Params.z_cavity_top - Params.z_floor_top + {mm(EXTRA)}", part="Shell", container=cs)
    b.cut("Shell_Body", outer, cavity, role="shell", part="Shell", container=cs)
    ctx.metrics.update(outer=[v["shell_len"], v["shell_wid"], v["shell_h"]], cavity=[v["cav_len"], v["cav_wid"]],
                       z=[v["z_floor_bot"], v["z_cavity_top"]])


# --------------------------------------------------------------------------------------------------------------------
# stage 4: standoffs
# --------------------------------------------------------------------------------------------------------------------

def _hole_positions(board: dict) -> list[tuple[float, float]]:
    """Mounting hole centres of ``board``: its ``positions`` list, else the expansion of its ``pattern``."""
    mh = board["mounting_holes"]
    pos = mh.get("positions") or []
    if pos:
        return [(float(p[0]), float(p[1])) for p in pos]
    pat = mh.get("pattern") or {}
    ox, oy = pat["offset"]
    dx, dy = float(pat["dx"]), float(pat["dy"])
    return [(ox, oy), (ox + dx, oy), (ox, oy + dy), (ox + dx, oy + dy)]


def _hole_exprs(v: dict, x: float, y: float) -> "tuple[str | None, str | None]":
    """Expressions of a hole centre in terms of ``hole_off`` / ``hole_dx`` / ``hole_dy`` when it sits on that lattice, else None."""
    off = v.get("hole_off")
    if off is None:
        return None, None

    def pick(c: float, step: str) -> "str | None":
        if _close(c, off):
            return "Params.hole_off"
        if step in v and _close(c, off + v[step]):
            return f"Params.hole_off + Params.{step}"
        return None

    return pick(x, "hole_dx"), pick(y, "hole_dy")


def stage_standoffs(ctx: StageCtx) -> None:
    """Board posts at every mounting hole, fused into the shell floor, each with a blind pilot hole.

    Posts: cylinders of diameter ``standoff_d`` from ``floor_t / 2`` below the floor top up to the PCB bottom (z 0); pilot
    holes of diameter ``standoff_pilot_d`` from the floor top up through the post. Hole centres come from the board data and
    are bound to ``hole_off`` / ``hole_dx`` / ``hole_dy`` where they lie on that lattice. One chain object (``Shell_Standoffs``).
    ``metrics``: ``{"count", "positions": [[x, y]], "post_d", "pilot_d", "height"}``.
    """
    v, b = ctx.v, ctx.b
    ctx.need("standoff_h", "standoff_d", "standoff_pilot_d", "floor_t", "z_floor_top")
    cs = ctx.container("Shell")
    base = ctx.tip("shell")
    holes = _hole_positions(ctx.board)
    posts, pilots = [], []
    for i, (x, y) in enumerate(holes, 1):
        ex, ey = _hole_exprs(v, x, y)
        pos = {"Placement.Base.x": ex, "Placement.Base.y": ey}
        posts.append(b.cylinder(
            f"Shell_StandoffPost{i}", v["standoff_d"] / 2, v["standoff_h"] + v["floor_t"] / 2, (x, y, v["z_floor_top"] - v["floor_t"] / 2),
            exprs={"Radius": "Params.standoff_d/2", "Height": f"Params.standoff_h + {FLOOR_OVERLAP_EXPR}",
                   "Placement.Base.z": f"Params.z_floor_top - {FLOOR_OVERLAP_EXPR}", **pos}, part="Shell", container=cs))
        pilots.append(b.cylinder(
            f"Shell_StandoffPilot{i}", v["standoff_pilot_d"] / 2, v["standoff_h"] + EXTRA, (x, y, v["z_floor_top"]),
            exprs={"Radius": "Params.standoff_pilot_d/2", "Height": f"Params.standoff_h + {mm(EXTRA)}",
                   "Placement.Base.z": "Params.z_floor_top", **pos}, part="Shell", container=cs))
    fused = b.fuse("Shell_StandoffFuse", base, b.union("Shell_StandoffPosts", posts, part="Shell", container=cs), part="Shell", container=cs)
    b.cut("Shell_Standoffs", fused, b.union("Shell_StandoffPilots", pilots, part="Shell", container=cs), role="shell", part="Shell",
          container=cs)
    ctx.metrics.update(count=len(holes), positions=[[x, y] for x, y in holes], post_d=v["standoff_d"], pilot_d=v["standoff_pilot_d"],
                       height=v["standoff_h"])


# --------------------------------------------------------------------------------------------------------------------
# openings (ports and user cut-outs)
# --------------------------------------------------------------------------------------------------------------------

def _user_cutouts(spec: "_spec.EnclosureSpec") -> list[dict]:
    """The valid ``spec.cutouts`` entries as openings (margin added around w x h) keeping ``part``, ``r`` and ``through``."""
    out = []
    for i, e in enumerate(spec.cutouts):
        if not isinstance(e, dict) or e.get("part") not in _spec.PARTS or e.get("face") not in _spec.FACES:
            continue
        nums = [e.get(k) for k in ("u", "v", "w", "h")]
        if not all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in nums):
            continue
        m = e["margin"] if isinstance(e.get("margin"), (int, float)) and not isinstance(e.get("margin"), bool) else 0.0
        r = e["r"] if isinstance(e.get("r"), (int, float)) and not isinstance(e.get("r"), bool) else 0.0
        out.append({"id": str(e.get("id") or f"cutout{i + 1}"), "kind": "cutout", "part": e["part"], "face": e["face"], "u": float(nums[0]),
                    "v": float(nums[1]), "w": float(nums[2]) + 2 * m, "h": float(nums[3]) + 2 * m, "r": float(r),
                    "blind": e.get("through") is False, "top_r": float(nums[1]) + float(nums[3]) / 2 + m})
    return out


def openings(ctx: StageCtx) -> list[dict]:
    """Every opening to cut, shell and lid: connector ports then user cut-outs.

    Each dict: ``{"id", "kind" ('port' | 'cutout'), "part" ('shell' | 'lid'), "face", "u", "v", "w", "h", "r", "blind", "top_r"}``
    in absolute world coordinates. A port on a ``top`` face belongs to the lid, every other port to the shell; ``top_r`` is
    the height of the receptacle (a port) or of the cut-out edge, used to decide which openings reach the lid skirt.
    """
    sp, board = ctx.spec, ctx.board
    t_board = float(board["pcb"]["thickness"])
    dz = ctx.pcb_t - t_board
    by_id = {c.get("id"): c for c in board.get("connectors", [])}
    rows = []
    for o in _spec.port_openings(sp, board, pcb_t=ctx.pcb_t):
        c = by_id.get(o["id"], {})
        z1 = c["box"]["z"][1] if c.get("box") else o["v"] + o["h"] / 2
        rows.append({"id": o["id"], "kind": "port", "part": "lid" if o["face"] == "top" else "shell", "face": o["face"], "u": o["u"],
                     "v": o["v"], "w": o["w"], "h": o["h"], "r": 0.0, "blind": False,
                     "top_r": z1 + dz if z1 >= t_board else z1})
    return rows + _user_cutouts(sp)


def _bounds(o: dict) -> "tuple[float, float, float, float]":
    return o["u"] - o["w"] / 2, o["u"] + o["w"] / 2, o["v"] - o["h"] / 2, o["v"] + o["h"] / 2


def stage_ports(ctx: StageCtx) -> None:
    """Cut the connector openings and the shell's user cut-outs through the walls (and the floor).

    One box per opening, sized by ``spec.port_openings`` (receptacle plus margin, at least the cable plug) and placed with its
    wall-normal extent bound to ``wall_t`` and the cavity faces. An opening whose top comes within ``min_web`` of the rim is
    extended up through the rim as a notch (the lid plate closes it from above) and reported as ``opening '<id>' merged with
    the rim (web <w> < <min_web>)``. A ``top`` opening belongs to the lid (cut by the ``vents`` stage). ``metrics``:
    ``{"openings": [{"id", "face", "u", "v", "w", "h", "notch"}], "rim_merged": [id]}``.
    """
    v, b = ctx.v, ctx.b
    ctx.need("cav_x0", "cav_x1", "cav_y0", "cav_y1", "wall_t", "z_cavity_top", "z_floor_bot", "floor_t")
    rim, web = v["z_cavity_top"], ctx.min_web
    rows = [o for o in openings(ctx) if o["part"] == "shell"]
    cutters, listed, merged = [], [], []
    if not rows:
        return
    cs = ctx.container("Shell")
    for o in rows:
        face = o["face"]
        span = C.through_span(v, "shell", face, blind=o["blind"])
        if span is None:
            ctx.warnings.append(f"opening '{o['id']}' is on the {face} face, which the shell does not have; it was skipped")
            continue
        u0, u1, v0, v1 = _bounds(o)
        exprs: dict = {}
        notch = False
        if face in WALL_FACES and v1 > rim - web + _spec.TOL:
            notch = True
            exprs["dv"] = f"Params.z_cavity_top + {mm(EXTRA)}{signed(-v0)}"
            v1 = rim + EXTRA
            merged.append(f"opening '{o['id']}' merged with the rim (web {_spec._n(rim - (o['v'] + o['h'] / 2))} < {_spec._n(web)})")
        cutters.append(C.rect_cutter(ctx, f"Shell_Port_{C.sanitize(o['id'])}", part="Shell", container=cs, face=face, span=span,
                                     u0=u0, u1=u1, v0=v0, v1=v1, exprs=exprs, rounded=o["r"]))
        listed.append({"id": o["id"], "kind": o["kind"], "face": face, "u": o["u"], "v": o["v"], "w": o["w"], "h": o["h"], "notch": notch})
    ctx.warnings.extend(merged)
    if cutters:
        tool = b.union("Shell_PortCutters", cutters, part="Shell", container=cs)
        b.cut("Shell_Ports", ctx.tip("shell"), tool, role="shell", part="Shell", container=cs)
    ctx.metrics.update(openings=listed, rim_merged=[m.split("'")[1] for m in merged])


# --------------------------------------------------------------------------------------------------------------------
# stage 6: lid
# --------------------------------------------------------------------------------------------------------------------

def stage_lid(ctx: StageCtx) -> None:
    """The cap lid: a plate of ``lid_t`` at the rim and a skirt of ``lip_h`` x ``lip_t`` wrapping the OUTSIDE of the shell walls.

    The plate (``Lid_Plate``) and the skirt ring (outer outline minus the shell outline grown by ``lid_fit``, hanging down from
    the plate underside) are fused, then ``Lid_Skirt`` is that fusion with a notch cut where an opening of a shell wall reaches
    into the skirt band (the receptacle top above the skirt bottom: the tall USB stacks of the Pi 4), so a cable plug can pass.
    The notches stop at the plate underside, so the plate stays closed. All outlines are
    Sketcher rounded rectangles bound to ``Params``. ``metrics``: ``{"outer": [len, wid], "skirt_z": [bottom, top], "notches": [id]}``.
    """
    v, b = ctx.v, ctx.b
    ctx.need("cav_x0", "cav_y0", "lid_len", "lid_wid", "shell_len", "shell_wid", "lid_t", "lip_h", "lip_t", "lid_fit", "wall_t", "corner_r",
             "z_cavity_top")
    cl = ctx.container("Lid")
    oz, fit, lip_t, lip_h = v["z_cavity_top"], v["lid_fit"], v["lip_t"], v["lip_h"]
    sk = b.rrect_sketch(
        "Lid_PlateSketch", width=v["lid_len"], height=v["lid_wid"], radius=v["corner_r"] + fit + lip_t,
        origin=(v["cav_x0"] - v["wall_t"] - fit - lip_t, v["cav_y0"] - v["wall_t"] - fit - lip_t, oz),
        exprs={"size_x": "Params.lid_len", "size_y": "Params.lid_wid", "rc": RC_LID,
               "Placement.Base.x": "Params.cav_x0 - Params.wall_t - Params.lid_fit - Params.lip_t",
               "Placement.Base.y": "Params.cav_y0 - Params.wall_t - Params.lid_fit - Params.lip_t",
               "Placement.Base.z": "Params.z_cavity_top"}, part="Lid", container=cl)
    plate = b.extrude("Lid_Plate", sk, length=v["lid_t"], length_expr="Params.lid_t", role="lid", part="Lid", container=cl)
    outer = b.extrude("Lid_SkirtOuter", sk, length=lip_h, length_expr="Params.lip_h", reverse=True, part="Lid", container=cl)
    sk_in = b.rrect_sketch(
        "Lid_SkirtInnerSketch", width=v["shell_len"] + 2 * fit, height=v["shell_wid"] + 2 * fit, radius=v["corner_r"] + fit,
        origin=(v["cav_x0"] - v["wall_t"] - fit, v["cav_y0"] - v["wall_t"] - fit, oz - lip_h - EXTRA),
        exprs={"size_x": "Params.shell_len + 2*Params.lid_fit", "size_y": "Params.shell_wid + 2*Params.lid_fit", "rc": RC_SKIRT_IN,
               "Placement.Base.x": "Params.cav_x0 - Params.wall_t - Params.lid_fit",
               "Placement.Base.y": "Params.cav_y0 - Params.wall_t - Params.lid_fit",
               "Placement.Base.z": f"Params.z_cavity_top - Params.lip_h - {mm(EXTRA)}"}, part="Lid", container=cl)
    inner = b.extrude("Lid_SkirtInner", sk_in, length=lip_h + EXTRA + FUSE_OVERLAP,
                      length_expr=f"Params.lip_h + {mm(EXTRA + FUSE_OVERLAP)}", part="Lid", container=cl)
    ring = b.cut("Lid_SkirtRing", outer, inner, part="Lid", container=cl)
    # skirt notches for tall openings (a plug must reach the connector through the skirt band)
    skirt_bottom = oz - lip_h
    notch_cutters, notch_ids = [], []
    for o in openings(ctx):
        if o["part"] != "shell" or o["face"] not in WALL_FACES or o["top_r"] <= skirt_bottom + PLATE_TOL:
            continue
        span = C.through_span(v, "lid", o["face"])
        u0, u1, v0, _v1 = _bounds(o)
        exprs: dict = {}
        if v0 < skirt_bottom - EXTRA:
            v0 = skirt_bottom - EXTRA
            origin_expr = f"Params.z_cavity_top - Params.lip_h - {mm(EXTRA)}"
            exprs.update(v_origin=origin_expr, dv=f"Params.z_cavity_top - ({origin_expr})")
        else:
            exprs["dv"] = f"Params.z_cavity_top{signed(-v0)}"
        notch_cutters.append(C.rect_cutter(ctx, f"Lid_Notch_{C.sanitize(o['id'])}", part="Lid", container=cl, face=o["face"], span=span,
                                           u0=u0, u1=u1, v0=v0, v1=oz, exprs=exprs))
        notch_ids.append(o["id"])
    skirted = b.fuse("Lid_SkirtBase" if notch_cutters else "Lid_Skirt", plate, ring, role="aux" if notch_cutters else "lid", part="Lid", container=cl)
    if notch_cutters:
        b.cut("Lid_Skirt", skirted, b.union("Lid_SkirtNotches", notch_cutters, part="Lid", container=cl), role="lid", part="Lid", container=cl)
    ctx.metrics.update(outer=[v["lid_len"], v["lid_wid"]], skirt_z=[skirt_bottom, oz], notches=notch_ids)


# --------------------------------------------------------------------------------------------------------------------
# stage 7: vents
# --------------------------------------------------------------------------------------------------------------------

def stage_vents(ctx: StageCtx) -> None:
    """Cut the vent array and every cut-out that belongs to the lid; shell-part vents go into the shell.

    The array (``spec.vents``) is ``count`` slots ``slot_l`` x ``slot_w`` along u at ``pitch`` across v (or round holes), through
    the ``part`` / ``face`` it names, centred on ``center`` (default: the face centre over the cavity, bound to ``Params``).
    Lid cut-outs (``cutouts[]`` with ``part == 'lid'``) and connector openings on a ``top`` face are cut into the lid in the same
    pass. A face the part does not have (a shell ``top``, a lid ``bottom``) is skipped with a warning. ``metrics``:
    ``{"vents": {"part", "face", "count", "pattern"} | None, "cutouts": [id]}``.
    """
    vs, b = ctx.spec.vents, ctx.b
    cutters: dict[str, list] = {"lid": [], "shell": []}
    info: "dict | None" = None
    if vs.enabled:
        span = C.through_span(ctx.v, vs.part, vs.face)
        if span is None:
            ctx.warnings.append(f"vents: the {vs.part} has no '{vs.face}' face, so the vents were skipped; use part 'lid' with face 'top' or a wall face")
        else:
            cutters[vs.part] += C.slot_cutters(ctx, f"{vs.part.capitalize()}_Vent", part=vs.part.capitalize(), container=ctx.container(vs.part.capitalize()),
                                               face=vs.face, span=span, center=vs.center, count=int(vs.count), slot_w=vs.slot_w, slot_l=vs.slot_l,
                                               pitch=vs.pitch, angle=float(vs.angle_deg), pattern=vs.pattern)
            info = {"part": vs.part, "face": vs.face, "count": int(vs.count), "pattern": vs.pattern}
    done = []
    for o in openings(ctx):
        if o["part"] != "lid":
            continue
        span = C.through_span(ctx.v, "lid", o["face"], blind=o["blind"])
        if span is None:
            ctx.warnings.append(f"opening '{o['id']}' is on the lid's {o['face']} face, which the lid does not have; it was skipped")
            continue
        u0, u1, v0, v1 = _bounds(o)
        cutters["lid"].append(C.rect_cutter(ctx, f"Lid_Cut_{C.sanitize(o['id'])}", part="Lid", container=ctx.container("Lid"), face=o["face"],
                                            span=span, u0=u0, u1=u1, v0=v0, v1=v1, rounded=o["r"]))
        done.append(o["id"])
    for part, objs in cutters.items():
        if not objs:
            continue
        label = part.capitalize()
        cont = ctx.container(label)
        tool = b.union(f"{label}_VentCutters", objs, part=label, container=cont)
        b.cut(f"{label}_Vents", ctx.tip(part), tool, role=part, part=label, container=cont)
    ctx.metrics.update(vents=info, cutouts=done)


# --------------------------------------------------------------------------------------------------------------------
# stage 8: hooks
# --------------------------------------------------------------------------------------------------------------------

def stage_hooks(ctx: StageCtx) -> None:
    """Snap-fit hooks: cantilever beams with barbs on the lid, blind catch pockets in the shell's outer walls.

    Positions come from ``spec.hook_positions`` (clear of every wall opening). Each beam continues the skirt downward by
    ``hooks.length``; its barb (``geometry.barb_profile``) protrudes inward by ``hooks.overhang`` and engages the pocket by
    ``overhang - fit``. The pocket is ``hooks.w + 0.4`` wide, ``barb_h + window_margin`` tall and ``overhang - fit + 0.1`` deep, so
    with the lid seated the barb top is ``window_margin`` below the pocket ceiling. Beam, barb and pocket positions are bound to
    the wall, ``lid_fit``, ``lip_h`` and ``hook_len`` rows so wall / clearance edits move them with the skirt. A lid style other
    than ``snap_fit`` builds nothing. ``metrics``: ``{"positions", "strain", "strain_limit", "pocket", "barb"}``.
    """
    sp, v, b = ctx.spec, ctx.v, ctx.b
    if sp.lid.style != "snap_fit":
        note = (f"lid style '{sp.lid.style}' has no snap hooks; the lid is a plain cap held by friction on the shell"
                + ("; screw bosses are not modelled, fasten it with tape or glue or choose snap_fit" if sp.lid.style == "screw" else "; lid.fit sets the grip"))
        ctx.metrics.update(positions=[], note=note)
        ctx.warnings.append(note)
        return
    ctx.need("cav_y0", "cav_y1", "wall_t", "lid_fit", "lip_h", "hook_len", "z_cavity_top")
    plans = G.hook_plans(sp, ctx.board, pcb_t=ctx.pcb_t)
    if len(plans) < sp.hooks.count:
        raise KitBuildError(f"hooks.count: only {len(plans)} positions are free of port openings on '{ctx.board.get('id')}'; reduce count or move ports")
    cl, cs = ctx.container("Lid"), ctx.container("Shell")
    z_b_expr = "Params.z_cavity_top - Params.lip_h - Params.hook_len"
    z_b = v["z_cavity_top"] - v["lip_h"] - v["hook_len"]
    fit, o_hang = v["lid_fit"], sp.hooks.overhang
    solids, pockets = [], []
    for i, p in enumerate(plans, 1):
        plus = p["wall"] == "+y"
        y_in = (v["cav_y1"] + v["wall_t"] + fit) if plus else (v["cav_y0"] - v["wall_t"] - fit)
        y_in_expr = ("Params.cav_y1 + Params.wall_t + Params.lid_fit" if plus else "Params.cav_y0 - Params.wall_t - Params.lid_fit")
        t, w = p["t"], p["w"]
        y_beam = y_in if plus else y_in - t
        beam = b.box(f"Lid_Hook{i}_Beam", (w, t, v["hook_len"] + FUSE_OVERLAP), (p["beam"]["x0"], y_beam, z_b),
                     exprs={"Placement.Base.y": y_in_expr if plus else y_in_expr + signed(-t), "Placement.Base.z": z_b_expr,
                            "Height": f"Params.hook_len + {mm(FUSE_OVERLAP)}"}, role="hook", part="Lid", container=cl)
        barb = b.static(f"Lid_Hook{i}_Barb", G.barb_shape(p["barb"], w, p["inward"]), (p["beam"]["x0"], y_in, z_b),
                        exprs={"Placement.Base.y": y_in_expr, "Placement.Base.z": z_b_expr}, role="hook", part="Lid", container=cl)
        solids += [beam, barb]
        pk = p["pocket"]
        depth_expr = f"({mm(o_hang)} - Params.lid_fit + {mm(_spec.POCKET_SEAT)})"
        depth = o_hang - fit + _spec.POCKET_SEAT
        outer = (v["cav_y1"] + v["wall_t"]) if plus else (v["cav_y0"] - v["wall_t"])
        y_lo = outer - depth if plus else outer - EXTRA
        y_expr = (f"Params.cav_y1 + Params.wall_t - {depth_expr}" if plus else f"Params.cav_y0 - Params.wall_t - {mm(EXTRA)}")
        pockets.append(b.box(f"Shell_Pocket{i}", (pk["w"], depth + EXTRA, pk["h"]), (pk["x0"], y_lo, z_b + pk["z_off"]),
                             exprs={"Placement.Base.y": y_expr, "Width": f"{depth_expr} + {mm(EXTRA)}",
                                    "Placement.Base.z": z_b_expr + signed(pk["z_off"])}, part="Shell", container=cs))
    hooks = b.union("Lid_HookSolids", solids, role="hook", part="Lid", container=cl)
    b.fuse("Lid_Hooks", ctx.tip("lid"), hooks, role="lid", part="Lid", container=cl)
    b.cut("Shell_Hooks", ctx.tip("shell"), b.union("Shell_Pockets", pockets, part="Shell", container=cs), role="shell", part="Shell", container=cs)
    prof = plans[0]["barb"]
    pk0 = plans[0]["pocket"]
    ctx.metrics.update(
        positions=[{"wall": p["wall"], "u": p["u"], "w": p["w"]} for p in plans], strain=_spec.hook_strain(sp), strain_limit=_spec.hook_strain_limit(sp),
        pocket={"w": pk0["w"], "h": pk0["h"], "depth": pk0["depth"], "z_floor": z_b + pk0["z_off"]},
        barb={"engagement": prof["engagement"], "ramp_eff_deg": round(prof["ramp_eff_deg"], 2), "retention_eff_deg": round(prof["retention_eff_deg"], 2),
              "flattened": prof["flattened"], "land": round(prof["land"], 4)})
    if prof["retention_eff_deg"] > 60:
        ctx.warnings.append(f"hooks: barb_h {sp.hooks.barb_h:g} mm limits the retention face to {prof['retention_eff_deg']:.0f} degrees from vertical; "
                            "the lid may be hard to open again, raise hooks.barb_h or lower hooks.overhang")
    if prof["ramp_eff_deg"] > 70:
        ctx.warnings.append(f"hooks: barb_h {sp.hooks.barb_h:g} mm limits the lead-in ramp to {prof['ramp_eff_deg']:.0f} degrees from vertical; "
                            "the lid may need a lot of force to close, raise hooks.barb_h")


# --------------------------------------------------------------------------------------------------------------------
# stage 9: assembly
# --------------------------------------------------------------------------------------------------------------------

def stage_assembly(ctx: StageCtx) -> None:
    """Group the parts: ``Assembly`` contains ``Part_Shell`` and ``Part_Lid``; ``Part_Lid.Placement.Base.z`` is bound to ``lid_lift``.

    The tips are shown and coloured by ``build_stage`` after the stage (``spec.colors``; headless there are no view objects).
    ``metrics``: ``{"lid_lift", "colors": {"shell", "lid", "ref"}}``.
    """
    b = ctx.b
    ctx.need("lid_lift")
    parts = {}
    for label in ("Shell", "Lid"):
        found = [o for o in ctx.doc.Objects if G.tags_of(o)["OscarKit"] == ctx.kit_id and G.tags_of(o)["OscarRole"] == "container"
                 and G.tags_of(o)["OscarPart"] == label]
        if not found:
            raise KitBuildError(f"the {label.lower()} part has not been built yet; build the earlier stages first (build_enclosure(stage='next'))")
        parts[label] = found[0]
    asm = b.new("App::Part", "Assembly", role="assembly", part="", hidden=False)
    for cont in parts.values():
        asm.addObject(cont)
    b.bind(parts["Lid"], "Placement.Base.z", "Params.lid_lift")
    ctx.metrics.update(lid_lift=ctx.v["lid_lift"], colors={k: list(map(float, c)) for k, c in ctx.spec.colors.items()})


STAGE_FUNCS = {"params": stage_params, "reference": stage_reference, "shell": stage_shell, "standoffs": stage_standoffs, "ports": stage_ports,
               "lid": stage_lid, "vents": stage_vents, "hooks": stage_hooks, "assembly": stage_assembly}
