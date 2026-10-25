"""Measuring an imported reference part from its EXACT B-rep.

FreeCAD module: every function that takes solids or an ``App.Document`` is GUI-thread only (tool handlers reach it through
``cad.dispatch``). The measurement works on the B-rep of a STEP / BREP import: bounding boxes, planes and the cylinder radius /
axis of the mounting holes come straight from ``Face.Surface``. It is deterministic, needs no subprocess and no tessellation,
which also keeps the ``Params`` values (and so ``spec_hash``) identical between runs. Meshes (STL) have no B-rep: see
``measure_mesh``.

What it finds, in this order:

1. the PCB: the largest thin axis-aligned solid (a slab with holes still fills its bounding box), or, for a model that was
   exported as ONE fused solid, the pair of opposite planar faces 0.4 .. 3.5 mm apart that covers the board;
2. the board FRAME: origin = lower-left corner of the PCB on its bottom surface, +X along the long edge, +Z up through
   the PCB towards the components. The "up" side is the side the components stand on; the +X direction is decided by the
   catalog board file when one is given (the orientation whose solids overlap the board's connector boxes best), else the
   positive direction of the long axis is kept. The transform is a rigid motion, so it is exactly reversible and applying it
   twice changes nothing: ``fit_frame`` of an already normalized reference is the identity;
3. in that frame: the PCB size and thickness, the mounting holes (cylindrical faces parallel to the board normal whose axis
   lies in empty space: a boss or a rounded corner has its axis inside the material), the corner radius, the connector
   candidates (solids that reach the PCB edge: side, offset along the wall, width, height, overhang) and the
   tallest / lowest z of everything.

Connector candidates need the connectors to be separate solids; a model fused into one lump reports the PCB and the holes only,
and says so in ``warnings``.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import FreeCAD as App
import Part

from . import identity

PCB_T_RANGE = (0.4, 3.5)        # accepted PCB thickness, mm (the board validator allows 0.4 .. 3.2)
THIN_RATIO = 0.12               # a slab: thickness / longest side below this
SLAB_FILL = 0.55                # a slab's volume must fill this share of its bounding box (so it is axis-aligned)
MIN_FOOTPRINT = 100.0           # smallest PCB footprint, mm2
HOLE_D_RANGE = (1.6, 8.0)       # diameters that count as holes
MOUNT_D_RANGE = (2.0, 6.5)      # diameters that count as MOUNTING holes (M2 .. M6)
EDGE_TOL = 0.5                  # a wall connector may sit this far inside the PCB edge
MIN_PART_VOLUME = 0.5           # mm3: smaller solids are ignored for connectors and heights
MAX_CYLINDER_FACES = 3000       # examined per PCB (real vendor models have thousands of vias)
TALL_ABOVE = 2.0                # a "tall component" stands at least this far above the PCB top
SIDES = ("-y", "+x", "+y", "-x")

_AXES = (App.Vector(1, 0, 0), App.Vector(0, 1, 0), App.Vector(0, 0, 1))


# --- records ---------------------------------------------------------------------

@dataclass
class SolidRecord:
    """One solid, in the frame the caller measures in. ``obj`` is the document object it came from (``""`` for a bare shape)."""

    name: str
    shape: Any
    obj: str = ""
    box: tuple = field(default=(), repr=False)
    volume: float = 0.0

    def __post_init__(self) -> None:
        if not self.box:
            self.box = bbox_of(self.shape)
        if not self.volume:
            try:
                self.volume = abs(float(self.shape.Volume))
            except Exception:  # noqa: BLE001 - a solid with no volume is simply ignored by the size rules
                self.volume = 0.0

    @property
    def size(self) -> tuple:
        return (self.box[3] - self.box[0], self.box[4] - self.box[1], self.box[5] - self.box[2])


def bbox_of(shape) -> tuple:
    """``(xmin, ymin, zmin, xmax, ymax, zmax)`` of a shape: the tight box when FreeCAD offers it, else ``BoundBox``."""
    box = None
    optimal = getattr(shape, "optimalBoundingBox", None)
    if optimal is not None:
        try:
            box = optimal(True, False)
        except Exception:  # noqa: BLE001
            box = None
    if box is None:
        box = shape.BoundBox
    return (box.XMin, box.YMin, box.ZMin, box.XMax, box.YMax, box.ZMax)


_SKIP_TYPES = frozenset({"App::Origin", "App::Line", "App::Plane", "App::Point"})


def _members(root) -> list:
    """The leaf objects of ``root``: a group / ``App::Part`` is flattened (depth first, document order), anything else is itself."""
    tid = getattr(root, "TypeId", "")
    if tid in _SKIP_TYPES:
        return []
    group = getattr(root, "Group", None)
    if tid in ("App::Part", "App::DocumentObjectGroup") or (group is not None and tid != "PartDesign::Body"):
        out = []
        for child in list(group or []):
            out.extend(_members(child))
        return out
    return [root]


def global_shape(obj):
    """A copy of the shape of ``obj`` placed in the GLOBAL frame (the shape of an object inside an ``App::Part`` is in the container frame)."""
    shape = obj.Shape.copy()
    shape.Placement = obj.getGlobalPlacement()
    return shape


def collect_solids(root) -> list:
    """The solids of ``root`` (a reference ``App::Part``, a group or a single object) as ``SolidRecord`` s in the GLOBAL frame.

    An object that holds several solids contributes one record each (``Label#1``, ``Label#2``); non-shape members (meshes,
    sketches, Origin features) are skipped. GUI-thread only.
    """
    records: list = []
    for obj in _members(root):
        shape = getattr(obj, "Shape", None)
        if shape is None or shape.isNull():
            continue
        try:
            placed = global_shape(obj)
            solids = list(placed.Solids)
        except Exception:  # noqa: BLE001 - one unreadable member must not hide the others
            continue
        label = str(getattr(obj, "Label", "") or obj.Name)
        for index, solid in enumerate(solids, 1):
            records.append(SolidRecord(label if len(solids) == 1 else f"{label}#{index}", solid, obj.Name))
    return records


# --- the PCB ---------------------------------------------------------------------

@dataclass
class Pcb:
    """The PCB found in a set of solids: ``axis`` (0, 1, 2) is its thin direction; ``lo`` / ``hi`` its bounding box."""

    axis: int
    lo: tuple
    hi: tuple
    method: str                  # "solid" (a slab solid) | "faces" (a fused model: the plane pair)
    index: int                   # index of the solid that is (or contains) the PCB
    name: str = ""

    @property
    def thickness(self) -> float:
        return self.hi[self.axis] - self.lo[self.axis]

    def extent(self, axis: int) -> float:
        return self.hi[axis] - self.lo[axis]


def _slab_pcb(solids: list) -> "Pcb | None":
    best = None
    for index, rec in enumerate(solids):
        sx, sy, sz = rec.size
        dims = sorted((sx, sy, sz))
        thin, longest = dims[0], dims[2]
        volume_box = sx * sy * sz
        if volume_box <= 0 or thin < PCB_T_RANGE[0] or thin > PCB_T_RANGE[1] or thin / longest > THIN_RATIO:
            continue
        if rec.volume / volume_box < SLAB_FILL or dims[1] * dims[2] < MIN_FOOTPRINT:
            continue
        footprint = dims[1] * dims[2]
        if best is None or footprint > best[0] + 1e-9:
            axis = (sx, sy, sz).index(thin)
            best = (footprint, Pcb(axis, rec.box[:3], rec.box[3:], "solid", index, rec.name))
    return best[1] if best else None


def _plane_faces(shape) -> dict:
    """Axis-aligned planar faces of a shape: ``{(axis, sign): [(area, offset, face_box), ...]}`` (sign = outward normal)."""
    found: dict = {}
    for face in shape.Faces:
        if not isinstance(face.Surface, Part.Plane):
            continue
        try:
            normal = face.normalAt(0, 0)
        except Exception:  # noqa: BLE001
            continue
        comps = (normal.x, normal.y, normal.z)
        axis = max(range(3), key=lambda i: abs(comps[i]))
        if abs(comps[axis]) < 0.9999:
            continue
        box = bbox_of(face)
        found.setdefault((axis, 1 if comps[axis] > 0 else -1), []).append((face.Area, box[axis], box))
    return found


def _face_pair_pcb(solids: list) -> "Pcb | None":
    """A fused model: the opposite planar faces 0.4 .. 3.5 mm apart that cover the board (largest overlap area wins)."""
    ranked = sorted(range(len(solids)), key=lambda i: -solids[i].volume)[:3]
    best = None
    for index in ranked:
        planes = _plane_faces(solids[index].shape)
        for axis in range(3):
            tops, bottoms = planes.get((axis, 1), []), planes.get((axis, -1), [])
            for top_area, top_off, top_box in sorted(tops, reverse=True)[:12]:
                for bot_area, bot_off, bot_box in sorted(bottoms, reverse=True)[:12]:
                    gap = top_off - bot_off
                    if gap < PCB_T_RANGE[0] or gap > PCB_T_RANGE[1]:
                        continue
                    score = min(top_area, bot_area)
                    if score < MIN_FOOTPRINT or (best is not None and score <= best[0] + 1e-9):
                        continue
                    lo, hi = [0.0] * 3, [0.0] * 3
                    for i in range(3):
                        if i == axis:
                            lo[i], hi[i] = bot_off, top_off
                        else:
                            lo[i], hi[i] = min(top_box[i], bot_box[i]), max(top_box[i + 3], bot_box[i + 3])
                    best = (score, Pcb(axis, tuple(lo), tuple(hi), "faces", index, solids[index].name))
    return best[1] if best else None


def find_pcb(solids: list) -> "Pcb | None":
    """The PCB among ``solids``: the largest thin slab solid, else the plane pair of a fused model, else ``None``."""
    solids = [s for s in solids if s.volume > 0]
    return _slab_pcb(solids) or _face_pair_pcb(solids) if solids else None


# --- the board frame -------------------------------------------------------------

def _transform_box(box: tuple, m) -> tuple:
    """Axis-aligned box of ``box`` after the rigid transform ``m`` (exact for the axis permutations used here)."""
    xs, ys, zs = [], [], []
    for x in (box[0], box[3]):
        for y in (box[1], box[4]):
            for z in (box[2], box[5]):
                p = m.multVec(App.Vector(x, y, z))
                xs.append(p.x)
                ys.append(p.y)
                zs.append(p.z)
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def _overlap(a: tuple, b: tuple) -> float:
    dx = min(a[3], b[3]) - max(a[0], b[0])
    dy = min(a[4], b[4]) - max(a[1], b[1])
    dz = min(a[5], b[5]) - max(a[2], b[2])
    return dx * dy * dz if dx > 0 and dy > 0 and dz > 0 else 0.0


def _board_boxes(board: "dict | None") -> list:
    out = []
    for conn in (board or {}).get("connectors") or []:
        try:
            box = conn["box"]
            out.append((box["x"][0], box["y"][0], box["z"][0], box["x"][1], box["y"][1], box["z"][1]))
        except (KeyError, TypeError, IndexError):
            continue
    return out


def _up_sign(solids: list, pcb: Pcb) -> int:
    """+1 when the components stand on the +axis side of the PCB, -1 when on the other side (ties: +1)."""
    a, a0, a1 = pcb.axis, pcb.lo[pcb.axis], pcb.hi[pcb.axis]
    above = below = 0.0
    if pcb.method == "faces":
        rec = solids[pcb.index]
        above, below = max(0.0, rec.box[a + 3] - a1), max(0.0, a0 - rec.box[a])
    else:
        for index, rec in enumerate(solids):
            if index == pcb.index or rec.volume < MIN_PART_VOLUME:
                continue
            lo, hi = rec.box[a], rec.box[a + 3]
            span = max(hi - lo, 1e-9)
            above += rec.volume * max(0.0, hi - max(lo, a1)) / span
            below += rec.volume * max(0.0, min(hi, a0) - lo) / span
    return -1 if below > above * 1.0000001 + 1e-9 else 1


@dataclass
class Frame:
    """The transform that takes the imported frame into the board frame and what was found while deriving it."""

    found: bool
    matrix: Any = None
    placement: Any = None
    is_identity: bool = True
    method: str = ""
    axes: str = ""
    translation: tuple = (0.0, 0.0, 0.0)
    pcb_size: tuple = ()              # (length, width, thickness) in the board frame
    pcb_name: str = ""
    note: str = ""

    def describe(self) -> str:
        if not self.found:
            return self.note or "no PCB found: the frame was left as imported"
        if self.is_identity:
            return "already in the board frame (origin = PCB bottom lower-left corner, long edge +X)"
        t = self.translation
        return (f"normalized into the board frame ({self.axes}; moved by {t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f} mm): "
                "origin = PCB bottom lower-left corner, long edge +X")


def _matrix(ex, ey, ez, origin) -> Any:
    """Matrix of p -> ((p - origin) . ex, (p - origin) . ey, (p - origin) . ez)."""
    return App.Matrix(ex.x, ex.y, ex.z, -origin.dot(ex),
                      ey.x, ey.y, ey.z, -origin.dot(ey),
                      ez.x, ez.y, ez.z, -origin.dot(ez),
                      0, 0, 0, 1)


def _axis_name(v) -> str:
    comps = (v.x, v.y, v.z)
    i = max(range(3), key=lambda k: abs(comps[k]))
    return ("+" if comps[i] > 0 else "-") + "XYZ"[i]


def _is_identity_matrix(m, tol: float = 1e-6) -> bool:
    eye = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    return all(abs(m.A[i] - eye[i]) <= tol for i in range(12))


def fit_frame(solids: list, board: "dict | None" = None) -> Frame:
    """Derive the transform into the board frame from the PCB (see the module docstring); never raises.

    ``solids`` are in the frame to normalize (typically global). Without a PCB the result has ``found=False`` and an
    identity placement. ``board`` is a catalog board dict: it only decides the sign of the long axis (and the quarter turn of a
    square board) by the overlap of the model's solids with the board's connector boxes.
    """
    try:
        pcb = find_pcb(solids)
    except Exception as exc:  # noqa: BLE001 - an exotic shape must not fail the import
        return Frame(False, App.Matrix(), App.Placement(), note=f"no PCB found ({type(exc).__name__}: {exc}): the frame was left as imported")
    if pcb is None:
        return Frame(False, App.Matrix(), App.Placement(),
                     note="no PCB found (no thin board solid or plane pair): the frame was left as imported")
    inplane = [i for i in range(3) if i != pcb.axis]
    long_axis, short_axis = (inplane[0], inplane[1]) if pcb.extent(inplane[0]) >= pcb.extent(inplane[1]) else (inplane[1], inplane[0])
    up = _up_sign(solids, pcb)
    ez = _AXES[pcb.axis] * up
    candidates = [(_AXES[long_axis], 1), (_AXES[long_axis] * -1, 1)]
    if abs(pcb.extent(long_axis) - pcb.extent(short_axis)) <= 0.01 * pcb.extent(long_axis):      # a square board: four turns
        candidates += [(_AXES[short_axis], 1), (_AXES[short_axis] * -1, 1)]
    boxes = _board_boxes(board)
    best = None
    for ex, _ in candidates:
        ey = ez.cross(ex)
        corners = [App.Vector(x, y, z) for x in (pcb.lo[0], pcb.hi[0]) for y in (pcb.lo[1], pcb.hi[1]) for z in (pcb.lo[2], pcb.hi[2])]
        origin = ex * min(c.dot(ex) for c in corners) + ey * min(c.dot(ey) for c in corners) + ez * min(c.dot(ez) for c in corners)
        matrix = _matrix(ex, ey, ez, origin)
        score = 0.0
        if boxes:
            for index, rec in enumerate(solids):
                if index == pcb.index or rec.volume < MIN_PART_VOLUME:
                    continue
                moved = _transform_box(rec.box, matrix)
                score += sum(_overlap(moved, b) for b in boxes)
        if best is None or score > best[0] + 1e-9:
            best = (score, ex, ey, origin, matrix)
    _score, ex, ey, origin, matrix = best
    size = (pcb.extent(long_axis), pcb.extent(short_axis), pcb.thickness)
    frame = Frame(True, matrix, App.Placement(matrix), _is_identity_matrix(matrix), pcb.method,
                  f"board +X = imported {_axis_name(ex)}, +Y = {_axis_name(ey)}, +Z = {_axis_name(ez)}",
                  (matrix.A14, matrix.A24, matrix.A34), size, pcb.name)
    if pcb.method == "faces":
        frame.note = "the model is one fused solid: the PCB was found as a plane pair"
    return frame


def apply_frame(solids: list, frame: Frame) -> list:
    """The ``solids`` moved into the board frame (copies; the input is untouched). An identity frame returns them as they are."""
    if frame.is_identity or frame.matrix is None:
        return list(solids)
    out = []
    for rec in solids:
        moved = rec.shape.copy()
        moved.transformShape(frame.matrix)
        out.append(SolidRecord(rec.name, moved, rec.obj))
    return out


# --- holes and corners ----------------------------------------------------------------

def _cylinders(rec: SolidRecord, z_lo: float, z_hi: float, axis_z_only: bool = True) -> list:
    """Cylindrical faces of one solid whose axis is parallel to Z and that lie between ``z_lo`` and ``z_hi``:
    ``[{"x", "y", "r", "convex": bool, "z0", "z1"}]``.

    ``convex`` says which way the material lies: the outward normal of the face points AWAY from the axis for a boss or a rounded
    corner (convex) and TOWARDS it for a hole (concave). Judged from the face normal, not from "is the axis inside the material":
    the centre of a corner arc may well lie inside a mounting hole next to it (Pi 4: corner radius 3.0 around (3, 3), hole at
    (3.5, 3.5) with radius 1.375).
    """
    found = []
    examined = 0
    for face in rec.shape.Faces:
        surf = face.Surface
        if not isinstance(surf, Part.Cylinder):
            continue
        examined += 1
        if examined > MAX_CYLINDER_FACES:
            break
        axis = surf.Axis
        if axis_z_only and abs(abs(axis.z) - 1.0) > 1e-6:
            continue
        box = bbox_of(face)
        if box[2] < z_lo - 0.05 or box[5] > z_hi + 0.05:
            continue
        try:
            u0, u1, v0, v1 = face.ParameterRange
            um, vm = (u0 + u1) / 2.0, (v0 + v1) / 2.0
            point, normal = face.valueAt(um, vm), face.normalAt(um, vm)
        except Exception:  # noqa: BLE001 - a face that cannot be sampled is skipped
            continue
        centre = surf.Center
        on_axis = centre + axis * ((point - centre).dot(axis))
        found.append({"x": on_axis.x, "y": on_axis.y, "r": float(surf.Radius), "convex": normal.dot(point - on_axis) > 0,
                      "z0": box[2], "z1": box[5]})
    return found


def _cluster(values: list, tol: float) -> list:
    """Sorted representative values (cluster means) of numbers closer than ``tol`` to their neighbour."""
    groups: list = []
    for v in sorted(values):
        if groups and v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def find_holes(rec: SolidRecord, thickness: float) -> tuple:
    """``(holes, corner_radius)`` of the PCB solid in the board frame.

    ``holes``: concave cylinders parallel to Z with a diameter inside ``HOLE_D_RANGE`` as ``{"x", "y", "d"}``, duplicates
    merged, sorted by (y, x). ``corner_radius``: the radius shared by convex corner cylinders (0 when the corners are square).
    """
    cyl = _cylinders(rec, -0.05, thickness + 0.05)
    seen, holes = set(), []
    for c in cyl:
        d = 2.0 * c["r"]
        if c["convex"] or not (HOLE_D_RANGE[0] <= d <= HOLE_D_RANGE[1]):
            continue
        key = (round(c["x"], 2), round(c["y"], 2), round(d, 2))
        if key in seen:
            continue
        seen.add(key)
        holes.append({"x": round(c["x"], 4), "y": round(c["y"], 4), "d": round(d, 4)})
    holes.sort(key=lambda h: (h["y"], h["x"]))
    radii = sorted(round(c["r"], 3) for c in cyl if c["convex"] and 0.5 <= c["r"] <= 0.5 * min(rec.size[0], rec.size[1]))
    corner = radii[len(radii) // 2] if len(radii) >= 2 else 0.0
    return holes, corner


def mounting_pattern(holes: list) -> "dict | None":
    """The mounting holes among ``holes`` and their pattern: ``{"d", "count", "holes", "pattern"}`` or ``None``.

    Mounting holes are the holes of the most common diameter inside ``MOUNT_D_RANGE`` (ties: the larger). ``pattern`` is
    ``{"dx", "dy", "offset": [x, y], "symmetric": bool}`` when they form a 2 x 2 grid, else ``None`` (an irregular layout
    such as the Arduino Uno keeps its ``holes``).
    """
    sized = [h for h in holes if MOUNT_D_RANGE[0] <= h["d"] <= MOUNT_D_RANGE[1]]
    if not sized:
        return None
    counts: dict = {}
    for h in sized:
        counts[round(h["d"], 2)] = counts.get(round(h["d"], 2), 0) + 1
    d = max(counts, key=lambda k: (counts[k], k))
    chosen = [h for h in sized if abs(h["d"] - d) <= 0.05]
    xs, ys = _cluster([h["x"] for h in chosen], 0.1), _cluster([h["y"] for h in chosen], 0.1)
    pattern = None
    if len(xs) == 2 and len(ys) == 2 and len(chosen) == 4:
        pattern = {"dx": round(xs[1] - xs[0], 4), "dy": round(ys[1] - ys[0], 4),
                   "offset": [round(xs[0], 4), round(ys[0], 4)], "symmetric": abs(xs[0] - ys[0]) <= 0.05}
    return {"d": d, "count": len(chosen), "holes": chosen, "pattern": pattern}


# --- connectors ----------------------------------------------------------------------------

def _iou(a: tuple, b: tuple) -> float:
    inter = _overlap(a, b)
    if inter <= 0:
        return 0.0
    va = (a[3] - a[0]) * (a[4] - a[1]) * (a[5] - a[2])
    vb = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2])
    return inter / (va + vb - inter)


def connector_candidates(solids: list, pcb_index: int, length: float, width: float, thickness: float,
                         board: "dict | None" = None) -> tuple:
    """``(connectors, tall)`` of the solids in the board frame.

    A connector candidate is a solid that reaches the PCB edge (flush within ``EDGE_TOL``, or overhanging): its ``side`` is
    the edge it protrudes past most, ``offset`` the centre along that wall, ``width`` its extent along the wall, ``height`` and
    ``z0`` / ``z1`` its vertical extent, ``overhang`` how far it reaches past the edge (0 when inset). With a board file each
    candidate names the board connector it overlaps (``matches``). ``tall`` lists the solids that stand away from the edges
    at least ``TALL_ABOVE`` mm above the PCB top (the GPIO header, a heat spreader): ``{"name", "x", "y", "z1", "height"}``.
    """
    boards = [(c.get("id", ""), c.get("face", ""), (c["box"]["x"][0], c["box"]["y"][0], c["box"]["z"][0],
                                                     c["box"]["x"][1], c["box"]["y"][1], c["box"]["z"][1]))
              for c in (board or {}).get("connectors") or [] if isinstance(c.get("box"), dict)]
    connectors, tall = [], []
    for index, rec in enumerate(solids):
        if index == pcb_index or rec.volume < MIN_PART_VOLUME:
            continue
        x0, y0, z0, x1, y1, z1 = rec.box
        reach = {"-x": -x0, "+x": x1 - length, "-y": -y0, "+y": y1 - width}
        side = max(SIDES, key=lambda s: (reach[s], -SIDES.index(s)))
        if reach[side] >= -EDGE_TOL:
            along = (x0, x1) if side in ("-y", "+y") else (y0, y1)
            item = {"name": rec.name, "side": side, "offset": round((along[0] + along[1]) / 2.0, 4),
                    "width": round(along[1] - along[0], 4), "height": round(z1 - z0, 4), "z0": round(z0, 4),
                    "z1": round(z1, 4), "overhang": round(max(0.0, reach[side]), 4),
                    "inset": round(max(0.0, -reach[side]), 4),
                    "box": {"x": [round(x0, 4), round(x1, 4)], "y": [round(y0, 4), round(y1, 4)], "z": [round(z0, 4), round(z1, 4)]}}
            best = max(((_iou(rec.box, b), cid) for cid, face, b in boards if face == side), default=(0.0, ""))
            if best[0] >= 0.5:
                item["matches"] = best[1]
            connectors.append(item)
        elif z1 - thickness >= TALL_ABOVE:
            tall.append({"name": rec.name, "x": round((x0 + x1) / 2.0, 4), "y": round((y0 + y1) / 2.0, 4),
                         "z1": round(z1, 4), "height": round(z1 - thickness, 4)})
    connectors.sort(key=lambda c: (SIDES.index(c["side"]), c["offset"]))
    tall.sort(key=lambda t: -t["z1"])
    return connectors, tall


# --- the measurement ------------------------------------------------------------------------

@dataclass
class Measurement:
    """Everything ``measure_reference_part`` knows (``to_dict`` is JSON-safe). Lengths in mm in the board frame."""

    kind: str = "brep"
    object: str = ""
    units: str = "mm"
    frame: dict = field(default_factory=dict)
    bbox: dict = field(default_factory=dict)
    solid_count: int = 0
    solids: list = field(default_factory=list)
    pcb: "dict | None" = None
    holes: list = field(default_factory=list)
    mounting: "dict | None" = None
    connectors: list = field(default_factory=list)
    tall: list = field(default_factory=list)
    tallest_z: "float | None" = None
    bottom_z: "float | None" = None
    identity: "dict | None" = None
    mesh: "dict | None" = None        # facts of a mesh reference (measure_mesh); None for a B-rep
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _bbox_dict(boxes) -> dict:
    lo = [min(b[i] for b in boxes) for i in range(3)]
    hi = [max(b[i + 3] for b in boxes) for i in range(3)]
    return {"min": [round(v, 4) for v in lo], "max": [round(v, 4) for v in hi], "size": [round(h - l, 4) for l, h in zip(lo, hi)]}


def solid_table(solids: list, top_n: int) -> list:
    """The ``top_n`` largest solids by volume as ``{"name", "size": [x, y, z], "volume", "z": [z0, z1]}``."""
    ranked = sorted(solids, key=lambda r: -r.volume)[:max(1, int(top_n))]
    return [{"name": r.name, "size": [round(v, 3) for v in r.size], "volume": round(r.volume, 2),
             "z": [round(r.box[2], 3), round(r.box[5], 3)]} for r in ranked]


def identity_of(measurement: "Measurement", board: "dict | None" = None, expected_bbox=None) -> "dict | None":
    """Plausibility check of the measured part against the catalog (a dict of ``identity.IdentityResult`` fields), or ``None``.

    ``expected_bbox`` (the catalog's STEP entry) is compared with the bounding box of the whole model; without it the PCB
    footprint is compared with the board outline (connector overhang allowed). Neither -> ``None``.
    """
    result = None
    if expected_bbox:
        result = identity.bbox_identity(measurement.bbox.get("size"), expected_bbox)
    elif board and measurement.pcb:
        result = identity.outline_identity((measurement.pcb["length"], measurement.pcb["width"], measurement.pcb["thickness"]), board)
    if result is None:
        return None
    return {"ok": result.ok, "status": result.status, "detail": result.detail}


def analyze(solids: list, *, board: "dict | None" = None, expected_bbox=None, top_n: int = 8, frame: "Frame | None" = None,
            object_name: str = "") -> Measurement:
    """Measure ``solids`` (``SolidRecord`` s in any frame) in the board frame; the frame is derived unless one is given.

    GUI-thread only. The solids are not modified. A model without a PCB still gets its bounding box, solid table and
    identity; ``pcb`` is ``None`` and a warning says that the frame, heights and connectors could not be derived.
    """
    m = Measurement(object=object_name)
    solids = [s for s in solids if s.volume > 0 and all(math.isfinite(v) and abs(v) < 1e7 for v in s.box)]
    m.solid_count = len(solids)
    if not solids:
        m.warnings.append("the reference holds no solid with a volume; nothing to measure")
        return m
    frame = frame or fit_frame(solids, board)
    placed = apply_frame(solids, frame)
    m.frame = {"found": frame.found, "applied": frame.found and not frame.is_identity, "axes": frame.axes,
               "translation": [round(v, 4) for v in frame.translation], "description": frame.describe()}
    m.bbox = _bbox_dict([r.box for r in placed])
    m.solids = solid_table(placed, top_n)
    if not frame.found:
        m.warnings.append(frame.note)
        m.identity = identity_of(m, board, expected_bbox)
        return m
    length, width, thickness = frame.pcb_size
    pcb_rec = placed[next((i for i, r in enumerate(placed) if r.name == frame.pcb_name), 0)]
    holes, corner = find_holes(pcb_rec, thickness)
    m.pcb = {"solid": frame.pcb_name, "method": frame.method, "length": round(length, 4), "width": round(width, 4),
             "thickness": round(thickness, 4), "z0": 0.0, "z1": round(thickness, 4), "corner_radius": round(corner, 4)}
    m.holes = holes
    m.mounting = mounting_pattern(holes)
    pcb_index = next((i for i, r in enumerate(placed) if r.name == frame.pcb_name), -1)
    if frame.method == "faces":
        m.warnings.append("the model is one fused solid: connector candidates cannot be separated; use a model with separate solids "
                          "or the nominal board data for the connector positions")
    else:
        m.connectors, m.tall = connector_candidates(placed, pcb_index, length, width, thickness, board)
    parts = [r for r in placed if r.volume >= MIN_PART_VOLUME] or placed
    m.tallest_z = round(max(r.box[5] for r in parts), 4)
    m.bottom_z = round(min(r.box[2] for r in parts), 4)
    if not holes:
        m.warnings.append("no mounting holes were found as cylindrical faces; the hole positions of the board data stay in use")
    m.identity = identity_of(m, board, expected_bbox)
    if m.identity and not m.identity["ok"]:
        m.warnings.append(m.identity["detail"])
    return m


def measure_object(root, *, board: "dict | None" = None, expected_bbox=None, top_n: int = 8, frame: "Frame | None" = None) -> Measurement:
    """``analyze(collect_solids(root), ...)`` for a reference object of a document (GUI-thread only)."""
    return analyze(collect_solids(root), board=board, expected_bbox=expected_bbox, top_n=top_n, frame=frame,
                   object_name=str(getattr(root, "Name", "")))


# --- text ------------------------------------------------------------------------------------

def _mm(v) -> str:
    text = f"{float(v):.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def render_text(m: Measurement, *, top_n: int = 8) -> str:
    """The bounded model-facing report of a measurement (about 20 lines at most, connector rows capped at ``top_n``)."""
    box = m.bbox
    lines = [f"Measured {m.object} (exact B-rep, {m.solid_count} solid{'s' if m.solid_count != 1 else ''}, units mm). "
             f"Frame: {m.frame.get('description', '')}."]
    if box:
        lo, hi, size = box["min"], box["max"], box["size"]
        lines.append(f"Bounding box: x {_mm(lo[0])}..{_mm(hi[0])}, y {_mm(lo[1])}..{_mm(hi[1])}, z {_mm(lo[2])}..{_mm(hi[2])} "
                     f"(size {_mm(size[0])} x {_mm(size[1])} x {_mm(size[2])} mm).")
    if m.pcb:
        p = m.pcb
        lines.append(f"PCB: {_mm(p['length'])} x {_mm(p['width'])} x {_mm(p['thickness'])} mm (solid '{p['solid']}', found as a "
                     f"{'slab solid' if p['method'] == 'solid' else 'plane pair'}), plane z {_mm(p['z0'])}..{_mm(p['z1'])}, "
                     f"corner radius {_mm(p['corner_radius'])}.")
    if m.mounting:
        mt = m.mounting
        points = ", ".join(f"({_mm(h['x'])}, {_mm(h['y'])})" for h in mt["holes"][:8])
        pat = mt["pattern"]
        extra = (f"; pattern {_mm(pat['dx'])} x {_mm(pat['dy'])} mm, offset {_mm(pat['offset'][0])}, {_mm(pat['offset'][1])}"
                 if pat else "; no 2 x 2 grid")
        lines.append(f"Mounting holes: {mt['count']} x diameter {_mm(mt['d'])} at {points}{extra}.")
    elif m.pcb:
        lines.append("Mounting holes: none found.")
    if m.connectors:
        shown = m.connectors[:top_n]
        lines.append(f"Connector candidates ({len(m.connectors)}; side, offset along the wall, width, height, z range, overhang):")
        for c in shown:
            tag = f" -> {c['matches']}" if c.get("matches") else ""
            lines.append(f"  {c['side']} offset {_mm(c['offset'])} width {_mm(c['width'])} height {_mm(c['height'])} "
                         f"z {_mm(c['z0'])}..{_mm(c['z1'])} overhang {_mm(c['overhang'])} ({c['name']}{tag})")
        if len(m.connectors) > len(shown):
            lines.append(f"  ... {len(m.connectors) - len(shown)} more (raise top_n, at most 30)")
    if m.tall:
        lines.append("Tall components away from the edges: " + "; ".join(
            f"{t['name']} at ({_mm(t['x'])}, {_mm(t['y'])}) top z {_mm(t['z1'])}" for t in m.tall[:3]) + ".")
    if m.tallest_z is not None:
        lines.append(f"tallest_z {_mm(m.tallest_z)}, bottom_z {_mm(m.bottom_z)} (from the PCB bottom).")
    if m.solids:
        lines.append("Largest solids: " + "; ".join(f"{s['name']} {_mm(s['size'][0])} x {_mm(s['size'][1])} x {_mm(s['size'][2])}"
                                                    for s in m.solids[:5]) + ".")
    if m.identity:
        lines.append(f"Identity: {m.identity['detail']}.")
    for warning in m.warnings:
        if not (m.identity and warning == m.identity.get("detail")):
            lines.append(f"Warning: {warning}.")
    return "\n".join(lines)
