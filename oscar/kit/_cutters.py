"""Cutter construction shared by the kit stages (private module of ``oscar.kit``; FreeCAD, GUI-thread only).

Every opening the kit cuts (connector ports, user cut-outs, vent slots, lid skirt notches, catch pockets) is a box (or a
cylinder) placed in the part's DESIGN frame. This module turns "face + (u, v) rectangle + how deep" into such a box with its
wall-normal placement expression-bound to ``Params`` (so wall thickness, clearance, standoff height ... edits move the cutters
with the walls; the in-plane position and size are literals taken from the board data and the spec).

Face frames: wall ``-y`` / ``+y``: ``u = x``, ``v = z``, normal ``y``; wall ``-x`` / ``+x``: ``u = y``, ``v = z``, normal ``x``;
``top`` / ``bottom``: ``u = x``, ``v = y``, normal ``z``.
"""

import math
from typing import Any

import FreeCAD as App
import Part
from FreeCAD import Vector as V

from . import geometry as G
from .geometry import EXTRA, KitGeometryError, mm, signed

# face -> (axis index of u, axis index of v, axis index of the normal)
FACE_AXES = {"-y": (0, 2, 1), "+y": (0, 2, 1), "-x": (1, 2, 0), "+x": (1, 2, 0), "top": (0, 1, 2), "bottom": (0, 1, 2)}
_DIM = ("Length", "Width", "Height")
_XYZ = ("x", "y", "z")
# face -> rotation axis of an in-plane rotation of (u, v): u x v
_ROT_AXIS = {"-y": V(0, -1, 0), "+y": V(0, -1, 0), "-x": V(1, 0, 0), "+x": V(1, 0, 0), "top": V(0, 0, 1), "bottom": V(0, 0, 1)}


def sanitize(text: str) -> str:
    """An identifier fragment for an object Name (letters, digits and underscores only)."""
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(text))
    return out or "x"


def through_span(v: dict, part: str, face: str, *, blind: bool = False) -> "dict | None":
    """How far a cutter must reach along the wall normal to open ``face`` of ``part`` (``shell`` or ``lid``).

    Returns ``{"lo", "len", "lo_expr", "len_expr"}`` (coordinate of the low end, length, and their ``Params`` expressions) or
    ``None`` when the part has no such face (a shell has no top, a lid no bottom). ``v`` is the Params value dict.
    A shell wall / floor is cut through its thickness plus ``EXTRA`` on both sides (``blind``: the outer half only); the lid
    plate through its thickness; a lid WALL face means the skirt: the band outside the shell wall, from the shell's outer face
    out past the skirt. Needs the rows ``cav_*``, ``wall_t``, ``floor_t``, ``lid_t``, ``lid_fit``, ``lip_t`` and the z levels.
    """
    ex = mm(EXTRA)
    if part == "shell":
        if face in ("-x", "+x", "-y", "+y"):
            a = face[1]
            sigma = -1 if face[0] == "-" else 1
            wt, inner = v["wall_t"], v[f"cav_{a}{0 if sigma < 0 else 1}"]
            outer = inner + sigma * wt
            if blind:
                depth = wt / 2
                if sigma < 0:
                    return {"lo": outer - EXTRA, "len": depth + EXTRA, "lo_expr": f"Params.cav_{a}0 - Params.wall_t - {ex}",
                            "len_expr": f"Params.wall_t/2 + {ex}"}
                return {"lo": outer - depth, "len": depth + EXTRA, "lo_expr": f"Params.cav_{a}1 + Params.wall_t/2",
                        "len_expr": f"Params.wall_t/2 + {ex}"}
            if sigma < 0:
                return {"lo": outer - EXTRA, "len": wt + 2 * EXTRA, "lo_expr": f"Params.cav_{a}0 - Params.wall_t - {ex}",
                        "len_expr": f"Params.wall_t + {mm(2 * EXTRA)}"}
            return {"lo": inner - EXTRA, "len": wt + 2 * EXTRA, "lo_expr": f"Params.cav_{a}1 - {ex}",
                    "len_expr": f"Params.wall_t + {mm(2 * EXTRA)}"}
        if face == "bottom":
            ft = v["floor_t"]
            return {"lo": v["z_floor_bot"] - EXTRA, "len": (ft / 2 if blind else ft) + (EXTRA if blind else 2 * EXTRA),
                    "lo_expr": f"Params.z_floor_bot - {ex}",
                    "len_expr": f"Params.floor_t/2 + {ex}" if blind else f"Params.floor_t + {mm(2 * EXTRA)}"}
        return None
    if part == "lid":
        if face == "top":
            lt = v["lid_t"]
            if blind:
                return {"lo": v["z_cavity_top"] + lt / 2, "len": lt / 2 + EXTRA, "lo_expr": "Params.z_cavity_top + Params.lid_t/2",
                        "len_expr": f"Params.lid_t/2 + {ex}"}
            return {"lo": v["z_cavity_top"] - EXTRA, "len": lt + 2 * EXTRA, "lo_expr": f"Params.z_cavity_top - {ex}",
                    "len_expr": f"Params.lid_t + {mm(2 * EXTRA)}"}
        if face in ("-x", "+x", "-y", "+y"):
            a = face[1]
            sigma = -1 if face[0] == "-" else 1
            band = v["lid_fit"] + v["lip_t"] + EXTRA
            outer = v[f"cav_{a}{0 if sigma < 0 else 1}"] + sigma * v["wall_t"]
            band_expr = f"Params.lid_fit + Params.lip_t + {ex}"
            if sigma < 0:
                # the band is a SUM: without the parentheses ``- fit + lip_t + 1 mm`` moved the notch INTO the cavity (the bound
                # expression overrides the literal placement), so a skirt notch on a -x / -y face never opened the skirt
                return {"lo": outer - band, "len": band, "lo_expr": f"Params.cav_{a}0 - Params.wall_t - ({band_expr})",
                        "len_expr": band_expr}
            return {"lo": outer, "len": band, "lo_expr": f"Params.cav_{a}1 + Params.wall_t", "len_expr": band_expr}
        return None
    return None


def default_center(v: dict, face: str) -> "tuple[float, float, str, str]":
    """The default centre of a vent array on ``face``: ``(cu, cv, cu_expr, cv_expr)``; the centre of the face over the cavity."""
    if face in ("top", "bottom"):
        return (v["cav_x0"] + v["cav_len"] / 2, v["cav_y0"] + v["cav_wid"] / 2,
                "Params.cav_x0 + Params.cav_len/2", "Params.cav_y0 + Params.cav_wid/2")
    zc = (v["z_floor_top"] + v["z_cavity_top"]) / 2
    zc_expr = "(Params.z_floor_top + Params.z_cavity_top)/2"
    if face in ("-y", "+y"):
        return v["cav_x0"] + v["cav_len"] / 2, zc, "Params.cav_x0 + Params.cav_len/2", zc_expr
    return v["cav_y0"] + v["cav_wid"] / 2, zc, "Params.cav_y0 + Params.cav_wid/2", zc_expr


def rect_cutter(ctx: Any, name: str, *, part: str, container: Any, face: str, span: dict, u0: float, u1: float, v0: float,
                v1: float, exprs: "dict[str, str] | None" = None, rounded: float = 0.0) -> Any:
    """A box cutter on ``face``: in-plane rectangle ``[u0, u1] x [v0, v1]`` (literals) through ``span`` (``through_span``).

    ``exprs`` may bind the in-plane edges: keys ``u_origin`` / ``v_origin`` (origin coordinate) and ``du`` / ``dv`` (extent). The wall-normal
    origin and length come from ``span`` (always bound). ``rounded`` > 0 rounds the four edges along the normal with that radius;
    such a cutter is a static shape (its placement is not expression-bound).
    """
    ua, va, na = FACE_AXES[face]
    size = [0.0, 0.0, 0.0]
    origin = [0.0, 0.0, 0.0]
    size[ua], size[va], size[na] = u1 - u0, v1 - v0, span["len"]
    origin[ua], origin[va], origin[na] = u0, v0, span["lo"]
    if min(size) <= 0:
        raise KitGeometryError(f"{name}: the cutter is empty ({size[0]:.3g} x {size[1]:.3g} x {size[2]:.3g} mm); check the opening size")
    b = ctx.b
    if rounded > 0:
        r = min(rounded, size[ua] / 2 - 1e-3, size[va] / 2 - 1e-3)
        shape = Part.makeBox(size[0], size[1], size[2], V(*origin))
        if r > 0:
            try:
                shape = shape.makeFillet(r, G.box_edges_along(shape, na))
            except Exception:    # noqa: BLE001 - a failed fillet leaves a sharp-cornered cutter, which still cuts
                pass
        return b.static(name, shape, role="aux", part=part, container=container)
    e = exprs or {}
    bind = {_DIM[na]: span["len_expr"], f"Placement.Base.{_XYZ[na]}": span["lo_expr"],
            f"Placement.Base.{_XYZ[ua]}": e.get("u_origin"), f"Placement.Base.{_XYZ[va]}": e.get("v_origin"),
            _DIM[ua]: e.get("du"), _DIM[va]: e.get("dv")}
    return b.box(name, tuple(size), tuple(origin), exprs=bind, role="aux", part=part, container=container)


def slot_cutters(ctx: Any, prefix: str, *, part: str, container: Any, face: str, span: dict, center: "list | None", count: int,
                 slot_w: float, slot_l: float, pitch: float, angle: float, pattern: str) -> list:
    """The cutters of one vent array: ``count`` slots (or round holes) of ``slot_l`` x ``slot_w`` along u at ``pitch`` across v.

    ``center`` ``[u, v]`` or None for the default (the face centre over the cavity: bound to ``Params`` when ``angle`` is 0).
    Slots lie along u at angle 0 (7.3 validation: ``vent_extent``); ``angle`` rotates the array about its centre. Holes
    (``pattern == 'holes'``) are cylinders of diameter ``slot_w`` in a single row along the (rotated) v direction.
    """
    v = ctx.v
    ua, va, na = FACE_AXES[face]
    cu0, cv0, cu_expr, cv_expr = default_center(v, face)
    if center is None:
        cu, cv, bound = cu0, cv0, angle == 0
    else:
        cu, cv, bound = float(center[0]), float(center[1]), False
    th = math.radians(angle)
    out = []
    for i in range(count):
        off = (i - (count - 1) / 2.0) * pitch
        su, sv = cu - off * math.sin(th), cv + off * math.cos(th)         # slot centre in (u, v)
        name = f"{prefix}{i + 1}"
        if pattern == "holes":
            out.append(_hole(ctx, name, part, container, face, span, su, sv, slot_w / 2))
            continue
        size = [0.0, 0.0, 0.0]
        size[ua], size[va], size[na] = slot_l, slot_w, span["len"]
        through = {f"Placement.Base.{_XYZ[na]}": span["lo_expr"], _DIM[na]: span["len_expr"]}     # the wall-normal extent is always bound
        if bound:
            origin = [0.0, 0.0, 0.0]
            origin[ua], origin[va], origin[na] = su - slot_l / 2, sv - slot_w / 2, span["lo"]
            ex = {**through, f"Placement.Base.{_XYZ[ua]}": cu_expr + signed(-slot_l / 2),
                  f"Placement.Base.{_XYZ[va]}": cv_expr + signed(off - slot_w / 2)}
            out.append(ctx.b.box(name, tuple(size), tuple(origin), exprs=ex, role="aux", part=part, container=container))
            continue
        center_w = [0.0, 0.0, 0.0]
        center_w[ua], center_w[va], center_w[na] = su, sv, span["lo"] + span["len"] / 2
        rot = App.Rotation(_ROT_AXIS[face], angle)
        origin_v = V(*center_w) - rot.multVec(V(size[0] / 2, size[1] / 2, size[2] / 2))
        out.append(ctx.b.box(name, tuple(size), (origin_v.x, origin_v.y, origin_v.z), exprs=through, rotation=rot, role="aux", part=part,
                             container=container))
    return out


def _hole(ctx: Any, name: str, part: str, container: Any, face: str, span: dict, su: float, sv: float, radius: float) -> Any:
    ua, va, na = FACE_AXES[face]
    center = [0.0, 0.0, 0.0]
    center[ua], center[va], center[na] = su, sv, span["lo"]
    if na == 2:
        rot = App.Rotation()
    elif na == 1:
        rot = App.Rotation(V(1, 0, 0), -90)      # +z -> +y
    else:
        rot = App.Rotation(V(0, 1, 0), 90)       # +z -> +x
    cyl = ctx.b.new("Part::Cylinder", name, role="aux", part=part, container=container)
    cyl.Radius, cyl.Height = float(radius), float(span["len"])
    cyl.Placement = App.Placement(V(*center), rot)
    ctx.b.bind_all(cyl, {"Height": span["len_expr"], f"Placement.Base.{_XYZ[na]}": span["lo_expr"]})
    return cyl

