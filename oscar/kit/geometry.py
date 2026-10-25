"""Geometry layer of the enclosure kit: tagging, scanning, the object builders and the snap-fit numbers.

FreeCAD module (GUI-thread only, never imports FreeCADGui). ``oscar.kit.build`` orchestrates the stages on top of it and
``oscar.kit._stages`` holds the nine stage builders; this module has no stage logic.

What lives here
---------------
* **Tags**: every object a stage creates carries the string properties ``OscarKit`` / ``OscarStage`` / ``OscarRole`` /
  ``OscarPart`` / ``OscarSpecHash`` (group ``Oscar``, persisted in the FCStd). ``tags_of``, ``kit_objects``, ``tip``,
  ``chain_objects``, ``stage_objects`` read them; objects without the tags are never touched by the kit.
* **Builder**: ``Builder`` creates tagged, hidden objects (Sketcher rounded rectangles with NAMED constraints bound to
  ``Params``, ``Part::Extrusion``, boxes, cylinders, static shapes, ``Part::Cut`` / ``Fuse`` / ``MultiFuse``) and binds
  placement / dimension properties to the ``Params`` sheet with ``setExpression``. Expression strings use the sheet's
  object Name (``Params.wall_t``), numeric literals carry units (``1 mm``).
* **Tips and labels**: ``refresh_tips`` keeps exactly one visible tip per part (Label ``Shell`` / ``Lid``, unique per
  document; a displaced tip gets its Name back as Label BEFORE the new tip is labelled, so FreeCAD never renames it to
  ``Shell001``), hides everything else and applies the spec colours (``getattr(obj, "ViewObject", None)`` guards: in
  ``freecadcmd`` there is no view provider).
* **Snap-fit numbers**: ``barb_profile`` (the barb cross-section), ``hook_plans`` (every number of every hook, built on
  ``spec.hook_positions``) and ``pocket_depth``. Pure functions, usable and testable without a document.
* ``single_solid`` / ``shape_facts``: the post-condition of every stage (one valid closed solid, positive volume).

Frame: world = board frame, PCB bottom lower-left corner at the origin, +X long edge, +Y short edge, +Z up. Objects
inside a part container (``Part_Shell`` / ``Part_Lid``) are built in the container's DESIGN frame (lid lift 0); the
assembly pose is the container placement, which is why every cross-object computation uses ``global_shape``.

Barb profile (7.2a made precise)
--------------------------------
A barb is a trapezoid prism in the (depth t, z) plane, ``t`` measured INWARD from the beam's inner face. Its root sits on
the beam (``t = 0``), its flat land at the tip (``t = overhang``). The lead-in ramp (lower side, ``ramp_deg``) and the
retention face (upper side, ``retention_deg``) make those angles with the MATING direction (vertical), so a face rises
``depth / tan(angle)`` over a depth: 90 degrees is a horizontal ledge (a permanent snap). ``barb_h`` is the height of the
barb where it meets the shell's outer face (the wall plane, ``t = fit``): that is what must fit the catch pocket
(``barb_h + window_margin`` tall). When the requested angles need more than ``0.9 * barb_h`` there (the
default angles of 30 / 45 degrees do, a true 30 degree lead-in over 0.55 mm of engagement is 0.95 mm tall by itself), BOTH
slopes are flattened by one common factor so a 10 % land remains; ``barb_profile`` reports the effective angles
(``ramp_eff_deg`` / ``retention_eff_deg``, from vertical) and ``flattened``. Engagement (``overhang - fit``) is never reduced.
"""

import math
from typing import Any, Iterable

import FreeCAD as App
import Part
import Sketcher
from FreeCAD import Vector as V

from . import spec as _spec
from .spec import STAGES, hook_positions  # noqa: F401  (hook_positions: re-exported contract name, 7.2a)

try:    # one definition of "is model geometry" / "global frame shape"; private equivalents until health exists
    from ..cad.health import global_shape, is_geometry  # noqa: F401
except ImportError:    # pragma: no cover - the health module is part of this build; kept for a partial checkout
    _ORIGIN = frozenset({"App::Origin", "App::Line", "App::Plane", "App::Point"})

    def is_geometry(obj: Any) -> bool:    # type: ignore[misc]
        shape = getattr(obj, "Shape", None)
        return (getattr(obj, "TypeId", "") not in _ORIGIN and shape is not None and not shape.isNull()
                and shape.ShapeType in ("Solid", "Compound", "CompSolid", "Shell"))

    def global_shape(obj: Any) -> Any:    # type: ignore[misc]
        s = obj.Shape.copy()
        s.Placement = obj.getGlobalPlacement()
        return s

SHEET = "Params"                 # object Name of the parameter sheet (expressions use the Name, 7.5)
TAG_PROPS = ("OscarKit", "OscarStage", "OscarRole", "OscarPart", "OscarSpecHash")
ROLES = ("params", "ref", "shell", "lid", "hook", "aux", "assembly", "container")
CHAIN_ROLES = ("shell", "lid")   # roles whose objects are valid single solids and form a part's feature chain
TIP_LABEL = {"shell": "Shell", "lid": "Lid"}
STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}

EXTRA = 1.0              # mm: how far a cutter overshoots a face it must cut cleanly through
FUSE_OVERLAP = 0.5       # mm: how far a fused feature reaches into the solid it joins (a touching face alone is fragile)
MIN_RADIUS = 0.01        # mm: floor of a sketch corner radius (a zero radius would collapse the arc constraints)
LAND_FRACTION = 0.10     # share of barb_h kept as a flat land at the barb tip when the slopes are flattened
MIN_VOLUME = 1e-6        # mm3: a stage solid below this is considered empty
PLATE_TOL = 0.05         # mm: tolerance of "this opening reaches into the skirt"

BBOX_KEYS = ("xmin", "ymin", "zmin", "xmax", "ymax", "zmax")


class KitGeometryError(ValueError):
    """A geometry operation of the kit failed; the message follows the model-facing error style."""


# --------------------------------------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------------------------------------

def num(x: float) -> str:
    """A number for an expression: fixed notation, at most 9 decimals, no exponent (``1e-05`` is not a FreeCAD literal)."""
    s = format(float(x), ".9f").rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def mm(x: float) -> str:
    """A length literal for an expression: ``mm(2.5) == '2.5 mm'``."""
    return f"{num(x)} mm"


def signed(x: float) -> str:
    """``' + 2.5 mm'`` or ``' - 2.5 mm'``: a literal appended to an expression."""
    return f" + {mm(x)}" if x >= 0 else f" - {mm(-x)}"


# --------------------------------------------------------------------------------------------------------------------
# tags and scanning
# --------------------------------------------------------------------------------------------------------------------

def set_tag(obj: Any, name: str, value: str) -> None:
    """Create the string property ``name`` (group ``Oscar``) when missing and set it."""
    if name not in obj.PropertiesList:
        obj.addProperty("App::PropertyString", name, "Oscar", f"kit tag: {name}")
    setattr(obj, name, str(value))


def tag(obj: Any, *, kit_id: str, stage: str, role: str, part: str = "", spec_hash: str = "") -> Any:
    """Tag ``obj`` as a product of ``stage`` (all five tag properties); returns ``obj``."""
    set_tag(obj, "OscarKit", kit_id)
    set_tag(obj, "OscarStage", stage)
    set_tag(obj, "OscarRole", role)
    set_tag(obj, "OscarPart", part)
    set_tag(obj, "OscarSpecHash", spec_hash)
    return obj


def tags_of(obj: Any) -> dict:
    """``{"OscarKit", "OscarStage", "OscarRole", "OscarPart", "OscarSpecHash"}`` of ``obj`` ('' for a missing tag)."""
    props = set(getattr(obj, "PropertiesList", ()) or ())
    return {p: (str(getattr(obj, p, "") or "") if p in props else "") for p in TAG_PROPS}


def is_kit_object(obj: Any, kit_id: "str | None" = None) -> bool:
    """True when ``obj`` carries an ``OscarKit`` tag (equal to ``kit_id`` when given)."""
    kit = tags_of(obj)["OscarKit"]
    return bool(kit) and (kit_id is None or kit == kit_id)


def kit_objects(doc: Any, kit_id: "str | None" = None) -> list:
    """Every kit-tagged object of ``doc`` in document order."""
    return [o for o in doc.Objects if is_kit_object(o, kit_id)]


def stage_objects(doc: Any, kit_id: str, stage: str) -> list:
    """The objects of ``stage`` (any role except part containers, which belong to no single stage)."""
    out = []
    for o in doc.Objects:
        t = tags_of(o)
        if t["OscarKit"] == kit_id and t["OscarStage"] == stage and t["OscarRole"] != "container":
            out.append(o)
    return out


def chain_objects(doc: Any, role: str, kit_id: "str | None" = None) -> list:
    """The feature-chain objects of a part (``role`` ``shell`` or ``lid``), oldest first."""
    return [o for o in doc.Objects if tags_of(o)["OscarRole"] == role and is_kit_object(o, kit_id)]


def tip(doc: Any, role: str, kit_id: "str | None" = None) -> Any:
    """The last object of a part's chain by ``OscarRole`` (``shell`` | ``lid`` | ``ref``), or None.

    "Last" is the highest stage index, ties broken by document order. For ``ref`` the result is the ``Ref_<ID>`` container
    (tagged ``ref``; an imported reference carries no tag, so an untagged ``Ref_*`` ``App::Part`` is the fallback).
    """
    best, best_key = None, (-1, -1)
    for pos, o in enumerate(doc.Objects):
        t = tags_of(o)
        if t["OscarRole"] != role or not t["OscarKit"] or (kit_id is not None and t["OscarKit"] != kit_id):
            continue
        key = (STAGE_INDEX.get(t["OscarStage"], -1), pos)
        if key > best_key:
            best, best_key = o, key
    if best is None and role == "ref":
        for o in doc.Objects:
            if o.Name.startswith("Ref_") and o.TypeId == "App::Part" and not tags_of(o)["OscarKit"]:
                return o
    return best


def ref_container(doc: Any, board_id: str) -> Any:
    """The ``Ref_<ID>`` container of ``board_id`` (Name first, then Label), or None. Tagged or imported alike."""
    name = ref_name(board_id)
    obj = doc.getObject(name)
    if obj is not None:
        return obj
    hits = doc.getObjectsByLabel(name)
    return hits[0] if hits else None


def ref_name(board_id: str) -> str:
    """``Ref_RPI4B`` for board ``rpi4b`` (identifier characters only)."""
    ident = "".join(ch if ch.isalnum() else "_" for ch in str(board_id)).upper()
    return f"Ref_{ident}"


# --------------------------------------------------------------------------------------------------------------------
# visibility, colours, labels
# --------------------------------------------------------------------------------------------------------------------

def set_visible(obj: Any, visible: bool) -> None:
    """Show or hide ``obj`` (``Visibility`` property; plus the view object when a GUI exists)."""
    try:
        obj.Visibility = bool(visible)
    except Exception:    # noqa: BLE001, S110 - objects without the property (sheets) are simply left alone
        pass
    vo = getattr(obj, "ViewObject", None)
    if vo is not None:
        try:
            vo.Visibility = bool(visible)
        except Exception:    # noqa: BLE001, S110
            pass


def set_color(obj: Any, rgb: "Iterable[float] | None") -> bool:
    """Give ``obj`` the shape colour ``rgb`` (three floats 0..1) when it has a view object; True when applied (headless: False)."""
    vo = getattr(obj, "ViewObject", None)
    if vo is None or rgb is None:
        return False
    try:
        vo.ShapeColor = tuple(float(c) for c in rgb)
        return True
    except Exception:    # noqa: BLE001
        return False


def _label_part(doc: Any, label: str, keep: Any) -> list[str]:
    """Reset the Label of every kit object other than ``keep`` that holds ``label`` to its Name; returns the names touched."""
    touched = []
    for o in doc.getObjectsByLabel(label):
        if o is not keep and is_kit_object(o):
            o.Label = o.Name
            touched.append(o.Name)
    return touched


def refresh_tips(doc: Any, kit_id: str, colors: "dict | None" = None) -> list[str]:
    """Make the chain tip of each part the one visible, labelled object; returns warnings (empty when all went well).

    For ``shell`` and ``lid``: every earlier chain object is hidden and gets its Name as Label, the tip is shown, its Label
    becomes ``Shell`` / ``Lid`` (a displaced holder of the label is reset first, so the document keeps exactly one hit for
    ``getObjectsByLabel('Shell')``) and, with a GUI, it gets the part colour. A label held by an object that is not ours
    cannot be taken: FreeCAD would then name the tip ``Shell001``, which is reported as a warning. The ``Ref_*`` container is hidden.
    """
    warnings: list[str] = []
    colors = colors or {}
    for role, label in TIP_LABEL.items():
        objs = chain_objects(doc, role, kit_id)
        last = tip(doc, role, kit_id)
        for o in objs:
            if o is not last:
                set_visible(o, False)
                if o.Label != o.Name:
                    o.Label = o.Name
        if last is None:
            continue
        _label_part(doc, label, last)
        last.Label = label
        if last.Label != label:
            warnings.append(f"the label '{label}' is used by another object, so the {role} tip is labelled '{last.Label}'; rename that object")
        set_visible(last, True)
        set_color(last, colors.get(role))
    ref = tip(doc, "ref", kit_id)
    if ref is not None:
        set_visible(ref, False)
        for child in getattr(ref, "Group", []) or []:
            set_visible(child, False)
            set_color(child, colors.get("ref"))
    return warnings


# --------------------------------------------------------------------------------------------------------------------
# shapes: the stage post-condition
# --------------------------------------------------------------------------------------------------------------------

def shape_facts(obj: Any) -> dict:
    """``{"solids", "valid", "closed", "volume", "bbox"}`` of the GLOBAL-frame shape of ``obj`` (None-safe, never raises)."""
    out: dict = {"solids": 0, "valid": False, "closed": False, "volume": 0.0, "bbox": None}
    try:
        shape = global_shape(obj)
        if shape.isNull():
            return out
        out["solids"] = len(shape.Solids)
        out["valid"] = bool(shape.isValid())
        out["closed"] = bool(all(s.isClosed() for s in shape.Shells)) if shape.Shells else False
        out["volume"] = float(shape.Volume)
        bb = shape.BoundBox
        out["bbox"] = [bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax]
    except Exception:    # noqa: BLE001
        pass
    return out


def single_solid(obj: Any) -> "tuple[bool, str]":
    """Is ``obj`` ONE valid, closed solid of positive volume? Returns ``(ok, reason)``; ``reason`` is '' when ok."""
    f = shape_facts(obj)
    if f["bbox"] is None:
        return False, f"'{obj.Name}' has no shape ({_status(obj)})"
    if f["solids"] != 1:
        return False, f"'{obj.Name}' has {f['solids']} solids, expected 1"
    if not f["valid"]:
        return False, f"'{obj.Name}' fails the BRep validity check"
    if not f["closed"]:
        return False, f"'{obj.Name}' is not a closed solid"
    if not f["volume"] > MIN_VOLUME:
        return False, f"'{obj.Name}' has no volume ({f['volume']:.6g} mm3)"
    return True, ""


def _status(obj: Any) -> str:
    try:
        return str(obj.getStatusString())
    except Exception:    # noqa: BLE001
        return "unknown state"


def common_volume(a: Any, b: Any) -> float:
    """Volume of the intersection of the global shapes of two objects (0.0 on failure)."""
    try:
        return float(global_shape(a).common(global_shape(b)).Volume)
    except Exception:    # noqa: BLE001
        return 0.0


# --------------------------------------------------------------------------------------------------------------------
# the Builder: tagged, hidden objects bound to the Params sheet
# --------------------------------------------------------------------------------------------------------------------

class Builder:
    """Creates the objects of ONE stage: tags them, hides them, records their names and binds them to ``Params``.

    ``spec_hash`` is the stage hash stamped on every object; ``created`` lists the object Names in creation order.
    Every ``bind`` failure is a ``KitGeometryError`` naming the object, the property and the expression.
    """

    def __init__(self, doc: Any, *, kit_id: str, stage: str, spec_hash: str, sheet: str = SHEET) -> None:
        self.doc, self.kit_id, self.stage, self.spec_hash, self.sheet = doc, kit_id, stage, spec_hash, sheet
        self.created: list[str] = []

    # --- plumbing ---
    def new(self, type_id: str, name: str, *, role: str = "aux", part: str = "", container: Any = None, hidden: bool = True) -> Any:
        """Create an object of ``type_id`` named ``name`` (FreeCAD may add a suffix when the Name is taken), tag it, put it
        into ``container`` and hide it. ``role`` is one of ``ROLES``."""
        if role not in ROLES:
            raise KitGeometryError(f"role {role!r} is not a kit role; use one of {', '.join(ROLES)}")
        obj = self.doc.addObject(type_id, name)
        tag(obj, kit_id=self.kit_id, stage=self.stage, role=role, part=part, spec_hash=self.spec_hash)
        if container is not None:
            container.addObject(obj)
        if hidden:
            set_visible(obj, False)
        self.created.append(obj.Name)
        return obj

    def bind(self, obj: Any, prop: str, expr: str) -> None:
        """``obj.setExpression(prop, expr)``; ``expr`` uses ``Params.<alias>`` references and unit literals."""
        try:
            obj.setExpression(prop, expr)
        except Exception as exc:    # noqa: BLE001
            raise KitGeometryError(f"cannot bind {obj.Name}.{prop} to '{expr}': {exc}; check that every Params row it names exists") from None

    def bind_all(self, obj: Any, exprs: "dict[str, str | None]") -> None:
        for prop, expr in exprs.items():
            if expr:
                self.bind(obj, prop, expr)

    # --- 2D ---
    def rrect_sketch(self, name: str, *, width: float, height: float, radius: float, origin: "tuple[float, float, float]",
                     exprs: "dict[str, str | None]", role: str = "aux", part: str = "", container: Any = None) -> Any:
        """A fully constrained rounded rectangle with its lower-left corner at ``origin`` (sketch placement), in the XY plane.

        The sketch has four lines, four tangent arcs of equal radius and the NAMED dimensional constraints ``size_x``,
        ``size_y`` and ``rc`` (the corner radius: ``Part::Fillet`` radii cannot be expression-bound, Sketcher ones can).
        ``exprs`` maps ``size_x`` / ``size_y`` / ``rc`` (constraints, written ``Constraints.<name>``) and
        ``Placement.Base.x`` / ``.y`` / ``.z`` to expression strings. A radius below ``MIN_RADIUS`` is raised to it and one above half
        the shorter side lowered to fit.
        """
        w, h = float(width), float(height)
        if w <= 0 or h <= 0:
            raise KitGeometryError(f"{name}: a {w:.3g} x {h:.3g} mm outline is empty; check the board and wall dimensions")
        r = min(max(float(radius), MIN_RADIUS), min(w, h) / 2 - 1e-3)
        sk = self.new("Sketcher::SketchObject", name, role=role, part=part, container=container)
        sk.Placement = App.Placement(V(*origin), App.Rotation())
        pi = math.pi
        sk.addGeometry([
            Part.LineSegment(V(r, 0, 0), V(w - r, 0, 0)),
            Part.LineSegment(V(w, r, 0), V(w, h - r, 0)),
            Part.LineSegment(V(w - r, h, 0), V(r, h, 0)),
            Part.LineSegment(V(0, h - r, 0), V(0, r, 0)),
            Part.ArcOfCircle(Part.Circle(V(w - r, r, 0), V(0, 0, 1), r), -pi / 2, 0),
            Part.ArcOfCircle(Part.Circle(V(w - r, h - r, 0), V(0, 0, 1), r), 0, pi / 2),
            Part.ArcOfCircle(Part.Circle(V(r, h - r, 0), V(0, 0, 1), r), pi / 2, pi),
            Part.ArcOfCircle(Part.Circle(V(r, r, 0), V(0, 0, 1), r), pi, 1.5 * pi),
        ], False)
        c = Sketcher.Constraint
        sk.addConstraint([
            c("Tangent", 0, 2, 4, 1), c("Tangent", 4, 2, 1, 1), c("Tangent", 1, 2, 5, 1), c("Tangent", 5, 2, 2, 1),
            c("Tangent", 2, 2, 6, 1), c("Tangent", 6, 2, 3, 1), c("Tangent", 3, 2, 7, 1), c("Tangent", 7, 2, 0, 1),
            c("Horizontal", 0), c("Horizontal", 2), c("Vertical", 1), c("Vertical", 3),
            c("Equal", 4, 5), c("Equal", 5, 6), c("Equal", 6, 7),
            c("PointOnObject", 3, 1, -2), c("PointOnObject", 0, 1, -1),
        ])
        for cname, constraint in (("rc", c("Radius", 4, r)), ("size_x", c("DistanceX", 3, 1, 1, 1, w)),
                                  ("size_y", c("DistanceY", 0, 1, 2, 1, h))):
            sk.renameConstraint(sk.addConstraint(constraint), cname)
        solved = sk.solve()
        if solved != 0:
            raise KitGeometryError(f"{name}: the rounded-rectangle sketch does not solve (code {solved}) for {w:.3g} x {h:.3g} mm, radius {r:.3g} mm; check the corner radius")
        for prop, expr in exprs.items():
            if expr:
                self.bind(sk, prop if prop.startswith("Placement") else f"Constraints.{prop}", expr)
        return sk

    # --- 3D primitives ---
    def extrude(self, name: str, sketch: Any, *, length: float, length_expr: "str | None", reverse: bool = False, role: str = "aux",
                part: str = "", container: Any = None) -> Any:
        """A solid ``Part::Extrusion`` of ``sketch`` along its normal (``reverse``: against it) by ``length`` mm."""
        ex = self.new("Part::Extrusion", name, role=role, part=part, container=container)
        ex.Base = sketch
        ex.DirMode = "Normal"
        ex.Solid = True
        ex.Reversed = bool(reverse)
        ex.LengthFwd = float(length)
        if length_expr:
            self.bind(ex, "LengthFwd", length_expr)
        return ex

    def box(self, name: str, size: "tuple[float, float, float]", origin: "tuple[float, float, float]", *,
            exprs: "dict[str, str | None] | None" = None, rotation: "App.Rotation | None" = None, role: str = "aux",
            part: str = "", container: Any = None) -> Any:
        """A ``Part::Box`` of ``size`` (x, y, z extents) with its corner at ``origin``; ``exprs`` binds ``Length`` / ``Width`` /
        ``Height`` / ``Placement.Base.x|y|z``."""
        b = self.new("Part::Box", name, role=role, part=part, container=container)
        b.Length, b.Width, b.Height = (float(s) for s in size)
        b.Placement = App.Placement(V(*origin), rotation or App.Rotation())
        self.bind_all(b, exprs or {})
        return b

    def cylinder(self, name: str, radius: float, height: float, origin: "tuple[float, float, float]", *,
                 exprs: "dict[str, str | None] | None" = None, role: str = "aux", part: str = "", container: Any = None) -> Any:
        """A ``Part::Cylinder`` along +Z with its base centre at ``origin``."""
        cyl = self.new("Part::Cylinder", name, role=role, part=part, container=container)
        cyl.Radius, cyl.Height = float(radius), float(height)
        cyl.Placement = App.Placement(V(*origin), App.Rotation())
        self.bind_all(cyl, exprs or {})
        return cyl

    def static(self, name: str, shape: Any, origin: "tuple[float, float, float]" = (0.0, 0.0, 0.0), *,
               exprs: "dict[str, str | None] | None" = None, role: str = "aux", part: str = "", container: Any = None) -> Any:
        """A ``Part::Feature`` holding ``shape`` (built around the local origin) placed at ``origin``; ``exprs`` may bind the placement."""
        f = self.new("Part::Feature", name, role=role, part=part, container=container)
        f.Shape = shape
        f.Placement = App.Placement(V(*origin), App.Rotation())
        self.bind_all(f, exprs or {})
        return f

    # --- booleans ---
    def cut(self, name: str, base: Any, tool: Any, *, role: str = "aux", part: str = "", container: Any = None) -> Any:
        """``Part::Cut`` of ``base`` minus ``tool`` with ``Refine`` (coplanar faces merged)."""
        c = self.new("Part::Cut", name, role=role, part=part, container=container)
        c.Base, c.Tool, c.Refine = base, tool, True
        return c

    def fuse(self, name: str, base: Any, tool: Any, *, role: str = "aux", part: str = "", container: Any = None) -> Any:
        """``Part::Fuse`` of ``base`` and ``tool`` with ``Refine``."""
        f = self.new("Part::Fuse", name, role=role, part=part, container=container)
        f.Base, f.Tool, f.Refine = base, tool, True
        return f

    def union(self, name: str, objs: list, *, role: str = "aux", part: str = "", container: Any = None) -> Any:
        """One object standing for the union of ``objs``: the object itself for one, a ``Part::MultiFuse`` for two or more."""
        if not objs:
            raise KitGeometryError(f"{name}: nothing to unite")
        if len(objs) == 1:
            return objs[0]
        m = self.new("Part::MultiFuse", name, role=role, part=part, container=container)
        m.Shapes = list(objs)
        m.Refine = True
        return m

    def container(self, name: str, part: str, *, role: str = "container") -> Any:
        """Get or create the ``App::Part`` container ``name`` of this kit (found by tags, so two kits do not share one)."""
        for o in self.doc.Objects:
            t = tags_of(o)
            if t["OscarKit"] == self.kit_id and t["OscarRole"] == role and t["OscarPart"] == part and o.TypeId == "App::Part":
                return o
        obj = self.new("App::Part", name, role=role, part=part, hidden=False)
        return obj


def polygon_area(points: list) -> float:
    """Area of a closed planar polygon of 3D vectors (Newell's method; any plane)."""
    nx = ny = nz = 0.0
    for p, q in zip(points, points[1:] + points[:1]):
        nx += (p.y - q.y) * (p.z + q.z)
        ny += (p.z - q.z) * (p.x + q.x)
        nz += (p.x - q.x) * (p.y + q.y)
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


# --------------------------------------------------------------------------------------------------------------------
# snap-fit numbers (pure)
# --------------------------------------------------------------------------------------------------------------------

def pocket_depth(overhang: float, fit: float) -> float:
    """Depth of the blind catch pocket: ``overhang - fit + 0.1`` (``spec.catch_pocket_depth``, 7.2a)."""
    return overhang - fit + _spec.POCKET_SEAT


def barb_profile(overhang: float, fit: float, barb_h: float, ramp_deg: float, retention_deg: float) -> dict:
    """The barb cross-section for the given numbers (module docstring: "Barb profile").

    Returns ``{"points": [(t, z), ...], "z_wall": z, "root_h": .., "land": .., "scale": s, "flattened": bool,
    "ramp_eff_deg": .., "retention_eff_deg": .., "engagement": overhang - fit}``. ``points`` is the closed polygon
    (counter-clockwise, ``t`` inward from the beam's inner face, ``z`` from the barb's root bottom = 0); the first and last two
    points are the root, lying 0.3 mm INSIDE the beam (``t < 0``) so the barb fuses with it. ``z_wall`` is the height of the
    barb's lower edge at the wall plane (``t = fit``): the catch pocket starts there. Raises ``KitGeometryError`` unless
    ``overhang > fit``, ``barb_h > 0`` and both angles are in (0, 90].
    """
    d = overhang - fit
    if d <= 0 or barb_h <= 0 or not 0 < ramp_deg <= 90 or not 0 < retention_deg <= 90:
        raise KitGeometryError(f"hooks: cannot form a barb from overhang {overhang:g}, fit {fit:g}, barb_h {barb_h:g}, ramp {ramp_deg:g}, "
                               f"retention {retention_deg:g}; the barb must reach past the gap (overhang > lid.fit) and the angles must be 0..90 degrees")
    k_l = 1.0 / math.tan(math.radians(min(ramp_deg, 89.999)))
    k_u = 1.0 / math.tan(math.radians(min(retention_deg, 89.999)))
    need = d * (k_l + k_u)
    avail = barb_h * (1.0 - LAND_FRACTION)
    s = min(1.0, avail / need) if need > 0 else 1.0
    rise_l, rise_u = s * k_l * d, s * k_u * d
    land = barb_h - rise_l - rise_u
    z_wall = s * k_l * fit
    root = 0.3
    pts = [(-root, 0.0), (0.0, 0.0), (overhang, s * k_l * overhang), (overhang, s * k_l * overhang + land),
           (0.0, z_wall + barb_h + s * k_u * fit), (-root, z_wall + barb_h + s * k_u * fit)]
    return {
        "points": pts, "z_wall": z_wall, "root_h": z_wall + barb_h + s * k_u * fit, "land": land, "scale": s,
        "flattened": s < 1.0 - 1e-9,
        "ramp_eff_deg": math.degrees(math.atan2(1.0, s * k_l)) if k_l * s > 0 else 90.0,
        "retention_eff_deg": math.degrees(math.atan2(1.0, s * k_u)) if k_u * s > 0 else 90.0,
        "engagement": d,
    }


def barb_shape(profile: dict, width: float, inward_sign: int) -> Any:
    """The barb as a solid: the profile extruded ``width`` along +X from x = 0; ``t`` maps to ``y = inward_sign * t`` (so
    ``-1`` for the ``+y`` wall, whose inside is toward -y) and ``z`` to z. Placement origin = (hook x0, beam inner face y, beam bottom z)."""
    pts = [V(0.0, inward_sign * t, z) for t, z in profile["points"]]
    if polygon_area(pts) <= 1e-9:
        raise KitGeometryError("the barb profile is degenerate; check hooks.overhang, hooks.barb_h and the angles")
    face = Part.Face(Part.makePolygon(pts + [pts[0]]))
    solid = face.extrude(V(float(width), 0, 0))
    if solid.Volume < 0:
        solid.reverse()
    if solid.Volume < 0:      # pragma: no cover - extrude orientation differs between OCC builds
        solid = Part.Solid(solid.Shells[0].reversed())
    return solid


def hook_plans(spec: "_spec.EnclosureSpec | dict", board: dict, *, pcb_t: "float | None" = None) -> list[dict]:
    """Every number of every snap hook of ``spec`` on ``board`` (pure; empty unless ``lid.style == 'snap_fit'``).

    One dict per ``spec.hook_positions`` entry, in that order: ``{"wall", "u", "w", "t", "length", "inward" (-1 for ``+y``, +1 for
    ``-y``), "beam": {"x0", "x1"}, "pocket": {"x0", "x1", "w", "h", "depth", "z_off"}, "barb": profile}`` where ``z_off`` is the
    pocket floor above the beam bottom and ``h = barb_h + window_margin``. The beam hangs ``length`` below the skirt; positions
    come from ``hook_positions`` (they avoid every wall opening, 7.2a).
    """
    s = _spec._as_spec(spec)
    h = s.hooks
    out = []
    positions = hook_positions(s, board, pcb_t=pcb_t)
    if not positions:
        return out
    prof = barb_profile(h.overhang, s.lid.fit, h.barb_h, h.ramp_deg, h.retention_deg)
    for p in positions:
        u, w = p["u"], p["w"]
        out.append({
            "wall": p["wall"], "u": u, "w": w, "t": h.t, "length": h.length, "inward": -1 if p["wall"] == "+y" else 1,
            "beam": {"x0": u - w / 2, "x1": u + w / 2},
            "pocket": {"x0": u - (w + 0.4) / 2, "x1": u + (w + 0.4) / 2, "w": w + 0.4, "h": h.barb_h + h.window_margin,
                       "depth": pocket_depth(h.overhang, s.lid.fit), "z_off": prof["z_wall"]},
            "barb": prof,
        })
    return out


def box_edges_along(shape: Any, axis: int) -> list:
    """The edges of a box-like ``shape`` parallel to coordinate ``axis`` (0, 1, 2), used to round a cutter's corners."""
    out = []
    for e in shape.Edges:
        a, b = e.Vertexes[0].Point, e.Vertexes[-1].Point
        d = [abs(b.x - a.x), abs(b.y - a.y), abs(b.z - a.z)]
        if d[axis] > 1e-9 and sum(1 for i in range(3) if i != axis and d[i] < 1e-9) == 2:
            out.append(e)
    return out
