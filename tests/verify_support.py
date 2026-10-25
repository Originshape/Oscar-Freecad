"""Shared fixtures of the verification tests: no test case on purpose (helper module, collected as a file only).

``build_enclosure`` makes a small but REAL enclosure in the kit's board frame without the kit builder: a PCB with a
tall component in a ``Ref_TEST`` container, a shell (floor, four walls, a standoff, a blind catch pocket), a cap lid
(plate, skirt, one snap hook with a barb) and a ``Params`` sheet. The numbers are chosen so every generic fit row has a
known answer (clearance 2.0, lid gap 2.0, barb engaging 0.0 mm3 when seated and 0.64 mm3 after a 0.5 mm lift); the
keyword switches break exactly one thing each, so a failing-validation example is one argument away.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Part  # noqa: E402

V = App.Vector


import contextlib  # noqa: E402


@contextlib.contextmanager
def hidden_module(name):
    """Make ``import <name>`` / ``from <parent> import <leaf>`` raise ``ModuleNotFoundError`` (a module that was not
    delivered): the module, the attribute on its parent package and the sys.modules entry are all hidden, then restored."""
    parent_name, _, leaf = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    saved_module = sys.modules.get(name)
    saved_attr = parent.__dict__.pop(leaf, None) if parent is not None else None
    sys.modules[name] = None
    try:
        yield
    finally:
        if saved_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved_module
        if parent is not None and saved_attr is not None:
            setattr(parent, leaf, saved_attr)


def box(x, y, z, dx, dy, dz):
    """``Part.makeBox`` from a corner and the three extents."""
    return Part.makeBox(dx, dy, dz, V(x, y, z))


def tag(obj, role="", stage="", part="", kit="enc1"):
    """The kit's ``Oscar*`` string properties."""
    for name, value in (("OscarKit", kit), ("OscarStage", stage), ("OscarRole", role), ("OscarPart", part)):
        if value:
            obj.addProperty("App::PropertyString", name, "Oscar", "")
            setattr(obj, name, value)


def feature(doc, name, shape, *, role="", stage="", part="", visible=True, label=None):
    """A ``Part::Feature`` holding ``shape`` (no boolean chain: the geometry is what the checks read)."""
    obj = doc.addObject("Part::Feature", name)
    obj.Shape = shape
    if label:
        obj.Label = label
    tag(obj, role, stage, part)
    if not visible:
        obj.Visibility = False
    return obj


def params(doc, rows, name="Params"):
    """The kit's ``Params`` layout: header row, value in B (aliased), unit C, source D, note E.
    ``rows``: ``(alias, formula, unit, source)``."""
    sheet = doc.addObject("Spreadsheet::Sheet", name)
    for col, title in zip("ABCDE", ("name", "value", "unit", "source", "note")):
        sheet.set(f"{col}1", title)
    for i, (alias, formula, unit, source) in enumerate(rows, start=2):
        sheet.set(f"A{i}", alias)
        sheet.set(f"B{i}", formula)
        sheet.setAlias(f"B{i}", alias)
        sheet.set(f"C{i}", unit)
        sheet.set(f"D{i}", source)
    doc.recompute()
    return sheet


DEFAULT_PARAMS = [
    ("wall_t", "=2 mm", "mm", "assumption"),
    ("clearance", "=2 mm", "mm", "user_requirement"),
    ("clearance_eff", "=2 mm", "mm", "derived"),
    ("lid_gap", "=2 mm", "mm", "assumption"),
    ("lid_lift", "=0 mm", "mm", "assumption"),
    ("z_cavity_top", "=14 mm", "mm", "derived"),
]


def build_enclosure(doc, *, hooks=True, lid_lift=0.0, overlap=False, wide_pcb=False, ref_poke=False,
                    nest=False, with_params=True, tagged=True, offset_z=0.0):
    """Build the fixture in ``doc`` and return ``{"shell", "lid", "plate", "pcb", "comp", "ref", "sheet", ...}``.

    ``hooks=False`` leaves the barb off the lid (retention must FAIL); ``lid_lift`` lifts the lid container AND
    sets ``Params.lid_lift``; ``overlap`` adds a block to the lid that cuts into the shell wall (a 40 mm3 interference FAIL);
    ``wide_pcb`` makes the board 62 mm wide (clearance FAIL); ``ref_poke`` adds a reference box through the -x wall;
    ``nest`` puts everything in an ``Assembly`` ``App::Part`` placed ``offset_z`` up (the global-frame trap);
    ``tagged=False`` drops the ``Oscar*`` tags (the Label fallback of the part finder)."""
    out = {}
    sheet = None
    if with_params:
        rows = [(a, f, u, s) for a, f, u, s in DEFAULT_PARAMS if a != "lid_lift"] + [
            ("lid_lift", f"={lid_lift} mm", "mm", "assumption")]
        sheet = params(doc, rows)
    out["sheet"] = sheet
    pcb_w = 62.0 if wide_pcb else 60.0
    ref = doc.addObject("App::Part", "Ref_TEST")
    pcb = feature(doc, "PCB", box(0, 0, 0, pcb_w, 40, 1.6), role="ref" if tagged else "", stage="reference", part="Ref")
    comp = feature(doc, "Comp", box(20, 15, 1.6, 10, 10, 10.4), role="ref" if tagged else "", stage="reference",
                   part="Ref")
    ref.addObject(pcb)
    ref.addObject(comp)
    out.update(ref=ref, pcb=pcb, comp=comp)
    if ref_poke:
        poke = feature(doc, "Poke", box(-6, 10, 2, 10, 4, 4), role="ref" if tagged else "", stage="reference",
                       part="Ref")
        ref.addObject(poke)
        out["poke"] = poke
    # shell: floor and walls (open top), a standoff under the board corner, a blind catch pocket in the -y wall
    shell_shape = box(-4, -4, -5, 68, 48, 19).cut(box(-2, -2, -3, 64, 44, 18))
    shell_shape = shell_shape.fuse(Part.makeCylinder(2.5, 3, V(3.5, 3.5, -3)))
    shell_shape = shell_shape.cut(box(8, -4, 2.9, 8, 0.7, 1.4))
    shell = feature(doc, "Shell", shell_shape, role="shell" if tagged else "", stage="shell", part="Shell")
    out["shell"] = shell
    # lid: plate on top, skirt around the outside of the wall top, optional hook beam + barb
    plate = feature(doc, "Lid_Plate", box(-6, -6, 14, 72, 52, 2), role="lid" if tagged else "",
                    stage="lid" if tagged else "", part="Lid", visible=False)
    lid_shape = plate.Shape.fuse(box(-6, -6, 10, 72, 52, 4).cut(box(-4.2, -4.2, 10, 68.4, 48.4, 4)))
    if hooks:
        hook_shape = box(8, -6, 2, 8, 1.8, 8).fuse(box(8, -4.2, 3.0, 8, 0.6, 1.0))
        lid_shape = lid_shape.fuse(hook_shape)
        out["hooks"] = feature(doc, "Lid_Hooks", hook_shape, role="hook" if tagged else "", stage="hooks",
                               part="Lid", visible=False)
    if overlap:
        lid_shape = lid_shape.fuse(box(30, -4.4, 9, 10, 1.4, 4))
    lid = feature(doc, "Lid", lid_shape.removeSplitter(), role="lid" if tagged else "", stage="lid", part="Lid")
    out.update(lid=lid, plate=plate)
    lid_part = doc.addObject("App::Part", "Part_Lid")
    lid_part.addObject(plate)
    lid_part.addObject(lid)
    lid_part.Placement = App.Placement(V(0, 0, lid_lift), App.Rotation())
    shell_part = doc.addObject("App::Part", "Part_Shell")
    shell_part.addObject(shell)
    out.update(lid_part=lid_part, shell_part=shell_part)
    if nest:
        assembly = doc.addObject("App::Part", "Assembly")
        for member in (shell_part, lid_part, ref):
            assembly.addObject(member)
        assembly.Placement = App.Placement(V(0, 0, offset_z), App.Rotation())
        out["assembly"] = assembly
    doc.recompute()
    return out


def build_parametric(doc, *, slot=8.0, role="shell"):
    """A parametric block with a slot: ``Body = Base - SlotTool``; ``Base`` is bound to Params.len / wid / hei and the
    tool's width to ``Params.slot + 1 mm``. With ``slot=8`` every +-10% edit leaves ONE solid; with ``slot=9.5`` the
    +10% slot (10.45 mm) cuts the 10 mm wide block in two (a FRAGMENTS error on the tagged body)."""
    sheet = params(doc, [("len", "=20 mm", "mm", "user_requirement"), ("wid", "=10 mm", "mm", "user_requirement"),
                         ("hei", "=5 mm", "mm", "user_requirement"), ("slot", f"={slot} mm", "mm", "assumption")])
    base = doc.addObject("Part::Box", "Base")
    base.setExpression("Length", "Params.len")
    base.setExpression("Width", "Params.wid")
    base.setExpression("Height", "Params.hei")
    cutter = doc.addObject("Part::Box", "SlotTool")
    cutter.Placement.Base = V(8, -1, -1)
    cutter.Length = 4
    cutter.Height = 7
    cutter.setExpression("Width", "Params.slot + 1 mm")
    body = doc.addObject("Part::Cut", "Body")
    body.Base, body.Tool = base, cutter
    tag(body, role, "shell" if role else "", "Shell" if role else "")
    doc.recompute()
    return {"sheet": sheet, "base": base, "tool": cutter, "body": body}


class FakeProject:
    """The Project methods ``run_validation`` / the verification tools call, over plain dicts."""

    def __init__(self, spec=None, reference_status="none", last_geometry_step=3):
        self.spec = spec
        self.reference_status = reference_status
        self.step = last_geometry_step
        self.validation = None
        self.recorded = []

    def read_spec(self):
        return self.spec

    def read_validation(self):
        return self.validation

    def last_geometry_step(self):
        return self.step

    def record_validation(self, validation):
        data = validation if isinstance(validation, dict) else dict(validation.__dict__)
        self.recorded.append(data)
        self.validation = data


class FakeLedger:
    """Records ``apply_validation`` calls; ``reference_status`` and ``last_geometry_step`` as the real Ledger has them."""

    def __init__(self, reference_status="none", last_geometry_step=3):
        self.reference_status = reference_status
        self.last_geometry_step = last_geometry_step
        self.applied = []

    def apply_validation(self, validation):
        self.applied.append(validation)
