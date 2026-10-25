"""Generate the deterministic Pi-4-like STEP fixtures of the reference-part tests, headlessly.

Run it under ``freecadcmd`` (a plain interpreter has no ``Part``)::

    freecadcmd -c "exec(open('tests/fixtures/refparts/build_pi4_step.py').read()); main()"

or import it from a test that already runs inside FreeCAD (``build_pi4_step.write_fixtures(dest)``). It writes

* ``pi4_like.step``         the board in the kit frame: a PCB box 85 x 56 x 1.4 mm with the four 2.75 mm mounting holes at
                            the 58 x 49 mm pattern (3.5, 3.5) and ONE solid per connector box of ``boards/rpi4b.json`` (the nine
                            connector boxes, GPIO header included), each labelled with its connector id;
* ``pi4_like_offset.step``  the same board with its origin at the board CENTRE and its long edge along +Y (a quarter turn about Z),
                            to prove that ``import_reference_part`` normalizes an arbitrary vendor frame into the kit frame;
* ``pi4_like.sha256``       ``sha256sum`` format, one line per STEP file;
* ``mirror/rpi4b.step``     a copy of ``pi4_like.step`` named like a catalog id: point ``OSCAR_REFPART_LOCAL_DIR`` at ``mirror/``.

Deterministic: geometry comes from the board file with fixed numbers; the two run-dependent texts of an Open CASCADE STEP file, the
``FILE_NAME`` time stamp and the process-wide counter that numbers ``NEXT_ASSEMBLY_USAGE_OCCURRENCE`` entries, are normalized, so two
runs (even in one interpreter) give identical bytes (the tests assert it) and the recorded hash verifies the committed files. The bytes can differ between Open CASCADE versions; the tests therefore check the committed files
against their recorded hash and generate fresh copies only to compare run against run.
"""

import hashlib
import json
import os
import re
import shutil
import sys

STAMP = "2026-01-01T00:00:00"
HOLE_D = 2.75
HOLE_PATTERN = ((3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5))
BOARD_LENGTH, BOARD_WIDTH, PCB_T = 85.0, 56.0, 1.4
_STAMP_RE = re.compile(r"(FILE_NAME\('[^']*',')\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(')")
_OCCURRENCE_RE = re.compile(r"(NEXT_ASSEMBLY_USAGE_OCCURRENCE\(')\d+(')")


def _here() -> str:
    path = globals().get("__file__")
    if path:
        return os.path.dirname(os.path.abspath(path))
    return os.path.abspath(os.path.join(os.getcwd(), "tests", "fixtures", "refparts"))


def board_file() -> dict:
    """``oscar/refparts/boards/rpi4b.json`` read straight from disk (the generator needs no ``oscar`` import)."""
    mod = os.path.abspath(os.path.join(_here(), "..", "..", ".."))
    with open(os.path.join(mod, "oscar", "refparts", "boards", "rpi4b.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def board_shapes() -> list:
    """``[(label, Part.Shape), ...]``: the PCB first, then one box per connector of the Pi 4 board file, in file order."""
    import FreeCAD as App
    import Part

    pcb = Part.makeBox(BOARD_LENGTH, BOARD_WIDTH, PCB_T)
    for x, y in HOLE_PATTERN:
        pcb = pcb.cut(Part.makeCylinder(HOLE_D / 2.0, PCB_T + 2.0, App.Vector(x, y, -1.0)))
    shapes = [("PCB", pcb)]
    for conn in board_file()["connectors"]:
        (x0, x1), (y0, y1), (z0, z1) = conn["box"]["x"], conn["box"]["y"], conn["box"]["z"]
        shapes.append((conn["id"], Part.makeBox(x1 - x0, y1 - y0, z1 - z0, App.Vector(x0, y0, z0))))
    return shapes


def _export(shapes: list, path: str, doc_name: str) -> None:
    import FreeCAD as App
    import Import

    doc = App.newDocument(doc_name)
    try:
        objects = []
        for label, shape in shapes:
            obj = doc.addObject("Part::Feature", re.sub(r"[^A-Za-z0-9_]", "_", label))
            obj.Label = label
            obj.Shape = shape
            objects.append(obj)
        doc.recompute()
        Import.export(objects, path)
    finally:
        App.closeDocument(doc.Name)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    text, count = _STAMP_RE.subn(lambda m: f"{m.group(1)}{STAMP}{m.group(2)}", text, count=1)
    if not count:
        raise RuntimeError(f"{path}: no FILE_NAME time stamp found; the STEP writer changed, update _STAMP_RE")
    # The writer numbers assembly occurrences with a counter that lives as long as the process (a second export in the same
    # interpreter continues where the first stopped): renumber them 1..n in file order so every run gives the same bytes.
    counter = iter(range(1, 100000))
    text = _OCCURRENCE_RE.sub(lambda m: f"{m.group(1)}{next(counter)}{m.group(2)}", text)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def offset_shapes() -> list:
    """The board with its origin at the board centre and the long edge along +Y (quarter turn about Z, +90 degrees)."""
    import FreeCAD as App

    centre = App.Vector(BOARD_LENGTH / 2.0, BOARD_WIDTH / 2.0, PCB_T / 2.0)
    rotation = App.Rotation(App.Vector(0, 0, 1), 90)
    out = []
    for label, shape in board_shapes():
        moved = shape.copy()
        moved.translate(App.Vector(-centre.x, -centre.y, -centre.z))
        moved.rotate(App.Vector(0, 0, 0), App.Vector(0, 0, 1), rotation.Angle * 180.0 / 3.141592653589793)
        out.append((label, moved))
    return out


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_fixtures(dest_dir: str | None = None) -> dict:
    """Write the fixture files into ``dest_dir`` (default: this directory) and return ``{"step", "offset", "mirror", "sha256", "hashes"}``.

    ``mirror`` is the directory to use as ``OSCAR_REFPART_LOCAL_DIR`` (it holds ``rpi4b.step``). Existing files are replaced.
    """
    dest = os.path.abspath(dest_dir or _here())
    os.makedirs(os.path.join(dest, "mirror"), exist_ok=True)
    step, offset = os.path.join(dest, "pi4_like.step"), os.path.join(dest, "pi4_like_offset.step")
    _export(board_shapes(), step, "Pi4Like")
    _export(offset_shapes(), offset, "Pi4LikeOffset")
    shutil.copyfile(step, os.path.join(dest, "mirror", "rpi4b.step"))
    hashes = {os.path.basename(p): sha256_file(p) for p in (step, offset)}
    sums = os.path.join(dest, "pi4_like.sha256")
    with open(sums, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(f"{digest}  {name}\n" for name, digest in sorted(hashes.items()))
    return {"step": step, "offset": offset, "mirror": os.path.join(dest, "mirror"), "sha256": sums, "hashes": hashes}


def main(dest_dir: str | None = None) -> None:
    """``write_fixtures`` plus one line on stdout (call it explicitly: ``freecadcmd -c`` runs the file as ``__main__`` with its own argv)."""
    result = write_fixtures(dest_dir)
    sys.stdout.write("wrote " + ", ".join(sorted(result["hashes"])) + f" into {os.path.dirname(result['step'])}\n")
