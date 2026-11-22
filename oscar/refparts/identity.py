"""What a downloaded file really is, and whether it is the part we asked for.

PURE module (stdlib only, no FreeCAD, no network). Two jobs:

1. Magic-byte sniffing. A download that ends in ``.step`` is routinely an HTML
   landing page, a login wall, a GitHub 404 or a git-LFS pointer (a few lines of
   text). ``sniff_bytes`` / ``sniff_file`` look at the CONTENT, never at the name,
   and return one of ``KNOWN_TYPES``; ``rejection_hint`` says what to tell the
   model when that content is not a usable file.
2. Identity. ``bbox_identity`` / ``outline_identity`` compare the bounding box of
   an imported reference with what the catalog expects. This is a plausibility
   check, not a proof: two boards with the same footprint (Raspberry Pi 3B+ and 4)
   cannot be told apart by size, and the result says so in its wording.
3. Hash policy. ``pin_verdict`` decides what to do with a catalog file whose sha256 no longer
   matches the pin (the vendor replaced the file): keep it with a warning while its
   bounding box and outline still match, refuse it otherwise.
"""

import math
import os
import struct
from dataclasses import dataclass

# --- detected types ---------------------------------------------------------

STEP = "step"
STL = "stl"            # ascii and binary alike (the flavour is not needed downstream)
ZIP = "zip"
PDF = "pdf"
HTML = "html"
LFS_POINTER = "lfs_pointer"
BREP = "brep"
IGES = "iges"
EMPTY = "empty"
UNKNOWN = "unknown"
KNOWN_TYPES = (STEP, STL, ZIP, PDF, HTML, LFS_POINTER, BREP, IGES, EMPTY, UNKNOWN)

# Types that are geometry the app can import (a download expected to be CAD must be one of these or a ZIP of them).
CAD_TYPES = (STEP, STL, BREP, IGES)

# Extensions the app can bring in as geometry (kept in step with ``attachments.CAD_EXTENSIONS``).
CAD_EXTENSIONS = frozenset({".step", ".stp", ".iges", ".igs", ".brep", ".brp", ".stl", ".obj",
                            ".3mf", ".ply", ".fcstd"})

# Extensions that are consistent with each detected type (a ``.3mf`` and a ``.FCStd`` are ZIP containers).
EXTENSIONS_FOR = {
    STEP: (".step", ".stp"),
    STL: (".stl",),
    ZIP: (".zip", ".3mf", ".fcstd"),
    PDF: (".pdf",),
    BREP: (".brep", ".brp"),
    IGES: (".iges", ".igs"),
}

# File extensions whose content ``sniff_*`` can verify (STEP / IGES / BREP / STL); other CAD formats (OBJ, PLY, DXF ...) cannot be sniffed.
EXTENSIONS_FOR_FORMATS = frozenset(ext for t in (STEP, STL, BREP, IGES) for ext in EXTENSIONS_FOR[t])

SNIFF_BYTES = 4096      # how much of the file head ``sniff_file`` reads

_BOM = b"\xef\xbb\xbf"
_HTML_STARTS = (b"<!doctype html", b"<html", b"<head", b"<body", b"<meta", b"<title")

_HINTS = {
    HTML: ("the server returned an HTML web page, not a file (a landing page, a login wall or an error page); "
           "use the direct download URL (for GitHub the raw URL https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>) "
           "or ask the user to download the file in a browser and attach it"),
    LFS_POINTER: ("the URL returned a git-LFS pointer (a few lines of text), not the real file; "
                  "use the GitHub 'media' URL https://media.githubusercontent.com/media/<owner>/<repo>/<branch>/<path> "
                  "or ask the user to download the file and attach it"),
    EMPTY: "the download is empty (0 bytes); the URL does not serve a file, ask the user to attach it",
}

_LABELS = {
    STEP: "a STEP file", STL: "an STL mesh", ZIP: "a ZIP archive", PDF: "a PDF document",
    HTML: "an HTML page", LFS_POINTER: "a git-LFS pointer", BREP: "an OpenCASCADE BREP file",
    IGES: "an IGES file", EMPTY: "an empty file", UNKNOWN: "a file of unrecognised type",
}


def describe_type(detected_type: str) -> str:
    """Human wording for a detected type (``"a STEP file"``); unknown values come back as ``"a file of unrecognised type"``."""
    return _LABELS.get(detected_type, _LABELS[UNKNOWN])


def canonical_extension(detected_type: str) -> str:
    """The extension to give a file of this type (``".step"``), or ``""`` when the type has none (unknown, html, ...)."""
    exts = EXTENSIONS_FOR.get(detected_type)
    return exts[0] if exts else ""


def extension_matches(ext: str, detected_type: str) -> bool:
    """True when ``ext`` (with the dot, any case) is a normal extension for ``detected_type``."""
    return (ext or "").lower() in EXTENSIONS_FOR.get(detected_type, ())


def rejection_hint(detected_type: str) -> str:
    """Why content of this type is never a usable CAD download, in the ``"<why>; <what to do>"`` style; ``""`` when it is acceptable.

    Only ``html``, ``lfs_pointer`` and ``empty`` are rejected on sight: every other type (including ``unknown``) may be a
    legitimate file, and callers that need a specific type say so with ``expect=`` (``netguard.download_verified``).
    """
    return _HINTS.get(detected_type, "")


# --- sniffing ----------------------------------------------------------------

def _is_binary_stl(head: bytes, size) -> bool:
    """Binary STL: 80-byte header, uint32 triangle count, then exactly 50 bytes per triangle."""
    if size is None or len(head) < 84:
        return False
    count = struct.unpack_from("<I", head, 80)[0]
    return count > 0 and 84 + 50 * count == size


def _is_ascii_stl(body: bytes) -> bool:
    low = body.lstrip().lower()
    if not low.startswith(b"solid"):
        return False
    window = body[:SNIFF_BYTES].lower()
    return b"facet" in window or b"endsolid" in window


def _is_iges(body: bytes) -> bool:
    first = body.split(b"\n", 1)[0].rstrip(b"\r")
    return len(first) >= 73 and first[72:73] == b"S" and first[73:80].strip().isdigit()


def sniff_bytes(head: bytes, size: int | None = None) -> str:
    """Classify a file from its first bytes. Returns one of ``KNOWN_TYPES``; never raises.

    ``head`` is the start of the file (``SNIFF_BYTES`` is plenty); ``size`` is the whole file's byte count when known. Binary
    STL can only be recognised with ``size`` (the header is free text and may even start with ``solid``); without it a binary
    STL comes back ``unknown``. A zero-length head with ``size`` of 0 or ``None`` is ``empty``.
    """
    head = bytes(head or b"")
    if not head:
        return EMPTY if size in (None, 0) else UNKNOWN
    body = head[3:] if head.startswith(_BOM) else head
    if body.startswith(b"version https://git-lfs.github.com/spec/") and (size is None or size < 2048):
        return LFS_POINTER
    if body.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return ZIP
    low = body[:1024].lstrip().lower()
    if low.startswith(_HTML_STARTS) or (low.startswith((b"<?xml", b"<!--")) and b"<html" in low):
        return HTML
    stripped = body.lstrip()
    if stripped.startswith(b"%PDF-") or b"%PDF-" in body[:1024]:
        return PDF
    if stripped[:13].upper() == b"ISO-10303-21;" or stripped[:12].upper() == b"ISO-10303-21":
        return STEP
    if stripped.startswith(b"DBRep_DrawableShape"):
        return BREP
    if _is_binary_stl(body, size):
        return STL
    if _is_ascii_stl(body):
        return STL
    if _is_iges(body):
        return IGES
    return UNKNOWN


def sniff_file(path: str) -> str:
    """``sniff_bytes`` of a file on disk (reads the first ``SNIFF_BYTES`` bytes). Raises ``OSError`` when it cannot be read."""
    with open(path, "rb") as fh:
        head = fh.read(SNIFF_BYTES)
    return sniff_bytes(head, os.path.getsize(path))


# --- identity ---------------------------------------------------------------

@dataclass(frozen=True)
class IdentityResult:
    """Outcome of an identity check.

    ``ok`` is False ONLY for a verdict of ``mismatch``: an ``unchecked`` result (nothing to compare against, or unusable
    numbers) is ``ok=True`` so that a missing expectation never blocks a build. ``detail`` is one sentence for the model
    and for the provenance note. ``measured`` / ``expected`` are the sorted (longest first) dimensions actually compared.
    """

    ok: bool
    status: str                 # "match" | "mismatch" | "unchecked"
    detail: str
    measured: tuple = ()
    expected: tuple = ()


# measured / expected ratio -> what probably went wrong (a STEP states its unit; a mis-read unit scales every side alike)
_UNIT_FACTORS = ((25.4, "the file may be in millimetres and read as inches"),
                 (1 / 25.4, "the file may be in inches and read as millimetres"),
                 (1000.0, "the file may be in millimetres and read as metres"),
                 (0.001, "the file may be in metres and read as millimetres"),
                 (10.0, "the file may be in millimetres and read as centimetres"),
                 (0.1, "the file may be in centimetres and read as millimetres"))


def _dims(values) -> tuple | None:
    """Positive finite numbers, longest first; None when ``values`` is not 2 or 3 of them."""
    try:
        values = list(values)
        nums = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if len(nums) not in (2, 3) or any(isinstance(v, bool) for v in values):
        return None
    if any(not math.isfinite(v) or v <= 0 for v in nums):
        return None
    return tuple(sorted(nums, reverse=True))


def _fmt(values, digits: int = 1) -> str:
    return " x ".join(f"{v:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{v:g}" for v in values)


def _unit_hint(measured: tuple, expected: tuple) -> str:
    ratios = [m / e for m, e in zip(measured, expected)]
    for factor, text in _UNIT_FACTORS:
        if all(abs(r / factor - 1.0) <= 0.03 for r in ratios):
            return f" (every side is x{factor:g} of the expected size: {text})"
    return ""


def bbox_identity(measured, expected, *, rel_tol: float = 0.08, abs_tol: float = 1.5) -> IdentityResult:
    """Compare a measured bounding box with an expected one, ignoring axis order and orientation.

    ``measured`` and ``expected`` are 3 numbers (x, y, z extents in mm) or 2 (an outline); both are sorted longest first and
    compared on the shorter of the two lengths, so a 3-D measurement can be checked against a 2-D outline. Each side may
    deviate by ``max(abs_tol, rel_tol * expected_side)``. Returns ``unchecked`` (``ok=True``) when either input is missing or
    not made of 2-3 positive finite numbers. A mismatch names the unit mix-ups it recognises (inches, metres, centimetres).
    """
    m = _dims(measured or ())
    e = _dims(expected or ())
    if m is None:
        return IdentityResult(True, "unchecked", "identity not checked: the measured bounding box is not 2 or 3 positive numbers")
    if e is None:
        return IdentityResult(True, "unchecked", "identity not checked: the catalog has no expected size for this part", m, ())
    n = min(len(m), len(e))
    m, e = m[:n], e[:n]
    worst = None
    for got, want in zip(m, e):
        tol = max(abs_tol, rel_tol * want)
        if abs(got - want) > tol:
            worst = (got, want, tol)
            break
    if worst is None:
        return IdentityResult(True, "match",
                              f"bounding box {_fmt(m)} mm matches the expected {_fmt(e, 0)} mm "
                              "(a size match is a plausibility check, not proof of identity)", m, e)
    got, want, tol = worst
    return IdentityResult(False, "mismatch",
                          f"bounding box {_fmt(m)} mm does not match the expected {_fmt(e, 0)} mm "
                          f"(a side measures {got:.1f} against {want:g}, tolerance {tol:.1f}){_unit_hint(m, e)}; "
                          "this is probably a different part", m, e)


def _board_overhang(board: dict) -> tuple[float, float]:
    """How far the connector boxes reach past the outline, summed over both sides: (along the length, along the width)."""
    outline = board.get("outline") or {}
    length, width = float(outline.get("length", 0.0)), float(outline.get("width", 0.0))
    xs0 = xs1 = ys0 = ys1 = None
    for conn in board.get("connectors") or []:
        box = conn.get("box") or {}
        try:
            (x0, x1), (y0, y1) = box["x"], box["y"]
        except (KeyError, TypeError, ValueError):
            continue
        xs0 = x0 if xs0 is None else min(xs0, x0)
        xs1 = x1 if xs1 is None else max(xs1, x1)
        ys0 = y0 if ys0 is None else min(ys0, y0)
        ys1 = y1 if ys1 is None else max(ys1, y1)
    over_x = (max(0.0, -xs0) + max(0.0, xs1 - length)) if xs0 is not None else 0.0
    over_y = (max(0.0, -ys0) + max(0.0, ys1 - width)) if ys0 is not None else 0.0
    return over_x, over_y


def outline_identity(measured, board: dict, *, rel_tol: float = 0.08, abs_tol: float = 1.5) -> IdentityResult:
    """Check the footprint of an imported reference against a board file (``catalog.load_board``), for parts with no expected_bbox.

    The two longest measured sides are compared with the board's length and width. A bare PCB measures exactly the outline;
    a full model with its connectors measures up to the outline plus the connector overhang, so each side is accepted from
    ``outline - tol`` up to ``outline + overhang + tol`` (overhang taken from the board's connector boxes). The third side
    (height) is ignored. Same plausibility caveat as ``bbox_identity``: boards with one footprint are indistinguishable.
    """
    m = _dims(measured or ())
    outline = (board or {}).get("outline") if isinstance(board, dict) else None
    try:
        length, width = float(outline["length"]), float(outline["width"])
    except (TypeError, KeyError, ValueError):
        return IdentityResult(True, "unchecked", "identity not checked: the board file has no outline size")
    if m is None:
        return IdentityResult(True, "unchecked", "identity not checked: the measured bounding box is not 2 or 3 positive numbers")
    over_x, over_y = _board_overhang(board)
    got_l, got_w = m[0], m[1]
    tol_l, tol_w = max(abs_tol, rel_tol * length), max(abs_tol, rel_tol * width)
    ok = (length - tol_l <= got_l <= length + over_x + tol_l) and (width - tol_w <= got_w <= width + over_y + tol_w)
    expected = (length, width)
    if ok:
        return IdentityResult(True, "match",
                              f"footprint {_fmt(m[:2])} mm is consistent with the {board.get('name', 'board')} outline "
                              f"{length:g} x {width:g} mm (plus connector overhang up to {over_x:g} x {over_y:g} mm); "
                              "a size match is a plausibility check, not proof of identity", m[:2], expected)
    return IdentityResult(False, "mismatch",
                          f"footprint {_fmt(m[:2])} mm does not match the {board.get('name', 'board')} outline "
                          f"{length:g} x {width:g} mm (accepted up to {length + over_x + tol_l:.1f} x {width + over_y + tol_w:.1f} mm)"
                          f"{_unit_hint(m[:2], expected)}; this is probably a different part", m[:2], expected)


# --- hash policy ----------------------------------------------------------------

def _as_check(item) -> "tuple[bool, str, str] | None":
    """``(ok, status, detail)`` of an ``IdentityResult`` or the dict form (``measure.identity_of``); ``None`` for anything else."""
    if isinstance(item, IdentityResult):
        return item.ok, item.status, item.detail
    if isinstance(item, dict) and "status" in item:
        return bool(item.get("ok", item.get("status") != "mismatch")), str(item["status"]), str(item.get("detail", ""))
    return None


def pin_verdict(pin: str, checks) -> tuple[str, str]:
    """What to do with a catalog file whose hash does (not) match the pin: ``(verdict, text)``.

    ``pin`` is ``fetch.pin_state``: ``"match"`` / ``"unpinned"`` / ``""`` -> ``("accepted", "")`` (the identity checks are advisory then).
    ``"mismatch"`` is the vendor-side-change case and is decided by ``checks`` (``IdentityResult`` s or their dicts, typically the
    bounding box against ``expected_bbox`` and the PCB footprint against the board outline):

    * every CHECKED result matches and at least one was checked -> ``("accepted_with_warning", warning)``;
    * any result is a ``mismatch`` -> ``("rejected", why)``, naming the first failing check;
    * nothing could be checked (no signature, no PCB found) -> ``("rejected", why)``: an unverifiable changed file is not trusted.
    """
    if pin != "mismatch":
        return "accepted", ""
    rows = [row for row in (_as_check(c) for c in (checks or [])) if row is not None and row[1] != "unchecked"]
    failed = [row for row in rows if not row[0] or row[1] == "mismatch"]
    if failed:
        return "rejected", f"the file differs from the verified copy and its geometry no longer matches: {failed[0][2]}"
    if not rows:
        return "rejected", ("the file differs from the verified copy and nothing could confirm that it is still the same part "
                            "(no bounding-box signature or PCB outline to compare)")
    return "accepted_with_warning", ("the file differs from the verified copy (the vendor changed it) but its bounding box / outline still "
                                     "match; accuracy of the changed details is not verified")
