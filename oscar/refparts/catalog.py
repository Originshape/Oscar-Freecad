"""The vetted reference-part catalog and the board library.

PURE module: stdlib only, no FreeCAD, no network, no Qt. It reads two kinds of
data that ship next to it:

* ``catalog.json``  - what can be sourced: id, aliases, vendor / datasheet pages,
  allowlisted STEP downloads (a URL is listed ONLY after the file was downloaded and
  verified: type, size, sha256, bounding box / outline signature, trust level and
  licence note; see ``step_for`` / ``trust_label`` and the ``step_urls`` schema in
  ``_validate_step_url``), a licence note and whether the entry is verified;
* ``boards/<id>.json`` - published board dimensions (outline, mounting holes,
  connector boxes) in the kit frame, each value with its source and a confidence
  from ``CONFIDENCE_LEVELS``. This is the single data source of the nominal
  fallback and of the enclosure kit.

Public API (the kit, the tools and the tests use only these names):
``list_entries``, ``get``, ``match`` (and ``match_scored``), ``allow_hosts``,
``list_boards``, ``load_board``, ``validate_board``, ``validate_catalog``,
``opening``, ``plug_size``, ``PLUG_DEFAULTS``, ``CONFIDENCE_LEVELS``, ``FACES``,
``STEP_TRUSTS``, ``step_for``, ``trust_of``, ``trust_label``.

Frame: origin = lower-left corner of the PCB on its BOTTOM surface,
+X along the long edge, +Y along the short edge, +Z up through the PCB, millimetres.
"""

import copy
import json
import math
import os
import re
import threading

SCHEMA = 1
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.json")

# measured: taken from an imported STEP by measure_reference_part; datasheet: read from the manufacturer's drawing;
# community: consistent across published designs, not yet verified against the drawing; approximate / assumed: estimates.
CONFIDENCE_LEVELS = ("measured", "datasheet", "community", "approximate", "assumed")
# Trust level of a downloadable STEP file: ``official`` = published by the manufacturer; ``community`` = a third-party model whose
# accuracy nobody guarantees. An entry without ``trust`` is treated as ``community`` (the safe reading).
STEP_TRUSTS = ("official", "community")
HASH_POLICIES = ("reject", "identity")
FACES = ("-x", "+x", "-y", "+y", "top", "bottom")
WALL_FACES = ("-x", "+x", "-y", "+y")
BOARD_KINDS = ("sbc", "mcu", "module", "hardware")
FRAME = {"origin": "pcb_bottom_lower_left", "x": "long_edge", "y": "short_edge", "z": "up_through_pcb", "units": "mm"}

# Plug sizes (width, height in mm) of the CABLE PLUG that must pass an opening, per connector kind. Approximate.
# Kinds that are not listed (rj45, usb_a_stack, microsd, ...) have receptacles that already contain their plug.
PLUG_DEFAULTS = {"usb_c": (12.0, 6.5), "micro_hdmi": (11.0, 7.0), "audio_jack": (9.0, 9.0)}

MAX_PROTRUSION = 6.0        # validator rule 4: more than this past the PCB edge is almost certainly a wrong sign
MIN_REACH = 2.0             # a wall connector's body must reach at least this far into the board
_EPS = 1e-9

_ID_RE = re.compile(r"^[a-z0-9_]+$")
_REQUIRED_BOARD_KEYS = ("schema", "id", "name", "aliases", "kind", "frame", "outline", "pcb", "mounting_holes",
                        "connectors", "height", "omitted", "verify_against", "verified")
_REQUIRED_CONNECTOR_KEYS = ("id", "name", "kind", "face", "box", "cutout", "margin_hint", "source", "confidence")

_LOCK = threading.Lock()
_CATALOG: dict | None = None
_BOARDS: dict[str, dict] = {}


# --- small helpers ----------------------------------------------------------

def _num(value) -> float | None:
    """A finite float for an int / float (not bool); None otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _fmt(value) -> str:
    """Number as written in messages: 17.0, 17.5, -1.5."""
    return repr(round(float(value), 4))


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"catalog data {os.path.basename(path)!r} cannot be read: {exc}; reinstall the Oscar module") from exc
    if not isinstance(data, dict):
        raise ValueError(f"catalog data {os.path.basename(path)!r} is not a JSON object; reinstall the Oscar module")
    return data


def _catalog() -> dict:
    global _CATALOG
    with _LOCK:
        if _CATALOG is None:
            _CATALOG = _read_json(CATALOG_PATH)
        return _CATALOG


def _reset_cache() -> None:
    """Forget everything read from disk (tests that patch the data files)."""
    global _CATALOG
    with _LOCK:
        _CATALOG = None
        _BOARDS.clear()


# --- catalog entries --------------------------------------------------------

def list_entries() -> list[dict]:
    """Every catalog entry, in catalog order, as fresh copies (callers may mutate them)."""
    return copy.deepcopy(list(_catalog().get("entries") or []))


def allow_hosts() -> list[str]:
    """Hosts a reference-part download may come from (the first URL and the final URL after redirects), lower case."""
    return [str(h).lower() for h in (_catalog().get("allow_hosts") or [])]


def get(entry_id: str) -> dict:
    """The catalog entry with this id (case-insensitive) as a copy. Raises ``KeyError`` listing the known ids when there is none."""
    key = str(entry_id or "").strip().lower()
    for entry in _catalog().get("entries") or []:
        if entry.get("id") == key:
            return copy.deepcopy(entry)
    raise KeyError(f"unknown reference part {entry_id!r}; known ids: {', '.join(e.get('id', '?') for e in _catalog().get('entries') or [])}")


# --- STEP sources: trust and hash policy -------------------------------------

def trust_of(step) -> str:
    """The trust level of a ``step_urls`` item (``official`` | ``community``); anything else, or no field, reads as ``community``."""
    value = step.get("trust") if isinstance(step, dict) else None
    return value if value in STEP_TRUSTS else "community"


def trust_label(trust: str, *, changed: bool = False) -> str:
    """The words that go after ``step-imported`` / ``step-measured`` so a Build Summary never overstates a model's reliability.

    ``community`` -> ``"community, unverified accuracy"``; ``official`` -> ``"official, vendor-published"``. ``changed`` (the file's
    hash differs from the verified copy but its outline still matched) appends ``", file changed since verified"``. An unknown
    level reads as ``community``. Pure text, no I/O.
    """
    base = "official, vendor-published" if trust == "official" else "community, unverified accuracy"
    return base + (", file changed since verified" if changed else "")


def step_for(entry: dict, *, url: str = "", sha256: str = "") -> dict:
    """The ``step_urls`` item of ``entry`` that ``url`` or ``sha256`` (archive or member hash) names, else the first one, else ``{}``.

    Returns the item itself (not a copy: ``entry`` is already a copy from ``get`` / ``list_entries``)."""
    steps = [s for s in (entry or {}).get("step_urls") or [] if isinstance(s, dict)]
    link, digest = str(url or "").strip(), str(sha256 or "").strip().lower()
    for step in steps:
        if link and step.get("url") == link:
            return step
    for step in steps:
        if digest and digest in (str(step.get("expected_sha256", "")).lower(), str(step.get("member_sha256", "")).lower()):
            return step
    return steps[0] if steps else {}


# --- matching ---------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+")


def _tokens(text) -> list[str]:
    """Lower-case words, with letters and digits split apart (``rpi4b`` -> rpi 4 b) and ``+`` read as the word plus."""
    return _TOKEN_RE.findall(str(text or "").lower().replace("+", " plus "))


def _contains(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def _score_phrase(query: list[str], phrase: list[str]) -> int:
    if not query or not phrase:
        return 0
    if phrase == query:
        return 1000 + len(phrase)
    if _contains(query, phrase):                       # an alias inside a longer request
        return 500 + 10 * len(phrase)
    if sum(len(t) for t in query) >= 3 and _contains(phrase, query):   # the query is a fragment of a name / alias
        return 300 + 10 * len(query)
    return 0


def match_scored(text: str) -> list[tuple[int, dict]]:
    """Catalog entries that match ``text``, best first, as ``(score, entry)`` pairs.

    Deterministic: the text is split into words (letters and digits apart, ``+`` read as "plus"), every alias / name / id of
    an entry is compared as a whole phrase, and the best phrase decides: equal (1000+) > an alias inside the text (500+,
    longer aliases win) > the text is a fragment of an alias (300+). Ties keep catalog order. Words never match partially
    (``pi 400`` does not match ``pi 4``), and an empty or unrelated text matches nothing.
    """
    query = _tokens(text)
    scored: list[tuple[int, int, dict]] = []
    for index, entry in enumerate(_catalog().get("entries") or []):
        phrases = [entry.get("id", ""), entry.get("name", "")] + list(entry.get("aliases") or [])
        best = max((_score_phrase(query, _tokens(p)) for p in phrases), default=0)
        if best > 0:
            scored.append((best, index, entry))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(score, copy.deepcopy(entry)) for score, _i, entry in scored]


def match(text: str, limit: int | None = None) -> list[dict]:
    """Entries matching ``text`` (a part name, or a whole request that mentions one), best first; see ``match_scored``.

    ``limit`` caps the list. Returns copies of the catalog entries; an empty list means "not in the catalog".
    """
    found = [entry for _score, entry in match_scored(text)]
    return found[:limit] if limit else found


# --- boards -----------------------------------------------------------------

class _Connector(dict):
    """A connector dict that remembers the outline of the board it came from (``board_length`` / ``board_width``).

    ``opening`` needs them for the protrusion on the ``+x`` / ``+y`` faces. They are plain attributes of a dict subclass:
    equality with an ordinary dict, ``json.dumps`` and ``copy.deepcopy`` behave exactly as for a dict.
    """


def _board_path(entry: dict) -> str:
    rel = str(entry.get("board") or "")
    return os.path.normpath(os.path.join(DATA_DIR, rel))


def list_boards() -> list[dict]:
    """The board library as ``{"id", "name", "aliases"}`` dicts, in catalog order (entries that carry board data)."""
    return [{"id": e["id"], "name": e.get("name", ""), "aliases": list(e.get("aliases") or [])}
            for e in _catalog().get("entries") or [] if e.get("board")]


def load_board(board_id: str) -> dict:
    """The board file for ``board_id`` (case-insensitive) as a fresh deep copy. Raises ``KeyError`` listing the known ids.

    Raises ``ValueError`` when the file is unreadable or its ``id`` is not the id it is filed under. The result is NOT
    validated (``validate_board`` does that); its ``connectors`` are dicts that ``opening`` can resolve on their own.
    """
    key = str(board_id or "").strip().lower()
    entries = [e for e in _catalog().get("entries") or [] if e.get("board")]
    entry = next((e for e in entries if e.get("id") == key), None)
    if entry is None:
        raise KeyError(f"unknown board {board_id!r}; known boards: {', '.join(e['id'] for e in entries)}")
    with _LOCK:
        raw = _BOARDS.get(key)
    if raw is None:
        raw = _read_json(_board_path(entry))
        if raw.get("id") != key:
            raise ValueError(f"board file {entry['board']!r} has id {raw.get('id')!r}, expected {key!r}; fix the file")
        with _LOCK:
            _BOARDS[key] = raw
    board = copy.deepcopy(raw)
    outline = board.get("outline") if isinstance(board.get("outline"), dict) else {}
    length, width = _num(outline.get("length")), _num(outline.get("width"))
    wrapped = []
    for conn in board.get("connectors") or []:
        if isinstance(conn, dict):
            conn = _Connector(conn)
            conn.board_length, conn.board_width = length, width
        wrapped.append(conn)
    board["connectors"] = wrapped
    return board


# --- derived openings -------------------------------------------------------

def plug_size(connector: dict) -> tuple[float, float]:
    """(width, height) in mm of the cable plug that must pass this connector's opening; (0, 0) when the receptacle holds its plug.

    A connector's own ``plug_w`` / ``plug_h`` win; otherwise ``PLUG_DEFAULTS[kind]``.
    """
    default_w, default_h = PLUG_DEFAULTS.get(str(connector.get("kind", "")), (0.0, 0.0))
    pw, ph = _num(connector.get("plug_w")), _num(connector.get("plug_h"))
    return (default_w if pw is None else pw, default_h if ph is None else ph)


def _parse_box(box) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    """((x0, x1), (y0, y1), (z0, z1)) from a connector box dict, or None when it is malformed."""
    if not isinstance(box, dict):
        return None
    out = []
    for axis in ("x", "y", "z"):
        pair = box.get(axis)
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        lo, hi = _num(pair[0]), _num(pair[1])
        if lo is None or hi is None:
            return None
        out.append((lo, hi))
    return out[0], out[1], out[2]


def _protrusion(face: str, box, length, width) -> float | None:
    """How far the box reaches past the PCB edge on its own wall face; None when ``length`` / ``width`` is needed and unknown."""
    (x0, x1), (y0, y1), _z = box
    if face == "-y":
        return max(0.0, -y0)
    if face == "-x":
        return max(0.0, -x0)
    if face == "+y":
        return None if width is None else max(0.0, y1 - width)
    if face == "+x":
        return None if length is None else max(0.0, x1 - length)
    return 0.0


def opening(connector: dict, margin: float | None = None, plug: bool = True, *, board: dict | None = None) -> dict:
    """The wall / lid / floor opening of a connector: ``{"face", "u", "v", "w", "h", "protrusion"}`` (7.1, frame 7.2).

    ``margin=None`` returns the RECEPTACLE geometry: for wall faces ``u`` = centre along the wall (x for ``+-y``, y for
    ``+-x``), ``v`` = centre height ``(z0+z1)/2``, ``w`` = extent along ``u``, ``h`` = ``z1 - z0``; for ``top`` / ``bottom``
    ``u`` / ``v`` are the x / y centres and ``w`` / ``h`` the x / y extents. ``protrusion`` is how far the box reaches past the
    PCB edge on its face (0 for top / bottom).
    With a ``margin`` (mm, >= 0) it returns what the kit cuts: ``w = max(w + 2*margin, plug_w)`` and the same for ``h``,
    where the plug size comes from ``plug_size`` (ignored with ``plug=False``).

    The ``+x`` / ``+y`` protrusion needs the board's length / width: connectors from ``load_board`` carry them; for any other
    connector dict pass ``board=`` (a board dict with an ``outline``). Raises ``ValueError`` for a malformed connector or when
    the board size is needed and unknown. Results are rounded to 1e-6.
    """
    cid = connector.get("id", "?") if isinstance(connector, dict) else "?"
    if not isinstance(connector, dict):
        raise ValueError(f"opening: expected a connector dict, got {type(connector).__name__}; pass an entry of board['connectors']")
    face = connector.get("face")
    if face not in FACES:
        raise ValueError(f"opening: connector {cid!r} has face {face!r}; expected one of {', '.join(FACES)}")
    box = _parse_box(connector.get("box"))
    if box is None:
        raise ValueError(f"opening: connector {cid!r} has no valid box; expected {{'x': [x0, x1], 'y': [y0, y1], 'z': [z0, z1]}} in mm")
    (x0, x1), (y0, y1), (z0, z1) = box
    if face in ("-y", "+y"):
        u, w = (x0 + x1) / 2.0, x1 - x0
        v, h = (z0 + z1) / 2.0, z1 - z0
    elif face in ("-x", "+x"):
        u, w = (y0 + y1) / 2.0, y1 - y0
        v, h = (z0 + z1) / 2.0, z1 - z0
    else:
        u, w = (x0 + x1) / 2.0, x1 - x0
        v, h = (y0 + y1) / 2.0, y1 - y0
    outline = (board or {}).get("outline") if isinstance(board, dict) else None
    length = _num(outline.get("length")) if isinstance(outline, dict) else getattr(connector, "board_length", None)
    width = _num(outline.get("width")) if isinstance(outline, dict) else getattr(connector, "board_width", None)
    protrusion = _protrusion(face, box, length, width)
    if protrusion is None:
        raise ValueError(f"opening: connector {cid!r} is on face {face!r}; its protrusion needs the board size, "
                         "pass board=<the board dict> (connectors returned by load_board carry it)")
    if margin is not None:
        m = _num(margin)
        if m is None or m < 0:
            raise ValueError(f"opening: margin {margin!r} is not a number >= 0; use for example 0.4")
        pw, ph = plug_size(connector) if plug else (0.0, 0.0)
        w, h = max(w + 2 * m, pw), max(h + 2 * m, ph)
    return {"face": face, "u": round(u, 6), "v": round(v, 6), "w": round(w, 6), "h": round(h, 6),
            "protrusion": round(protrusion, 6)}


# --- the board validator ----------------------------------------------------

def _provenance(path: str, node: dict, errors: list[str]) -> None:
    """Rule 1 for one node: non-empty source, a known confidence, a source_url for datasheet / measured values."""
    source = node.get("source")
    if not isinstance(source, str) or not source.strip():
        errors.append(f"{path}: 'source' is missing or empty; say where the numbers come from")
    confidence = node.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        errors.append(f"{path}: confidence {confidence!r} is not one of {', '.join(CONFIDENCE_LEVELS)}")
    elif confidence in ("datasheet", "measured"):
        url = node.get("source_url")
        if not isinstance(url, str) or not url.strip():
            errors.append(f"{path}: confidence '{confidence}' requires a 'source_url'; add the drawing's URL or lower the confidence")


def _pattern_points(pattern) -> list[tuple[float, float]] | None:
    """The hole centres a ``{"offset": [x, y], "dx", "dy"}`` pattern expands to (unique, row by row), or None when malformed."""
    if not isinstance(pattern, dict):
        return None
    offset = pattern.get("offset")
    if not isinstance(offset, (list, tuple)) or len(offset) != 2:
        return None
    ox, oy, dx, dy = _num(offset[0]), _num(offset[1]), _num(pattern.get("dx")), _num(pattern.get("dy"))
    if None in (ox, oy, dx, dy):
        return None
    points: list[tuple[float, float]] = []
    for j in (0, 1):
        for i in (0, 1):
            p = (ox + i * dx, oy + j * dy)
            if not any(abs(p[0] - q[0]) <= 0.01 and abs(p[1] - q[1]) <= 0.01 for q in points):
                points.append(p)
    return points


def _circle_hits_rect(cx: float, cy: float, r: float, xs: tuple[float, float], ys: tuple[float, float]) -> bool:
    dx = max(xs[0] - cx, 0.0, cx - xs[1])
    dy = max(ys[0] - cy, 0.0, cy - ys[1])
    return math.hypot(dx, dy) < r - _EPS


def validate_board(board: dict) -> tuple[bool, list[str], list[str]]:
    """Check a board dict against schema 1 and its rules; returns ``(ok, errors, warnings)``. Never raises.

    ``ok`` is True when there is no error. Messages name the connector id and the numbers and follow the
    ``"<what failed>: <why>; <what to do>"`` style. Rules: required keys, confidence vocabulary, non-empty sources, a
    ``source_url`` for datasheet / measured values, unique connector ids; ``length >= width > 0``, corner radius within
    ``0..width/2``, PCB thickness 0.4..3.2; at least 2 mounting holes inside the outline by the pad radius and equal to the
    pattern expansion (0.01); every box has ``lo < hi`` on each axis, wall boxes inside the outline along the wall, at least
    2 mm into the board and at most 6 mm past the edge, top / bottom boxes inside the outline; no two boxes overlap in all
    three axes; a connector appears once, with one face; ``height`` equals the max top / min bottom of the boxes (0.01);
    no mounting-hole pad overlaps a box in plan view. Warnings: same-wall connectors closer than 0.4 mm along the wall,
    a box nearer the opposite edge than its face, a wall connector that protrudes without a cutout, ``verified`` set while
    some value is not datasheet / measured, overlapping hole pads. The optional per-connector ``plug_w`` / ``plug_h`` must be
    positive numbers.
    """
    errors: list[str] = []
    warnings: list[str] = []
    try:
        _validate(board, errors, warnings)
    except Exception as exc:  # a validator must never take a tool down; report it as a finding
        errors.append(f"board: the validator failed unexpectedly ({type(exc).__name__}: {exc}); check the board file structure")
    return (not errors), errors, warnings


def _validate(board, errors: list[str], warnings: list[str]) -> None:
    if not isinstance(board, dict):
        errors.append(f"board: expected a JSON object, got {type(board).__name__}; see refparts/boards/rpi4b.json for the schema")
        return
    for key in _REQUIRED_BOARD_KEYS:
        if key not in board:
            errors.append(f"board: missing required key '{key}'; add it (schema {SCHEMA})")
        elif board[key] is None:
            errors.append(f"board: required key '{key}' is null; give it a value (schema {SCHEMA})")
    if "schema" in board and board["schema"] != SCHEMA:
        errors.append(f"schema: {board['schema']!r} is not supported; expected {SCHEMA}")
    bid = board.get("id")
    if "id" in board and not (isinstance(bid, str) and _ID_RE.match(bid)):
        errors.append(f"id: {bid!r} is not a valid board id; use lower-case letters, digits and underscores")
    if "name" in board and not (isinstance(board["name"], str) and board["name"].strip()):
        errors.append("name: must be a non-empty string")
    if "aliases" in board and not (isinstance(board["aliases"], list)
                                   and all(isinstance(a, str) and a.strip() for a in board["aliases"])):
        errors.append("aliases: must be a list of non-empty strings")
    if "kind" in board and board["kind"] not in BOARD_KINDS:
        errors.append(f"kind: {board['kind']!r} is not one of {', '.join(BOARD_KINDS)}")
    if "frame" in board and board["frame"] != FRAME:
        errors.append(f"frame: {board['frame']!r} is not the kit frame {FRAME!r}; the frame is a constant")
    if "omitted" in board and not (isinstance(board["omitted"], list) and all(isinstance(o, str) for o in board["omitted"])):
        errors.append("omitted: must be a list of strings (what the file leaves out)")
    if "verify_against" in board and not (isinstance(board["verify_against"], str) and board["verify_against"].strip()):
        errors.append("verify_against: must be the URL of the drawing to verify against")
    if "verified" in board and not isinstance(board["verified"], bool):
        errors.append("verified: must be true or false")
    cross = board.get("step_cross_check")
    if cross is not None:
        if not (isinstance(cross, dict) and all(isinstance(cross.get(k), str) and cross[k].strip() for k in ("source", "result", "date"))
                and cross.get("trust") in STEP_TRUSTS and _DATE_RE.match(cross.get("date", ""))
                and isinstance(cross.get("discrepancies", []), list) and all(isinstance(d, str) for d in cross.get("discrepancies", []))):
            errors.append("step_cross_check: must be {'source', 'trust' (official|community), 'date' (YYYY-MM-DD), 'result', "
                          "'discrepancies': [str]}; it records a comparison with a STEP file and never overrides the values above")

    length, width = _check_outline(board.get("outline"), errors)
    _check_pcb(board.get("pcb"), errors)
    pad_r, holes = _check_holes(board.get("mounting_holes"), length, width, errors, warnings)
    boxes = _check_connectors(board.get("connectors"), length, width, errors, warnings)
    _check_height(board.get("height"), boxes, errors)
    if holes and pad_r is not None:
        for cid, face, box in boxes:
            (x0, x1), (y0, y1), _z = box
            for n, (hx, hy) in enumerate(holes, start=1):
                if _circle_hits_rect(hx, hy, pad_r, (x0, x1), (y0, y1)):
                    errors.append(f"connector '{cid}': overlaps the pad of mounting hole {n} at ({_fmt(hx)}, {_fmt(hy)}) "
                                  f"(pad radius {_fmt(pad_r)} mm) in plan view; move the connector or shrink pad_diameter")
    if board.get("verified") is True:
        weak = [n for n, node in _confidence_nodes(board) if node.get("confidence") not in ("datasheet", "measured")]
        if weak:
            warnings.append(f"verified: true but {', '.join(weak[:6])} {'is' if len(weak) == 1 else 'are'} not datasheet or measured; "
                            "lower 'verified' or raise those confidences with evidence")


def _confidence_nodes(board: dict):
    for key in ("outline", "pcb", "mounting_holes"):
        node = board.get(key)
        if isinstance(node, dict):
            yield key, node
    for conn in board.get("connectors") or []:
        if isinstance(conn, dict):
            yield f"connector '{conn.get('id', '?')}'", conn


def _check_outline(outline, errors: list[str]) -> tuple[float | None, float | None]:
    if not isinstance(outline, dict):
        if outline is not None:
            errors.append("outline: must be an object with shape, length, width, corner_radius, source, confidence")
        return None, None
    if outline.get("shape") != "rounded_rect":
        errors.append(f"outline: shape {outline.get('shape')!r} is not supported; use 'rounded_rect'")
    length, width, radius = _num(outline.get("length")), _num(outline.get("width")), _num(outline.get("corner_radius"))
    if length is None or width is None:
        errors.append("outline: length and width must be numbers (mm)")
        length = width = None
    elif not (length >= width > 0):
        errors.append(f"outline: length {_fmt(length)} and width {_fmt(width)} must satisfy length >= width > 0; "
                      "the long edge runs along +x")
    if radius is None:
        errors.append("outline: corner_radius must be a number (mm)")
    elif width is not None and not (0 <= radius <= width / 2 + _EPS):
        errors.append(f"outline: corner_radius {_fmt(radius)} is outside 0..{_fmt(width / 2)} (half the width); use a smaller radius")
    _provenance("outline", outline, errors)
    return length, width


def _check_pcb(pcb, errors: list[str]) -> None:
    if not isinstance(pcb, dict):
        if pcb is not None:
            errors.append("pcb: must be an object with thickness, source, confidence")
        return
    thickness = _num(pcb.get("thickness"))
    if thickness is None or not (0.4 <= thickness <= 3.2):
        errors.append(f"pcb: thickness {pcb.get('thickness')!r} is outside 0.4..3.2 mm; typical boards are 1.0 to 1.6")
    _provenance("pcb", pcb, errors)


def _check_holes(holes, length, width, errors: list[str], warnings: list[str]):
    """Rule 3. Returns (pad_radius, [(x, y), ...]) for the plan-view overlap check (None / [] when unusable)."""
    if not isinstance(holes, dict):
        if holes is not None:
            errors.append("mounting_holes: must be an object with diameter, pad_diameter, thread, positions, source, confidence")
        return None, []
    diameter, pad = _num(holes.get("diameter")), _num(holes.get("pad_diameter"))
    if diameter is None or diameter <= 0:
        errors.append(f"mounting_holes: diameter {holes.get('diameter')!r} must be a positive number (mm)")
    if pad is None or pad <= 0:
        errors.append(f"mounting_holes: pad_diameter {holes.get('pad_diameter')!r} must be a positive number (mm)")
    elif diameter is not None and pad < diameter:
        errors.append(f"mounting_holes: pad_diameter {_fmt(pad)} is smaller than the hole diameter {_fmt(diameter)}; "
                      "the pad is the keep-out ring around the hole")
    if not isinstance(holes.get("thread"), str):
        errors.append("mounting_holes: thread must be a string (for example 'M2.5')")
    _provenance("mounting_holes", holes, errors)
    raw = holes.get("positions")
    points: list[tuple[float, float]] = []
    if not isinstance(raw, list) or not all(isinstance(p, (list, tuple)) and len(p) == 2 and None not in (_num(p[0]), _num(p[1]))
                                            for p in raw):
        errors.append("mounting_holes: positions must be a list of [x, y] numbers (mm, board frame)")
        raw = []
    else:
        points = [(float(p[0]), float(p[1])) for p in raw]
        if len(points) < 2:
            errors.append(f"mounting_holes: {len(points)} hole(s) listed; at least 2 are needed to hold a board")
    pad_r = pad / 2.0 if pad is not None and pad > 0 else None
    for n, (hx, hy) in enumerate(points, start=1):
        if pad_r is not None and length is not None and width is not None and not (
                pad_r - _EPS <= hx <= length - pad_r + _EPS and pad_r - _EPS <= hy <= width - pad_r + _EPS):
            errors.append(f"mounting_holes: hole {n} at ({_fmt(hx)}, {_fmt(hy)}) is closer than the pad radius {_fmt(pad_r)} mm "
                          f"to the board edge (outline 0..{_fmt(length)} x 0..{_fmt(width)}); check the position")
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]) < 1e-6:
                errors.append(f"mounting_holes: holes {i + 1} and {j + 1} are at the same position ({_fmt(points[i][0])}, {_fmt(points[i][1])})")
            elif pad is not None and pad > 0 and math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]) < pad - _EPS:
                warnings.append(f"mounting_holes: the pads of holes {i + 1} and {j + 1} overlap; pad_diameter {_fmt(pad)} is larger than their spacing")
    pattern = holes.get("pattern")
    if pattern is not None:
        expected = _pattern_points(pattern)
        if expected is None:
            errors.append("mounting_holes: pattern must be {'offset': [x, y], 'dx': mm, 'dy': mm}")
        elif points:
            unmatched = [p for p in expected if not any(abs(p[0] - q[0]) <= 0.01 and abs(p[1] - q[1]) <= 0.01 for q in points)]
            extra = [q for q in points if not any(abs(p[0] - q[0]) <= 0.01 and abs(p[1] - q[1]) <= 0.01 for p in expected)]
            if unmatched or extra or len(points) != len(expected):
                errors.append("mounting_holes: positions do not equal the pattern expansion "
                              f"{[[round(a, 4), round(b, 4)] for a, b in expected]}; fix the positions or the pattern (tolerance 0.01 mm)")
    return pad_r, points


def _check_connectors(connectors, length, width, errors: list[str], warnings: list[str]):
    """Rules 1, 4-7 for the connector list. Returns [(id, face, box), ...] of the usable connectors."""
    if not isinstance(connectors, list) or not connectors:
        if connectors is not None:
            errors.append("connectors: must be a non-empty list; list at least the connectors on the walls and the tallest part")
        return []
    seen_ids: dict[str, int] = {}
    seen_names: dict[str, tuple[str, str]] = {}
    usable = []
    for index, conn in enumerate(connectors):
        if not isinstance(conn, dict):
            errors.append(f"connectors[{index}]: must be an object")
            continue
        cid = conn.get("id")
        label = cid if isinstance(cid, str) else f"#{index}"
        for key in _REQUIRED_CONNECTOR_KEYS:
            if key not in conn:
                errors.append(f"connector '{label}': missing required key '{key}'")
        if not (isinstance(cid, str) and _ID_RE.match(cid)):
            errors.append(f"connector '{label}': id must match [a-z0-9_]+")
        elif cid in seen_ids:
            errors.append(f"connector '{cid}': id is used twice (entries {seen_ids[cid]} and {index}); ids must be unique")
        else:
            seen_ids[cid] = index
        name = conn.get("name")
        if not (isinstance(name, str) and name.strip()):
            errors.append(f"connector '{label}': name must be a non-empty string")
        elif name.strip().lower() in seen_names:
            first_id, first_face = seen_names[name.strip().lower()]
            errors.append(f"connector '{label}': name {name!r} is already used by '{first_id}' (face '{first_face}'); "
                          "a connector appears once, with exactly one face")
        else:
            seen_names[name.strip().lower()] = (label, str(conn.get("face")))
        if not (isinstance(conn.get("kind"), str) and conn["kind"].strip()):
            errors.append(f"connector '{label}': kind must be a non-empty string")
        if "cutout" in conn and not isinstance(conn["cutout"], bool):
            errors.append(f"connector '{label}': cutout must be true or false")
        hint = conn.get("margin_hint")
        if "margin_hint" in conn and (_num(hint) is None or _num(hint) < 0):
            errors.append(f"connector '{label}': margin_hint {hint!r} must be a number >= 0 (mm)")
        for key in ("plug_w", "plug_h"):
            if key in conn and (_num(conn[key]) is None or _num(conn[key]) <= 0):
                errors.append(f"connector '{label}': {key} {conn[key]!r} must be a positive number (mm)")
        if "note" in conn and not isinstance(conn["note"], str):
            errors.append(f"connector '{label}': note must be a string")
        _provenance(f"connector '{label}'", conn, errors)
        face = conn.get("face")
        if face not in FACES:
            errors.append(f"connector '{label}': face {face!r} is not one of {', '.join(FACES)}; "
                          "a cut-out needs a wall face (-x, +x, -y, +y) or top / bottom for a lid / floor opening")
            continue
        box = _parse_box(conn.get("box"))
        if box is None:
            errors.append(f"connector '{label}': box must be {{'x': [x0, x1], 'y': [y0, y1], 'z': [z0, z1]}} in mm")
            continue
        flat = False
        for axis, (lo, hi) in zip("xyz", box):
            if not lo < hi:
                errors.append(f"connector '{label}': box {axis} {_fmt(lo)}..{_fmt(hi)} is empty; {axis}0 must be below {axis}1")
                flat = True
        if flat:
            continue
        usable.append((label, face, box))
        _check_box_placement(label, face, box, conn, length, width, errors, warnings)
    _check_overlaps(usable, errors, warnings)
    return usable


def _check_box_placement(cid, face, box, conn, length, width, errors: list[str], warnings: list[str]) -> None:
    (x0, x1), (y0, y1), _z = box
    if length is None or width is None:
        return
    if face in WALL_FACES:
        lo, hi, limit, axis = ((x0, x1, length, "x") if face in ("-y", "+y") else (y0, y1, width, "y"))
        if lo < -_EPS or hi > limit + _EPS:
            errors.append(f"connector '{cid}': box {axis} {_fmt(lo)}..{_fmt(hi)} leaves the outline 0..{_fmt(limit)} along the '{face}' wall; "
                          "move it onto the board edge")
        reach = {"-y": y1, "-x": x1, "+y": width - y0, "+x": length - x0}[face]
        if reach < MIN_REACH - _EPS:
            errors.append(f"connector '{cid}': box reaches only {_fmt(reach)} mm into the board from face '{face}'; "
                          f"a wall connector must reach at least {_fmt(MIN_REACH)} mm past the PCB edge")
        protrusion = _protrusion(face, box, length, width) or 0.0
        if protrusion > MAX_PROTRUSION + _EPS:
            errors.append(f"connector '{cid}': protrusion {_fmt(protrusion)} mm on face '{face}'; probably a wrong sign")
        centre = {"-y": (y0 + y1) / 2 - width / 2, "+y": width / 2 - (y0 + y1) / 2,
                  "-x": (x0 + x1) / 2 - length / 2, "+x": length / 2 - (x0 + x1) / 2}[face]
        if centre > _EPS:
            warnings.append(f"connector '{cid}': its box lies nearer the opposite edge than its face '{face}'; check the face")
        if protrusion > 0.5 and conn.get("cutout") is False:
            warnings.append(f"connector '{cid}': protrudes {_fmt(protrusion)} mm past the '{face}' edge but has no cutout; "
                            "the wall would collide with it")
    else:
        if x0 < -_EPS or x1 > length + _EPS or y0 < -_EPS or y1 > width + _EPS:
            errors.append(f"connector '{cid}': a '{face}' box must lie inside the outline (0..{_fmt(length)} x 0..{_fmt(width)}), "
                          f"got x {_fmt(x0)}..{_fmt(x1)}, y {_fmt(y0)}..{_fmt(y1)}")


def _check_overlaps(usable, errors: list[str], warnings: list[str]) -> None:
    """Rule 5: no two boxes overlap in all three axes; same-wall connectors within 0.4 mm along the wall warn."""
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            (a_id, a_face, a), (b_id, b_face, b) = usable[i], usable[j]
            spans = [min(a[k][1], b[k][1]) - max(a[k][0], b[k][0]) for k in range(3)]
            if all(s > _EPS for s in spans):
                errors.append(f"connectors '{a_id}' and '{b_id}' overlap: x {_fmt(a[0][0])}..{_fmt(a[0][1])} / {_fmt(b[0][0])}..{_fmt(b[0][1])}, "
                              f"y {_fmt(a[1][0])}..{_fmt(a[1][1])} / {_fmt(b[1][0])}..{_fmt(b[1][1])}, "
                              f"z {_fmt(a[2][0])}..{_fmt(a[2][1])} / {_fmt(b[2][0])}..{_fmt(b[2][1])}; move or resize one of them")
            if a_face == b_face and a_face in WALL_FACES:
                k = 0 if a_face in ("-y", "+y") else 1
                gap = max(a[k][0], b[k][0]) - min(a[k][1], b[k][1])      # negative when the u-ranges overlap
                if gap < 0.4 - _EPS:                                       # each range widened by 0.2 mm would overlap
                    warnings.append(f"connectors '{a_id}' and '{b_id}' are on the same '{a_face}' wall and closer than 0.4 mm along it; "
                                    "their openings would merge")


def _check_height(height, usable, errors: list[str]) -> None:
    """Rule 8: height.max_top_z / min_bottom_z equal the max z1 / min z0 over the connector boxes (0.01)."""
    if not isinstance(height, dict):
        if height is not None:
            errors.append("height: must be {'max_top_z': mm, 'min_bottom_z': mm}")
        return
    top, bottom = _num(height.get("max_top_z")), _num(height.get("min_bottom_z"))
    if top is None or bottom is None:
        errors.append("height: max_top_z and min_bottom_z must be numbers (mm)")
        return
    if not usable:
        return
    max_top = max(b[2][1] for _c, _f, b in usable)
    min_bottom = min(b[2][0] for _c, _f, b in usable)
    if abs(top - max_top) > 0.01:
        errors.append(f"height.max_top_z {_fmt(top)} != {_fmt(max_top)} (max of connector tops)")
    if abs(bottom - min_bottom) > 0.01:
        errors.append(f"height.min_bottom_z {_fmt(bottom)} != {_fmt(min_bottom)} (min of connector bottoms)")


# --- the catalog validator --------------------------------------------------

_HOST_RE = re.compile(r"^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _https_host(url) -> str | None:
    """Lower-case host of an https URL; None when it is not one."""
    if not isinstance(url, str):
        return None
    m = re.match(r"^https://([^/?#:@]+)(?::\d+)?(?:[/?#]|$)", url.strip(), re.IGNORECASE)
    return m.group(1).lower() if m else None


def validate_catalog(catalog: dict | None = None) -> tuple[bool, list[str], list[str]]:
    """Check ``catalog.json`` (or a given dict): ids, aliases, board files, URLs and the allowlist; returns ``(ok, errors, warnings)``.

    Errors: wrong schema; an id that is not ``[a-z0-9_]+`` or repeats; an alias shared by two entries (a match would be
    ambiguous); an unknown kind; a ``board`` that is missing, outside ``boards/`` or whose file has another id; a non-https
    vendor / datasheet / step URL; a step URL whose host is not in ``allow_hosts``; a malformed ``expected_sha256`` /
    ``expected_bbox``. Warnings: a step URL with no pinned hash, ``verified`` without ``last_verified``. Never raises.
    """
    errors: list[str] = []
    warnings: list[str] = []
    try:
        _validate_catalog(catalog if catalog is not None else _catalog(), errors, warnings)
    except Exception as exc:
        errors.append(f"catalog: the validator failed unexpectedly ({type(exc).__name__}: {exc})")
    return (not errors), errors, warnings


def _validate_catalog(cat, errors: list[str], warnings: list[str]) -> None:
    if not isinstance(cat, dict):
        errors.append("catalog: expected a JSON object")
        return
    if cat.get("schema") != SCHEMA:
        errors.append(f"catalog: schema {cat.get('schema')!r} is not supported; expected {SCHEMA}")
    hosts = cat.get("allow_hosts")
    if not isinstance(hosts, list) or not all(isinstance(h, str) and _HOST_RE.match(h) and h == h.lower() for h in hosts):
        errors.append("allow_hosts: must be a list of lower-case host names without scheme, port or path")
        hosts = []
    entries = cat.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("entries: must be a non-empty list")
        return
    ids: set[str] = set()
    alias_owner: dict[tuple, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entries[{index}]: must be an object")
            continue
        eid = entry.get("id")
        label = eid if isinstance(eid, str) else f"#{index}"
        if not (isinstance(eid, str) and _ID_RE.match(eid)):
            errors.append(f"entry '{label}': id must match [a-z0-9_]+")
        elif eid in ids:
            errors.append(f"entry '{eid}': id is used twice")
        else:
            ids.add(eid)
        if not (isinstance(entry.get("name"), str) and entry["name"].strip()):
            errors.append(f"entry '{label}': name must be a non-empty string")
        if entry.get("kind") not in BOARD_KINDS:
            errors.append(f"entry '{label}': kind {entry.get('kind')!r} is not one of {', '.join(BOARD_KINDS)}")
        aliases = entry.get("aliases")
        if not (isinstance(aliases, list) and aliases and all(isinstance(a, str) and a.strip() for a in aliases)):
            errors.append(f"entry '{label}': aliases must be a non-empty list of strings")
            aliases = []
        for alias in aliases:
            key = tuple(_tokens(alias))
            if key in alias_owner and alias_owner[key] != label:
                errors.append(f"entry '{label}': alias {alias!r} is also an alias of '{alias_owner[key]}'; a match would be ambiguous")
            alias_owner.setdefault(key, label)
        _validate_entry_board(label, entry, errors)
        for field in ("vendor_urls", "datasheet_urls"):
            urls = entry.get(field)
            if not isinstance(urls, list) or any(_https_host(u) is None for u in urls):
                errors.append(f"entry '{label}': {field} must be a list of https URLs")
        steps = entry.get("step_urls")
        if not isinstance(steps, list):
            errors.append(f"entry '{label}': step_urls must be a list (empty when no verified STEP file is known)")
            steps = []
        for n, step in enumerate(steps):
            _validate_step_url(label, n, step, hosts, errors, warnings)
        if not (isinstance(entry.get("license"), str) and entry["license"].strip()):
            errors.append(f"entry '{label}': license must be a non-empty string")
        last = entry.get("last_verified", "")
        if not (isinstance(last, str) and (last == "" or _DATE_RE.match(last))):
            errors.append(f"entry '{label}': last_verified must be '' or YYYY-MM-DD")
        if not isinstance(entry.get("verified"), bool):
            errors.append(f"entry '{label}': verified must be true or false")
        elif entry["verified"] and not last:
            warnings.append(f"entry '{label}': verified is true but last_verified is empty")


def _validate_entry_board(label: str, entry: dict, errors: list[str]) -> None:
    rel = entry.get("board")
    if rel in (None, ""):
        return
    if not (isinstance(rel, str) and rel.startswith("boards/") and rel.endswith(".json") and ".." not in rel and "\\" not in rel):
        errors.append(f"entry '{label}': board {rel!r} must be a path like 'boards/<id>.json'")
        return
    stem = os.path.splitext(os.path.basename(rel))[0]
    if stem != entry.get("id"):
        errors.append(f"entry '{label}': board file {rel!r} must be named after the entry id '{label}'")
    path = os.path.join(DATA_DIR, rel)
    if not os.path.isfile(path):
        errors.append(f"entry '{label}': board file {rel!r} does not exist")
        return
    try:
        raw = _read_json(path)
    except ValueError as exc:
        errors.append(f"entry '{label}': {exc}")
        return
    if raw.get("id") != entry.get("id"):
        errors.append(f"entry '{label}': board file {rel!r} has id {raw.get('id')!r}")


def _validate_step_url(label: str, n: int, step, hosts: list[str], errors: list[str], warnings: list[str]) -> None:
    where = f"entry '{label}': step_urls[{n}]"
    if not isinstance(step, dict):
        errors.append(f"{where}: must be an object with url, mirror, expected_sha256, expected_bbox, license")
        return
    host = _https_host(step.get("url"))
    if host is None:
        errors.append(f"{where}: url {step.get('url')!r} must be an https URL")
    elif host not in hosts:
        errors.append(f"{where}: host '{host}' is not in allow_hosts ({', '.join(hosts)}); add it deliberately or use an allowed host")
    if not isinstance(step.get("mirror"), bool):
        errors.append(f"{where}: mirror must be true or false")
    sha = step.get("expected_sha256", "")
    if not (isinstance(sha, str) and (sha == "" or _SHA_RE.match(sha))):
        errors.append(f"{where}: expected_sha256 must be '' or 64 lower-case hex characters")
    elif sha == "":
        warnings.append(f"{where}: no expected_sha256 pinned; the file is verified by type and size only")
    bbox = step.get("expected_bbox", [])
    if not (isinstance(bbox, list) and (not bbox or (len(bbox) == 3 and all(_num(v) is not None and _num(v) > 0 for v in bbox)))):
        errors.append(f"{where}: expected_bbox must be [] or three positive numbers [l, w, h] in mm")
    if not (isinstance(step.get("license"), str) and step["license"].strip()):
        errors.append(f"{where}: license must be a non-empty string")
    _validate_step_extras(where, step, errors)


def _validate_step_extras(where: str, step: dict, errors: list[str]) -> None:
    """The optional registry fields of a ``step_urls`` item (a legacy item without them stays valid).

    ``trust`` (``official`` | ``community``), ``size`` (bytes of the download, a positive int), ``member`` (file name inside a ZIP,
    one plain name) with ``member_sha256`` / ``member_bytes`` (the extracted STEP), ``on_hash_mismatch`` (``reject`` = a file whose
    hash differs from the pin is refused, the default; ``identity`` = it is kept with a warning when its bounding box / outline
    still matches, which needs ``expected_bbox``), ``last_verified`` (YYYY-MM-DD), ``note`` (the cross-check result, a string).
    """
    trust = step.get("trust")
    if trust is not None and trust not in STEP_TRUSTS:
        errors.append(f"{where}: trust {trust!r} must be one of {', '.join(STEP_TRUSTS)}")
    for key in ("size", "member_bytes"):
        if key in step and not (isinstance(step[key], int) and not isinstance(step[key], bool) and step[key] > 0):
            errors.append(f"{where}: {key} must be a positive integer (bytes)")
    member = step.get("member")
    if member is not None and not (isinstance(member, str) and member.strip() and "/" not in member and "\\" not in member
                                   and member not in (".", "..")):
        errors.append(f"{where}: member must be one plain file name inside the archive")
    if "member_sha256" in step and not (isinstance(step["member_sha256"], str) and _SHA_RE.match(step["member_sha256"])):
        errors.append(f"{where}: member_sha256 must be 64 lower-case hex characters")
    if ("member_sha256" in step or "member_bytes" in step) and not member:
        errors.append(f"{where}: member_sha256 / member_bytes need a 'member' name")
    policy = step.get("on_hash_mismatch")
    if policy is not None and policy not in HASH_POLICIES:
        errors.append(f"{where}: on_hash_mismatch {policy!r} must be one of {', '.join(HASH_POLICIES)}")
    elif policy == "identity" and not step.get("expected_bbox"):
        errors.append(f"{where}: on_hash_mismatch 'identity' needs expected_bbox (the signature that replaces the hash)")
    last = step.get("last_verified")
    if last is not None and not (isinstance(last, str) and _DATE_RE.match(last)):
        errors.append(f"{where}: last_verified must be YYYY-MM-DD")
    if "note" in step and not isinstance(step["note"], str):
        errors.append(f"{where}: note must be a string")
