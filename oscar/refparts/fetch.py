"""Getting a reference file: catalog entry -> at most ONE allowlisted download -> verification -> import-ready.

This is the logic behind ``source_reference_part``; ``oscar/tools_ref.py`` is the thin tool layer on top
of it (arguments, permission dialog, provenance, result text). PURE of FreeCAD and Qt: the only I/O is the file system,
the ``OSCAR_REFPART_LOCAL_DIR`` mirror and, through ``netguard.download_verified``, the network.

The order of preference for ONE call is fixed:

1. the offline mirror ``$OSCAR_REFPART_LOCAL_DIR/<catalog id>.step`` (test hook and air-gapped installs; no dialog, no
   network, but the file goes through the SAME verification as a download);
2. the URL the caller gave (it must be on the catalog allowlist);
3. the first ``step_urls`` item of the catalog entry (the catalog lists its preferred source first; there is NO second
   try, a second download attempt would be a retry loop, which is ruled out);
4. nothing: the result is ``nominal`` (the catalog's published board data is used, no file).

Pinned hashes and the identity policy. A catalog STEP item pins the download (``expected_sha256``; for a ZIP also its member:
``member`` + ``member_sha256``). A vendor may later replace the file behind a stable URL, which must not strand the user. An item
with ``on_hash_mismatch: "identity"`` (and an ``expected_bbox`` signature) therefore degrades a hash mismatch to a WARNING: the
result is ``obtained`` with ``pin == "mismatch"`` and ``warnings``, and ``import_reference_part`` then imports it ONLY if the
measured bounding box AND board outline still match (``identity.pin_verdict``; otherwise the import is refused and the document is
unchanged). Items without that policy keep the strict rule: a mismatch is ``failed``. Every obtained result also carries the
trust level of its catalog source (``trust``: ``official`` | ``community``; ``trust_label`` is the wording for the Build Summary).

Results are one of ``STATUSES``: ``obtained`` (a verified file), ``declined`` (the user or the caller said no),
``failed`` (the download or the verification failed; the text says why) and ``nominal`` (nothing to download). Every
result says whether board data exists (``has_board``), because ``declined`` / ``failed`` with a board fall back to the
nominal data and without one leave the reference unresolved.

The module also keeps two small in-process registries, both thread-safe: the file ``obtained`` for each reference id (so
``import_reference_part(id=...)`` finds it without a Project) and the per-turn download attempts (so a turn can never
download twice after a failure).
"""

import collections
import hashlib
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Callable
from urllib.parse import urlsplit

from . import catalog, identity

ENV_LOCAL_DIR = "OSCAR_REFPART_LOCAL_DIR"
STATUSES = ("obtained", "declined", "failed", "nominal")
DOWNLOAD_TYPES = (identity.STEP, identity.STL, identity.BREP, identity.IGES, identity.ZIP)   # what a reference download may be
GEOMETRY_TYPES = (identity.STEP, identity.STL, identity.BREP, identity.IGES)                  # what the importer can read
MIRROR_EXTENSIONS = (".step", ".stp", ".stl", ".brep")     # tried in this order for ``<id>`` in the offline mirror
MAX_REFERENCE_BYTES = 150 * 1024 * 1024
MAX_ATTEMPTS_PER_TURN = 3                                  # successful downloads per turn; a failed / declined one ends them
_TURNS_KEPT = 32

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class SourceResult:
    """Outcome of ``fetch``; ``as_dict`` is JSON-safe.

    ``status`` is one of ``STATUSES``. ``path`` is the verified geometry file (``""`` when none). ``origin`` is
    ``"local_mirror"``, ``"download"`` or ``"none"``. ``archive`` is set when the file came out of a ZIP (``path`` of the
    archive, its ``sha256`` and the ``member`` chosen). ``note`` is one or two model-facing sentences (what happened and
    what to do next); ``cached`` is True when the result is the answer of an earlier attempt of the same turn (the tool layer then
    records no second provenance entry: ``provenance`` still names the first one).
    """

    status: str = "failed"
    ref_id: str = ""
    ref_name: str = ""
    has_board: bool = False
    origin: str = "none"
    path: str = ""
    sha256: str = ""
    bytes: int = 0
    detected_type: str = ""
    url: str = ""
    final_url: str = ""
    archive: dict = field(default_factory=dict)
    note: str = ""
    cached: bool = False
    provenance: str = ""         # the provenance entry id the tool layer recorded for this result (set by the caller, kept for a cached answer)
    expected_bbox: list = field(default_factory=list)
    license: str = ""
    trust: str = ""              # trust level of the catalog source of an obtained file ("official" | "community"); "" = not a catalog source
    trust_label: str = ""        # the wording for the Build Summary, e.g. "community, unverified accuracy" (``catalog.trust_label``)
    pin: str = ""                # "match" | "mismatch" | "unpinned" (the file against the catalog's pinned hash) | "" (no catalog source)
    warnings: list = field(default_factory=list)   # model-facing warnings (a pinned hash that differs, ...)

    def as_dict(self) -> dict:
        """The result as a plain dict."""
        return asdict(self)


# --- names and entries --------------------------------------------------------

def slug_id(text: str, default: str = "part") -> str:
    """A reference id (``[a-z0-9_]+``, at most 32 characters) from a file name or title: ``"Pi 4 (v2).step"`` -> ``pi_4_v2``."""
    stem = os.path.splitext(os.path.basename(str(text or "")))[0] if re.search(r"\.[A-Za-z0-9]{2,5}$", str(text or "")) else str(text or "")
    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")[:32].strip("_")
    return slug or default


def ref_object_name(ref_id: str) -> str:
    """The document object name of a reference: ``Ref_<ID upper case>`` (``rpi4b`` -> ``Ref_RPI4B``)."""
    return "Ref_" + re.sub(r"[^A-Za-z0-9_]", "_", str(ref_id or "part")).upper()


_OBJECT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def valid_object_name(name: str) -> bool:
    """True for a name FreeCAD accepts for a document object that this package creates (letters, digits, underscores; 64 at most)."""
    return bool(_OBJECT_NAME_RE.match(str(name or "")))


def resolve_entry(ref_id: str = "", hint: str = "", request: str = "") -> tuple[dict | None, list[dict]]:
    """The catalog entry for ``ref_id`` or, without an id, the part named in ``hint`` / ``request`` (in that order).

    Returns ``(entry, candidates)``: ``entry`` is a catalog entry copy or ``None``; ``candidates`` are the other matches
    (best first, for the "did you mean" list of an error). Raises ``KeyError`` (message names the known ids) when an
    explicit ``ref_id`` is not in the catalog.
    """
    if ref_id and str(ref_id).strip():
        return catalog.get(ref_id), []
    for text in (hint, request):
        found = catalog.match(text or "")
        if found:
            return found[0], found[1:]
    return None, []


def entry_summary(entry: dict) -> dict:
    """What the catalog knows about an entry, for ``find_reference_part``: ids, data kinds and honest reliability.

    ``{"id", "name", "kind", "board": bool, "connectors": int, "confidence": {level: count}, "weakest": level,
    "verified": bool, "step_urls": int, "datasheets": [urls], "vendor": [urls], "license"}``. A board file that cannot be
    read is reported as ``board: False`` (never raises).
    """
    out = {"id": entry.get("id", ""), "name": entry.get("name", ""), "kind": entry.get("kind", ""), "board": False,
           "connectors": 0, "confidence": {}, "weakest": "", "verified": bool(entry.get("verified")),
           "step_urls": len(entry.get("step_urls") or []),
           "step_trust": [catalog.trust_of(st) for st in entry.get("step_urls") or [] if isinstance(st, dict)], "datasheets": list(entry.get("datasheet_urls") or []),
           "vendor": list(entry.get("vendor_urls") or []), "license": entry.get("license", "")}
    if not entry.get("board"):
        return out
    try:
        board = catalog.load_board(entry["id"])
    except (KeyError, ValueError):
        return out
    counts: dict[str, int] = {}
    for node in (board.get("outline"), board.get("pcb"), board.get("mounting_holes"), *(board.get("connectors") or [])):
        level = node.get("confidence") if isinstance(node, dict) else None
        if level:
            counts[level] = counts.get(level, 0) + 1
    order = {level: i for i, level in enumerate(catalog.CONFIDENCE_LEVELS)}
    out.update(board=True, connectors=len(board.get("connectors") or []), confidence=counts,
               weakest=max(counts, key=lambda lv: order.get(lv, 99)) if counts else "",
               verified=bool(entry.get("verified") and board.get("verified")))
    return out


# --- files ---------------------------------------------------------------------

def file_facts(path: str) -> dict:
    """``{"sha256", "bytes", "detected_type"}`` of a file on disk (raises ``OSError`` when it cannot be read)."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size, "detected_type": identity.sniff_file(path)}


def local_mirror_dir() -> str:
    """``$OSCAR_REFPART_LOCAL_DIR`` (read at CALL time, ``~`` expanded) or ``""`` when unset."""
    value = os.environ.get(ENV_LOCAL_DIR, "").strip()
    return os.path.abspath(os.path.expanduser(value)) if value else ""


def find_local_mirror(ref_id: str) -> str:
    """The offline copy ``<mirror dir>/<ref_id>.step`` (also ``.stp`` / ``.stl`` / ``.brep``) or ``""`` when there is none."""
    base = local_mirror_dir()
    key = str(ref_id or "").strip().lower()
    if not base or not re.fullmatch(r"[a-z0-9_]+", key):
        return ""
    for ext in MIRROR_EXTENSIONS:
        path = os.path.join(base, key + ext)
        if os.path.isfile(path):
            return path
    return ""


def verify_file(path: str, *, expected_sha256: str = "", expect=GEOMETRY_TYPES, max_bytes: int = MAX_REFERENCE_BYTES) -> tuple[dict, str]:
    """Check a reference file: readable, not empty, not over ``max_bytes``, a type in ``expect`` and (when given) the pinned hash.

    Returns ``(facts, error)``; ``facts`` is ``file_facts`` (empty when unreadable) and ``error`` is ``""`` or one model-facing
    sentence in the ``<what failed>: <why>; <what to do>`` style. Never raises.
    """
    try:
        facts = file_facts(path)
    except OSError as exc:
        return {}, f"file cannot be read: {os.path.basename(path)!r}: {exc}; check the path or ask the user to attach the file"
    kind = facts["detected_type"]
    hint = identity.rejection_hint(kind)
    if hint:
        return facts, f"file rejected: {os.path.basename(path)!r}: {hint}"
    if facts["bytes"] > max_bytes:
        return facts, f"file too large: {facts['bytes'] / 1e6:.1f} MB exceeds the {max_bytes / 1e6:.0f} MB limit; ask the user for a smaller file"
    if kind not in tuple(expect):
        wanted = " or ".join(identity.describe_type(t) for t in expect)
        return facts, (f"file rejected: {os.path.basename(path)!r} is {identity.describe_type(kind)}, expected {wanted}; "
                       "use a STEP or STL file")
    pinned = (expected_sha256 or "").strip().lower()
    if pinned and facts["sha256"] != pinned:
        return facts, (f"hash mismatch: the file has sha256 {facts['sha256'][:12]}..., the catalog pins {pinned[:12]}...; the published file "
                       "changed or this is not the official one, so it is not used")
    return facts, ""


# --- pins, trust, identity policy ----------------------------------------------

def hash_policy(step) -> str:
    """``"identity"`` when a hash mismatch of this catalog item degrades to a warning (it must also carry an ``expected_bbox``), else ``"reject"``."""
    if isinstance(step, dict) and step.get("on_hash_mismatch") == "identity" and step.get("expected_bbox"):
        return "identity"
    return "reject"


def _file_pin(step) -> str:
    """The sha256 that the STEP FILE itself (not its archive) is pinned to, or ``""``: the member hash of a ZIP item, else ``expected_sha256``."""
    if not isinstance(step, dict):
        return ""
    pin = step.get("member_sha256") if step.get("member") else step.get("expected_sha256")
    return str(pin).lower() if isinstance(pin, str) and _SHA_RE.match(pin) else ""


def pin_state(entry: "dict | None", sha256: str) -> str:
    """How the geometry file with this hash stands against the catalog pin: ``"match"``, ``"mismatch"``, ``"unpinned"`` (the entry's
    source pins nothing for a file) or ``""`` (the entry lists no STEP source). Pure."""
    step = catalog.step_for(entry or {}, sha256=sha256)
    if not step:
        return ""
    pinned = _file_pin(step)
    if not pinned:
        return "unpinned"
    return "match" if str(sha256 or "").lower() == pinned else "mismatch"


def trust_info(entry: "dict | None", sha256: str) -> dict:
    """``{"trust", "label", "pin", "policy", "last_verified", "license"}`` for the file with this hash from a catalog source, or ``{}``.

    A file whose hash is not the pinned one is still described by the entry's (first) source, with ``pin == "mismatch"`` and a label
    that says the file changed. The caller decides whether the file is user-attached (then none of this applies).
    """
    step = catalog.step_for(entry or {}, sha256=sha256)
    if not step:
        return {}
    pin = pin_state(entry, sha256)
    trust = catalog.trust_of(step)
    return {"trust": trust, "label": catalog.trust_label(trust, changed=pin == "mismatch"), "pin": pin, "policy": hash_policy(step),
            "last_verified": str(step.get("last_verified") or ""), "license": str(step.get("license") or "")}


def _judge_pin(result: "SourceResult", step: dict, sha: str, pinned: str, subject: str) -> str:
    """Compare ``sha`` with ``pinned`` for ``subject`` ("file" / "archive"). Returns ``""`` (accepted: pin match, no pin, or a mismatch
    degraded to a warning by the identity policy) or the model-facing refusal. Sets ``result.pin`` and appends warnings."""
    if not pinned:
        result.pin = result.pin or ("unpinned" if step else "")
        return ""
    if sha.lower() == pinned.lower():
        result.pin = result.pin if result.pin == "mismatch" else "match"
        return ""
    result.pin = "mismatch"
    if hash_policy(step) == "identity":
        result.warnings.append(
            f"the {subject} has sha256 {sha[:12]}..., the catalog pins {pinned[:12]}... (verified {step.get('last_verified') or 'earlier'}): the "
            "published file changed; it is used only if its measured bounding box and outline still match, which import_reference_part checks")
        return ""
    if subject == "archive":
        return f"hash mismatch: the archive has sha256 {sha[:12]}..., the catalog pins {pinned[:12]}...; it is not used"
    return (f"hash mismatch: the file has sha256 {sha[:12]}..., the catalog pins {pinned[:12]}..., the published file "
            "changed or this is not the official one, so it is not used")


def _set_trust(result: "SourceResult", step: dict) -> None:
    if step:
        result.trust = catalog.trust_of(step)
        result.trust_label = catalog.trust_label(result.trust, changed=result.pin == "mismatch")


BREP_EXTENSIONS = (".step", ".stp", ".iges", ".igs", ".brep", ".brp")
MESH_EXTENSIONS = (".stl",)
IMPORTABLE_EXTENSIONS = BREP_EXTENSIONS + MESH_EXTENSIONS


def check_importable(path: str) -> tuple[str, dict, str]:
    """Is ``path`` a file ``import_reference_part`` can bring in? Returns ``(kind, facts, error)``.

    ``kind`` is ``"brep"`` (STEP / IGES / BREP: exact solids) or ``"mesh"`` (STL); ``facts`` is ``file_facts``;
    ``error`` is ``""`` or one model-facing sentence. The extension and the CONTENT must agree (a ``.step`` that holds an HTML page
    or an STL mesh is refused with what it really is), an empty file, a git-LFS pointer and a ZIP are refused. Never raises.
    """
    name = os.path.basename(str(path))
    ext = os.path.splitext(name)[1].lower()
    if not os.path.isabs(str(path)):
        return "", {}, f"'path' must be absolute, got {path!r}; use the absolute path of the attached file"
    if not os.path.isfile(path):
        return "", {}, f"file not found: {path!r}; check the path of the attached file"
    if ext == ".zip":
        return "", {}, f"{name!r} is a ZIP archive; unzip it and attach the STEP or STL file inside (source_reference_part(url=...) extracts archives itself)"
    if ext not in IMPORTABLE_EXTENSIONS:
        return "", {}, (f"unsupported reference file type {ext or '(no extension)'!r}; supported: "
                        f"{', '.join(e.lstrip('.') for e in IMPORTABLE_EXTENSIONS)}")
    facts, error = verify_file(path, expect=GEOMETRY_TYPES)
    if error:
        return "", facts, error
    detected = facts["detected_type"]
    if not identity.extension_matches(ext, detected):
        return "", facts, (f"{name!r} has the extension {ext!r} but holds {identity.describe_type(detected)}; "
                           "attach the real file or rename it to match its content")
    return ("mesh" if detected == identity.STL else "brep"), facts, ""


# --- ZIP archives ----------------------------------------------------------------

def pick_cad_member(files, hint: str = "") -> str:
    """The CAD file to use out of an extracted archive (``files`` are absolute paths): STEP first, then BREP / IGES, then STL.

    Within a class the file whose name shares most words with ``hint`` wins, then the larger file, then the name (so the
    choice is deterministic). Returns ``""`` for an empty list.
    """
    rank = {".step": 0, ".stp": 0, ".brep": 1, ".brp": 1, ".iges": 1, ".igs": 1, ".stl": 2}
    words = set(re.findall(r"[a-z0-9]+", str(hint or "").lower()))
    scored = []
    for path in files or []:
        ext = os.path.splitext(path)[1].lower()
        if ext not in rank:
            continue
        shared = len(words & set(re.findall(r"[a-z0-9]+", os.path.basename(path).lower())))
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        scored.append((rank[ext], -shared, -size, path))
    return min(scored)[3] if scored else ""


# --- attempts of one turn ----------------------------------------------------------

_LOCK = threading.Lock()
_OBTAINED: dict[str, dict] = {}
_ATTEMPTS: "collections.OrderedDict[str, list]" = collections.OrderedDict()


def register_obtained(ref_id: str, path: str, **facts) -> None:
    """Remember the verified file of ``ref_id`` in this process (``obtained_file`` returns it while the file exists)."""
    with _LOCK:
        _OBTAINED[str(ref_id).lower()] = {"path": path, **facts}


def obtained_file(ref_id: str) -> dict:
    """``{"path", ...facts}`` registered for ``ref_id`` and still on disk, else ``{}``."""
    with _LOCK:
        info = dict(_OBTAINED.get(str(ref_id or "").lower()) or {})
    return info if info.get("path") and os.path.isfile(info["path"]) else {}


def obtained_by_sha(sha256: str) -> dict:
    """The registered obtained file whose content hash is ``sha256`` (``{"ref_id", "path", ...}``), or ``{}``.

    Lets ``import_reference_part`` recognise the ``file:`` line of a ``SOURCE RESULT`` when the model passes it back as ``path``."""
    digest = str(sha256 or "").lower()
    if not digest:
        return {}
    with _LOCK:
        items = list(_OBTAINED.items())
    for key, info in reversed(items):
        if str(info.get("sha256", "")).lower() == digest and info.get("path") and os.path.isfile(info["path"]):
            return {"ref_id": key, **info}
    return {}


def latest_obtained() -> dict:
    """The most recently registered obtained file that still exists (``{"ref_id", "path", ...}``), or ``{}``."""
    with _LOCK:
        items = list(_OBTAINED.items())
    for key, info in reversed(items):
        if info.get("path") and os.path.isfile(info["path"]):
            return {"ref_id": key, **info}
    return {}


def forget_all() -> None:
    """Clear both registries (tests)."""
    with _LOCK:
        _OBTAINED.clear()
        _ATTEMPTS.clear()


def previous_attempt(turn_id: str, key: str) -> "SourceResult | None":
    """What an earlier ``fetch`` of THIS turn answered for the same ``key`` (cached copy), else ``None``."""
    if not turn_id:
        return None
    with _LOCK:
        for k, result in _ATTEMPTS.get(turn_id) or []:
            if k == key:
                return SourceResult(**{**result.as_dict(), "cached": True})
    return None


def attempt_budget(turn_id: str) -> str:
    """``""`` when this turn may download again, else the model-facing reason it may not.

    A failed or declined attempt ends the turn's downloads (the retry loop is the failure mode this prevents); successful
    ones are limited to ``MAX_ATTEMPTS_PER_TURN``. Turns without an id (bare ``run_tool`` calls) are never limited.
    """
    if not turn_id:
        return ""
    with _LOCK:
        done = list(_ATTEMPTS.get(turn_id) or [])
    if any(result.status in ("declined", "failed") for _k, result in done):
        return ("only one download attempt is made per turn and the earlier one did not succeed; do not try another URL, "
                "ask the user to download the file and attach it")
    if len(done) >= MAX_ATTEMPTS_PER_TURN:
        return f"{MAX_ATTEMPTS_PER_TURN} downloads were already made this turn; ask the user to attach any further file"
    return ""


def record_attempt(turn_id: str, key: str, result: "SourceResult") -> None:
    """Remember the answer of a ``fetch`` that touched the network (or was declined) in this turn."""
    if not turn_id:
        return
    with _LOCK:
        _ATTEMPTS.setdefault(turn_id, []).append((key, result))
        _ATTEMPTS.move_to_end(turn_id)
        while len(_ATTEMPTS) > _TURNS_KEPT:
            _ATTEMPTS.popitem(last=False)


# --- fetch ---------------------------------------------------------------------------

def _host(url: str) -> str:
    return (urlsplit(url or "").hostname or "").lower()


def check_url(url: str) -> str:
    """``""`` when ``url`` is an https URL on the catalog allowlist, else one model-facing reason."""
    if not isinstance(url, str) or not url.strip().lower().startswith(("http://", "https://")):
        return f"only http(s) URLs can be downloaded, got {url!r}; use an official vendor URL or ask the user to attach the file"
    if urlsplit(url.strip()).scheme.lower() != "https":
        return f"download refused: {url!r} is not https; use the https URL of the official page or ask the user to attach the file"
    hosts = catalog.allow_hosts()
    if _host(url) not in hosts:
        return (f"download refused: host '{_host(url)}' is not on the allowlist ({', '.join(hosts)}); "
                "use an official vendor URL or ask the user to attach the file")
    return ""


def _download_target(entry: dict | None, url: str) -> tuple[dict, str]:
    """``(target, error)``: ``target`` is ``{"url", "expected_sha256", "expected_bbox", "license", "mirror", "step"}`` (``step`` = the
    catalog item, ``{}`` for a bare URL) or ``{}``."""
    if url:
        problem = check_url(url)
        base = {"url": url.strip(), "expected_sha256": "", "expected_bbox": [], "license": "", "mirror": False, "step": {}}
        for step in (entry or {}).get("step_urls") or []:           # the caller's URL may be one the catalog pins
            if isinstance(step, dict) and step.get("url") == url.strip():
                base.update(expected_sha256=step.get("expected_sha256", ""), expected_bbox=list(step.get("expected_bbox") or []),
                            license=step.get("license", ""), mirror=bool(step.get("mirror")), step=step)
        return ({} if problem else base), problem
    for step in (entry or {}).get("step_urls") or []:
        if isinstance(step, dict) and step.get("url"):
            problem = check_url(step["url"])
            target = {"url": step["url"], "expected_sha256": step.get("expected_sha256", ""),
                      "expected_bbox": list(step.get("expected_bbox") or []), "license": step.get("license", ""),
                      "mirror": bool(step.get("mirror")), "step": step}
            return ({} if problem else target), problem
    return {}, ""


HEAVY_IMPORT_BYTES = 40 * 1024 * 1024     # an unpacked STEP above this freezes the GUI for minutes while it imports


def is_heavy(step: dict) -> bool:
    """True for a catalog STEP source that is too big to import on the GUI thread (``heavy: true``, or an unpacked member /
    file of at least ``HEAVY_IMPORT_BYTES``). Such a model is opt-in: the catalog's board data is used unless the caller
    passes ``allow_heavy`` (or the user attaches the file themselves)."""
    if not isinstance(step, dict):
        return False
    if step.get("heavy") is True:
        return True
    try:
        return int(step.get("member_bytes") or step.get("size") or 0) >= HEAVY_IMPORT_BYTES
    except (TypeError, ValueError):
        return False


def fetch(entry: dict | None, *, ref_id: str = "", url: str = "", allow_download: bool = True, allow_heavy: bool = False,
          confirm: "Callable[[str, str], bool] | None" = None, dest_dir: str = "", reason: str = "",
          should_stop: "Callable[[], bool] | None" = None, max_bytes: int = MAX_REFERENCE_BYTES, timeout: float = 90.0,
          turn_id: str = "", downloader=None, extractor=None) -> SourceResult:
    """Obtain the reference file for ``entry`` (or for a bare ``url``): the order of the module docstring.

    Never raises on network, file or verification problems (``netguard.download_verified`` itself never raises).

    ``confirm(summary, detail[, url=...]) -> bool`` is the permission channel (``url`` is passed when the callback accepts it);
    without one a download is ``declined``. ``dest_dir`` defaults to ``netguard.default_download_dir()``. ``turn_id`` enables
    the one-failed-attempt-per-turn rule and the cache (an identical second call returns the first answer with
    ``cached=True`` and touches nothing). ``downloader`` / ``extractor`` replace ``netguard.download_verified`` /
    ``netguard.extract_zip`` (tests).
    """
    from .. import netguard                                    # lazy: netguard imports this package

    entry = entry or None
    rid = str(ref_id or (entry or {}).get("id") or "").strip().lower()
    if not rid and url:
        rid = slug_id(_path_stem(url), "download")
    result = SourceResult(ref_id=rid, ref_name=(entry or {}).get("name", "") or rid, license=(entry or {}).get("license", ""))
    if entry and entry.get("board"):
        try:
            catalog.load_board(entry["id"])
            result.has_board = True
        except (KeyError, ValueError):
            result.has_board = False
    fallback = " The catalog's published board data is used instead." if result.has_board else ""

    mirror = find_local_mirror(rid) if rid else ""
    key = f"mirror:{mirror}" if mirror else f"url:{url.strip()}" if url else f"id:{rid}"
    cached = previous_attempt(turn_id, key)
    if cached is not None:
        cached.note = f"already attempted in this turn, no second download is made: {cached.note}"
        return cached

    if mirror:                                                  # 1. the offline mirror: verified like a download
        step = catalog.step_for(entry or {})
        pinned = _file_pin(step)
        strict_pin = pinned if hash_policy(step) == "reject" else ""
        facts, error = verify_file(mirror, expected_sha256=strict_pin, max_bytes=max_bytes)
        result.origin, result.url, result.license = "local_mirror", "", ""
        if not error:
            if strict_pin:                                      # verify_file already refused a different hash
                result.pin = "match"
            else:
                error = _judge_pin(result, step, facts["sha256"], pinned, "file")
        if error:
            result.status, result.note = "failed", f"the local mirror file is not usable: {error}.{fallback}"
            return result
        _fill(result, "obtained", mirror, facts)
        result.expected_bbox = _expected_bbox(entry)
        _set_trust(result, step)
        result.note = (f"STEP file taken from the local mirror ({ENV_LOCAL_DIR}) and verified."
                       + "".join(f" WARNING: {w}." for w in result.warnings))
        register_obtained(rid, mirror, **facts, trust=result.trust, trust_label=result.trust_label, pin=result.pin)
        return result

    target, problem = _download_target(entry, url)
    if problem:
        result.status, result.note = "failed", _sentence(problem, fallback)     # refused before any network use: costs no attempt
        return result
    if not target:
        if result.has_board:
            result.status = "nominal"
            result.note = ("the catalog lists no downloadable STEP file for this part; its published board data (outline, mounting "
                           "holes, connector boxes) is used. If you have the manufacturer's STEP file, attach it to the chat "
                           "and import it with import_reference_part(path=...).")
        else:
            result.status = "failed"
            result.note = ("the catalog has no STEP file and no board data for this part; ask the user to attach its STEP or STL file "
                           "(import_reference_part with path), or to give the dimensions.")
        return result
    result.url, result.license = target["url"], target["license"] or result.license
    _set_trust(result, target.get("step") or {})                # the permission dialog and a declined result already say what the source is
    result.expected_bbox = target["expected_bbox"] or _expected_bbox(entry)
    if not allow_download:
        result.status = "declined"
        result.note = f"the download was not made (allow_download=false).{fallback}"
        return result
    step = target.get("step") or {}
    if result.has_board and not allow_heavy and not url and is_heavy(step):
        megabytes = int(step.get("member_bytes") or step.get("size") or 0) / (1024 * 1024)
        result.status = "nominal"
        result.note = (f"the catalog's STEP file for this part is large (about {megabytes:.0f} MB unpacked); importing it would freeze the "
                       "application for minutes, so it is NOT downloaded automatically. The catalog's board data (outline, mounting holes, "
                       "connector boxes) is used instead"
                       + (" - it was measured from that official model" if str(step.get("trust", "")).lower() == "official" else "")
                       + ". To use the real model anyway call source_reference_part again with allow_heavy=true, or ask the user to "
                       "attach the file and import it with import_reference_part(path=...).")
        return result
    budget = attempt_budget(turn_id)
    if budget:
        result.status, result.note = "failed", f"{budget}.{fallback}"
        return result
    if confirm is None or not _ask(confirm, result, reason):
        result.status = "declined"
        result.note = ("the user declined the download. Do not ask again and do not try another URL; ask the user to download "
                       f"{target['url']} in a browser and attach the file, or continue with the nominal data.{fallback}")
        record_attempt(turn_id, key, result)
        return result
    return _download(result, target, dest_dir or netguard.default_download_dir(), turn_id, key, fallback,
                     should_stop=should_stop, max_bytes=max_bytes, timeout=timeout, downloader=downloader, extractor=extractor)


def _download(result: SourceResult, target: dict, dest_dir: str, turn_id: str, key: str, fallback: str, *, should_stop,
              max_bytes: int, timeout: float, downloader, extractor) -> SourceResult:
    from .. import netguard

    fetcher = downloader or netguard.download_verified
    got = fetcher(target["url"], dest_dir, "", max_bytes=max_bytes, timeout=timeout, expect=DOWNLOAD_TYPES,
                  allowed_hosts=catalog.allow_hosts(), should_stop=should_stop)
    result.final_url = got.final_url or target["url"]
    if not got.ok:
        result.status, result.sha256, result.bytes, result.detected_type = "failed", got.sha256, got.bytes, got.detected_type
        stopped = "stopped" in (got.error or "")
        result.note = _sentence(got.error, fallback) if not stopped else "the download was stopped."
        record_attempt(turn_id, key, result)
        return result
    path, facts = got.path, {"sha256": got.sha256, "bytes": got.bytes, "detected_type": got.detected_type}
    step = target.get("step") or {}
    if got.detected_type == identity.ZIP:
        member, archive, error = _open_archive(got.path, dest_dir, result.ref_name, extractor, max_bytes, str(step.get("member") or ""))
        if error:
            result.status, result.note = "failed", f"{error}.{fallback}"
            result.sha256, result.bytes, result.detected_type = got.sha256, got.bytes, got.detected_type
            record_attempt(turn_id, key, result)
            return result
        result.archive = {"path": got.path, "sha256": got.sha256, "bytes": got.bytes, "member": os.path.basename(member), **archive}
        path = member
        problem = _judge_pin(result, step, got.sha256, str(target["expected_sha256"] or "").lower(), "archive")
        if problem:
            result.status = "failed"
            result.note = f"{problem}.{fallback}"
            record_attempt(turn_id, key, result)
            return result
        facts, error = verify_file(path, max_bytes=max_bytes)          # the catalog pins the archive; its member is judged next
        if not error and step.get("member_sha256") and str(step.get("member") or "") == os.path.basename(path):
            error = _judge_pin(result, step, facts["sha256"], str(step["member_sha256"]).lower(), "file")
    else:
        strict_pin = target["expected_sha256"] if hash_policy(step) == "reject" else ""
        facts, error = verify_file(path, expected_sha256=strict_pin, max_bytes=max_bytes)
        if not error:
            if strict_pin:
                result.pin = "match"
            else:
                error = _judge_pin(result, step, facts["sha256"], str(target["expected_sha256"] or "").lower(), "file")
    if error:
        result.status, result.note = "failed", f"{error}.{fallback}"
        result.sha256, result.bytes, result.detected_type = facts.get("sha256", ""), facts.get("bytes", 0), facts.get("detected_type", "")
        record_attempt(turn_id, key, result)
        return result
    _fill(result, "obtained", path, facts)
    result.origin = "download"
    _set_trust(result, step)
    result.note = (f"downloaded from {_host(result.final_url)} and verified ({identity.describe_type(facts['detected_type'])}, "
                   f"sha256 {facts['sha256'][:12]}...)"
                   + (f"; extracted {os.path.basename(path)!r} from the archive" if result.archive else "") + "."
                   + "".join(f" WARNING: {w}." for w in result.warnings))
    register_obtained(result.ref_id, path, **facts, trust=result.trust, trust_label=result.trust_label, pin=result.pin)
    record_attempt(turn_id, key, result)
    return result


def _open_archive(zip_path: str, dest_dir: str, hint: str, extractor, max_bytes: int, member_name: str = "") -> tuple[str, dict, str]:
    """Extract a downloaded ZIP into a fresh sibling directory; returns ``(member path, extra facts, error)``.

    ``member_name`` (the catalog's ``member``) is preferred when the archive holds a file of that name; otherwise the best CAD file
    is chosen by ``pick_cad_member``."""
    from .. import netguard

    with open(zip_path, "rb") as fh:
        tag = hashlib.sha256(fh.read(1 << 20)).hexdigest()[:8]
    target = os.path.join(dest_dir, f"{os.path.splitext(os.path.basename(zip_path))[0]}-{tag}")
    extract = extractor or netguard.extract_zip
    done = extract(zip_path, target, max_total_bytes=max_bytes)
    if not done.ok:
        return "", {}, f"the downloaded archive was refused: {done.error}; ask the user to attach the CAD file itself"
    pool = done.cad_files or done.files
    member = next((f for f in pool if member_name and os.path.basename(f) == member_name), "") or pick_cad_member(pool, hint)
    if not member:
        return "", {}, ("the downloaded archive holds no STEP, IGES, BREP or STL file "
                        f"({len(done.files)} files); ask the user to attach the CAD file")
    return member, {"extracted_to": done.dest_dir, "files": len(done.files)}, ""


def _sentence(text: str, tail: str = "") -> str:
    """``text`` as one sentence ending in a single full stop, followed by ``tail`` (no ``..`` when ``text`` already ends in one)."""
    return text.rstrip().rstrip(".") + "." + tail


def _fill(result: SourceResult, status: str, path: str, facts: dict) -> None:
    result.status, result.path = status, path
    result.sha256, result.bytes, result.detected_type = facts["sha256"], facts["bytes"], facts["detected_type"]


def _expected_bbox(entry: dict | None) -> list:
    for step in (entry or {}).get("step_urls") or []:
        if isinstance(step, dict) and step.get("expected_bbox"):
            return list(step["expected_bbox"])
    return []


def _path_stem(url: str) -> str:
    from urllib.parse import unquote

    return os.path.splitext(unquote(urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]))[0]


def _accepts_url(confirm) -> bool:
    """True when ``confirm`` takes a ``url`` keyword (or ``**kwargs``): the structured way to tell the permission layer WHICH
    URL is being asked about, so a per-host grant never has to be scraped back out of the dialog text."""
    import inspect

    try:
        params = inspect.signature(confirm).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "url" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


def _ask(confirm, result: SourceResult, reason: str) -> bool:
    summary, detail = _permission_text(result, reason)
    if _accepts_url(confirm):
        return bool(confirm(summary, detail, url=result.url))
    return bool(confirm(summary, detail))                       # a two-argument callback (tests, older callers)


def _permission_text(result: SourceResult, reason: str) -> tuple[str, str]:
    """``(summary, detail)`` of the permission dialog: what is fetched, from where, why, and what protects the user."""
    part = result.ref_name or result.ref_id or "a reference part"
    summary = ("Oscar wants to download a reference file for your design:\n"
               f"• {result.url}\n• Part: {part}")
    if result.trust:
        summary += f"\n• Source: {result.trust} model"
    detail = f"Reason: {reason.strip() or 'reference geometry for the design'}\n"
    if result.trust_label:
        detail += f"Trust: {result.trust_label}.\n"
    detail += (f"Only allowlisted hosts are contacted ({', '.join(catalog.allow_hosts())}). "
               "The file is checked (type, size, hash) before it is used, and at most one download is made.")
    return summary, detail
