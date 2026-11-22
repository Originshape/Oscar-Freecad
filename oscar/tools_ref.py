"""Reference-part tools: ``find_reference_part``, ``source_reference_part``, ``import_reference_part``, ``measure_reference_part``.

The sourcing flow of a design that wraps a real part (a Raspberry Pi, a motor):

1. ``find_reference_part``   no network; what the vetted catalog knows and how reliable it is;
2. ``source_reference_part`` catalog entry (or one official URL) -> AT MOST ONE allowlisted download after the user said yes ->
   verified file, or the catalog's published board data (``nominal``); returns ``obtained | declined | failed | nominal``;
3. ``import_reference_part`` the file becomes ONE hidden ``Ref_<ID>`` ``App::Part`` in the document, already normalized into the
   board frame (a geometry tool: health, delta, snapshot and checkpoint run for it);
4. ``measure_reference_part`` exact B-rep measurements, optionally written into the ``Params`` sheet (``step_measurement``).

This module is the thin tool layer. The logic lives in ``oscar.refparts`` (``fetch``: sourcing and verification; ``importref``
and ``measure*``: FreeCAD work on the GUI thread; ``catalog`` / ``identity``: pure data). Handlers run on the agent thread; every
document access goes through ``cad.dispatch`` with NO lock held and the context captured BEFORE dispatching. Provenance goes into
the bound ``Project`` through its API only (a failure there is logged and never fails the call; without a project the tools still
work and the result says nothing was recorded). Handlers talk to ``run_tool`` through ``ctx.state["extra"]["reference_result"]``
(read by ``Ledger.on_tool_result``) and, for ``import_reference_part``, ``ctx.state["args_override"]``.
"""

import os
import re

from . import netguard
from . import turnlog as _turnlog
from .project import ProjectError
from .refparts import catalog, fetch, identity
from .tools import ToolError, current_context, tool

_GUI_TIMEOUT_S = 240.0
_IMPORT_TIMEOUT_MAX_S = 1200.0


def _import_timeout(nbytes: int) -> float:
    """GUI-thread budget of ``import_reference_part``: 240 s, more for a big STEP (the official Pi 5 model, 77 MB, 2689 solids, takes
    about 200 s on a laptop: roughly 6 s per MB plus 60 s), never more than 20 minutes."""
    return max(_GUI_TIMEOUT_S, min(_IMPORT_TIMEOUT_MAX_S, 60.0 + 6.0 * max(0, int(nbytes or 0)) / 1e6))
_MAX_FIND = 10
_FOUND_PHASES = ("INTAKE", "SOURCE", "PLAN")
_REF_STATUS_RANK = {"none": 0, "catalog-nominal": 1, "step-imported": 2, "user-attached": 2, "step-measured": 3}
_PROV_TYPES = ("step", "stl", "zip", "pdf", "brep", "iges")


# --- shared helpers ---------------------------------------------------------------------------------------------

def _fmt_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _mm(value) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _gui_call(fn, op: str, timeout: float = _GUI_TIMEOUT_S):
    """Run ``fn`` on the GUI thread (inline in console runs); returns ``(result, "")`` or ``(None, error text)``. Never raises."""
    try:
        from .cad import dispatch

        result, failed = dispatch.dispatch_stoppable(fn, timeout, op)
    except Exception as exc:  # noqa: BLE001 - a dispatcher problem is a tool failure, not a crash
        return None, f"{type(exc).__name__}: {exc}"
    if failed:
        return None, dispatch.failure_text(result)
    return result, ""


def _require_doc(doc_name: str, tool_name: str) -> str:
    """The real document name for ``doc_name`` (a ``ToolError`` listing the open documents otherwise)."""
    name = str(doc_name or "").strip()
    if not name:
        raise ToolError(f"{tool_name}: 'doc_name' is empty; use the name returned by create_document")
    from . import tools

    return tools._require_document(name)


def _project_call(project, what: str, fn, *args, **kwargs):
    """``fn(*args, **kwargs)`` against the Project, guarded: returns ``(value, "")`` or ``(None, error text)``; logs the failure."""
    if project is None:
        return None, "no project is bound to this chat"
    try:
        return fn(*args, **kwargs), ""
    except (ProjectError, OSError, ValueError) as exc:
        _turnlog.log_event("project_error", hook=what, error=f"{type(exc).__name__}: {exc}"[:200])
        return None, str(exc)[:200]
    except Exception as exc:  # noqa: BLE001 - recording must never fail the call
        _turnlog.log_event("project_error", hook=what, error=f"{type(exc).__name__}: {exc}"[:200])
        return None, f"{type(exc).__name__}: {exc}"[:200]


def _reference_status(project) -> str:
    try:
        return str(project.reference_status) if project is not None else "none"
    except Exception:  # noqa: BLE001
        return "none"


def _stronger(current: str, wanted: str) -> bool:
    """True when ``wanted`` is a step up from ``current`` (a re-run of an earlier tool never downgrades the status)."""
    return _REF_STATUS_RANK.get(wanted, 0) > _REF_STATUS_RANK.get(current, 0)


_SAME_KEYS = ("kind", "status", "source", "sha256", "claim", "url")


def _repeated_entry(project, fields: dict) -> str:
    """The id of the NEWEST provenance entry when it says exactly what ``fields`` would (a model that repeats a call must not
    fill the Build Summary with identical rows), else ``""``."""
    data, _error = _project_call(project, "read_provenance", project.read_provenance if project is not None else None)
    entries = (data or {}).get("entries") or []
    if not entries:
        return ""
    last = entries[-1]
    return last.get("id", "") if all(str(last.get(k, "")) == str(fields.get(k, "")) for k in _SAME_KEYS if k != "sha256" or fields.get("sha256")) else ""


def _record(project, fields: dict, ref_id: str, ref_name: str, status_wanted: str = "") -> tuple[str, str]:
    """Write one provenance entry and (re)set the project's reference. Returns ``(entry id or "", note)``.

    An identical repeat of the newest entry is not written again (its id is returned)."""
    repeated = _repeated_entry(project, fields)
    if repeated:
        return repeated, ""
    current = _reference_status(project)
    if status_wanted and _stronger(current, status_wanted):
        fields = {**fields, "reference_status": status_wanted}
    entry, error = _project_call(project, "provenance_add", project.provenance_add if project is not None else None, **fields)
    if entry is None:
        return "", error
    _project_call(project, "set_reference", project.set_reference, ref_id, ref_name, _reference_status(project), entry)
    return entry, ""


def _ref_name(ref_id: str) -> str:
    """The catalog name of a reference id, else the id."""
    try:
        return catalog.get(ref_id)["name"]
    except (KeyError, ValueError):
        return ref_id


def _provenance_type(detected: str) -> str:
    return detected if detected in _PROV_TYPES else "none"


def _ext_dir(project, local_rel: str, fallback: str) -> str:
    """The absolute path of the project's ``refs/`` copy of a file, else ``fallback``."""
    if project is not None and local_rel:
        try:
            path = project.path(local_rel)
            if os.path.isfile(path):
                return path
        except Exception:  # noqa: BLE001
            pass
    return fallback


def _set_extra(ctx, **values) -> None:
    """``ctx.state["extra"]["reference_result"]``: the channel ``Ledger.on_tool_result`` reads."""
    extra = ctx.state.setdefault("extra", {})
    extra["reference_result"] = values


# --- find_reference_part --------------------------------------------------------------------------------------

@tool(
    "find_reference_part",
    "Search the vetted reference-part catalog (boards, modules, hardware) by name. Returns matching entries with their ids, "
    "what data exists (STEP download, board data) and how reliable it is. Use it before sourcing a part; it needs no network.",
    {"type": "object", "properties": {
        "query": {"type": "string", "description": "e.g. 'raspberry pi 4 model b'"},
        "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10}},
     "required": ["query"]},
    category="source", phases=_FOUND_PHASES,
)
def _find_reference_part(query: str, limit: int = 5):
    """List the catalog entries matching ``query`` (best first) with their data and reliability; no network, no document."""
    text = str(query or "").strip()
    if not text:
        raise ToolError("find_reference_part: 'query' is empty; give a part name such as 'raspberry pi 4 model b'.")
    count = max(1, min(int(limit or 5), _MAX_FIND))
    found = catalog.match_scored(text)
    if not found:
        known = ", ".join(f"{e['id']} ({e['name']})" for e in catalog.list_entries())
        return (f"No catalog entry matches '{text}'. Known parts: {known}. If the part is not listed, ask the user to attach its STEP or "
                "STL file (import_reference_part with path) or to give the dimensions; never invent a URL.")
    shown = found[:count]
    lines = [f"{len(found)} catalog match{'es' if len(found) != 1 else ''} for '{text}'" + (f" (showing {len(shown)})" if len(found) > len(shown) else "") + ":"]
    for rank, (_score, entry) in enumerate(shown, 1):
        info = fetch.entry_summary(entry)
        data = []
        if info["board"]:
            levels = ", ".join(f"{n} {level}" for level, n in sorted(info["confidence"].items(), key=lambda kv: catalog.CONFIDENCE_LEVELS.index(kv[0]) if kv[0] in catalog.CONFIDENCE_LEVELS else 99))
            data.append(f"board data ({info['connectors']} connectors; {levels})")
        else:
            data.append("no board data")
        data.append(f"STEP download: {info['step_urls']} source{'s' if info['step_urls'] != 1 else ''} listed "
                    f"({', '.join(sorted(set(info['step_trust']))) or 'unknown'} trust)" if info["step_urls"]
                    else "STEP download: none listed (nominal board data only)")
        data.append("verified against the drawing" if info["verified"] else "NOT verified against the drawing")
        lines.append(f"{rank}. {info['id']} - {info['name']} [{info['kind']}]: " + "; ".join(data) + ".")
        if info["datasheets"]:
            lines.append(f"   datasheet: {info['datasheets'][0]}")
    best = shown[0][1]["id"]
    lines.append(f"Next: source_reference_part(id=\"{best}\", reason=\"...\") obtains the part (one allowlisted download after the user agrees, "
                 "else the published board data).")
    return "\n".join(lines)


# --- source_reference_part -------------------------------------------------------------------------------------

def _confirm_scoped(summary: str, detail: str, url: str = "") -> bool:
    """The permission question of ``source_reference_part``, scoped to the host of the URL being fetched (so 'Always allow' is
    THIS host for THIS tool). ``url`` arrives structurally from ``fetch.fetch``; only an older caller without it falls back to
    the first URL in the dialog text. When no host can be determined nothing is asked and the answer is no: an unscoped
    grant must never be what the user ends up giving."""
    import re as _re
    from urllib.parse import urlsplit as _urlsplit

    def host_of(text: str) -> str:
        try:
            return (_urlsplit(text).hostname or "").lower()
        except ValueError:
            return ""

    host = host_of(url)
    if not host:
        found = _re.search(r"https?://[^\s'\"<>)]+", summary or "")
        host = host_of(found.group(0)) if found else ""
    if not host:
        _turnlog.log_event("net_permission", decision="deny_unscoped", tool="source_reference_part")
        return False
    return netguard.request_permission(summary, detail=detail, tool="source_reference_part", host=host) in ("allow", "always")



@tool(
    "source_reference_part",
    "Obtain a reference part for the design. Uses the catalog entry (id) or one URL you are SURE is official; makes at most one "
    "allowlisted download (the user is asked first), verifies the file (type, hash, size) and records provenance. If nothing can "
    "be downloaded it falls back to the catalog's published board data and says so (status nominal). If you give neither id nor "
    "url, the app matches the part named in the request. Never invent a URL.",
    {"type": "object", "properties": {
        "id": {"type": "string", "description": "Catalog id from find_reference_part, e.g. 'rpi4b'. Optional: omitted -> the board matched "
               "from the request text (ledger.reference_hint) through catalog.match; an error lists the candidates when nothing matches."},
        "url": {"type": "string", "description": "Alternative to id: an official http(s) URL of a STEP/STL/ZIP file."},
        "reason": {"type": "string", "description": "Why this is needed; shown to the user in the permission dialog."},
        "allow_download": {"type": "boolean", "default": True},
        "allow_heavy": {"type": "boolean", "default": False,
                        "description": "also fetch a very large catalog STEP model (tens of MB; importing it freezes the app for minutes). "
                                       "Default false: the catalog's board data is used instead."}},
     "required": ["reason"]},
    category="source", mutating=True, phases=_FOUND_PHASES,
)
def _source_reference_part(reason: str, id: str = "", url: str = "", allow_download: bool = True,  # noqa: A002 - the schema names it `id`
                           allow_heavy: bool = False):
    """Resolve the part, obtain its file (local mirror, the given URL or the catalog URL: at most ONE download), record provenance and report.

    Returns the ``SOURCE RESULT`` block, whose first lines are fixed: status, reference_status, reference, file, provenance, note.
    Raises ``ToolError`` only for arguments the model can fix (empty reason, unknown id, nothing named). A declined or failed download
    is a normal result (``declined`` / ``failed``), never an error.
    """
    ctx = current_context()
    project, ledger = ctx.project, ctx.ledger
    why = str(reason or "").strip()
    problems = []
    if not why:
        problems.append("'reason' is empty; say why the part is needed (the user sees it in the permission dialog)")
    link = str(url or "").strip()
    if link and not link.lower().startswith(("http://", "https://")):
        problems.append(f"'url' must be an http(s) URL, got {link!r}")
    if problems:
        raise ToolError(f"source_reference_part: {'; '.join(problems)}.")
    try:
        entry, others = fetch.resolve_entry(str(id or "").strip(), getattr(ledger, "reference_hint", "") or "", getattr(ledger, "request", "") or "")
    except KeyError as exc:
        raise ToolError(f"source_reference_part: {exc.args[0]}; call find_reference_part(query=...) to see the catalog.") from None
    if entry is None and not link:
        known = ", ".join(f"{e['id']} ({e['name']})" for e in catalog.list_entries())
        raise ToolError("source_reference_part: no 'id' or 'url' was given and the request does not name a catalog part; call "
                        f"find_reference_part(query=...) first. Known parts: {known}. If the part is not listed, ask the user to attach its "
                        "STEP or STL file (import_reference_part with path); never invent a URL.")
    if entry is not None and not str(id or "").strip() and others:
        _turnlog.log_event("reference_ambiguous", picked=entry["id"], others=[e["id"] for e in others[:4]])

    result = fetch.fetch(entry, ref_id=(entry or {}).get("id", ""), url=link, allow_download=bool(allow_download), allow_heavy=bool(allow_heavy), confirm=_confirm_scoped,
                         reason=why, should_stop=ctx.should_stop, turn_id=ctx.turn_id, dest_dir=netguard.default_download_dir())
    if result.cached and result.provenance:
        entry_id, record_note = result.provenance, ""                  # the first answer of this turn was recorded already
    else:
        entry_id, record_note = _record_source(project, result, entry)
        result.provenance = entry_id
    return _source_text(ctx, project, result, entry, entry_id, record_note)


def _board_claim(entry: dict) -> tuple[str, str, str, str]:
    """``(claim, source, url, confidence)`` of the nominal board data of a catalog entry (read from its board file)."""
    board = catalog.load_board(entry["id"])
    outline, holes = board.get("outline") or {}, board.get("mounting_holes") or {}
    pattern = holes.get("pattern") or {}
    claim = f"Board outline {_mm(outline.get('length', 0))} x {_mm(outline.get('width', 0))} mm"
    if pattern:
        offset = pattern.get("offset") or [0, 0]
        claim += f", mounting holes {_mm(pattern.get('dx', 0))} x {_mm(pattern.get('dy', 0))} mm at ({_mm(offset[0])}, {_mm(offset[1])})"
    claim += ", connector positions"
    return (claim, f"catalog:{entry['id']} board.json v{board.get('schema', 1)} ({outline.get('source', 'published drawing')})",
            str(outline.get("source_url") or (entry.get("datasheet_urls") or [""])[0]), str(outline.get("confidence") or "assumed"))


def _record_source(project, result: "fetch.SourceResult", entry: "dict | None") -> tuple[str, str]:
    """Write the provenance entry of a sourcing result; returns ``(entry id, note)``."""
    rid, name = result.ref_id, result.ref_name or result.ref_id
    base = {"kind": "download", "status": result.status, "detected_type": _provenance_type(result.detected_type)}
    if result.status == "nominal" or (result.status in ("declined", "failed") and result.has_board):
        claim, source, url, confidence = _board_claim(entry)
        weakest = fetch.entry_summary(entry).get("weakest", "")
        fields = {**base, "kind": "reference", "claim": claim, "source": source, "url": url, "confidence": confidence,
                  "status": result.status,
                  "note": ("No STEP file was obtained; nominal published dimensions are used." if result.status == "nominal"
                           else f"Download {result.status}: nominal published dimensions are used. {result.note}")[:600]
                  + (f" Weakest board data: {weakest}." if weakest else "")}
        return _record(project, fields, rid, name, "catalog-nominal")
    if result.status == "obtained":
        verified = bool(entry and entry.get("verified") and any(s.get("expected_sha256") for s in entry.get("step_urls") or []))
        host = _host_of(result.final_url or result.url)
        origin = {"local_mirror": f"local-mirror:{rid} {os.path.basename(result.path)}", "download": f"catalog:{rid} download from {host}"
                  if entry else f"url:{rid} download from {host}"}[result.origin]
        trusted = result.trust == "official" and result.pin == "match"      # a changed or community file is never "datasheet"
        label = f" [{result.trust_label}]" if result.trust_label else ""
        fields = {**base, "claim": f"Reference geometry file for {name}: {os.path.basename(result.path)} "
                                   f"({identity.describe_type(result.detected_type)}, {_fmt_size(result.bytes)}){label}",
                  "source": origin, "url": result.final_url or result.url, "local_path": result.path, "sha256": result.sha256,
                  "confidence": "datasheet" if trusted or (verified and not result.trust) else "community",
                  "note": (result.note + (f" Licence: {result.license}." if result.license else ""))[:600]}
        if result.archive:
            fields["note"] = (fields["note"] + f" Extracted from archive {os.path.basename(result.archive.get('path', ''))} "
                              f"(sha256 {str(result.archive.get('sha256', ''))[:12]}).")[:600]
        return _record(project, fields, rid, name)
    fields = {**base, "claim": f"No file was obtained for {name} ({result.status})", "source": f"source_reference_part {('catalog:' + rid) if entry else 'url'}",
              "url": result.url, "confidence": "assumed", "note": result.note[:600]}
    return _record(project, fields, rid, name)


def _host_of(url: str) -> str:
    """The host of ``url`` for a provenance source string (``"?"`` when there is none)."""
    from urllib.parse import urlsplit

    return (urlsplit(url or "").hostname or "?").lower()


def _source_text(ctx, project, result: "fetch.SourceResult", entry: "dict | None", entry_id: str, record_note: str) -> str:
    """The ``SOURCE RESULT`` block (fixed first lines) and the ``reference_result`` hand-over to the ledger."""
    fallback = result.status in ("nominal", "declined", "failed") and result.has_board
    expected = "catalog-nominal" if fallback else "none"
    status_now = _reference_status(project)
    shown_status = status_now if _stronger(expected, status_now) else expected
    if result.status == "obtained":
        local = ""
        if project is not None and entry_id:
            rows = {e.get("id"): e for e in (_project_call(project, "read_provenance", project.read_provenance)[0] or {}).get("entries", [])}
            local = (rows.get(entry_id) or {}).get("local_path", "")
        shown = _ext_dir(project, local, result.path)
        file_line = f"{shown} (sha256 {result.sha256}, {_fmt_size(result.bytes)}, {result.detected_type})"
    else:
        file_line = "none"
    name = result.ref_name or result.ref_id or "?"
    prov = entry_id or f"none ({record_note or 'no project is bound to this chat, provenance is not recorded'})"
    nxt = {"obtained": " Next: create_document if no document is open, then import_reference_part(doc_name=..., id=\"%s\")." % result.ref_id,
           "nominal": " Next: skip the import and measurement (there is no STEP file); the kit builds the board from this data "
                      "(enclosure_spec, then build_enclosure).",
           "declined": " Say honestly that the download was declined and continue with the nominal data." if fallback else
                       " Ask the user to attach the file, or to give the dimensions.",
           "failed": " Say honestly that no file could be obtained and continue with the nominal data." if fallback else
                     " Ask the user to attach the file, or to give the dimensions."}[result.status]
    _set_extra(ctx, status=result.status, reference_status=expected if fallback else "none", id=result.ref_id, name=name,
               has_board=result.has_board, board=result.has_board, sha256=result.sha256, bytes=result.bytes,
               detected_type=result.detected_type, path=result.path, origin=result.origin, provenance=entry_id,
               reason=result.note, cached=result.cached, trust=result.trust, trust_label=result.trust_label, pin=result.pin)
    _turnlog.log_event("reference_source", id=result.ref_id, status=result.status, origin=result.origin, bytes=result.bytes,
                       sha=result.sha256[:12], host=_host_of(result.final_url or result.url), cached=result.cached)
    lines = [
        f"SOURCE RESULT: status={result.status}",
        f"reference_status: {shown_status}",
        f"reference: {result.ref_id or '?'} ({name})",
        f"file: {file_line}",
        f"provenance: {prov}",
    ]
    if result.status == "obtained" and result.trust:
        lines.append(f"trust: {result.trust_label}" + (f"; licence: {result.license}" if result.license else ""))
    lines.append(f"note: {result.note}{nxt}")
    return "\n".join(lines)


# --- import_reference_part -----------------------------------------------------------------------------------

def _obtained_for_id(project, ref_id: str) -> dict:
    """The file obtained for ``ref_id``: ``{"path", "sha256", "bytes", "detected_type"}`` from this process, the project's provenance
    or the offline mirror; ``{}`` when there is none."""
    got = fetch.obtained_file(ref_id)
    if got:
        return got
    if project is not None:
        data = (_project_call(project, "read_provenance", project.read_provenance)[0]) or {}
        tokens = (f"catalog:{ref_id}", f"local-mirror:{ref_id}", f"url:{ref_id}")
        for row in reversed(data.get("entries") or []):
            if row.get("status") == "obtained" and row.get("local_path") and str(row.get("source", "")).startswith(tokens):
                path = _ext_dir(project, row["local_path"], "")
                if path:
                    return {"path": path, "sha256": row.get("sha256", ""), "bytes": row.get("bytes", 0), "detected_type": row.get("detected_type", "")}
    mirror = fetch.find_local_mirror(ref_id)
    return {"path": mirror} if mirror else {}


def _latest_obtained(project) -> dict:
    """The newest obtained reference file of this process or project (``{"ref_id", "path", ...}``), else ``{}``."""
    got = fetch.latest_obtained()
    if got:
        return got
    if project is not None:
        data = (_project_call(project, "read_provenance", project.read_provenance)[0]) or {}
        for row in reversed(data.get("entries") or []):
            source = str(row.get("source", ""))
            m = re.match(r"(?:catalog|local-mirror|url):([a-z0-9_]+)", source)
            if row.get("status") == "obtained" and row.get("local_path") and m:
                path = _ext_dir(project, row["local_path"], "")
                if path:
                    return {"ref_id": m.group(1), "path": path, "sha256": row.get("sha256", "")}
    return {}


def _obtained_id_by_sha(project, sha: str) -> str:
    """The reference id of an obtained file with this content hash (this process first, then the project's provenance), else ``""``."""
    got = fetch.obtained_by_sha(sha)
    if got:
        return got.get("ref_id", "")
    if project is not None and sha:
        data = (_project_call(project, "read_provenance", project.read_provenance)[0]) or {}
        for row in reversed(data.get("entries") or []):
            m = re.match(r"(?:catalog|local-mirror|url):([a-z0-9_]+)", str(row.get("source", "")))
            if m and row.get("status") == "obtained" and str(row.get("sha256", "")).lower() == sha.lower():
                return m.group(1)
    return ""


def _resolve_import_source(project, ref_id: str, path: str) -> tuple[str, str, bool, str]:
    """``(ref_id, file path, user_attached, sha256)``; raises ``ToolError`` with what to do when no file can be found."""
    if path:
        facts = fetch.file_facts(path) if os.path.isfile(path) else {}
        known = _obtained_id_by_sha(project, facts.get("sha256", ""))
        if known:                                   # the model passed back the file line of SOURCE RESULT: not an attachment
            return known, path, False, facts.get("sha256", "")
        return ref_id or fetch.slug_id(path, "part"), path, True, facts.get("sha256", "")
    if ref_id:
        got = _obtained_for_id(project, ref_id)
        if got.get("path"):
            return ref_id, got["path"], False, got.get("sha256", "")
        raise ToolError(f"import_reference_part: no file was obtained for '{ref_id}'; call source_reference_part(id=\"{ref_id}\", reason=...) first "
                        "(status nominal means there is no STEP file: skip the import and measurement), or pass path=<absolute path of an attached file>.")
    got = _latest_obtained(project)
    if got.get("path"):
        return got.get("ref_id", ""), got["path"], False, got.get("sha256", "")
    raise ToolError("import_reference_part: neither 'id' nor 'path' was given and no reference file was obtained yet; call source_reference_part "
                    "first, or pass path=<absolute path of an attached STEP/STL file>.")


@tool(
    "import_reference_part",
    "Import the sourced (or user-attached) reference file into the document as ONE hidden App::Part named Ref_<ID>. Returns a bounded "
    "summary: solid count, overall bounding box, the largest solids, units.",
    {"type": "object", "properties": {
        "doc_name": {"type": "string"},
        "id": {"type": "string", "description": "Catalog id whose file was obtained by source_reference_part"},
        "path": {"type": "string", "description": "Alternative: absolute path of an attached STEP/STL file"},
        "name": {"type": "string", "description": "Object name; default Ref_<ID in upper case>"}},
     "required": ["doc_name"]},
    category="source", mutating=True, geometry=True, phases=("SOURCE",),
)
def _import_reference_part(doc_name: str, id: str = "", path: str = "", name: str = ""):  # noqa: A002
    """Import the reference file as ``Ref_<ID>`` (hidden ``App::Part``, normalized into the board frame) and report a bounded summary.

    ``id`` finds the file ``source_reference_part`` obtained; ``path`` is an attached file (the reference then counts as
    user-attached). A ``path`` that is the obtained file itself is treated like ``id``. Raises ``ToolError`` for everything the model can fix.
    """
    ctx = current_context()
    project = ctx.project
    ref_id = str(id or "").strip().lower()
    target = str(path or "").strip()
    if target.startswith("~"):
        target = os.path.expanduser(target)
    label = name.strip() if name else ""
    if label and not fetch.valid_object_name(label):
        raise ToolError(f"import_reference_part: invalid object name {label!r}; use letters, digits and underscores, starting with a letter "
                        "(for example 'Ref_PI4').")
    ref_id, file_path, attached, sha_known = _resolve_import_source(project, ref_id, target)    # pure checks first, then the document
    kind, facts, error = fetch.check_importable(file_path)
    if error:
        raise ToolError(f"import_reference_part: {error}.")
    actual = _require_doc(doc_name, "import_reference_part")
    entry, board, expected = None, None, None
    try:
        entry = catalog.get(ref_id)
        board = catalog.load_board(ref_id) if entry.get("board") else None
        expected = next((s["expected_bbox"] for s in entry.get("step_urls") or [] if s.get("expected_bbox")), None)
    except (KeyError, ValueError):
        entry = None
    trust = fetch.trust_info(entry, facts.get("sha256") or sha_known) if entry is not None and not attached else {}
    strict = trust.get("pin") == "mismatch"          # a catalog file the vendor changed: imported only if its outline still matches
    if not attached and target:
        ctx.state["args_override"] = {"id": ref_id, "path": ""}          # the file line of SOURCE RESULT is not a user attachment (5.5 item 4)

    def task():
        from .refparts import importref

        try:
            doc = _doc_by_name(actual)
            if doc is None:
                return {"error": f"document '{actual}' is not open"}
            return importref.import_reference(doc, file_path, ref_id=ref_id, name=label, board=board, expected_bbox=expected,
                                              sha256=facts.get("sha256", sha_known), strict_identity=strict,
                                              trust_label=trust.get("label", ""))
        except importref.ReferenceImportError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"import_reference_part: the import failed: {type(exc).__name__}: {exc}"}

    summary, failure = _gui_call(task, "import_reference_part", _import_timeout(facts.get("bytes", 0)))
    if failure:
        raise ToolError(f"import_reference_part: {failure}")
    if not isinstance(summary, dict) or summary.get("error"):
        message = (summary or {}).get("error") or "the importer returned nothing"
        raise ToolError(message if message.startswith("import_reference_part") else f"import_reference_part: {message}")
    status = "user-attached" if attached else "step-imported"
    ref_name = (entry or {}).get("name") or ref_id
    base = os.path.basename(file_path)
    noun = ("mesh object" if kind == "mesh" else "solid") + ("" if summary["solids"] == 1 else "s")
    claim = (f"Imported {base} as {summary['object']}: {summary['solids']} {noun}, "
             f"bounding box {' x '.join(_mm(v) for v in summary['bbox'].get('size', []))} mm")
    note = summary.get("frame", {}).get("description", "")
    if summary.get("identity"):
        note += f". Identity: {summary['identity']['detail']}"
    label = trust.get("label", "")
    if label:
        claim += f" [{label}]"
    source = f"user attachment: {base}" if attached else f"import_reference_part: {base}"
    if label:                           # the Build Summary prints the first 90 characters of this: the trust level must come first
        source = f"{status} ({label}): {source}"
    if summary.get("pin_check"):
        note += f". {summary['pin_check']['text']}"
    fields = {"kind": "attachment" if attached else "reference", "claim": claim, "status": "attached" if attached else "obtained",
              "source": source, "sha256": facts.get("sha256", ""),
              "bytes": facts.get("bytes", 0), "detected_type": _provenance_type(facts.get("detected_type", "")),
              "confidence": "datasheet" if trust.get("trust") == "official" and trust.get("pin") == "match" else "community",
              "note": note[:600]}
    if attached:
        fields["local_path"] = file_path
    entry_id, record_note = _record(project, fields, ref_id, ref_name, status)
    _set_extra(ctx, status="imported", reference_status=status, id=ref_id, object=summary["object"], provenance=entry_id,
               attached=attached, kind=kind, trust=trust.get("trust", ""), trust_label=label, pin=trust.get("pin", ""))
    _turnlog.log_event("reference_import", id=ref_id, object=summary["object"], file_type=kind, solids=summary["solids"], attached=attached)
    return _import_text(summary, kind, actual, ref_id, attached, entry_id, record_note, label)


def _doc_by_name(name: str):
    import FreeCAD as App

    docs = App.listDocuments()
    if name in docs:
        return docs[name]
    lowered = {n.lower(): n for n in docs}
    return docs[lowered[name.lower()]] if name.lower() in lowered else None


def _import_text(summary: dict, kind: str, doc_name: str, ref_id: str, attached: bool, entry_id: str, record_note: str,
                 trust_label: str = "") -> str:
    box, obj = summary.get("bbox") or {}, summary["object"]
    size = " x ".join(_mm(v) for v in box.get("size", []))
    lines = []
    if kind == "mesh":
        lines.append(f"Imported the mesh as {obj} (hidden App::Part) in '{doc_name}': {summary['facets']} triangles, overall bounding box {size} mm, units mm.")
    else:
        lines.append(f"Imported {obj} (hidden App::Part) in '{doc_name}': {summary['solids']} solid{'s' if summary['solids'] != 1 else ''}, "
                     f"overall bounding box {size} mm (x {_mm(box['min'][0])}..{_mm(box['max'][0])}, y {_mm(box['min'][1])}..{_mm(box['max'][1])}, "
                     f"z {_mm(box['min'][2])}..{_mm(box['max'][2])}), units mm.")
        if summary.get("largest"):
            lines.append("Largest solids: " + "; ".join(f"{s['name']} {' x '.join(_mm(v) for v in s['size'])}" for s in summary["largest"][:5]) + ".")
    lines.append(f"Frame: {summary.get('frame', {}).get('description', '')}.")
    ident = summary.get("identity")
    if ident:
        lines.append(("Identity: " if ident["ok"] else "WARNING identity: ") + ident["detail"] + ("." if not ident["detail"].endswith(".") else ""))
        if not ident["ok"]:
            lines.append("Tell the user the file may be the wrong part and continue only if they confirm.")
    for warning in summary.get("warnings") or []:
        if not (ident and warning == ident["detail"]):
            lines.append(f"Warning: {warning}.")
    if trust_label:
        lines.append(f"Trust: {trust_label}; say so when you report dimensions taken from this model.")
    lines.append(f"reference: {ref_id} ({'user-attached' if attached else 'step-imported'}); provenance: "
                 + (entry_id or f"none ({record_note or 'no project is bound'})"))
    lines.append(f"Next: measure_reference_part(doc_name=\"{doc_name}\", write_params=true) for exact board dimensions"
                 + ("." if kind == "brep" else " (a mesh is only bounded, not measured)."))
    return "\n".join(lines)


# --- measure_reference_part --------------------------------------------------------------------------------------

@tool(
    "measure_reference_part",
    "Measure the imported reference: bounding box, PCB plane and thickness, mounting holes (cylindrical faces), and connector "
    "candidates with side, offset, width, height and overhang. With write_params=true the board dimensions are written into the "
    "Params sheet with source step_measurement.",
    {"type": "object", "properties": {
        "doc_name": {"type": "string"},
        "object": {"type": "string", "description": "Reference object; default the latest Ref_*"},
        "write_params": {"type": "boolean", "default": False},
        "top_n": {"type": "integer", "default": 8, "minimum": 1, "maximum": 30}},
     "required": ["doc_name"]},
    category="source", mutating=True, phases=("SOURCE", "PARAMS"),
)
def _measure_reference_part(doc_name: str, object: str = "", write_params: bool = False, top_n: int = 8):  # noqa: A002
    """Measure the reference in the board frame; optionally write ``Params`` (``step_measurement``) and record the measurement.

    Works on the exact B-rep of a STEP / BREP reference (and, with fewer results, on an STL mesh). The nominal board built by the kit is
    refused (there is nothing to measure). The frame of an imported reference is already normalized by ``import_reference_part``; a
    reference that is not gets it applied here and the result says so. Raises ``ToolError`` for everything the model can fix.
    """
    ctx = current_context()
    project = ctx.project
    actual = _require_doc(doc_name, "measure_reference_part")
    wanted = str(object or "").strip()
    count = max(1, min(int(top_n or 8), 30))
    write = bool(write_params)

    def task():
        try:
            return _measure_task(actual, wanted, write, count)
        except Exception as exc:  # noqa: BLE001 - a GUI-thread task must hand back an error, not raise
            return {"error": f"measure_reference_part: {type(exc).__name__}: {exc}"}

    result, failure = _gui_call(task, "measure_reference_part")
    if failure:
        raise ToolError(f"measure_reference_part: {failure}")
    if not isinstance(result, dict) or result.get("error"):
        message = (result or {}).get("error") or "the measurement returned nothing"
        raise ToolError(message if message.startswith("measure_reference_part") else f"measure_reference_part: {message}")
    return _measure_text(ctx, project, result, actual, write)


def _find_reference(doc, wanted: str):
    """The reference object to measure: ``wanted`` (name or label) or the latest ``Ref_*`` / ``OscarRole == ref`` ``App::Part``."""
    if wanted:
        obj = doc.getObject(wanted)
        if obj is None:
            hits = [o for o in doc.Objects if o.Label.lower() == wanted.lower() or o.Name.lower() == wanted.lower()]
            obj = hits[0] if hits else None
        if obj is None:
            refs = [o.Name for o in doc.Objects if o.TypeId == "App::Part" and o.Name.startswith("Ref_")]
            raise ValueError(f"measure_reference_part: '{wanted}' is not an object of '{doc.Name}'; reference objects: {', '.join(refs) or 'none'}")
        return obj
    refs = [o for o in doc.Objects if o.TypeId == "App::Part" and (o.Name.startswith("Ref_") or getattr(o, "OscarRole", "") == "ref")]
    if not refs:
        raise ValueError(f"measure_reference_part: '{doc.Name}' has no reference part (Ref_*); call import_reference_part first "
                         "(with nominal data there is no STEP file to measure)")
    return refs[-1]


def _measure_task(doc_name: str, wanted: str, write: bool, top_n: int) -> dict:
    """GUI thread: find the reference, normalize its frame when needed, measure, optionally write ``Params``; one undo transaction."""
    from .refparts import measure, measure_mesh, measure_params

    doc = _doc_by_name(doc_name)
    if doc is None:
        return {"error": f"measure_reference_part: document '{doc_name}' is not open"}
    root = _find_reference(doc, wanted)
    props = getattr(root, "PropertiesList", [])
    if "OscarKit" in props and getattr(root, "OscarKit", "") and "OscarRefSha256" not in props:
        return {"error": f"measure_reference_part: '{root.Name}' is the nominal board built from catalog data, not an imported file; there is "
                         "nothing to measure. Import a STEP file first (source_reference_part, then import_reference_part)"}
    ref_id = getattr(root, "OscarRefId", "") if "OscarRefId" in props else ""
    ref_id = ref_id or re.sub(r"^ref_", "", root.Name, flags=re.IGNORECASE).lower()
    board = expected = None
    try:
        entry = catalog.get(ref_id)
        board = catalog.load_board(ref_id) if entry.get("board") else None
        expected = next((s["expected_bbox"] for s in entry.get("step_urls") or [] if s.get("expected_bbox")), None)
    except (KeyError, ValueError):
        pass
    is_ref_part = root.TypeId == "App::Part" and ("OscarRole" in props and getattr(root, "OscarRole", "") == "ref" or root.Name.startswith("Ref_"))
    mesh_like = any(getattr(o, "TypeId", "") == "Mesh::Feature" for o in measure._members(root))
    previous = doc.UndoMode
    doc.UndoMode = 1
    doc.openTransaction("Oscar: measure reference")
    committed = False
    try:
        if mesh_like:
            m = measure_mesh.analyze_mesh(root, top_n=top_n, object_name=root.Name)
            text, sync, frame_applied = measure_mesh.render_text(m), {"written": [], "skipped": [], "diff": [], "names": [], "sheet": False}, False
        else:
            solids = measure.collect_solids(root)
            frame = measure.fit_frame(solids, board)
            frame_applied = False
            if frame.found and not frame.is_identity and is_ref_part:
                root.Placement = frame.placement * root.Placement
                doc.recompute()
                solids, frame_applied = measure.collect_solids(root), True
                frame = measure.fit_frame(solids, board)
            m = measure.analyze(solids, board=board, expected_bbox=expected, top_n=top_n, frame=frame, object_name=root.Name)
            sync = measure_params.sync(doc, m, write=write)
            text = measure.render_text(m, top_n=top_n)
        committed = True
    finally:
        if committed:
            doc.commitTransaction()
        else:
            doc.abortTransaction()
        doc.UndoMode = previous
    return {"text": text, "m": m.to_dict(), "sync": sync, "object": root.Name, "frame_applied": frame_applied, "is_ref_part": is_ref_part,
            "ref_id": ref_id, "kind": m.kind, "sha256": getattr(root, "OscarRefSha256", "") if "OscarRefSha256" in props else "",
            "trust_label": getattr(root, "OscarRefTrust", "") if "OscarRefTrust" in props else ""}


def _params_lines(sync: dict, write: bool, has_pcb: bool) -> list:
    lines = []
    if write:
        if sync["written"]:
            lines.append("Params: wrote " + ", ".join(sync["written"]) + " (source step_measurement; pcb_t, tallest_z and bottom_z are written together).")
        elif not has_pcb:
            lines.append("Params: nothing written (no PCB was found, so there are no board dimensions to write).")
        if sync["skipped"]:
            lines.append("Params skipped: " + "; ".join(f"{s['name']} ({s['why']})" for s in sync["skipped"]) + ".")
    if sync["diff"]:
        lines.append(("Params differ from the measurement (now updated): " if write and sync["written"] else
                      "Params differ from the measurement: ") + ", ".join(
            f"{d['name']} sheet {_mm(d['sheet'])} vs measured {_mm(d['measured'])} (source {d['source'] or '?'})" for d in sync["diff"][:8])
            + ("" if write else "; apply them with write_params=true, then rebuild the affected stages") + ".")
    return lines


def _measure_text(ctx, project, result: dict, doc_name: str, write: bool) -> str:
    m = result["m"]
    sync = result["sync"]
    status_now = _reference_status(project)
    wanted = "step-measured" if (result["kind"] == "brep" and m.get("pcb") and result["is_ref_part"]) else ""
    ledger_status = getattr(ctx.ledger, "reference_status", "") if ctx.ledger is not None else ""
    base_status = status_now if status_now != "none" else (ledger_status or "none")
    new_status = "step-measured" if (wanted and base_status == "step-imported") else base_status
    claim = f"Measured {result['object']}: "
    if m.get("pcb"):
        p = m["pcb"]
        claim += f"PCB {_mm(p['length'])} x {_mm(p['width'])} x {_mm(p['thickness'])} mm"
        if m.get("mounting"):
            claim += f", {m['mounting']['count']} mounting holes d{_mm(m['mounting']['d'])}"
        claim += f", {len(m.get('connectors') or [])} connector candidates"
    else:
        claim += "bounding box " + " x ".join(_mm(v) for v in (m.get("bbox") or {}).get("size", []))
    note = str((m.get("frame") or {}).get("description", ""))
    label = "" if "user-attached" in (status_now, ledger_status) else str(result.get("trust_label") or "")
    source = f"measure_reference_part on {result['object']} ({'exact B-rep' if result['kind'] == 'brep' else 'mesh bounding box'})"
    if label:                           # never state a measurement of a community model without saying what the model is
        claim += f" [{label}]"
        if new_status in ("step-imported", "step-measured"):
            source = f"{new_status} ({label}): {source}"
    fields = {"kind": "measurement", "claim": claim[:600], "status": "obtained", "detected_type": "step" if result["kind"] == "brep" else "stl",
              "source": source, "sha256": result.get("sha256", ""), "confidence": "measured" if result["kind"] == "brep" and m.get("pcb") else "approximate",
              "note": (note + (f" Params written: {', '.join(sync['written'])}." if sync["written"] else ""))[:600]}
    if len(fields["sha256"]) != 64:
        fields["sha256"] = ""
    entry_id, record_note = _record(project, fields, result["ref_id"], _ref_name(result["ref_id"]), new_status if new_status != base_status else "")
    extra = {"status": "measured", "reference_status": new_status, "object": result["object"], "kind": result["kind"], "provenance": entry_id,
             "written": list(sync["written"])}
    _set_extra(ctx, **extra)
    if write and sync.get("names"):
        ctx.state.setdefault("extra", {})["param_names"] = list(sync["names"])
    lines = [result["text"]]
    if result["frame_applied"]:
        lines.append(f"Frame applied to {result['object']} (its Placement now maps the imported file into the board frame).")
    elif not result["is_ref_part"]:
        lines.append(f"Note: '{result['object']}' is not a reference part, so its frame was not changed; the numbers above are in the board frame.")
    lines.extend(_params_lines(sync, write, bool(m.get("pcb"))))
    lines.append(f"reference_status: {new_status}; provenance: " + (entry_id or f"none ({record_note or 'no project is bound'})"))
    if not write and m.get("pcb") and result["kind"] == "brep":
        lines.append(f"Next: measure_reference_part(doc_name=\"{doc_name}\", write_params=true) writes the board dimensions into Params, then enclosure_spec.")
    return "\n".join(lines)
