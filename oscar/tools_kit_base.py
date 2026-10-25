"""Shared plumbing of the kit tools (``tools_kit`` and ``tools_kit_params``).

* GUI hand-off: ``gui_call`` runs a function on the GUI thread (inline in console runs) and turns a dispatcher failure into a
  ``ToolError``; the context is captured BEFORE dispatching and no lock is held across it.
* The spec store: the EnclosureSpec dict lives in the bound Project (``spec.json`` through ``save_spec`` / ``read_spec``). A bare
  ``run_tool`` call without a Project (a test, the bridge) keeps it in a small in-process cache keyed by document name, so the
  kit tools work in both cases. The Project is the source of truth when it has a spec.
* ``valid_subset``: the "apply the defaults plus whatever part of the patch is valid" rule of ``enclosure_spec``.
* Small formatters and the ``Params`` name list that every parameter tool reports through ``ctx.state["extra"]``.

Pure Python at import (FreeCAD only inside the functions that need it), so ``oscar.tools`` can import the tool modules in any
interpreter.
"""

import copy
import difflib
import re
import threading
from typing import Any, Callable

from . import turnlog as _turnlog
from .kit import spec as _spec
from .tools import ToolError

GUI_TIMEOUT_S = 240.0
DEFAULT_BOARD = "rpi4b"

_SPECS: "dict[str, dict]" = {}
_OWNER: "dict[str, str]" = {}      # Project root -> name of the document its spec was last stored for
_LOCK = threading.Lock()
_MAX_CACHE = 64


# --------------------------------------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------------------------------------

def fmt(x: Any) -> str:
    """A number for model-facing text: up to 3 decimals, no trailing zeros (``2.4``, ``94.9``, ``0``)."""
    try:
        text = f"{float(x):.3f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(x)
    return "0" if text in ("", "-0") else text


def limit_list(items: "list[str]", cap: int = 6) -> str:
    """``a, b, c`` with ``and N more`` after ``cap`` entries."""
    items = [str(i) for i in items]
    return ", ".join(items[:cap]) + (f" and {len(items) - cap} more" if len(items) > cap else "")


def suggest(name: str, choices: "list[str]") -> str:
    """`` (similar: a, b)`` for the closest ``choices`` to ``name`` (empty when nothing is close)."""
    close = difflib.get_close_matches(str(name), list(choices), n=3, cutoff=0.6)
    return f" (similar: {', '.join(close)})" if close else ""


# --------------------------------------------------------------------------------------------------------------------
# GUI hand-off and documents
# --------------------------------------------------------------------------------------------------------------------

def gui_call(tool_name: str, fn: Callable[[], Any], timeout: float = GUI_TIMEOUT_S) -> Any:
    """``fn()`` on the GUI thread; raises ``ToolError`` with the failure text when the dispatcher fails (timeout, Stop, exception)."""
    try:
        from .cad import dispatch

        result, failed = dispatch.dispatch_stoppable(fn, timeout, tool_name)
    except Exception as exc:  # noqa: BLE001 - a dispatcher problem is a tool failure, not a crash
        raise ToolError(f"{tool_name}: {type(exc).__name__}: {exc}") from None
    if failed:
        raise ToolError(f"{tool_name}: {dispatch.failure_text(result)}")
    return result


def require_doc(doc_name: Any, tool_name: str) -> str:
    """The real name of the open document ``doc_name`` (``ToolError`` listing the open documents otherwise)."""
    name = str(doc_name or "").strip()
    if not name:
        raise ToolError(f"{tool_name}: 'doc_name' is empty; use the name returned by create_document")
    from . import tools

    return tools._require_document(name)


def open_doc(name: str) -> Any:
    """The ``App.Document`` called ``name`` or None (GUI thread)."""
    import FreeCAD as App

    return App.listDocuments().get(name)


def ctx_project(ctx: Any) -> Any:
    return getattr(ctx, "project", None)


def project_call(ctx: Any, what: str, fn_name: str, *args: Any, **kwargs: Any) -> "tuple[Any, str]":
    """``project.<fn_name>(...)`` guarded: ``(value, "")`` or ``(None, error text)``; failures are logged, never raised."""
    project = ctx_project(ctx)
    if project is None:
        return None, "no project is bound to this chat"
    try:
        return getattr(project, fn_name)(*args, **kwargs), ""
    except Exception as exc:  # noqa: BLE001 - recording must never fail a tool call
        _turnlog.log_event("project_error", hook=what, error=f"{type(exc).__name__}: {exc}"[:200])
        return None, f"{type(exc).__name__}: {exc}"[:200]


def reference_status(ctx: Any) -> str:
    """The reference status of this turn: the Project's, else the Ledger's, else ``none``."""
    for holder in (ctx_project(ctx), getattr(ctx, "ledger", None)):
        try:
            status = str(getattr(holder, "reference_status", "") or "")
        except Exception:  # noqa: BLE001
            status = ""
        if status:
            return status
    return "none"


def record_assumption(ctx: Any, text: str) -> str:
    """Write an ``assumption`` note into the Project (``""`` when none is bound or the write failed)."""
    entry, _err = project_call(ctx, "note", "note", "assumption", text[:2000])
    return entry or ""


# --------------------------------------------------------------------------------------------------------------------
# the spec store
# --------------------------------------------------------------------------------------------------------------------

def load_spec(ctx: Any, doc: str = "") -> "dict | None":
    """The stored EnclosureSpec dict (a copy) or None: the Project's ``spec.json`` (and ONLY that while a Project is
    bound and readable), else the in-process cache for ``doc`` / ``""`` (project-less sessions, replays)."""
    data, err = project_call(ctx, "read_spec", "read_spec")
    if isinstance(data, dict) and data:
        return copy.deepcopy(data)
    if ctx_project(ctx) is not None and not err:
        # The Project is the authority of this chat: it has no spec yet. The process-wide cache holds whatever
        # ANOTHER chat or an earlier document of the same name stored last (its wall_t, its material ...), and
        # must never seed a fresh build.
        return None
    with _LOCK:
        found = _SPECS.get(doc) or _SPECS.get("")
    return copy.deepcopy(found) if found else None


def store_spec(ctx: Any, doc: str, spec: dict) -> str:
    """Save ``spec``: through ``project.save_spec`` when a Project is bound and in the in-process cache. Returns a note
    (``""`` when everything worked, else why the Project copy was not written)."""
    data = copy.deepcopy(spec)
    with _LOCK:
        if len(_SPECS) >= _MAX_CACHE and doc not in _SPECS:
            _SPECS.pop(next(iter(_SPECS)))
        _SPECS[doc] = data
        _SPECS[""] = copy.deepcopy(data)
        root = _project_root(ctx)
        if root and doc:
            if len(_OWNER) >= _MAX_CACHE and root not in _OWNER:
                _OWNER.pop(next(iter(_OWNER)))
            _OWNER[root] = doc
    if ctx_project(ctx) is None:
        return ""
    _value, err = project_call(ctx, "save_spec", "save_spec", copy.deepcopy(data))
    return f"(the spec could not be written to the project: {err})" if err else ""


def forget_specs() -> None:
    """Empty the in-process spec cache and the owner map (tests)."""
    with _LOCK:
        _SPECS.clear()
        _OWNER.clear()


def _project_root(ctx: Any) -> str:
    return str(getattr(ctx_project(ctx), "root", "") or "")


def spec_owner(ctx: Any) -> str:
    """The name of the document the bound Project's spec was last stored for (``""`` when unknown, e.g. after a restart)."""
    root = _project_root(ctx)
    with _LOCK:
        return _OWNER.get(root, "") if root else ""


def _has_kit(doc: Any, kit_id: str) -> bool:
    """Does ``doc`` hold built kit geometry? The tagged ``Params`` sheet alone does not count: every ``set_parameters`` call tags the
    sheet it creates (``ensure_sheet``), in a bracket document as well."""
    from .kit import geometry as G

    try:
        return any(G.tags_of(o).get("OscarRole") != "params" for o in G.kit_objects(doc, kit_id))
    except Exception:  # noqa: BLE001 - an unreadable document is simply not a kit document
        return False


def spec_applies(doc: Any, spec: "dict | None", owner: str = "") -> bool:
    """Does the stored ``spec`` describe ``doc``? (GUI thread.) A Project keeps ONE spec for the whole chat, but a chat may hold other
    documents (a bracket with its own ``Params`` sheet and a ``wall_t`` row): their edits must be neither validated against the
    enclosure's rules nor written back into it. True when ``doc`` carries kit objects, or is the document the spec was stored for
    (``owner``); with an unknown owner, when no OTHER open document carries the kit (so a spec set before the first build still applies)."""
    if not spec or doc is None:
        return False
    kit_id = str(spec.get("kit_id") or "enc1")
    if _has_kit(doc, kit_id):
        return True
    if owner:
        return owner == getattr(doc, "Name", "")
    try:
        import FreeCAD as App

        return not any(other is not doc and _has_kit(other, kit_id) for other in App.listDocuments().values())
    except Exception:  # noqa: BLE001
        return True


# --------------------------------------------------------------------------------------------------------------------
# board and spec helpers
# --------------------------------------------------------------------------------------------------------------------

def known_boards() -> "list[str]":
    return _spec._known_board_ids()


def _board_key(text: Any) -> str:
    """``text`` lower-cased with ``-`` / ``_`` / runs of spaces folded to one space (``'RPi-4_B'`` -> ``'rpi 4 b'``)."""
    return " ".join(re.sub(r"[-_\s]+", " ", str(text or "").strip().lower()).split())


def resolve_board(text: Any) -> str:
    """The catalog board id that ``text`` names: an id, or EXACTLY a board name or alias (case, ``-``, ``_`` and spacing are ignored), so
    'Raspberry Pi 4 Model B' and 'RPi 4' find ``rpi4b``. Anything else, and a name that two boards share, comes back stripped and
    lower-cased for the validator to report (a board is never guessed). Never raises."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    key = _board_key(raw)
    try:
        from .refparts import catalog

        hits = [str(b.get("id")) for b in catalog.list_boards()
                if key in {_board_key(n) for n in (b.get("id"), b.get("name"), *(b.get("aliases") or [])) if n}]
    except Exception:  # noqa: BLE001 - a catalog problem must not hide the validator's own message
        hits = []
    return hits[0] if len(set(hits)) == 1 else raw.lower()


def load_board(board_id: str, tool_name: str) -> dict:
    """The board data of ``board_id`` or a ``ToolError`` that lists the known boards."""
    try:
        return _spec.load_board(board_id)
    except KeyError as exc:
        raise ToolError(f"{tool_name}: {exc.args[0] if exc.args else exc}") from None
    except (LookupError, ValueError) as exc:
        raise ToolError(f"{tool_name}: the board '{board_id}' cannot be used: {exc}") from None


def spec_errors(d: dict, board: "dict | None" = None) -> "tuple[bool, list[str], list[str]]":
    """``spec.validate`` that never raises (an internal failure is reported as one error)."""
    try:
        return _spec.validate(d, board)
    except Exception as exc:  # noqa: BLE001
        return False, [f"the spec could not be checked ({type(exc).__name__}: {exc})"], []


def normalized(d: dict) -> dict:
    """``d`` coerced through ``from_dict`` / ``to_dict`` (numeric strings become numbers, ids lower-cased)."""
    return _spec.to_dict(_spec.from_dict(d))


def flatten(d: Any, prefix: str = "") -> "dict[str, Any]":
    """Dotted leaf paths of a nested dict (lists are leaves)."""
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and v:
                out.update(flatten(v, path))
            else:
                out[path] = v
    elif prefix:
        out[prefix] = d
    return out


def nest(path: str, value: Any) -> dict:
    """``{"a": {"b": value}}`` for ``"a.b"``."""
    parts = path.split(".")
    node: Any = value
    for key in reversed(parts):
        node = {key: node}
    return node


def diff(before: dict, after: dict) -> "list[tuple[str, Any, Any]]":
    """``[(path, old, new)]`` for every leaf that differs."""
    a, b = flatten(before), flatten(after)
    out = []
    for path in list(a) + [p for p in b if p not in a]:
        if a.get(path) != b.get(path):
            out.append((path, a.get(path), b.get(path)))
    return out


def valid_subset(base: dict, patch: dict, board: "dict | None" = None) -> "tuple[dict, list[str], list[str]]":
    """``base`` plus every leaf of ``patch`` that keeps the spec valid: ``(spec, kept paths, dropped "path: reason")``.

    Leaves are tried in order; a leaf whose merge introduces a validation error that ``base`` did not have is dropped with the
    first such error as its reason. ``base`` itself is returned unchanged when it is not valid to begin with."""
    spec = copy.deepcopy(base)
    kept: list[str] = []
    dropped: list[str] = []
    _ok, base_errors, _w = spec_errors(spec, board)
    known = set(base_errors)
    for path, value in flatten(patch).items():
        trial = _spec.merge(spec, nest(path, value))
        try:
            ok, errors, _w = spec_errors(trial, board)          # the RAW merge: an unknown key is an error here, not silently stripped
        except Exception:  # noqa: BLE001
            ok, errors = False, ["could not be applied"]
        new = [e for e in errors if e not in known]
        if ok or not new:
            spec = _spec.to_dict(_spec.from_dict(trial))
            kept.append(path)
        else:
            dropped.append(f"{path}: {new[0]}")
    return spec, kept, dropped


def param_names(doc: Any, sheet: str = "Params") -> "list[str]":
    """The ``Params`` names present in ``doc`` (GUI thread; ``[]`` without a sheet)."""
    try:
        from .kit import params as _params

        return _params.Params(doc, sheet).names()
    except Exception:  # noqa: BLE001
        return []


def set_extra(ctx: Any, **values: Any) -> None:
    """``ctx.state["extra"]`` entries that ``run_tool`` moves into ``ToolOutcome.extra``."""
    ctx.state.setdefault("extra", {}).update(values)


def default_board_id(ctx: Any, doc_name: str = "") -> str:
    """The board a spec-less tool should default to: the Project's reference, a ``Ref_<ID>`` object of the document, the board the
    request names (``ledger.reference_hint``) or ``rpi4b``. The document scan runs on the GUI thread."""
    boards = known_boards()
    ref = ""
    try:
        ref = str(((getattr(ctx_project(ctx), "manifest", None) or {}).get("reference") or {}).get("id") or "").lower()
    except Exception:  # noqa: BLE001
        ref = ""
    if ref in boards:
        return ref
    if doc_name:
        def scan() -> str:
            doc = open_doc(doc_name)
            for obj in (doc.Objects if doc is not None else []):
                if obj.Name.startswith("Ref_") and obj.Name[4:].lower() in boards:
                    return obj.Name[4:].lower()
            return ""

        try:
            found = gui_call("enclosure_spec", scan, 20.0)
        except ToolError:
            found = ""
        if found:
            return found
    hint = str(getattr(getattr(ctx, "ledger", None), "reference_hint", "") or "")
    if hint:
        try:
            from .refparts import catalog

            for entry in catalog.match(hint, 3):
                if entry.get("id") in boards:
                    return entry["id"]
        except Exception:  # noqa: BLE001
            pass
    return DEFAULT_BOARD


# --------------------------------------------------------------------------------------------------------------------
# a spec for a tool that needs one
# --------------------------------------------------------------------------------------------------------------------

def default_spec_dict(board_id: str) -> dict:
    """The board-sized default spec of ``board_id`` as a dict (every catalogue board validates, 7.3)."""
    return _spec.to_dict(_spec.default_spec(board_id))


def spec_summary(d: dict, board: "dict | None" = None) -> str:
    """One line of the numbers a model needs: board, material, wall, clearance, lid style and the shell size."""
    parts = [f"board '{d.get('board')}'", str(d.get("material")), f"wall {fmt(d.get('wall_t'))} mm",
             f"clearance {fmt(d.get('clearance'))} mm", f"{(d.get('lid') or {}).get('style')} lid"]
    try:
        dv = _spec.derived(d, board if board is not None else _spec.load_board(str(d.get("board"))))
        parts.append(f"shell {fmt(dv['shell_len']['value'])} x {fmt(dv['shell_wid']['value'])} x {fmt(dv['shell_h']['value'])} mm")
    except Exception:  # noqa: BLE001 - the summary is a courtesy
        pass
    return ", ".join(parts)


def defaults_note(d: dict) -> str:
    """The sentence recorded (as an assumption) and shown when the board defaults were applied."""
    return (f"No enclosure spec was set, so the defaults for board '{d.get('board')}' were applied ({spec_summary(d)}); "
            "change them with enclosure_spec(action='set') or set_parameter.")


def ensure_spec(ctx: Any, doc_name: str, tool_name: str, *, board_id: str = "") -> "tuple[dict, str]":
    """``(spec dict, note)``: the stored spec (``note == ""``) or, when none exists, the defaults for ``board_id`` / the detected
    board, stored and recorded as an ``assumption`` (``note`` says so). A default spec that does not validate is a ``ToolError``."""
    stored = load_spec(ctx, doc_name)
    if stored:
        return stored, ""
    board_name = (board_id or default_board_id(ctx, doc_name)).strip().lower()
    load_board(board_name, tool_name)      # a ToolError listing the known boards for an unknown id
    d = default_spec_dict(board_name)
    ok, errors, _warns = spec_errors(d, load_board(board_name, tool_name))
    if not ok:
        raise ToolError(f"{tool_name}: the default spec of board '{board_name}' does not validate ({'; '.join(errors[:2])}); "
                        "set the fields with enclosure_spec(action='set').")
    note = defaults_note(d)
    store_spec(ctx, doc_name, d)
    record_assumption(ctx, note)
    return d, note


def merge_validate(current: dict, patch: dict) -> "tuple[dict | None, list[str], list[str]]":
    """``(candidate, errors, warnings)`` for ``current`` with ``patch`` merged in.

    The RAW merge is validated (so an unknown key such as ``wal_t`` is an error with a suggestion, not silently dropped by the
    coercion); ``candidate`` is the normalized spec, or None when there are errors or the merge itself failed."""
    try:
        raw = _spec.merge(current, patch)
    except Exception as exc:  # noqa: BLE001 - a patch of the wrong shape
        return None, [f"the spec could not be merged ({type(exc).__name__}: {exc}); pass fields like {{\"wall_t\": 2.4}}"], []
    ok, errors, warnings = spec_errors(raw)
    if not ok:
        return None, errors, warnings
    try:
        return normalized(raw), errors, warnings
    except Exception as exc:  # noqa: BLE001
        return None, [f"the spec could not be normalized ({type(exc).__name__}: {exc})"], warnings
