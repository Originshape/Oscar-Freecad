"""GUI-thread helpers of :mod:`oscar.project` (private; nothing else imports this module).

Every function here touches ``App.Document`` objects, shapes or views, so each one is GUI-thread ONLY:
``oscar.project`` calls them inside ``oscar.cad.dispatch.dispatch`` tasks (inline in console runs).
FreeCAD is imported lazily inside the functions so ``oscar.project`` stays importable without it.

The geometry predicates and the parameter-sheet reader are NOT defined here: ``is_geometry``,
``global_shape`` and ``read_param_table`` have exactly one definition, ``oscar.cad.health``,
reached through a guarded lazy import (``health_module``). A document is therefore judged the same way by
health, the Project tree / summary, the replay equality check and the export (Origin features have an
infinite bounding box, a shape inside an ``App::Part`` is in the container's local frame).
"""

import os
import uuid
import zipfile
from typing import Any
from . import fsretry as _fsretry

# ---------------------------------------------------------------------------
# guarded access to cad.health

_HEALTH: Any = False   # False = not loaded yet, None = absent


def health_module():
    """``oscar.cad.health`` or None when that module does not exist yet (a missing dependency INSIDE it raises)."""
    global _HEALTH
    if _HEALTH is False:
        try:
            from .cad import health  # noqa: PLC0415 - lazy by design

            _HEALTH = health
        except ModuleNotFoundError as e:
            if e.name != f"{__package__}.cad.health":
                raise
            _HEALTH = None
    return _HEALTH


# ---------------------------------------------------------------------------
# dispatching to the GUI thread

_SIGNATURES: dict[Any, bool] = {}   # dispatcher function -> accepts the ``should_stop`` keyword


def load_dispatch():
    """``oscar.cad.dispatch.dispatch`` (imported lazily: it pulls in Qt). Tests replace this function with one
    that returns a fake dispatcher."""
    from .cad import dispatch  # noqa: PLC0415

    return dispatch.dispatch


def _accepts_should_stop(fn) -> bool:
    key = getattr(fn, "__func__", fn)   # a bound method is a new object on every access; its function is stable
    if key not in _SIGNATURES:
        import inspect

        try:
            _SIGNATURES[key] = "should_stop" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            _SIGNATURES[key] = False
    return _SIGNATURES[key]


def gui_call(fn, timeout: float, op: str, should_stop=None) -> tuple[Any, str]:
    """Run ``fn`` on the GUI thread through ``cad.dispatch.dispatch``; returns ``(result, "")`` on success and
    ``(None, error text)`` on a failure, a timeout (the text contains ``did not complete within``) or a stop.
    Never raises. ``should_stop`` is forwarded when the dispatcher accepts it. No lock
    of any module may be held by the caller: the GUI thread must stay free to drain its queue."""
    try:
        dispatch = load_dispatch()
        kwargs = {"should_stop": should_stop} if should_stop is not None and _accepts_should_stop(dispatch) else {}
        result, failed = dispatch(fn, timeout=max(0.1, float(timeout)), op=op, **kwargs)
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    if failed:
        error = result.get("error") if isinstance(result, dict) else result
        return None, str(error or "GUI task failed")
    return result, ""


# ---------------------------------------------------------------------------
# documents and objects

_CONTAINER_TYPES = frozenset({"App::Part", "App::DocumentObjectGroup"})
_ORIGIN_TYPES = frozenset({"App::Origin", "App::Line", "App::Plane", "App::Point"})   # the Origin of a Part / Body
TEMP_PREFIX = "_oscar_tmp"


def resolve_doc(doc_name: str):
    """The open document ``doc_name`` names, tolerating the model's near-misses (exact, case-insensitive,
    unique prefix); ``""`` means the active document. None when there is no such document."""
    import FreeCAD as App

    docs = App.listDocuments()
    if not doc_name:
        return App.ActiveDocument
    if doc_name in docs:
        return docs[doc_name]
    lowered = {name.lower(): name for name in docs}
    if doc_name.lower() in lowered:
        return docs[lowered[doc_name.lower()]]
    matches = [name for name in docs if name.lower().startswith(doc_name.lower())]
    return docs[matches[0]] if len(matches) == 1 else None


def _health():
    """``oscar.cad.health``; raises ``RuntimeError`` when it cannot be imported (inside a GUI task that becomes a
    visible failure of that task, never a silent private re-implementation of the geometry rules)."""
    module = health_module()
    if module is None:
        raise RuntimeError("oscar.cad.health is not available")
    return module


def is_geometry(obj) -> bool:
    """The model-geometry predicate (``cad.health.is_geometry``): False for Origin features, containers without a
    shape, empty shapes and shapes with an infinite bounding box. Never raises."""
    try:
        return bool(_health().is_geometry(obj))
    except Exception:  # noqa: BLE001 - one odd object must not hide the others
        return False


def is_container(obj) -> bool:
    """An ``App::Part`` / group (FreeCAD 1.1 gives ``App::Part`` a compound ``Shape`` of its children: it
    must never be counted as a part of its own). The ``App::Origin`` that every Part / Body creates is NOT a
    container the user sees (it also has a ``Group`` attribute): it and its axis / plane / point features are
    plumbing, listed only among the document's object names."""
    tid = getattr(obj, "TypeId", "")
    if tid in _CONTAINER_TYPES:
        return True
    if tid in _ORIGIN_TYPES:
        return False
    return tid != "PartDesign::Body" and hasattr(obj, "Group")


def global_shape(obj):
    """The object's shape in the GLOBAL frame (``cad.health.global_shape``: a shape inside an App::Part is in the
    container's local frame otherwise)."""
    return _health().global_shape(obj)


def _solid_count(obj) -> int:
    try:
        return len(obj.Shape.Solids)
    except Exception:
        return 0


def geometry_objects(doc) -> list:
    """Model geometry of ``doc`` in document order: ``is_geometry`` objects that are not containers and not
    temporary scratch objects."""
    return [o for o in doc.Objects
            if not o.Name.startswith(TEMP_PREFIX) and not is_container(o) and is_geometry(o)]


def is_ref(obj) -> bool:
    """True for the hidden reference part: ``OscarRole == "ref"``, a ``Ref_*`` name, or a member of a ``Ref_*`` container."""
    if str(getattr(obj, "OscarRole", "") or "") == "ref" or obj.Name.startswith("Ref_"):
        return True
    frontier, seen = [obj], {obj.Name}
    for _ in range(4):
        parents = []
        for node in frontier:
            for parent in getattr(node, "InList", []) or []:
                if parent.Name in seen:
                    continue
                seen.add(parent.Name)
                if parent.Name.startswith("Ref_") or str(getattr(parent, "OscarRole", "") or "") == "ref":
                    return True
                parents.append(parent)
        frontier = parents
        if not frontier:
            break
    return False


def visible_tips(doc) -> list:
    """The visible solid TIPS of ``doc``: geometry objects with at least one solid that are visible, are not
    part of a reference (``Ref_*``) and that no other geometry object builds on (a boolean's inputs and the
    features inside a PartDesign Body are not tips; the boolean result and the Body are). Structural
    (InList based): headless sessions never hide a boolean's inputs, so ``Visibility`` alone cannot decide."""
    candidates = [o for o in geometry_objects(doc) if _solid_count(o) > 0 and not is_ref(o)]
    names = {o.Name for o in candidates}
    tips = []
    for obj in candidates:
        if not getattr(obj, "Visibility", True):
            continue
        if any(parent.Name in names for parent in (getattr(obj, "InList", []) or [])):
            continue
        tips.append(obj)
    return tips


def object_names(doc) -> tuple[list[str], list[str]]:
    """``(geometry-or-container names, every object name)`` of ``doc`` (the checkpoint sidecar's fields)."""
    primary = [o.Name for o in doc.Objects if is_container(o) or is_geometry(o)]
    return primary, [o.Name for o in doc.Objects]


def has_geometry(doc) -> bool:
    """True when ``doc`` holds at least one model-geometry object (a Params-only document does not)."""
    return any(True for _ in geometry_objects(doc))


def activate(doc) -> bool:
    """Make ``doc`` the active document of the App and, in a GUI session, of the Gui. True when it is active."""
    import FreeCAD as App

    try:
        App.setActiveDocument(doc.Name)
        if getattr(App, "GuiUp", False):
            import FreeCADGui as Gui

            Gui.setActiveDocument(doc.Name)
    except Exception:
        pass
    active = App.ActiveDocument
    return active is not None and active.Name == doc.Name


# ---------------------------------------------------------------------------
# saving

def save_copy(doc, dest_abs: str) -> tuple[bool, str]:
    """Atomically write ``doc`` as an FCStd to ``dest_abs`` with ``doc.saveCopy``.

    FreeCAD silently appends ``.FCStd`` to any name that does not end in it, writes there and returns None,
    so the temp name MUST end in ``.FCStd``; ``zipfile.is_zipfile`` is the only success signal; then
    ``os.replace``. A unique temp name also avoids the ``*.FCBak`` backup FreeCAD writes when ``saveCopy``
    overwrites an existing file. Returns ``(ok, error)``; never raises.
    """
    directory = os.path.dirname(dest_abs) or "."
    base = os.path.basename(dest_abs)
    stem = base[: -len(".FCStd")] if base.endswith(".FCStd") else base
    tmp = os.path.join(directory, f"{stem}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}.FCStd")
    try:
        os.makedirs(directory, exist_ok=True)
        doc.saveCopy(tmp)
        if not os.path.isfile(tmp) or not zipfile.is_zipfile(tmp):
            return False, "saveCopy did not produce an FCStd file"
        _fsretry.replace(tmp, dest_abs)
        return True, ""
    except Exception as e:  # noqa: BLE001 - reported, never raised into the agent
        return False, f"{type(e).__name__}: {e}"
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# snapshots

def is_blank_grid(rows: list[list[tuple[int, int, int]]], tolerance: int = 6, blank_ratio: float = 0.995) -> bool:
    """Blank-frame test on a sampled pixel grid (``rows`` of ``(r, g, b)``).

    A sample is "background" when every channel is within ``tolerance`` of the per-ROW median colour
    (tolerant of gradient and anti-aliased backgrounds); the frame is blank when more than ``blank_ratio``
    of all samples are background. An empty grid counts as blank.
    """
    total = background = 0
    for row in rows:
        if not row:
            continue
        median = tuple(sorted(px[c] for px in row)[len(row) // 2] for c in range(3))
        for px in row:
            total += 1
            if all(abs(px[c] - median[c]) <= tolerance for c in range(3)):
                background += 1
    return total == 0 or background / total > blank_ratio


def _sample_image(path: str, columns: int = 64, rows: int = 48):
    """A ``rows x columns`` grid of ``(r, g, b)`` samples of the PNG at ``path`` (None when it cannot be read)."""
    try:
        from PySide import QtGui

        image = QtGui.QImage(path)
        if image.isNull() or image.width() < 1 or image.height() < 1:
            return None
        width, height = image.width(), image.height()
        grid = []
        for gy in range(rows):
            y = min(height - 1, int((gy + 0.5) * height / rows))
            line = []
            for gx in range(columns):
                px = image.pixel(min(width - 1, int((gx + 0.5) * width / columns)), y)
                line.append(((px >> 16) & 255, (px >> 8) & 255, px & 255))
            grid.append(line)
        return grid
    except Exception:
        return None


def capture_snapshot(doc_name: str, abs_path: str, width: int, height: int) -> tuple[bool, str]:
    """The default ``Project`` snapshot function (GUI thread only): isometric capture of ``doc_name`` at
    ``width x height`` into ``abs_path`` through ``cad.views.capture``, which restores the user's camera.

    Gated on ``App.GuiUp`` first: in ``freecadcmd`` ``FreeCADGui`` has no ``getDocument`` /
    ``ActiveDocument`` and ``obj.ViewObject`` is None, so a headless session returns
    ``(False, "no 3D view (headless session)")`` without touching Gui. A blank frame (see
    ``is_blank_grid``) gets one retry after ``fit_view()`` + ``Gui.updateGui()``; still blank gives
    ``(False, "blank frame")``. ``cad.views.capture_checked`` does exactly that (and reports its own reason) when
    it exists; the private test below is the fallback for a build without it.
    """
    import FreeCAD as App

    if not getattr(App, "GuiUp", False):
        return False, "no 3D view (headless session)"
    try:
        import FreeCADGui as Gui

        from .cad import views
    except Exception as e:  # noqa: BLE001
        return False, f"no 3D view ({type(e).__name__})"
    checked = getattr(views, "capture_checked", None)
    if checked is not None:   # cad.views already does the blank-frame test + retry and says WHY it failed
        try:
            ok, reason = checked(abs_path, "Isometric", width, height, None, doc_name)
        except Exception as e:  # noqa: BLE001
            return False, f"capture failed: {type(e).__name__}: {e}"
        return (True, "") if ok else (False, reason or "no 3D view available")
    try:
        if not views.capture(abs_path, "Isometric", width, height, None, doc_name):
            return False, "no 3D view available"
        grid = _sample_image(abs_path)
        if grid is not None and is_blank_grid(grid):
            views.fit_view()
            Gui.updateGui()
            if not views.capture(abs_path, "Isometric", width, height, None, doc_name):
                return False, "no 3D view available"
            grid = _sample_image(abs_path)
            if grid is not None and is_blank_grid(grid):
                return False, "blank frame"
    except Exception as e:  # noqa: BLE001
        return False, f"capture failed: {type(e).__name__}: {e}"
    return True, ""


# ---------------------------------------------------------------------------
# parameters

def param_table(doc) -> dict:
    """``{"sheet": name | "", "rows": [...]}`` of the document's ``Params`` / ``Parameters`` sheet: exactly the
    rows of ``cad.health.read_param_table`` (the one reader of that sheet; each row is ``{"name", "value", "unit",
    "expr", "source", "note", "cell", "bound", "needs_rebuild"}``). GUI thread only."""
    table = _health().read_param_table(doc)
    return {"sheet": str(table.get("sheet", "") or ""), "rows": list(table.get("rows") or [])}


# ---------------------------------------------------------------------------
# model tree (Build Summary)

def model_tree(doc) -> list[dict]:
    """Rows ``{"name","type","solids","volume","bbox","tip"}`` for the model geometry of ``doc`` (sketches and
    other solid-less shapes are not listed): visible tips first, then by descending volume. The bounding box
    is ``[xmin, ymin, zmin, xmax, ymax, zmax]`` in the GLOBAL frame (a part inside an App::Part would
    otherwise report its container-local frame)."""
    tips = {o.Name for o in visible_tips(doc)}
    rows = []
    for obj in geometry_objects(doc):
        tid = getattr(obj, "TypeId", "")
        try:
            if tid == "Mesh::Feature":
                box = obj.Mesh.BoundBox
                rows.append({"name": obj.Name, "type": tid, "solids": 0, "volume": None,
                             "bbox": [box.XMin, box.YMin, box.ZMin, box.XMax, box.YMax, box.ZMax],
                             "tip": False})
                continue
            solids = _solid_count(obj)
            if solids == 0 and obj.Shape.ShapeType != "Shell":
                continue
            shape = global_shape(obj)
            box = shape.BoundBox
            rows.append({"name": obj.Name, "type": tid, "solids": solids,
                         "volume": float(shape.Volume) if solids else None,
                         "bbox": [box.XMin, box.YMin, box.ZMin, box.XMax, box.YMax, box.ZMax],
                         "tip": obj.Name in tips})
        except Exception:
            continue
    rows.sort(key=lambda r: (not r["tip"], -(r["volume"] or 0.0)))
    return rows


def live_volumes(doc) -> dict[str, float | None]:
    """``{object name: volume mm3 | None}`` of the ``is_geometry`` objects of ``doc`` that hold solids (None =
    no solid): the comparable fingerprint of a model, used by ``replay_summary`` and the rebuild check.
    Origin features and containers are excluded (their volume is uninitialised garbage)."""
    out: dict[str, float | None] = {}
    for obj in geometry_objects(doc):
        try:
            solids = _solid_count(obj)
            out[obj.Name] = float(obj.Shape.Volume) if solids else None
        except Exception:
            out[obj.Name] = None
    return out


# ---------------------------------------------------------------------------
# restore

def restore_from_file(doc, path: str, label: str) -> dict:
    """Replace the content of the LIVE document ``doc`` with the FCStd at ``path`` (GUI thread only).

    The checkpoint is opened HIDDEN (no extra MDI window), the live objects are removed (reverse creation
    order) and ``doc.copyObject`` brings the checkpoint's objects over: names, expressions and values come
    back and ``doc.Name`` never changes. The whole replacement is ONE undo transaction named
    ``Oscar restore <label>`` (``UndoMode`` is forced to 1 around it, otherwise ``abortTransaction`` undoes
    nothing in ``freecadcmd``). The temp document is always closed and the previous ``UndoMode`` restored;
    closing it leaves BOTH the App and the Gui without an active document, so the live document is
    re-activated afterwards (``reactivated``).

    Returns ``{"ok", "error", "committed", "reactivated", "names_after", "invalid"}``; ``committed`` False
    means the transaction was aborted and the document is as before.
    """
    import FreeCAD as App

    out: dict[str, Any] = {"ok": False, "error": "", "committed": False, "reactivated": False,
                           "names_after": [], "invalid": []}
    tmp = None
    prev_undo = doc.UndoMode
    in_transaction = False
    try:
        try:
            tmp = App.openDocument(path, True)
        except TypeError:   # builds whose openDocument has no `hidden` argument
            tmp = App.openDocument(path)
        doc.UndoMode = 1
        doc.openTransaction(f"Oscar restore {label}")
        in_transaction = True
        for name in [o.Name for o in reversed(doc.Objects)]:
            if doc.getObject(name) is not None:
                try:
                    doc.removeObject(name)
                except Exception:
                    pass
        doc.copyObject(list(tmp.Objects), False)
        doc.commitTransaction()
        in_transaction = False
        out["committed"] = True
    except Exception as e:  # noqa: BLE001 - reported through the result
        out["error"] = f"{type(e).__name__}: {e}"
        if in_transaction:
            try:
                doc.abortTransaction()
            except Exception:
                pass
    finally:
        if tmp is not None:
            try:
                App.closeDocument(tmp.Name)
            except Exception:
                pass
        try:
            doc.UndoMode = prev_undo
        except Exception:
            pass
        out["reactivated"] = activate(doc)
    if out["committed"]:
        try:
            doc.recompute()
        except Exception as e:  # noqa: BLE001
            out["error"] = f"recompute failed: {type(e).__name__}: {e}"
        out["names_after"] = [o.Name for o in doc.Objects]
        out["invalid"] = [o.Name for o in doc.Objects if "Invalid" in (getattr(o, "State", None) or [])]
        out["ok"] = not out["error"]
    else:
        out["names_after"] = [o.Name for o in doc.Objects]
    return out
