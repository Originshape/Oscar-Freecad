"""CadEngine — the operation surface behind Oscar's tools and the bridge.

Every public method is callable from any thread: document and view work is
ferried to the GUI thread by ``dispatch``; results follow one convention
everywhere — a dict with ``success`` plus either the payload or an ``error``
message models and bridge clients can read directly.

Object creation accepts plain JSON specs::

    {"Name": "Box", "Type": "Part::Box", "Properties": {...}, "Analysis": …}

Native engine types go straight through; parametric primitives (Draft) and
FEM objects are built through their workbench factories; anything else falls
back to the generic object creator.
"""

import base64
import os
import tempfile
import threading
import uuid
import zipfile
from typing import Any

import FreeCAD as App

from . import dispatch, imported, scripting, views
from .. import paths
from .headless import run as run_headless
from .properties import apply_properties, apply_properties_report, restore_properties
from .serialize import collect_objects, object_to_dict

ENGINE_VERSION = "0.1.0"

# Every undo transaction Oscar opens is labelled with one of these prefixes ("Oscar: ..." by txn_task,
# scripted execution, the feature-plan builder and background jobs; "Oscar ..." / "Oscar: ..." by the kit, the
# reference tools and the Project restore). revert_turn uses them to tell a turn's own undo steps from the user's.
TXN_PREFIX = "Oscar:"
TXN_PREFIXES = (TXN_PREFIX, "Oscar")


def is_oscar_step(label: Any) -> bool:
    """True when an undo-stack entry was opened by Oscar (its label starts with one of ``TXN_PREFIXES``)."""
    return str(label).startswith(TXN_PREFIXES)


# Undo steps that ``Document.undo()`` must never pop: FreeCAD 1.1 can crash (SIGSEGV in Transaction::apply) undoing the
# PartDesign transactions of a FeaturePlan build, and a Project restore copies whole PartDesign trees in. A turn that
# holds one is reverted from its turn checkpoint instead (``revert_turn`` reports ``undo_unsafe``).
UNDO_UNSAFE_PREFIXES = ("Oscar: plan ", "Oscar restore ")


def is_undo_unsafe(label: Any) -> bool:
    """True when an undo-stack entry must not be undone with ``Document.undo()`` (``UNDO_UNSAFE_PREFIXES``)."""
    return str(label).startswith(UNDO_UNSAFE_PREFIXES)

_engine: "CadEngine | None" = None
_engine_lock = threading.Lock()


def get_engine() -> "CadEngine":
    """The process-wide ``CadEngine`` (created on first use; thread-safe)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = CadEngine()
        return _engine


# ---------------------------------------------------------------------------
# factory helpers



def _draft_module():
    try:
        import Draft

        return Draft
    except ImportError:
        import draft

        return draft


def _draft_fn(draft, snake: str, camel: str):
    """Draft ships snake_case functions with camelCase legacy aliases."""
    fn = getattr(draft, snake, None) or getattr(draft, camel, None)
    if fn is None:
        raise ValueError(f"Draft factory '{snake}' not available")
    return fn


def _with_supported_kwargs(fn, kwargs: dict):
    """Drop kwargs the installed Draft signature does not know.

    Draft's keyword names differ between builds (close/closed,
    length/width); probing beats guessing.
    """
    import inspect

    try:
        params = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return fn(*kwargs.values())
    accepted = {k: v for k, v in kwargs.items() if k in params}
    return fn(**accepted)


def _draft_points(points) -> list:
    """Vertices from [{"x":..,"y":..,"z"?:..}, ...] or [[x, y(, z)], ...]."""
    vertices = []
    for point in points or []:
        if isinstance(point, dict) and "x" in point and "y" in point:
            coords = (point["x"], point["y"], point.get("z", 0))
        elif isinstance(point, (list, tuple)) and len(point) in (2, 3):
            coords = tuple(point) + (0,) * (3 - len(point))
        else:
            raise ValueError(
                f"point {point!r} is not {{x, y, z}} or [x, y, z] (z is optional)"
            )
        vertices.append(App.Vector(*(float(c) for c in coords)))
    return vertices


def _make_draft(doc, type_name: str, name: str, properties: dict):
    """Build the parametric primitives the Draft workbench implements."""
    draft = _draft_module()
    if type_name == "Draft::Circle":
        obj = _draft_fn(draft, "make_circle", "makeCircle")(float(properties.get("Radius", 5)))
    elif type_name == "Draft::Rectangle":
        fn = _draft_fn(draft, "make_rectangle", "makeRectangle")
        length = float(properties.get("Length", 10))
        height = float(properties.get("Height", 10))
        obj = _with_supported_kwargs(
            fn, {"length": length, "width": length, "height": height}
        )
    elif type_name in ("Draft::Polygon", "Draft::Wire"):
        vertices = _draft_points(properties.get("Points"))
        closed = bool(properties.get("Closed", type_name == "Draft::Polygon"))
        if vertices:
            # Explicit vertices become a wire; Draft's polygon factory only
            # knows (sides, radius) and would drop them.
            if len(vertices) < 2:
                raise ValueError(f"{type_name} needs at least 2 Points")
            fn = _draft_fn(draft, "make_wire", "makeWire")
            obj = _with_supported_kwargs(fn, {"pointslist": vertices, "closed": closed})
        elif type_name == "Draft::Polygon":
            faces = int(properties.get("FacesNumber", 6))
            if faces < 3:
                raise ValueError("a Draft::Polygon needs FacesNumber >= 3")
            fn = _draft_fn(draft, "make_polygon", "makePolygon")
            obj = _with_supported_kwargs(
                fn, {"nfaces": faces, "radius": float(properties.get("Radius", 5))}
            )
        else:
            raise ValueError(
                "Draft::Wire needs Points: a list of at least 2 {x, y, z} vertices"
            )
    else:
        return None
    obj.Label = name
    # Everything the factory did not consume (Placement, ViewObject, …)
    # still applies — it used to be dropped silently.
    consumed = {"Radius", "Length", "Height", "Points", "Closed", "FacesNumber"}
    rest = {k: v for k, v in properties.items() if k not in consumed}
    if rest:
        apply_properties(obj, rest)
    return obj


# The Gmsh mesh object has no C++ type of its own; the model spells it either
# way and ObjectsFem builds it with makeMeshGmsh.
_GMSH_TYPES = ("FemMeshGmsh", "MeshGmsh")


def _is_gmsh(type_name: str) -> bool:
    return type_name.split("::")[-1] in _GMSH_TYPES


def _make_fem(doc, type_name: str, name: str, properties: dict, analysis):
    """Build FEM objects through the FEM workbench factories."""
    import ObjectsFem

    short = type_name.split("::", 1)[1]
    # type suffix -> candidate factory names on ObjectsFem (first hit wins)
    maker_names = {
        "AnalysisPython": ("makeAnalysis",),
        "Analysis": ("makeAnalysis",),
        "MaterialCommon": ("makeMaterialCommon", "makeMaterialSolid", "makeMaterial"),
        "FemMeshGmsh": ("makeMeshGmsh",),
        "MeshGmsh": ("makeMeshGmsh",),
    }.get(short, (f"make{short}", f"make{short}Python"))
    maker = next(
        (getattr(ObjectsFem, candidate, None) for candidate in maker_names
         if getattr(ObjectsFem, candidate, None) is not None),
        None,
    )

    obj = maker(doc, name) if maker else doc.addObject(type_name, name)
    if obj is None:
        raise ValueError(f"FEM factory for {type_name} returned nothing")

    properties = {k: v for k, v in properties.items() if k != "Analysis"}
    gmsh = _is_gmsh(type_name)
    if gmsh and "Part" in properties and "Part" not in obj.PropertiesList:
        # the geometry link is 'Shape'; older FreeCAD called it 'Part'
        properties = {("Shape" if k == "Part" else k): v for k, v in properties.items()}
    apply_properties(obj, properties)

    if gmsh:
        if getattr(obj, "Shape", None) is None:
            raise ValueError(
                'a Gmsh mesh needs the solid to mesh: Properties {"Shape": "<object name>"}'
            )
        length_max = properties.get(
            "CharacteristicLengthMax", properties.get("ElementSizeMax", 10)
        )
        length_min = properties.get(
            "CharacteristicLengthMin", properties.get("ElementSizeMin", 0.1)
        )
        if hasattr(obj, "CharacteristicLengthMax"):
            obj.CharacteristicLengthMax = length_max
            obj.CharacteristicLengthMin = length_min
        elif hasattr(obj, "ElementSizeMax"):
            obj.ElementSizeMax = length_max
            obj.ElementSizeMin = length_min
        if analysis is not None:
            analysis.addObject(obj)
        from femmesh import gmshtools

        # The analysis is read from the mesh's parent group, which is why the
        # mesh joins it first. create_mesh() blocks until Gmsh exits and
        # reports nothing, so the outcome is read off the mesh itself.
        gmshtools.GmshTools(obj).create_mesh()
        if obj.FemMesh.NodeCount == 0:
            raise ValueError(
                f"Gmsh produced no nodes for '{obj.Shape.Name}'; check that it is a "
                "solid and that CharacteristicLengthMax/Min fit its size (the "
                "report view has Gmsh's own message)"
            )
    return obj


def _ccx_errors(fea) -> str:
    """The ``*ERROR`` lines CalculiX printed during a failed run (first few)."""
    text = f"{getattr(fea, 'ccx_stderr', '')}\n{getattr(fea, 'ccx_stdout', '')}"
    lines = [ln.strip(" *") for ln in text.splitlines() if "ERROR" in ln.upper()]
    return "; ".join(lines[:3])


def _valid(obj) -> bool:
    """Recompute sanity: no error state, and the object validates if it can."""
    states = getattr(obj, "State", []) or []
    if any(s in states for s in ("Invalid", "Error", "Touched")):
        if hasattr(obj, "isValid"):
            try:
                if not obj.isValid():
                    return False
            except Exception:
                pass
    return True


def _resolve_doc(doc_name: str):
    """Find a document by name, tolerating the model's near-misses:
    exact, then case-insensitive exact, then a unique case-insensitive
    prefix ("Gear_Sys" -> "Gear_System"). Returns the document or None."""
    docs = App.listDocuments()
    if doc_name in docs:
        return docs[doc_name]
    by_lower = {name.lower(): name for name in docs}
    exact = by_lower.get(doc_name.lower())
    if exact:
        return docs[exact]
    prefix_matches = [n for n in docs if n.lower().startswith(doc_name.lower())]
    if len(prefix_matches) == 1:
        return docs[prefix_matches[0]]
    return None


def _discard_created(doc, before: set[str], visibility: dict[str, bool]) -> None:
    """Roll back a failed create: remove every object the attempt added and
    restore visibility of existing objects (a boolean hides its inputs the
    moment its links are set). A retry then starts from a clean document —
    no stray half-built object, no '001'-suffixed rename."""
    added = [o.Name for o in doc.Objects if o.Name not in before]
    for name in reversed(added):
        try:
            if doc.getObject(name) is not None:
                doc.removeObject(name)
        except Exception:
            pass
    for name, visible in visibility.items():
        obj = doc.getObject(name)
        if obj is not None:
            try:
                if bool(obj.Visibility) != visible:
                    obj.Visibility = visible
            except Exception:
                pass
    try:
        doc.recompute()
    except Exception:
        pass


def _visibility(doc) -> dict[str, bool]:
    states = {}
    for obj in doc.Objects:
        try:
            states[obj.Name] = bool(obj.Visibility)
        except Exception:
            pass
    return states


def _link_objects(value):
    """The document objects inside a link value: an object, a (object, subnames) tuple, or a list of those."""
    if isinstance(value, App.DocumentObject):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _link_objects(item)


def _bodies(doc) -> list:
    return [o for o in doc.Objects if o.TypeId == "PartDesign::Body"]


def _active_body(doc):
    """The PartDesign Body the user has active in the GUI for ``doc``, or None (none active, another document in
    front, or a console session)."""
    try:
        import FreeCADGui as Gui

        if Gui.ActiveDocument.Document.Name == doc.Name:
            return Gui.ActiveDocument.ActiveView.getActiveObject("pdbody")
    except Exception:  # noqa: BLE001 - console session: no active view
        pass
    return None


def _requested_body(doc, requested: Any):
    """The ``PartDesign::Body`` the caller named (``body`` argument or ``Properties.Body``: a Name, then a Label),
    or None when none was named. Raises ValueError for a name that is not a Body; the message lists the Bodies."""
    if not requested:
        return None
    bodies = _bodies(doc)
    wanted = str(requested)
    chosen = next((o for o in bodies if o.Name == wanted), None) or next(
        (o for o in bodies if o.Label == wanted), None)
    if chosen is None:
        names = ", ".join(o.Name for o in bodies) or "none"
        raise ValueError(f"body '{wanted}' is not a PartDesign::Body of '{doc.Name}' "
                         f"(bodies: {names}); create it first with create_object PartDesign::Body")
    return chosen


def _target_body(doc, type_name: str, requested: Any, feature=None, just_made=None):
    """The ``PartDesign::Body`` a new object must be put into, or None (it stays top-level).

    ``requested`` is the ``body`` argument (a Name or Label) and always wins. A PartDesign feature without one goes
    into, in this order: the Body that already owns what it builds on (its Profile / BaseFeature / Support), the Body
    the agent just created (unless the user has since activated another Body), the Body the user has active in the
    GUI, the only Body, else the most recently created one (a model building part after part works on the newest).
    A feature that needs a Profile but has none, in a document with several Bodies, has nothing to follow: that call
    is ambiguous and ValueError lists the Bodies. A Sketcher object is routed only when a body was named. Raises
    ValueError for a name that is not a Body.

    ``feature`` is the object just created (its links are read) and ``just_made`` is (name of the last Body the agent
    created in ``doc``, name of the Body that was active then). A Body made through the API does not become the GUI's
    active one, so without ``just_made`` the Body the user had active earlier would take the feature meant for the new
    part; once the user switches Body that is their decision and it wins."""
    if requested:
        return _requested_body(doc, requested)
    if not (type_name.startswith("PartDesign::") and type_name != "PartDesign::Body"):
        return None
    bodies = _bodies(doc)
    if not bodies:
        return None
    if feature is not None:
        for prop in ("Profile", "BaseFeature", "Support"):
            for linked in _link_objects(getattr(feature, prop, None)):
                parent = linked.getParentGeoFeatureGroup()
                if parent is not None and parent.TypeId == "PartDesign::Body":
                    return parent
    if len(bodies) > 1 and feature is not None and "Profile" in (getattr(feature, "PropertiesList", None) or []) \
            and not list(_link_objects(getattr(feature, "Profile", None))):
        names = ", ".join(o.Name for o in bodies)
        raise ValueError(f"'{type_name}' needs a Body and '{doc.Name}' has several ({names}); "
                         "pass body=<name> to choose one")
    active = _active_body(doc)
    if just_made is not None:
        made = doc.getObject(just_made[0])
        if (made is not None and made.TypeId == "PartDesign::Body"
                and (active.Name if active is not None else None) == just_made[1]):
            return made
    if active is not None:
        return active
    return bodies[-1]


def _groups(container, obj) -> bool:
    """True when ``container`` merely groups ``obj`` (it holds it in its ``Group``)."""
    try:
        return obj in (getattr(container, "Group", None) or [])
    except Exception:  # noqa: BLE001
        return False


def _follows_in_body(user, target) -> bool:
    """True when ``user`` is only the next feature after ``target`` in its Body. That BaseFeature link is the Body's
    to manage: it relinks the chain (and moves the Tip) when ``target`` goes, so it is no reason to refuse a delete."""
    try:
        base = getattr(user, "BaseFeature", None)
        if base is None or base.Name != target.Name:
            return False
        body = target.getParentGeoFeatureGroup()
        owner = user.getParentGeoFeatureGroup()
        return (body is not None and body.TypeId == "PartDesign::Body"
                and owner is not None and owner.Name == body.Name)
    except Exception:  # noqa: BLE001
        return False


def _links_only(user, current) -> bool:
    """True when ``user`` links ``current`` without depending on it: a container that merely groups it, or the next
    feature of its Body (see ``_follows_in_body``)."""
    return _groups(user, current) or _follows_in_body(user, current)


def _dependents(obj) -> list:
    """Objects that depend on ``obj``, dependents before what they depend on (deletion order).

    Walks ``InList`` transitively but never through a container that only groups the object (a
    Body or Part holds its children; deleting a child must not count the Body as a dependent) nor through
    the next feature of the same Body (the Body relinks that chain itself)."""
    found: dict[str, Any] = {}
    frontier = [obj]
    while frontier:
        current = frontier.pop()
        for user in getattr(current, "InList", None) or []:
            if user.Name in found or user is obj or _links_only(user, current):
                continue
            found[user.Name] = user
            frontier.append(user)
    ordered: list = []
    pending = dict(found)
    while pending:
        ready = [o for o in pending.values()
                 if not any(u.Name in pending and u is not o and not _links_only(u, o)
                            for u in (getattr(o, "InList", None) or []))]
        if not ready:                 # cannot happen in a DAG; never loop forever
            ready = list(pending.values())
        for o in ready:
            ordered.append(o)
            pending.pop(o.Name, None)
    return ordered


def _remove(doc, obj) -> None:
    """Delete ``obj`` from ``doc``; a PartDesign Body child leaves its Body properly first (the
    Body fixes its Tip and BaseFeature chain), then the document drops the object."""
    try:
        parent = obj.getParentGeoFeatureGroup()
        if parent is not None and parent.TypeId == "PartDesign::Body":
            parent.removeObject(obj)
    except Exception:  # noqa: BLE001 - the plain removal below still works
        pass
    if doc.getObject(obj.Name) is not None:
        doc.removeObject(obj.Name)


def _max_undo_size() -> int:
    """The undo stack depth of GUI documents (preference ``MaxUndoSize``, default 20); 0 in a
    console session, where documents run with undo off and nothing can be counted on."""
    if not App.GuiUp:
        return 0
    try:
        return int(App.ParamGet("User parameter:BaseApp/Preferences/Document").GetInt("MaxUndoSize", 20))
    except Exception:  # noqa: BLE001
        return 20


def _undo_count(doc) -> int:
    return int(getattr(doc, "UndoCount", 0) or 0)


def _undo_cap() -> int:
    """FreeCAD's undo-stack limit (the GUI applies Preferences/Document MaxUndoSize, default 20); 0 or less means
    unlimited. Unlike ``_max_undo_size`` this answers 20 in a console session too, because a console document
    switched to ``UndoMode 1`` keeps the same 20 steps."""
    try:
        return int(App.ParamGet("User parameter:BaseApp/Preferences/Document").GetInt("MaxUndoSize", 20))
    except Exception:  # noqa: BLE001
        return 20


def _fcstd_problem(path: str) -> str | None:
    """Why ``path`` cannot be read back as a FreeCAD document, or None when it can. An FCStd is a zip archive, so a
    file a killed writer cut short has no central directory and a damaged one fails its checksums."""
    if not os.path.isfile(path):
        return "no longer exists on disk"
    try:
        with zipfile.ZipFile(path) as archive:
            if "Document.xml" not in archive.namelist():
                return "is not a FreeCAD document (it has no Document.xml)"
            damaged = archive.testzip()
    except (zipfile.BadZipFile, OSError) as e:
        return f"cannot be read ({e})"
    if damaged is not None:
        return f"is damaged ({damaged} fails its checksum)"
    return None


def _gui_modified(doc_name: str) -> bool:
    """True when the GUI says the document has unsaved changes; False in a console session (no GUI document)."""
    try:
        import FreeCADGui as Gui

        return bool(Gui.getDocument(doc_name).Modified)
    except Exception:  # noqa: BLE001
        return False




def _error_text(result) -> str:
    """The message in a failed task's result: an error string, or the error-dict a dispatch timeout returns."""
    return dispatch.failure_text(result, "the GUI task failed")


def _gui_failure(op: str, result) -> RuntimeError:
    """The error for a read whose GUI task failed (or timed out): an empty list or a missing object would read as
    "nothing there" and send the model off rebuilding what a busy GUI thread merely could not show."""
    return RuntimeError(f"{op} failed: {_error_text(result)}")


def _close_transaction(doc_name: str, commit: bool, previous_mode: int | None) -> None:
    """Commit or abort the transaction ``txn_task`` opened on ``doc_name`` and restore its UndoMode.

    A document the task closed or replaced (reload, revert) is simply skipped."""
    doc = App.listDocuments().get(doc_name)
    if doc is None:
        return
    try:
        if commit:
            doc.commitTransaction()
        else:
            doc.abortTransaction()
    except Exception as e:  # noqa: BLE001 - a transaction problem must never mask the task's result
        App.Console.PrintWarning(f"Oscar: transaction close failed on '{doc_name}': {e}\n")
    if previous_mode is not None:
        try:
            if doc.UndoMode != previous_mode:
                doc.UndoMode = previous_mode
        except Exception:  # noqa: BLE001
            pass


def txn_task(task, label: str = "Oscar", doc_name: str | None = None):
    """Wrap a GUI-thread task in a document undo transaction that rolls back on failure.

    FreeCAD only records undo data inside transactions — without this
    wrapper nothing a tool does can be reverted. Empty transactions are
    discarded by FreeCAD, so wrapping read-mostly tasks is harmless.
    ``doc_name`` targets a specific document (resolved tolerantly); None
    targets the active one; documents created inside the task are handled
    by revert's close-list.

    The transaction is COMMITTED when the task returns normally and ABORTED when it raises or
    returns a dict with ``success`` False, so a failed tool call leaves the document exactly as
    it was (created objects gone, edited values and expressions back, deleted objects
    restored). Rollback needs ``UndoMode 1``: the GUI default, but ``freecadcmd`` documents run
    at 0 where ``abortTransaction`` undoes nothing, so the mode is raised for the duration of the
    task and restored afterwards.
    """

    def wrapped():
        doc = _resolve_doc(doc_name) if doc_name else App.ActiveDocument
        if doc is None:
            return task()
        name = doc.Name
        previous = None
        try:
            previous = doc.UndoMode
            if previous != 1:
                doc.UndoMode = 1
        except Exception:  # noqa: BLE001 - no undo support: the task still runs
            previous = None
        doc.openTransaction(label)
        committed = False
        try:
            result = task()
            committed = not (isinstance(result, dict) and result.get("success") is False)
            return result
        finally:
            _close_transaction(name, committed, previous)

    return wrapped


# ---------------------------------------------------------------------------
# the engine


def _meshes_to_solids(doc, names: list[str]) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    """Each ``Mesh::Feature`` among ``names`` replaced by a ``Part::Feature`` solid of it (same label, the mesh
    removed): ``(names now, a note per mesh that stayed a mesh, {solid name: {from_triangles, valid}})``.
    GUI-thread only."""
    out, notes, made = [], [], {}
    for name in names:
        obj = doc.getObject(name)
        if obj is None or getattr(obj, "TypeId", "") != imported.MESH_TYPE:
            out.append(name)
            continue
        solid, why = imported.solid_from_mesh(obj.Mesh)
        if solid is None:
            notes.append(f"{obj.Label} stays a mesh: {why}")
            out.append(name)
            continue
        label, triangles = obj.Label, int(obj.Mesh.CountFacets)
        doc.removeObject(name)
        feature = doc.addObject("Part::Feature", name)
        feature.Label = label
        feature.Shape = solid
        out.append(feature.Name)
        made[feature.Name] = {"from_triangles": triangles, "valid": bool(solid.isValid())}
    doc.recompute()
    return out, notes, made


class CadEngine:
    """Thread-safe CAD operations for the chat tools and the bridge."""

    ENGINE_VERSION = ENGINE_VERSION

    def __init__(self):
        # Document.Uid of every document Oscar itself created: a revert may
        # close those, never one it cannot attribute to the agent.
        self._agent_uids: set[str] = set()
        # Document.Uid -> (the Body the agent made last there, the Body that
        # was active then): where a PartDesign feature with no named Body goes
        self._new_bodies: dict[str, tuple[str, str | None]] = {}
        self._last_snapshot: dict[str, Any] | None = None
        self._snapshot_seq = 0

    # ---------------------------------------------------------------- basics

    def ping(self) -> bool:
        """Liveness probe for the bridge: always True."""
        return True

    def status(self) -> dict[str, Any]:
        """Application name, engine version, build and the names of the open documents (bridge surface)."""
        return {
            "success": True,
            "application": "Oscar",
            "engine_version": ENGINE_VERSION,
            "engine_build": App.Version()[:3],
            "documents": list(App.listDocuments().keys()),
        }

    def list_documents(self) -> list[str]:
        """Names of the open documents (dispatched to the GUI thread). Raises ``RuntimeError`` when the GUI thread
        did not answer: an empty list would read as "nothing is open"."""
        result, failed = dispatch.dispatch(
            lambda: list(App.listDocuments().keys()), op="list_documents"
        )
        if failed:
            raise _gui_failure("list_documents", result)
        return list(result or [])

    def find_document(self, doc_name: str) -> str | None:
        """The actual name of the open document ``doc_name`` resolves to
        (tolerating case and unique-prefix near-misses), or None."""
        def task():
            doc = _resolve_doc(doc_name or "")
            return doc.Name if doc is not None else None

        result, failed = dispatch.dispatch(task, op="find_document")
        return result if isinstance(result, str) and not failed else None

    def create_document(self, name: str = "Model") -> dict[str, Any]:
        """Create a document; the result carries the ACTUAL name (FreeCAD sanitises and de-duplicates it)."""
        def task():
            doc = App.newDocument(name)
            self._agent_uids.add(str(doc.Uid))
            doc.recompute()
            return {"success": True, "document_name": doc.Name}

        result, failed = dispatch.dispatch(task, op="create_document")
        return result if isinstance(result, dict) else {"success": False, "error": str(result)}

    def reload_document(self, doc_name: str, discard_unsaved: bool = False) -> dict[str, Any]:
        """Close ``doc_name`` and open its file again; returns the actual new document name.

        The open document is the only good copy of its unsaved work until the file is known to be readable, so
        nothing is closed before that is checked: the file name and the document name are captured BEFORE the
        document is closed (the old code read ``doc.FileName`` from a closed document) and an unreadable file
        (missing, truncated by a killed writer, damaged) never costs the open document. Unsaved changes are only
        given up when the caller says so (``discard_unsaved``): a GUI document the user modified is left open
        otherwise. Whenever unsaved changes are discarded they are first kept in a rescue copy
        (``rescue_copy`` in the result). If the reopen still fails the error says the document was closed and
        names the rescue copy (or nothing).
        """
        def task():
            doc = _resolve_doc(doc_name)
            if doc is None:
                return {"success": False, "error": f"document '{doc_name}' is not open"}
            name, file_name = doc.Name, doc.FileName
            if not file_name:
                return {
                    "success": False,
                    "error": f"'{doc_name}' was never saved; nothing to reload",
                }
            problem = _fcstd_problem(file_name)
            if problem is not None or not zipfile.is_zipfile(file_name):
                return {
                    "success": False,
                    "error": (f"'{file_name}' is missing or is not a valid .FCStd file"
                              f"{f' ({problem})' if problem else ''}, so the open document '{name}' was left as it is: "
                              "it is still open and untouched - it may hold the only good copy, so save it "
                              "again (or restore the file) instead"),
                }
            modified = _gui_modified(name)
            if modified and not discard_unsaved:
                return {
                    "success": False,
                    "error": (f"'{name}' has unsaved changes that reloading would discard; it was left open. "
                              "Ask the user to save them, or to throw them away with File > Revert, then "
                              "continue from there (or pass discard_unsaved=true once the user agreed)"),
                }
            rescue = self._rescue_copy(doc, file_name) if (modified or self._has_unsaved_changes(doc)) else ""
            was_active = App.ActiveDocument is not None and App.ActiveDocument.Name == name
            uid = str(doc.Uid)
            App.closeDocument(name)
            self._agent_uids.discard(uid)
            try:
                new = App.openDocument(file_name)
            except Exception as e:  # noqa: BLE001
                kept = f"; your unsaved version was kept at {rescue}" if rescue else ""
                return {"success": False,
                        "error": f"document '{name}' was closed but could not be reopened from {file_name}: {e}{kept}"}
            if was_active:
                try:
                    App.setActiveDocument(new.Name)
                    if App.GuiUp:
                        import FreeCADGui as Gui

                        Gui.setActiveDocument(new.Name)
                except Exception:  # noqa: BLE001 - activation is cosmetic
                    pass
            result = {"success": True, "document_name": new.Name, "previous_name": name,
                      "rescue_copy": rescue}
            if modified:
                result["warning"] = ("unsaved changes in the old copy were discarded"
                                     + (f" (a copy of them was kept at {rescue})" if rescue else ""))
            return result

        result, failed = dispatch.dispatch_stoppable(task, timeout=120.0, op="reload_document")
        return result if isinstance(result, dict) else {"success": False, "error": str(result)}

    @staticmethod
    def _has_unsaved_changes(doc) -> bool:
        """True when ``doc`` holds changes the file on disk does not have (GUI: the document's
        Modified flag; always: objects still waiting for a recompute)."""
        try:
            if doc.isTouched():
                return True
            if App.GuiUp:
                import FreeCADGui as Gui

                gui_doc = Gui.getDocument(doc.Name)
                return bool(gui_doc is not None and gui_doc.Modified)
        except Exception:  # noqa: BLE001
            pass
        return False

    @staticmethod
    def _rescue_copy(doc, file_name: str) -> str:
        """Save ``doc`` next to the cache as ``<stem>.rescue-<pid>-<rand>.FCStd``; "" when it fails.

        The temp name MUST end in .FCStd: ``saveCopy`` appends that extension to any other name
        and writes there instead; ``zipfile.is_zipfile`` is the only success signal."""
        stem = os.path.splitext(os.path.basename(file_name))[0] or "document"
        target = os.path.join(paths.cache_dir("reload"),
                              f"{stem}.rescue-{os.getpid()}-{uuid.uuid4().hex[:8]}.FCStd")
        try:
            doc.saveCopy(target)
        except Exception:  # noqa: BLE001
            return ""
        return target if zipfile.is_zipfile(target) else ""

    # --------------------------------------------------------------- objects

    def create_object(self, doc_name: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Create one object from ``spec`` (``Type``, ``Name``, ``Properties``, optional ``Analysis`` / ``Body``) in one undo transaction.

        A failure rolls the whole attempt back (no stray object, visibility restored) and returns
        ``{"success": False, "error": ...}``; success returns ``object_name``, ``document_name`` and ``body``.
        """
        def task():
            doc = _resolve_doc(doc_name)
            if doc is None:
                return {"success": False, "error": f"document '{doc_name}' is not open"}
            type_name = spec.get("Type", "")
            name = spec.get("Name", "Object")
            properties = spec.get("Properties") or {}
            analysis = None
            analysis_name = spec.get("Analysis")
            if analysis_name:
                analysis = doc.getObject(analysis_name)
                if analysis is None:
                    return {"success": False, "error": f"analysis '{analysis_name}' not found"}
            if not isinstance(properties, dict):
                return {
                    "success": False,
                    "error": "Properties must be a JSON object (property name -> value)",
                }
            # The target Body is an argument of the call, not a property of the feature: it may come as
            # ``Body`` in the spec or as ``Properties.Body``. A name that is not a Body fails here, before
            # anything is created.
            requested_body = spec.get("Body")
            if (type_name.startswith(("PartDesign::", "Sketcher::")) and type_name != "PartDesign::Body"
                    and "Body" in properties):
                properties = dict(properties)
                requested_body = properties.pop("Body", None) or requested_body
            try:
                target_body = _requested_body(doc, requested_body)
            except ValueError as e:
                return {"success": False, "error": str(e)}

            before = {o.Name for o in doc.Objects}
            visibility = _visibility(doc)

            def fail(error: str) -> dict[str, Any]:
                _discard_created(doc, before, visibility)
                return {"success": False, "error": error}

            try:
                if type_name.startswith("Draft::"):
                    obj = _make_draft(doc, type_name, name, properties)
                    if obj is None:
                        return fail(
                            f"{type_name} needs Points/Radius/etc.; use execute_code for exotic shapes"
                        )
                elif type_name.startswith("Fem::"):
                    obj = _make_fem(doc, type_name, name, properties, analysis)
                else:
                    try:
                        obj = doc.addObject(type_name, name)
                    except Exception as e:
                        return fail(
                            f"cannot create '{type_name}': {e}. Use a C++ type such as "
                            "Part::Box, Part::Cylinder, Part::Cut, Part::MultiFuse, "
                            "PartDesign::Body — Python-implemented types (e.g. "
                            "Part::Tube, arrays) need execute_code"
                        )
                    if obj is None:
                        return fail(f"cannot create '{type_name}'")
                    apply_properties(obj, properties)
                    try:
                        target_body = _target_body(doc, type_name, requested_body, obj,
                                                   self._new_bodies.get(str(doc.Uid)))
                    except ValueError as e:
                        return fail(str(e))
                    if target_body is not None and obj not in target_body.Group:
                        # PartDesign features (and sketches created with body=...) belong
                        # inside their Body; a failure here is reported, never swallowed.
                        try:
                            target_body.addObject(obj)
                        except Exception as e:  # noqa: BLE001
                            return fail(f"cannot add '{obj.Name}' to body '{target_body.Name}': {e}")
            except Exception as e:
                return fail(f"{type(e).__name__}: {e}")

            if analysis is not None and not _is_gmsh(type_name) and obj not in analysis.Group:
                try:
                    analysis.addObject(obj)
                except Exception:
                    pass
            try:
                doc.recompute()
            except Exception as e:
                return fail(f"recompute failed: {e}")
            if not _valid(obj):
                error = f"object '{obj.Name}' ended in an invalid state; check its dependencies"
                if type_name.startswith("PartDesign::"):
                    error += (
                        " — PartDesign features need a Body plus a Profile "
                        "(e.g. a sketch); for profile shapes prefer the Draft "
                        "factories (Draft::Wire/Circle) or build with execute_code"
                    )
                return fail(error + " (the object was removed; fix the inputs and retry)")
            if type_name == "PartDesign::Body":
                active = _active_body(doc)
                self._new_bodies[str(doc.Uid)] = (
                    obj.Name, active.Name if active is not None else None
                )
            App.Console.PrintLog(f"Oscar: created '{obj.Name}' ({type_name})\n")
            created = {"success": True, "object_name": obj.Name, "document_name": doc.Name}
            if target_body is not None:
                created["body"] = target_body.Name
            return created

        result, failed = dispatch.dispatch_stoppable(
            txn_task(task, "Oscar: create_object", doc_name), op="create_object"
        )
        return result if isinstance(result, dict) else {"success": False, "error": str(result)}

    def edit_object(self, doc_name: str, obj_name: str, properties: dict[str, Any]) -> dict[str, Any]:
        """Apply ``properties`` to one object atomically (see ``properties.apply_properties_report``) and recompute, in one undo transaction.

        Returns ``{"success": True, "object_name", "applied", "notes"}`` or ``{"success": False, "error"}``
        with nothing changed. An edit whose recompute leaves a VALID object invalid is rolled back and fails
        (``"... rolled back"``); an object that was already invalid and still is succeeds with a ``warning``.
        """
        def task():
            doc = _resolve_doc(doc_name)
            if doc is None:
                return {"success": False, "error": f"document '{doc_name}' is not open"}
            obj = doc.getObject(obj_name)
            if obj is None:
                return {"success": False, "error": f"object '{obj_name}' not found"}
            was_valid = _valid(obj)
            # The undo transaction cannot be relied on to undo a failed edit
            # (console sessions run without one), so the edit keeps its own
            # record of what to put back.
            log: list = []
            try:
                report = apply_properties_report(obj, properties, undo_log=log)
                doc.recompute()
            except Exception as e:
                restore_properties(log)
                try:
                    doc.recompute()
                except Exception:
                    pass
                return {"success": False, "error": f"{type(e).__name__}: {e}"}
            result = {"success": True, "object_name": obj.Name,
                      "applied": report["applied"], "notes": report["notes"]}
            if _valid(obj):
                return result
            state = ", ".join(getattr(obj, "State", []) or [])
            if was_valid:
                restore_properties(log)
                doc.recompute()
                return {
                    "success": False,
                    "error": (
                        f"edit left '{obj.Name}' in an invalid state ({state}); "
                        "the edit was rolled back - check the values (negative or "
                        "zero sizes, bad references)"
                    ),
                }
            result["warning"] = f"'{obj.Name}' is still in an invalid state ({state})"
            return result

        result, failed = dispatch.dispatch_stoppable(
            txn_task(task, "Oscar: edit_object", doc_name), op="edit_object"
        )
        return result if isinstance(result, dict) else {"success": False, "error": str(result)}

    def delete_object(self, doc_name: str, obj_name: str, cascade: bool = False,
                      force: bool = False) -> dict[str, Any]:
        """Delete one object; refuses when other objects depend on it unless ``cascade`` or ``force``.

        Dependents are the objects that link to it (``InList``: a boolean's Base / Tool, the
        feature that uses a sketch, objects bound by expression to a spreadsheet) EXCEPT the
        containers that merely group it (a Body or Part holding the object) and the next feature of its
        Body (the Body relinks that chain and moves the Tip itself). Without ``cascade`` / ``force``
        the call fails and names them; with ``cascade`` they are deleted first, dependents before what
        they depend on; with ``force`` only the object itself goes and its dependents are left (invalid).
        A Body owns its features, so deleting a Body deletes them with it (they are listed in
        ``removed_children``; leaving them behind would orphan them). Inputs a deleted boolean / feature hid
        are shown again (the GUI's own delete does this) unless something else still consumes them.
        """
        def task():
            doc = _resolve_doc(doc_name)
            if doc is None:
                return {"success": False, "error": f"document '{doc_name}' is not open"}
            obj = doc.getObject(obj_name)
            if obj is None:
                return {"success": False, "error": f"object '{obj_name}' not found"}
            children = list(obj.Group) if obj.TypeId == "PartDesign::Body" else []
            child_names = [o.Name for o in children]
            own = {obj.Name, *child_names}
            dependents: list = []
            for root in [obj, *children]:
                for user in _dependents(root):
                    if user.Name not in own and all(user.Name != d.Name for d in dependents):
                        dependents.append(user)
            if dependents and not (cascade or force):
                listing = ", ".join(f"'{o.Name}' ({o.TypeId})" for o in dependents[:8])
                more = f" and {len(dependents) - 8} more" if len(dependents) > 8 else ""
                return {"success": False, "error": (
                    f"cannot delete '{obj.Name}': {listing}{more} depend(s) on it; delete those "
                    "first, or pass cascade=true to delete them together with it")}
            if not cascade:
                dependents = []
            doomed = list(dependents) + [obj]
            removed_names = {o.Name for o in doomed} | set(child_names)
            cascaded = [o.Name for o in dependents]      # a removed object no longer has a Name
            inputs = []
            for victim in doomed:
                for source in getattr(victim, "OutList", []) or []:
                    if source.Name not in removed_names and source.Name not in inputs:
                        inputs.append(source.Name)
            for victim in doomed:
                if victim.TypeId == "PartDesign::Body" and victim.Group:
                    victim.removeObjectsFromDocument()    # the Body's features go with it
                _remove(doc, victim)
            # A deleted boolean/feature leaves its inputs hidden (the GUI's
            # own delete re-shows them); show inputs nothing else consumes.
            restored = []
            for name in inputs:
                source = doc.getObject(name)
                if source is None or getattr(source, "InList", None):
                    continue
                try:
                    if not source.Visibility:
                        source.Visibility = True
                        restored.append(name)
                except Exception:
                    pass
            doc.recompute()
            result = {"success": True, "object_name": obj_name, "restored": restored,
                      "cascaded": cascaded}
            if child_names:
                result["removed_children"] = child_names
            return result

        result, failed = dispatch.dispatch_stoppable(
            txn_task(task, "Oscar: delete_object", doc_name), op="delete_object"
        )
        return result if isinstance(result, dict) else {"success": False, "error": str(result)}

    def get_objects(self, doc_name: str) -> list[dict[str, Any]]:
        """Every object of the document as a dict (the bridge form; the ``get_objects`` TOOL
        renders a compact, paged table from ``_object_page`` instead). ``[]`` for a document that is not open;
        raises ``RuntimeError`` when the GUI thread did not answer (an empty list would read as an empty model)."""
        def task():
            doc = _resolve_doc(doc_name)
            if doc is None:
                return []
            return [object_to_dict(obj) for obj in doc.Objects]

        result, failed = dispatch.dispatch(task, op="get_objects")
        if failed:
            raise _gui_failure("get_objects", result)
        return list(result or [])

    def _object_page(self, doc_name: str, filter: str = "", limit: int | None = 50,
                     offset: int = 0) -> dict[str, Any]:
        """One filtered page of objects for the table: ``collect_objects`` output, or
        ``{"error": ...}`` when the GUI thread could not answer (never mistaken for an empty
        document). Private: it is not part of the bridge surface."""
        def task():
            doc = _resolve_doc(doc_name)
            if doc is None:
                return {"error": f"document '{doc_name}' is not open"}
            return collect_objects(doc, filter=filter, limit=limit, offset=offset)

        result, failed = dispatch.dispatch_stoppable(task, timeout=60.0, op="get_objects")
        if failed or not isinstance(result, dict):
            return {"error": f"could not read the objects of '{doc_name}': {dispatch.failure_text(result)}"}
        if "error" not in result and "objects" not in result:
            return {"error": str(result.get("error") or "the GUI thread did not answer")}
        return result

    def get_object(self, doc_name: str, obj_name: str, detail: bool = False) -> dict[str, Any] | None:
        """One object as a dict (None when the document or the object does not exist); ``detail=True`` adds
        ``PropertiesList``, links and the parent. Raises ``RuntimeError`` when the GUI thread did not answer: ``None``
        would read as "no such object"."""
        def task():
            doc = _resolve_doc(doc_name)
            if doc is None:
                return None
            obj = doc.getObject(obj_name)
            return object_to_dict(obj, detail=detail) if obj is not None else None

        result, failed = dispatch.dispatch(task, op="get_object")
        if failed:
            raise _gui_failure("get_object", result)
        return result if isinstance(result, dict) else None

    def _active_document_name(self) -> str:
        """Name of the active document ("" when none is open). Private helper of run_tool."""
        def task():
            return App.ActiveDocument.Name if App.ActiveDocument is not None else ""

        result, failed = dispatch.dispatch(task, op="active_document")
        return result if isinstance(result, str) and not failed else ""

    # ------------------------------------------------------------- scripting

    def execute_code(self, code: str, timeout: Any = None) -> dict[str, Any]:
        """Run a script on the GUI thread through ``scripting.run_script`` with a ``timeout`` budget in seconds (default 90, cap 1800).

        A successful run is checked by the merge guard (parts lost from the tree add a WARNING to ``message``).
        """
        try:
            budget = float(timeout) if timeout is not None else 90.0
            if budget <= 0:
                raise ValueError
        except (TypeError, ValueError):
            return {"success": False, "error": f"invalid timeout: {timeout!r}"}
        before = self._shapes_on_gui_thread()
        known = self._document_uids()
        result = scripting.run_script(code, timeout=min(budget, 1800.0))
        self._adopt_new_documents(known)
        if result.get("success"):
            # A failed or timed-out script is judged by run_script / run_tool (rollback, STILL
            # RUNNING); asking the busy GUI thread for a second snapshot would only stack a wait.
            merge_note = self._merge_guard_note(before)
            if merge_note:
                result["message"] = (result.get("message") or "") + merge_note
        return result

    @staticmethod
    def _document_uids() -> "set[str] | None":
        """Uids of the open documents, taken ON the GUI thread (None when it is busy)."""
        result, failed = dispatch.dispatch(
            lambda: {str(doc.Uid) for doc in App.listDocuments().values()},
            timeout=10.0, op="document_uids",
        )
        return result if isinstance(result, set) and not failed else None

    def _adopt_new_documents(self, known: "set[str] | None") -> None:
        """Documents a script created belong to the agent (``revert_turn`` may close them). Scripts run in
        transactions, but a document's undo stack says nothing about who made it, so the engine remembers which
        documents appeared during the script (``known`` = the uids before it ran)."""
        if known is None:
            return

        def task():
            for doc in App.listDocuments().values():
                if str(doc.Uid) not in known:
                    self._agent_uids.add(str(doc.Uid))

        dispatch.dispatch(task, timeout=10.0, op="adopt_documents")

    def _shapes_on_gui_thread(self) -> "set[str] | None":
        """Names of the shape-bearing objects of every open document, taken ON the GUI thread
        (a document must never be read from the agent thread); None when it could not be taken."""
        result, failed = dispatch.dispatch(self._snapshot_shapes, timeout=15.0, op="merge_guard")
        return result if isinstance(result, set) and not failed else None

    def _merge_guard_note(self, before: "set[str] | None" = None) -> str:
        """Detect parts LOST from the tree by scripted fusing.

        The classic LLM-CAD failure this guards: asked to add a second
        part, the model fuses it into the first (or Part.show()s a merged
        shape and deletes the originals) — one blob replaces the parts and
        the document tree no longer shows them individually. ``before`` is the shape-bearing
        object names taken before ``run_script`` (``_shapes_on_gui_thread``); a second snapshot
        is compared after, and a shrinking tree appends a model-facing warning so the next round
        fixes it (new part as its own object) instead of shipping a merged blob. Temporary
        ``_oscar_tmp*`` objects never count.
        """
        try:
            if before is None:
                return ""
            after = self._shapes_on_gui_thread()
            if after is None:
                return ""
            lost = before - after
            if not lost:
                return ""
            return (
                f"\nWARNING: the tree LOST shape object(s) {sorted(lost)} — "
                "parts were merged or deleted. Unless the user explicitly "
                "asked for a single merged solid, this is wrong: undo this "
                "step (or revert the turn) and add the new part as its OWN "
                "object (own Body / Part::Feature) so every part stays "
                "individually present in the tree. Tell the user what was "
                "lost if you cannot restore it."
            )
        except Exception:
            return ""

    @staticmethod
    def _snapshot_shapes() -> set[str]:
        """``"<doc>/<object>"`` of every shape-bearing object (``_oscar_tmp*`` helpers ignored).
        GUI-thread only."""
        names = set()
        for doc in App.listDocuments().values():
            for obj in getattr(doc, "Objects", []) or []:
                if obj.Name.startswith("_oscar_tmp"):
                    continue
                if getattr(obj, "Shape", None) is not None:
                    names.add(f"{doc.Name}/{obj.Name}")
        return names

    def execute_code_async(self, code: str) -> dict[str, Any]:
        """Start a background job (``scripting.start_job``); returns ``job_id`` to poll with ``get_async_status``."""
        return scripting.start_job(code)

    def get_async_status(self, job_id: str = "") -> dict[str, Any]:
        """State of one background job, or the list of running and recent jobs when ``job_id`` is empty."""
        return scripting.job_status(job_id)

    def execute_code_headless(self, code: str, timeout: float = 600.0) -> dict[str, Any]:
        """Run a standalone script in a separate headless process (``cad.headless.run``) with a ``timeout`` in seconds."""
        return run_headless(code, timeout)

    # ----------------------------------------------------------------- views

    def screenshot(
        self,
        orientation: str = "Isometric",
        width: int | None = None,
        height: int | None = None,
        focus: str | None = None,
        doc_name: str | None = None,
    ) -> str | None:
        """PNG (base64) of ``doc_name``'s 3D view — the document a tool just
        changed — or of the active view when no document is named.

        ``None`` when no valid picture could be taken: a console session (no 3D view), a view
        that cannot be captured, a blank frame or a busy GUI thread. An error text is NEVER
        returned in place of an image (it used to be, and ``get_view`` then reported
        "Screenshot captured." with a picture of the error message)."""
        return self._screenshot_with_reason(orientation, width, height, focus, doc_name)[0]

    def _screenshot_with_reason(
        self,
        orientation: str = "Isometric",
        width: int | None = None,
        height: int | None = None,
        focus: str | None = None,
        doc_name: str | None = None,
    ) -> tuple[str | None, str]:
        """``(png_base64 | None, reason)``: the reason says why there is no picture ("" when
        there is one). The result must be real PNG data (magic bytes checked). Private (not on
        the bridge); the tools use it to tell the model why a screenshot is missing."""
        def task():
            if not App.GuiUp:
                return None, views.HEADLESS_REASON
            target = None
            if doc_name:
                doc = _resolve_doc(doc_name)
                target = doc.Name if doc is not None else None
            return views.capture_base64_checked(orientation, width, height, focus, target)

        result, failed = dispatch.dispatch_stoppable(task, timeout=120.0, op="screenshot")
        if failed:
            detail = result.get("error") if isinstance(result, dict) else result
            return None, f"screenshot failed on the GUI thread: {detail}"
        if isinstance(result, dict):   # the dispatcher's timeout answer
            return None, str(result.get("error") or "the GUI thread did not answer")
        if not (isinstance(result, tuple) and len(result) == 2):
            return None, "screenshot failed: unexpected answer from the GUI thread"
        image, reason = result
        if image is None:
            return None, str(reason or "no screenshot")
        if not self._is_png_base64(image):
            return None, "screenshot failed: the data is not a PNG image"
        return image, ""

    @staticmethod
    def _is_png_base64(data: Any) -> bool:
        """True when ``data`` is base64 text that decodes to the 8-byte PNG signature."""
        if not isinstance(data, str) or len(data) < 12:
            return False
        try:
            return base64.b64decode(data[:12], validate=True)[:8] == b"\x89PNG\r\n\x1a\n"
        except ValueError:
            return False

    # ----------------------------------------------------------- parts library

    @staticmethod
    def _parts_library_root() -> str:
        """The parts library folder: the user Mod folder (where the Addon Manager installs it), else the
        application's Mod folder (a bundled copy). The first one that exists wins."""
        roots = [os.path.join(App.getUserAppDataDir(), "Mod", "parts_library"),
                 os.path.join(App.getResourceDir(), "Mod", "parts_library")]
        return next((r for r in roots if os.path.isdir(r)), roots[-1])

    def parts_list(self) -> list[str]:
        """Relative paths of the parts of the bundled library, sorted."""
        root = self._parts_library_root()
        found = []
        for base, _dirs, files in os.walk(root):
            for filename in files:
                if filename.lower().endswith(".fcstd"):
                    relative = os.path.relpath(os.path.join(base, filename), root)
                    found.append(relative)
        return sorted(found)

    def insert_part(self, relative_path: str) -> dict[str, Any]:
        """Insert a part of the bundled library (``relative_path`` from ``parts_list``) into the active document.

        A path outside the library, or a file that does not exist, returns ``{"success": False, "error": ...}``."""
        root = self._parts_library_root()
        source = os.path.normpath(os.path.join(root, relative_path))
        # resolve must stay inside the library root
        if os.path.commonpath([source, root]) != root or not os.path.isfile(source):
            return {"success": False, "error": f"library part not found: {relative_path}"}

        def task():
            target = App.ActiveDocument
            if target is None:
                target = App.newDocument("Library")
                self._agent_uids.add(str(target.Uid))
            donor = App.openDocument(source)
            try:
                for obj in donor.Objects:
                    target.copyObject(obj, True)
            finally:
                App.closeDocument(donor.Name)
            target.recompute()
            return {"success": True, "message": f"inserted {relative_path} into {target.Name}"}

        result, failed = dispatch.dispatch_stoppable(
            txn_task(task, "Oscar: insert_part"), timeout=120.0, op="insert_part"
        )
        return result if isinstance(result, dict) else {"success": False, "error": str(result)}

    # ------------------------------------------------------------- file import

    def import_file(self, path: str, doc_name: str = "", as_solid: bool = False) -> dict[str, Any]:
        """Import a geometry/mesh/drawing file, or open an FCStd document.

        STEP/IGES/BREP land as solids (Import), STL/OBJ/3MF/PLY as meshes
        (Mesh), DXF/SVG as drawing geometry (Draft importers), FCStd opens
        as its own document. Solids/meshes go into ``doc_name`` if given and
        open, else the active document, else a new one named after the file.

        ``as_solid`` makes each imported mesh a Part solid in its place
        (``imported.solid_from_mesh``), in the same undo step; a mesh that
        cannot be one stays a mesh and ``solid_notes`` says why. ``parts``
        holds ``imported.facts`` of every new object.
        """
        source = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(source):
            return {"success": False, "error": f"file not found: {path}"}
        ext = os.path.splitext(source)[1].lower().lstrip(".")

        if ext == "fcstd":
            def open_task():
                for doc in App.listDocuments().values():
                    if doc.FileName and os.path.abspath(doc.FileName) == source:
                        App.setActiveDocument(doc.Name)
                        return {
                            "success": True,
                            "document_name": doc.Name,
                            "objects": [o.Name for o in doc.Objects],
                            "message": "document already open",
                        }
                doc = App.openDocument(source)
                doc.recompute()
                return {
                    "success": True,
                    "document_name": doc.Name,
                    "objects": [o.Name for o in doc.Objects],
                }

            result, _ = dispatch.dispatch_stoppable(open_task, timeout=120.0, op="import_file")
            return result if isinstance(result, dict) else {"success": False, "error": str(result)}

        def _importer():
            # Importers are probed, not assumed: the bundled set varies.
            if ext in ("step", "stp", "iges", "igs"):
                import Import

                return lambda doc: Import.insert(source, doc.Name)
            if ext in ("brep", "brp"):
                # the Import module has no BREP reader; Part does
                import Part

                return lambda doc: Part.insert(source, doc.Name)
            if ext in ("stl", "obj", "3mf", "ply", "ast"):
                import Mesh

                return lambda doc: Mesh.insert(source, doc.Name)
            if ext == "dxf":
                import importDXF

                return lambda doc: importDXF.insert(source, doc.Name)
            if ext == "svg":
                import importSVG

                return lambda doc: importSVG.insert(source, doc.Name)
            return None

        caller = _importer()
        if caller is None:
            return {
                "success": False,
                "error": (
                    f"unsupported file type '.{ext}' — supported: step, stp, iges, "
                    "igs, brep, brp, stl, obj, 3mf, ply, dxf, svg, fcstd"
                ),
            }

        def task():
            doc = _resolve_doc(doc_name) if doc_name else None
            if doc_name and doc is None:
                return {"success": False, "error": f"document '{doc_name}' is not open"}
            if doc is None:
                doc = App.ActiveDocument
            if doc is None:
                stem = os.path.splitext(os.path.basename(source))[0] or "Import"
                doc = App.newDocument(stem)
                self._agent_uids.add(str(doc.Uid))
            before = {o.Name for o in doc.Objects}
            try:
                caller(doc)
                doc.recompute()
            except Exception as e:
                return {"success": False, "error": f"import failed: {type(e).__name__}: {e}"}
            created = [o.Name for o in doc.Objects if o.Name not in before]
            solid_notes, made = [], {}
            if as_solid and ext in imported.MESH_EXTENSIONS:
                created, solid_notes, made = _meshes_to_solids(doc, created)

            # Make imported geometry SHOW UP: activate the document and fit
            # the 3D view to it. Console sessions have no view — that's fine.
            try:
                import FreeCADGui as Gui

                Gui.setActiveDocument(doc.Name)
                from .views import fit_view

                fit_view()
            except Exception:
                pass

            # What landed and where, so the model (and the user) can work
            # from real sizes and positions, not from the picture alone.
            parts = []
            for name in created:
                obj = doc.getObject(name)
                found = imported.facts(obj) if obj is not None else None
                if found is not None:
                    parts.append({**found, **made.get(name, {})})
            return {
                "success": True,
                "document_name": doc.Name,
                "objects": created,
                "count": len(created),
                "parts": parts,
                "solid_notes": solid_notes,
            }

        result, _ = dispatch.dispatch_stoppable(
            txn_task(task, "Oscar: import_file", doc_name or None),
            timeout=180.0,
            op="import_file",
        )
        return result if isinstance(result, dict) else {"success": False, "error": str(result)}

    # ------------------------------------------------------------- turn revert

    def begin_turn_snapshot(self) -> dict[str, Any]:
        """Record the state of every open document at turn start, so the turn's CAD changes can be undone later
        (``revert_turn``).

        ``docs`` maps each document to its ``UndoCount``; ``max_undo`` to its undo-stack limit (the GUI default is
        20: a turn with more undo steps than that cannot be undone from the stack, which ``revert_turn`` reports as
        ``incomplete``; 0 = unknown, e.g. in a console session where undo is off anyway); ``undo_mode`` to its
        ``UndoMode``. The snapshot also carries what lets a revert undo the turn's own changes and nothing else:
        ``seq`` (turn order), ``uids`` (a name may later belong to another document), ``undo_cap``, ``objects`` and
        ``undo_names`` (what a full-history document looked like).

        Starting a turn also closes out the previous snapshot (``end_docs`` / ``end_uids``, on the SAME dict the
        panel keeps on its card): a revert of that older turn can then tell its own changes from the ones a later
        turn or the user made on top."""
        def task():
            cap = _undo_cap()
            size = _max_undo_size()
            docs = App.listDocuments()
            depth, max_undo, modes = {}, {}, {}
            for name, doc in docs.items():
                depth[name] = _undo_count(doc)
                max_undo[name] = int(getattr(doc, "MaxUndoSize", 0) or 0) or size
                modes[name] = int(getattr(doc, "UndoMode", 0) or 0)
            self._snapshot_seq += 1
            snapshot = {
                "success": True,
                "seq": self._snapshot_seq,
                "docs": depth,
                "max_undo": max_undo,
                "undo_mode": modes,
                "undo_cap": cap,
                "uids": {name: str(doc.Uid) for name, doc in docs.items()},
                # what a full-history document looked like, to recognise a
                # turn that changed it although its undo count cannot move
                "objects": {name: [o.Name for o in doc.Objects] for name, doc in docs.items()},
                "undo_names": {
                    name: list(doc.UndoNames) for name, doc in docs.items()
                    if cap > 0 and depth[name] >= cap
                },
            }
            previous = self._last_snapshot
            if previous is not None:
                previous.setdefault("end_docs", dict(depth))
                previous.setdefault("end_uids", dict(snapshot["uids"]))
            self._last_snapshot = snapshot
            return snapshot

        result, _ = dispatch.dispatch(task, op="turn_snapshot")
        if isinstance(result, dict) and result.get("success") is True:
            return result
        return {"success": False, "docs": {}}

    @staticmethod
    def _usable_snapshot(snapshot: Any) -> bool:
        """A turn-start snapshot that can drive a revert (``begin_turn_snapshot`` returned it successfully)."""
        return (isinstance(snapshot, dict) and snapshot.get("success") is True
                and isinstance(snapshot.get("docs"), dict))

    def revert_turn(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Undo the changes one turn made, and nothing else.

        A missing or failed snapshot (``{}``, ``success`` False, no ``docs``) is refused: it used to read as "no
        documents were open" and closed every open document.

        Only the documents the turn touched count (``touched_docs``, see ``_turn_touched``). When the undo stack of
        one of them may not have held the whole turn (its stack is full, i.e. at its recorded ``max_undo``: the
        oldest steps fall off the end), or when the turn's undo steps there include one ``Document.undo()`` must not
        pop (``UNDO_UNSAFE_PREFIXES``: FreeCAD can crash undoing a FeaturePlan build), NOTHING is reverted and the
        result is ``{"success": False, "incomplete": True, "incomplete_docs": [...], "touched_docs": [...]}`` (plus
        ``"undo_unsafe": True`` and ``unsafe_docs`` for the second reason) so the caller can restore the turn
        checkpoint instead; a half-undone turn is worse than none, and a crash loses everything. Where the limit is
        unknown (console sessions) the revert is done as far as the history reaches and the shortfall is reported as
        a warning.

        Each document's undo stack is popped back to the depth recorded at the turn's start — but only while the top
        step is one Oscar made, and only if nothing changed the document after the turn. Documents the turn created
        are closed only when they can be attributed to the agent and hold no work of the user's. Whatever could not
        be undone is reported (``partial``, ``conflicts``, ``warnings``, ``kept_docs``) instead of being passed off
        as a clean revert (``success`` is then False); a clean one has ``success`` True, ``partial`` False and
        ``incomplete`` False."""
        if not self._usable_snapshot(snapshot):
            return {"success": False,
                    "error": "no turn snapshot was recorded, so nothing can be reverted safely"}
        start = snapshot["docs"]
        max_undo = snapshot.get("max_undo") if isinstance(snapshot.get("max_undo"), dict) else {}

        def task():
            live = App.listDocuments()
            touched, incomplete, unsafe = [], [], []
            for name in start:
                doc = live.get(name)
                if doc is None or not self._turn_touched(doc, name, snapshot):
                    continue
                touched.append(name)
                limit = int(max_undo.get(name, 0) or 0)
                if limit > 0 and _undo_count(doc) >= limit:
                    incomplete.append(name)
                labels = self._turn_undo_labels(doc, int(start[name] or 0), snapshot)
                if any(is_undo_unsafe(label) for label in labels):
                    unsafe.append(name)
            if incomplete or unsafe:
                reasons = []
                if incomplete:
                    reasons.append("the undo stack cannot hold this turn (" + ", ".join(incomplete) + ")")
                if unsafe:
                    reasons.append("FreeCAD's undo can crash on the PartDesign features this turn built ("
                                   + ", ".join(unsafe) + ")")
                return {"success": False, "incomplete": True, "undo_unsafe": bool(unsafe),
                        "incomplete_docs": incomplete + [n for n in unsafe if n not in incomplete],
                        "unsafe_docs": unsafe, "touched_docs": touched, "undone": {}, "closed_docs": [],
                        "error": "; ".join(reasons) + "; restore the turn checkpoint instead"}
            report: dict[str, Any] = {
                "undone": {}, "closed_docs": [], "kept_docs": [],
                "conflicts": [], "warnings": [], "incomplete": False, "touched_docs": touched,
            }
            for name, depth in start.items():
                doc = live.get(name)
                if doc is not None:
                    self._undo_document(doc, int(depth), snapshot, report)
            self._close_new_documents(live, snapshot, report)
            problems = report["conflicts"] + report["warnings"]
            report["partial"] = bool(problems)
            report["success"] = not problems
            if problems:
                report["error"] = "the turn was only partly reverted: " + "; ".join(problems)
            return report

        result, _ = dispatch.dispatch(task, timeout=120.0, op="turn_revert")
        return result if isinstance(result, dict) else {"success": False, "error": str(result)}

    def close_turn_documents(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Close the documents a turn created, under the same ownership rules as ``revert_turn``: a document saved to
        disk, holding changes made outside Oscar, created by a later turn, changed after the turn, or not attributable
        to Oscar stays open and is reported in ``kept_docs`` (``{"doc", "reason"}``). The panel's checkpoint fallback
        uses this after it restored the turn's checkpoints. A missing or failed snapshot is refused and closes
        nothing. Returns ``{"success", "closed_docs", "kept_docs", "conflicts", "warnings"}`` (+ ``error``)."""
        empty = {"closed_docs": [], "kept_docs": [], "conflicts": [], "warnings": []}
        if not self._usable_snapshot(snapshot):
            return {"success": False, **empty,
                    "error": "no turn snapshot was recorded, so no document can be attributed to the turn"}

        def task():
            report: dict[str, Any] = {key: [] for key in empty}
            self._close_new_documents(App.listDocuments(), snapshot, report)
            problems = report["conflicts"] + report["warnings"]
            report["success"] = not problems
            if problems:
                report["error"] = "; ".join(problems)
            return report

        result, failed = dispatch.dispatch(task, timeout=60.0, op="close_turn_documents")
        if failed or not isinstance(result, dict):
            return {"success": False, **empty, "error": _error_text(result)}
        return result

    def changed_since_turn(self, snapshot: dict[str, Any], names) -> dict[str, str]:
        """Which of the documents ``names`` (open at the turn's start) were changed AFTER the turn, with the reason:
        restoring the turn checkpoint would throw that work away too. The tests are the ones the undo path applies
        (``_undo_document``): another document now carries the name, a later turn or the user changed the document
        after the turn ended (``end_docs``), or the newest undo step above the turn's start is not one of Oscar's.
        ``{}`` means nothing was found; a document that cannot be checked is reported as such (ask, never assume)."""
        wanted = [str(name) for name in (names or ()) if name]
        if not wanted or not self._usable_snapshot(snapshot):
            return {name: "the turn's starting state was not recorded" for name in wanted}

        def task():
            live = App.listDocuments()
            changed: dict[str, str] = {}
            for name in wanted:
                doc = live.get(name)
                if doc is None:
                    continue   # restoring re-creates it: nothing of the user's is open there
                uid = (snapshot.get("uids") or {}).get(name)
                if uid and uid != str(doc.Uid):
                    changed[name] = "another document is now open under this name"
                    continue
                current = _undo_count(doc)
                end = (snapshot.get("end_docs") or {}).get(name)
                if end is not None and current != end:
                    changed[name] = "it was changed after this turn"
                    continue
                entries = list(doc.UndoNames)
                if not entries or is_oscar_step(entries[0]):
                    continue
                depth = int(snapshot["docs"].get(name, 0) or 0)
                recorded = (snapshot.get("undo_names") or {}).get(name)
                moved = recorded is not None and entries != list(recorded)   # a full stack: the count cannot move
                if current > depth or moved:
                    changed[name] = f"your own step '{entries[0]}' came after this turn"
            return changed

        result, failed = dispatch.dispatch(task, timeout=30.0, op="changed_since_turn")
        if failed or not isinstance(result, dict):
            return {name: f"it could not be checked ({_error_text(result)})" for name in wanted}
        return result

    @staticmethod
    def _turn_touched(doc, name: str, snapshot: dict) -> bool:
        """True when ``doc`` (open at the turn's start as ``name``) may hold changes made since: its undo count moved,
        its full history no longer matches the one recorded at the start, or its object names changed. A document
        whose name now belongs to another document (the uid differs) is not the turn's to revert."""
        uid = (snapshot.get("uids") or {}).get(name)
        if uid and uid != str(doc.Uid):
            return False
        if _undo_count(doc) != int((snapshot.get("docs") or {}).get(name, 0) or 0):
            return True
        recorded = (snapshot.get("undo_names") or {}).get(name)
        if recorded is not None and list(doc.UndoNames) != list(recorded):
            return True
        before = (snapshot.get("objects") or {}).get(name)
        return before is not None and sorted(o.Name for o in doc.Objects) != sorted(before)

    @staticmethod
    def _turn_undo_labels(doc, depth: int, snapshot: dict) -> list[str]:
        """The undo labels the turn left on ``doc``, newest first: the entries above the turn-start ``depth``. Empty
        when a later turn or the user changed the document after the turn ended (``end_docs``): the undo path leaves
        such a document alone anyway."""
        current = _undo_count(doc)
        end = (snapshot.get("end_docs") or {}).get(doc.Name)
        if end is not None and current != end:
            return []
        return [str(label) for label in list(doc.UndoNames)[:max(0, current - depth)]]

    def _close_new_documents(self, live: dict, snapshot: dict, report: dict) -> None:
        """``_close_turn_document`` for every open document that did not exist when the turn began. GUI thread only."""
        start = snapshot.get("docs") or {}
        for name, doc in list(live.items()):
            if name not in start:
                self._close_turn_document(doc, snapshot, report)

    def _undo_document(self, doc, depth: int, snapshot: dict, report: dict) -> None:
        """Pop ``doc``'s undo stack back to ``depth``, never past a step
        that is not Oscar's."""
        name = doc.Name
        uid = (snapshot.get("uids") or {}).get(name)
        if uid and uid != str(doc.Uid):
            return  # the name now belongs to a different document
        current = _undo_count(doc)
        end = (snapshot.get("end_docs") or {}).get(name)
        if end is not None and current != end:
            # a later turn or the user changed it after this turn ended;
            # popping would undo THEIR work first
            if end != depth:
                report["conflicts"].append(
                    f"'{name}' was changed after this turn, so nothing was undone there"
                )
            return

        wanted = current - depth
        undone, blocked, broken = 0, None, None
        while undone < wanted:
            top = list(doc.UndoNames)
            if not top or not is_oscar_step(top[0]):
                blocked = top[0] if top else "?"
                break
            try:
                doc.undo()
            except Exception as e:
                broken = e
                break
            undone += 1
        if undone:
            report["undone"][name] = undone
        if blocked is not None:
            report["warnings"].append(
                f"your own step '{blocked}' sits on top of this turn's changes in "
                f"'{name}', so {wanted - undone} of its step(s) were left in place"
            )
            return
        if broken is not None:
            report["warnings"].append(
                f"undo failed in '{name}' ({broken}); {wanted - undone} of this "
                "turn's step(s) were left in place"
            )
            return

        # FreeCAD forgets the oldest undo steps once the stack is full; if
        # that happened during the turn its early changes cannot be undone.
        cap = snapshot.get("undo_cap")
        cap = cap if isinstance(cap, int) else _undo_cap()
        # An overrun is remembered on the snapshot: the revert that found it
        # leaves the stack shorter than the cap, so a second click on the same
        # card would otherwise see nothing wrong and report a clean success
        # with the turn's objects still there.
        overrun = snapshot.setdefault("overrun", {})
        if cap > 0 and (depth >= cap or current >= cap or name in overrun):
            detail = self._history_overrun(doc, depth, snapshot)
            if detail is None:
                overrun.pop(name, None)
            else:
                overrun[name] = True
                done = f"undid {undone} step(s), but " if undone else ""
                report["warnings"].append(
                    f"in '{name}' {done}the undo history holds only {cap} steps, so "
                    f"earlier changes from this turn may still be present{detail} "
                    "- check the document tree"
                )

    @staticmethod
    def _history_overrun(doc, depth: int, snapshot: dict) -> str | None:
        """After a revert in a document whose undo history is full: None when
        it matches its state at the turn's start, else a description of how
        it still differs."""
        name = doc.Name
        before = (snapshot.get("objects") or {}).get(name)
        now = [o.Name for o in doc.Objects]
        if before is None:
            return ""  # nothing to compare against: assume the worst
        extra = sorted(set(now) - set(before))
        missing = sorted(set(before) - set(now))
        if extra or missing:
            parts = []
            if extra:
                parts.append("still present: " + ", ".join(extra[:8]) + ("…" if len(extra) > 8 else ""))
            if missing:
                parts.append("removed and not restorable: " + ", ".join(missing[:8]))
            return " (" + "; ".join(parts) + ")"
        recorded = (snapshot.get("undo_names") or {}).get(name)
        if recorded is not None and list(doc.UndoNames) != recorded:
            return ""  # a full history that no longer matches: edits only
        return None

    def _close_turn_document(self, doc, snapshot: dict, report: dict) -> None:
        """Close a document that did not exist when the turn began — unless it
        holds work that is not provably the agent's."""
        name, uid = doc.Name, str(doc.Uid)
        end_uids = snapshot.get("end_uids")
        if end_uids is not None and uid not in end_uids.values():
            return  # created by a later turn
        end = (snapshot.get("end_docs") or {}).get(name)
        if end is not None and _undo_count(doc) != end:
            report["conflicts"].append(
                f"'{name}' was created in this turn but changed afterwards, so it was left open"
            )
            return

        entries = list(doc.UndoNames)
        keep = None
        if doc.FileName:
            keep = "it was saved to disk"
        elif any(not is_oscar_step(entry) for entry in entries):
            keep = "it holds changes made outside Oscar"
        elif not (uid in self._agent_uids or entries or len(doc.Objects) == 0):
            keep = "it was not created by Oscar"
        if keep is not None:
            report["kept_docs"].append({"doc": name, "reason": keep})
            if uid in self._agent_uids:
                # the turn's own document is still open: not a finished revert
                report["warnings"].append(
                    f"'{name}' was created in this turn but left open ({keep})"
                )
            return
        try:
            App.closeDocument(name)
        except Exception as e:
            reason = f"it could not be closed: {e}"
            report["kept_docs"].append({"doc": name, "reason": reason})
            report["warnings"].append(f"'{name}' was created in this turn but left open ({reason})")
            return
        self._agent_uids.discard(uid)
        report["closed_docs"].append(name)

    # -------------------------------------------------------------------- FEM

    def run_fem(self, doc_name: str, analysis_name: str, timeout: int = 600) -> dict[str, Any]:
        """Run the CalculiX analysis ``analysis_name`` of a document (``timeout`` seconds); returns stress and displacement maxima."""
        def task():
            return self._run_fem_on_gui(doc_name, analysis_name)

        result, failed = dispatch.dispatch_stoppable(
            txn_task(task, "Oscar: run_fem", doc_name),
            timeout=float(timeout),
            op="run_fem",
        )
        return result if isinstance(result, dict) else {"success": False, "error": str(result)}

    def _run_fem_on_gui(self, doc_name: str, analysis_name: str) -> dict[str, Any]:
        doc = _resolve_doc(doc_name)
        if doc is None:
            return {"success": False, "error": f"document '{doc_name}' is not open"}
        analysis = doc.getObject(analysis_name)
        if analysis is None:
            return {"success": False, "error": f"analysis '{analysis_name}' not found"}

        try:
            import ObjectsFem
            from femtools import ccxtools, femutils

            # A ccx-tools solver is a Fem::FemSolverObjectPython: only its
            # Proxy.Type says which kind it is, and is_of_type reads that.
            solver = next(
                (m for m in analysis.Group if femutils.is_of_type(m, "Fem::SolverCcxTools")),
                None,
            )
            if solver is None:
                for maker_name in ("makeSolverCalculiXCcxTools", "makeSolverCalculixCcxTools"):
                    maker = getattr(ObjectsFem, maker_name, None)
                    if maker is not None:
                        solver = maker(doc, "SolverCcx")
                        break
            if solver is None:
                return {"success": False, "error": "could not create a CalculiX solver"}
            if solver not in analysis.Group:
                analysis.addObject(solver)
            # Both given: with no arguments FemToolsCcx solves the "active"
            # analysis, which it cannot resolve when a document has several.
            fea = ccxtools.FemToolsCcx(analysis, solver)
        except Exception as e:
            return {"success": False, "error": f"solver setup failed: {e}"}

        try:
            fea.update_objects()
            # run() checks the same prerequisites but reports a failure only
            # as a modal dialog in the GUI, so check first to give the model
            # the reason. The check wants an existing working directory;
            # run() sets up the real one (a stand-in avoids a dead temp dir).
            fea.working_dir = tempfile.gettempdir()
            problem = fea.check_prerequisites()
            if problem:
                return {"success": False, "error": f"prerequisites: {problem.strip()}"}
            # True when CalculiX solved and the results were read in
            if not fea.run():
                detail = _ccx_errors(fea)
                return {
                    "success": False,
                    "error": "CalculiX failed" + (f": {detail}" if detail else "")
                    + " (the report view has the solver output)",
                    "working_dir": fea.working_dir,
                }
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

        summary: dict[str, Any] = {"success": True, "working_dir": fea.working_dir}
        try:
            for member in analysis.Group:
                if not femutils.is_of_type(member, "Fem::ResultMechanical"):
                    continue
                displacement = getattr(member, "DisplacementLengths", None)
                if displacement:
                    summary["max_displacement_mm"] = max(float(v) for v in displacement)
                    summary["node_count"] = len(getattr(member, "NodeNumbers", []) or [])
                stress = getattr(member, "vonMises", None)
                if stress:
                    # solver writes MPa for the stress norm in results
                    summary["max_von_mises_MPa"] = max(float(v) for v in stress)
        except Exception as e:
            summary["stats_error"] = str(e)
        return summary
