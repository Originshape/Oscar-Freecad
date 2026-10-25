"""Oscar's tool registry — the CAD operations the chat can invoke.

Each tool pairs a JSON Schema with a prompt-engineered description and an
in-process handler. Handlers call the shared CadEngine (``oscar.cad``),
which ferries document/GUI work onto the main thread — so tools are safe to
run from the agent's worker thread.

Handlers return ``(text, image_b64)`` where the image is a viewport
screenshot for vision turns, and raise ``ToolError`` when the operation
failed. Failures reach the model as ``is_error=True`` results (and the panel
marks the step ✗) so the model can read the error and self-correct.

The registry also carries each tool's metadata (category, mutating, geometry,
phase mask), the scoped ``specs()`` the agent sends to the
model, and ``run_tool``: the one pipeline every call goes through (argument
repair, handler, health check of the model, Project step record, ledger
auto-tick). ``specs`` / ``run_tool`` and the rest of the registry API are
implemented in ``oscar/toolrun.py`` and re-exported here; pure helpers (clipping, titles,
argument validation, health text) live in ``oscar/toolpipe.py``.
"""

import importlib
import json
import os
from typing import Any, Callable, Iterable

from . import cad as _cad
from .cad import imported as _imported
from . import codeguard
from . import netguard
from . import toolpipe
from . import turnlog as _turnlog
from .providers import ToolSpec
from .toolctx import current_context, use_context, ToolContext, ToolOutcome  # noqa: F401
from .toolpipe import clip_text  # noqa: F401  (public: tools.clip_text)

VIEW_NAMES = [
    "Isometric", "Front", "Top", "Right", "Back", "Left", "Bottom", "Dimetric", "Trimetric",
]


class ToolError(Exception):
    """A tool ran but the operation failed; the message is model-facing."""


def _engine():
    return _cad.get_engine()


def _screenshot(view_name: str = "Isometric", doc_name: str | None = None) -> str | None:
    """Screenshot of the document the tool just worked on (not whichever
    document happens to be active — that one may show nothing new).

    Returns the PNG as base64, or ``None`` when no valid picture could be taken (headless
    session, no 3D view in front, blank frame, GUI busy): never an error string that a
    caller could mistake for an image. Handlers that want to tell the model WHY there is no
    picture use ``_shot`` instead."""
    return _screenshot_with_reason(view_name, doc_name)[0]


def _screenshot_with_reason(view_name: str = "Isometric",
                            doc_name: str | None = None) -> tuple[str | None, str]:
    """``(png_base64 | None, reason)``; ``reason`` is ``""`` when an image was captured."""
    return _engine()._screenshot_with_reason(view_name, doc_name=doc_name)


# Reasons that are not worth telling the model about: a console session simply has no 3D view.
_QUIET_REASONS = ("no 3D view (headless session)",)


def _shot(text: str, include: bool, view_name: str = "Isometric",
          doc_name: str | None = None) -> tuple[str, str | None]:
    """Result tail of a handler that offers ``include_screenshot``: ``(text, image_b64 | None)``.

    When a screenshot was wanted but none could be taken in a GUI session, one line says
    why (``(no screenshot: blank frame)``), so the model knows it is not seeing the result
    instead of assuming the picture it did not get showed nothing wrong.
    """
    if not include:
        return text, None
    image, reason = _screenshot_with_reason(view_name, doc_name)
    if image is None and reason and reason not in _QUIET_REASONS:
        text += f"\n(no screenshot: {reason})"
    return text, image


def _read(call, *args, **kwargs):
    """Run an engine read. The engine raises RuntimeError when the GUI task
    failed or timed out; that must reach the model as an error, never as an
    'empty document' or 'not found' answer."""
    try:
        return call(*args, **kwargs)
    except RuntimeError as e:
        raise ToolError(f"Oscar's GUI thread did not answer: {e}") from e


def _warned(text: str, res: dict) -> str:
    """Append a warning the engine attached to a successful result: the
    operation went through, but the model must know what it left behind."""
    warning = res.get("warning")
    return f"{text} WARNING: {warning}." if warning else text


def _require_document(doc_name: str) -> str:
    """The real name of an open document, or a ToolError listing the open
    ones — so a wrong name is never mistaken for an empty document."""
    actual = _engine().find_document(doc_name)
    if actual:
        return actual
    open_docs = _read(_engine().list_documents)
    listing = ", ".join(open_docs) if open_docs else "none — create one with create_document"
    raise ToolError(f"Document '{doc_name}' is not open. Open documents: {listing}.")


def _download_dir() -> str:
    """Where ``download_file`` saves: ``netguard.default_download_dir()`` (env
    ``OSCAR_DOWNLOAD_DIR``) when it exists, else ``~/Downloads/Oscar``."""
    resolver = getattr(netguard, "default_download_dir", None)
    if callable(resolver):
        try:
            path = resolver()
            if path:
                return path
        except Exception:  # noqa: BLE001 - fall back rather than fail the download
            pass
    return os.path.join(os.path.expanduser("~"), "Downloads", "Oscar")


ALL_PHASES = ("INTAKE", "SOURCE", "PLAN", "PARAMS", "BUILD", "VERIFY", "DELIVER")
CATEGORIES = ("session", "read", "build", "params", "source", "verify", "deliver",
              "plan", "fem", "net")


# The universal optional argument of every tool (the registry adds it to the schema of mutating
# tools, ``run_tool`` removes it before the handler is bound; handlers never declare it).
UNIVERSAL_ARGS = {
    "step_title": {
        "type": "string",
        "description": "Optional 3-8 word title for this step, shown to the user "
                       "(for example 'Cut the USB-C opening').",
    },
}

# Optional properties dropped from the schema copy of ``lite`` specs (handlers keep their defaults).
_LITE_DROPPED_PROPERTIES = ("step_title", "view_name", "timeout", "include_screenshot")


class Tool:
    """A registered tool: schema, handler and the metadata the flow code reads."""

    def __init__(self, name, description, parameters, handler, *, category="read",
                 mutating=False, geometry=False, phases=None, title=None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.category = category                      # one of CATEGORIES
        self.geometry = bool(geometry)                # model-changing: health + delta + snapshot + checkpoint
        self.mutating = bool(mutating or geometry)    # recorded as a step record
        self.phases = tuple(phases) if phases else ALL_PHASES
        self.title = title                            # Callable[[dict], str] | None (row title from the arguments)

    def spec(self, *, lite: bool = False) -> ToolSpec:
        """The ToolSpec the model sees. The stored ``parameters`` are never modified.

        A mutating tool's schema gains the universal optional ``step_title``. With ``lite`` the
        description shrinks to its first sentence (<= 200 chars) and the optional properties
        ``step_title``, ``view_name``, ``timeout`` and ``include_screenshot`` are removed
        (handlers keep their defaults; ``run_tool`` forces ``include_screenshot=False`` in lite
        turns). Required properties are never removed.
        """
        schema = dict(self.parameters)
        properties = dict(schema.get("properties") or {})
        if self.mutating:
            for key, value in UNIVERSAL_ARGS.items():
                properties.setdefault(key, value)
        description = self.description
        if lite:
            required = set(schema.get("required") or ())
            for key in _LITE_DROPPED_PROPERTIES:
                if key not in required:
                    properties.pop(key, None)
            description = toolpipe.first_sentence(description, 200)
        schema["properties"] = properties
        return ToolSpec(self.name, description, schema)


_HANDLERS: dict[str, Tool] = {}


def _P(mask: str) -> tuple[str, ...]:
    """Phase mask (``-X--X--`` over INTAKE SOURCE PLAN PARAMS BUILD VERIFY DELIVER)
    as the tuple of phases it introduces the tool in."""
    return tuple(phase for phase, flag in zip(ALL_PHASES, mask) if flag == "X")


def tool(name: str, description: str, parameters: dict[str, Any], *, category="read",
         mutating=False, geometry=False, phases=None, title=None):
    """Register a tool: ``@tool(name, description, schema, category=..., mutating=..., geometry=..., phases=..., title=...)``.

    The metadata is the tool's row of the phase matrix; ``geometry`` implies ``mutating``. The
    decorated handler is returned unchanged and stored in ``_HANDLERS`` (registration order = the
    order of ``specs()``).
    """
    def register(fn: Callable[..., tuple[str, str | None]]):
        _HANDLERS[name] = Tool(name, description, parameters, fn, category=category,
                               mutating=mutating, geometry=geometry, phases=phases, title=title)
        return fn

    return register


# --------------------------------------------------------------------------
# Schemas


_OBJ_TYPE_NOTES = """``obj_type`` must name a type registered in the C++ type system
("Part::", "PartDesign::", "Fem::", ...). These Python-implemented types are
supported through dedicated factories:

    Draft::Circle     Radius
    Draft::Rectangle  Length, Height
    Draft::Polygon    FacesNumber, Radius (or Points)
    Draft::Wire       Points (list of {x, y, z}), optional Closed

FEM types (analysis containers, materials, constraints, Gmsh meshes) are
built through the FEM factories; Gmsh meshes are generated on creation and
added to the named analysis (a Gmsh mesh needs Properties {"Shape": "<solid
name>"}, optionally CharacteristicLengthMax/Min in mm). Other Python-implemented types must be built
with execute_code instead. Draft factories name objects themselves; the
returned object_name differs from the requested one. Always use the returned
name in later calls. Hollow shapes like tubes: two cylinders + a boolean
cut, or execute_code. Booleans: Part::Cut/Part::Fuse/Part::Common take
"Base" and "Tool" object names; Part::MultiFuse/Part::MultiCommon take
"Shapes": ["A", "B", …]; their inputs are hidden automatically. A failed
create leaves nothing behind, so a corrected retry can reuse the name."""

_INCLUDE_SHOT_DOC = (
    "include_screenshot: return a screenshot of the model (default True). Set False to "
    "save tokens when the image would not be informative; orient with view_name "
    '(default "Isometric") to show the change being made.'
)
_READ_SHOT_DOC = (
    "include_screenshot: also return a screenshot (default False — the JSON is "
    "usually what you need)."
)

_SCHEMA_DOC_NAME = {"type": "string", "description": "Name of the Oscar document"}
_SCHEMA_OBJ_NAME = {"type": "string", "description": "Internal Name of the object (not the Label)"}


# --------------------------------------------------------------------------
# Handlers


@tool(
    "create_document",
    "Create a new document. Returns the document's actual name — Oscar "
    "sanitises and de-duplicates names ('My Doc' -> 'My_Doc', 'Doc' -> 'Doc001'); "
    "always use the returned name in later calls.",
    {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    category="session", mutating=True, phases=_P("XXXXX--"),
)
def _create_document(name: str):
    res = _engine().create_document(name)
    if res.get("success"):
        return f"Document '{res['document_name']}' created and active.", None
    raise ToolError(f"Failed to create document: {res.get('error')}")


@tool(
    "create_object",
    f"Create a new object in a document. {_OBJ_TYPE_NOTES}\n\n"
    "Example properties: Height/Radius/Placement for primitives "
    '({"Base": {"x":…, "y":…, "z":…}, "Rotation": {"Axis": {"x":0,"y":0,"z":1}, "Angle": 45}}, '
    "Angle in degrees); "
    'booleans reference parts by name ("Base": "Box", "Tool": "Cylinder"); '
    '"ViewObject": {"ShapeColor": [r, g, b]} (0–1 floats) sets appearance. '
    f"{_INCLUDE_SHOT_DOC}",
    {
        "type": "object",
        "properties": {
            "doc_name": _SCHEMA_DOC_NAME,
            "obj_type": {"type": "string", "description": "e.g. 'Part::Box', 'Part::Cut', 'PartDesign::Body', 'Fem::AnalysisPython'"},
            "obj_name": {"type": "string"},
            "analysis_name": {"type": "string", "description": "FEM analysis container to add the object to (FEM types only)"},
            "obj_properties": {"type": "object", "description": "Property name -> value map"},
            "body": {"type": "string",
                     "description": "PartDesign::Body that receives a Sketcher / PartDesign feature "
                                    "(default: the document's only Body; required when it has several)"},
            "include_screenshot": {"type": "boolean", "default": True},
            "view_name": {"type": "string", "enum": VIEW_NAMES, "default": "Isometric"},
        },
        "required": ["doc_name", "obj_type", "obj_name"],
    },
    category="build", geometry=True, phases=_P("----XX-"),
)
def _create_object(doc_name: str, obj_type: str, obj_name: str, analysis_name: str | None = None,
                   obj_properties: dict | None = None, include_screenshot: bool = True,
                   view_name: str = "Isometric", body: str | None = None):
    obj_data = {
        "Name": obj_name,
        "Type": obj_type,
        "Properties": obj_properties or {},
        "Analysis": analysis_name,
    }
    if body:
        obj_data["Body"] = body
    res = _engine().create_object(doc_name, obj_data)
    if not res.get("success"):
        raise ToolError(f"Failed to create object: {res.get('error')}")
    text = f"Object '{res['object_name']}' created in '{res.get('document_name', doc_name)}'."
    if res.get("object_name") != obj_name:
        text += f" (named '{res['object_name']}' — use that name from now on)"
    if res.get("body"):
        text += f" It is part of body '{res['body']}'."
    return _shot(text, include_screenshot, view_name, doc_name)


@tool(
    "edit_object",
    "Edit properties of an existing object. Placement is edited as "
    '{"Base": {"x":…, "y":…, "z":…}, "Rotation": {"Axis": {…}, "Angle": …}} '
    "(degrees); links (Base/Tool/Shapes/Source/Profile) accept object names; "
    'References take [{"object_name": …, "face"/"edge": …}]. '
    f"{_INCLUDE_SHOT_DOC}",
    {
        "type": "object",
        "properties": {
            "doc_name": _SCHEMA_DOC_NAME,
            "obj_name": _SCHEMA_OBJ_NAME,
            "obj_properties": {"type": "object"},
            "include_screenshot": {"type": "boolean", "default": True},
            "view_name": {"type": "string", "enum": VIEW_NAMES, "default": "Isometric"},
        },
        "required": ["doc_name", "obj_name", "obj_properties"],
    },
    category="build", geometry=True, phases=_P("----XX-"),
)
def _edit_object(doc_name: str, obj_name: str, obj_properties: dict,
                 include_screenshot: bool = True, view_name: str = "Isometric"):
    res = _engine().edit_object(doc_name, obj_name, obj_properties)
    if not res.get("success"):
        raise ToolError(f"Failed to edit object: {res.get('error')}")
    text = f"Object '{res['object_name']}' edited."
    if res.get("applied"):
        text += f" Set: {', '.join(res['applied'])}."
    if res.get("notes"):
        text += " Note: " + "; ".join(str(n).rstrip(". ") for n in res["notes"]) + "."
    text = _warned(text, res)
    return _shot(text, include_screenshot, view_name, doc_name)


@tool(
    "delete_object",
    "Delete an object from a document. An object that other objects depend on (a boolean's "
    "input, a sketch used by a feature, a spreadsheet that drives expressions) is NOT deleted "
    "unless cascade=true, which deletes those dependents first. "
    f"{_INCLUDE_SHOT_DOC}",
    {
        "type": "object",
        "properties": {
            "doc_name": _SCHEMA_DOC_NAME,
            "obj_name": _SCHEMA_OBJ_NAME,
            "cascade": {"type": "boolean", "default": False,
                        "description": "Also delete every object that depends on this one"},
            "include_screenshot": {"type": "boolean", "default": True},
            "view_name": {"type": "string", "enum": VIEW_NAMES, "default": "Isometric"},
        },
        "required": ["doc_name", "obj_name"],
    },
    category="build", geometry=True, phases=_P("----XX-"),
)
def _delete_object(doc_name: str, obj_name: str, include_screenshot: bool = True,
                   view_name: str = "Isometric", cascade: bool = False):
    res = _engine().delete_object(doc_name, obj_name, cascade=cascade)
    if not res.get("success"):
        raise ToolError(f"Failed to delete object: {res.get('error')}")
    text = f"Object '{res['object_name']}' deleted."
    if res.get("cascaded"):
        text += f" Also deleted (they depended on it): {', '.join(res['cascaded'])}."
    if res.get("removed_children"):
        text += f" Also deleted its features: {', '.join(res['removed_children'])}."
    if res.get("restored"):
        text += f" Its inputs are visible again: {', '.join(res['restored'])}."
    return _shot(text, include_screenshot, view_name, doc_name)


@tool(
    "execute_code",
    "Execute arbitrary Python in the application's GUI thread with the FreeCAD, App, "
    "FreeCADGui and Gui API modules pre-imported; printed output is returned (on an "
    "error: the failing line, the traceback and the output printed before it). The best "
    "tool for repeated or computed geometry (bolt circles, arrays, patterns) and anything "
    "the structured tools cannot express; documents with pending changes are "
    "recomputed after the script. Use execute_code_async for heavy pure-geometry work and "
    "execute_code_headless for OCCT work that can crash. "
    "After saving a document another process modified, call reload_document. "
    f"{_INCLUDE_SHOT_DOC}",
    {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "include_screenshot": {"type": "boolean", "default": True},
            "view_name": {"type": "string", "enum": VIEW_NAMES, "default": "Isometric"},
            "timeout": {"type": "number", "description": "Seconds for each queue+run budget (default 90, cap 1800)"},
        },
        "required": ["code"],
    },
    category="build", geometry=True, phases=_P("-X-XXX-"),
)
def _execute_code(code: str, include_screenshot: bool = True, view_name: str = "Isometric",
                  timeout: float | None = None):
    allowed, message = codeguard.gate_execution(code, 'execute_code')   # scripts are off by default / asked per call
    if not allowed:
        raise ToolError(message)
    allowed, message = netguard.gate_code(code, 'execute_code')
    if not allowed:
        raise ToolError(message)
    res = _engine().execute_code(code, timeout)
    if not res.get("success"):
        raise ToolError(f"Failed to execute code: {res.get('error')}")
    text = f"Code executed. {res.get('message', '')}".rstrip()
    return _shot(text, include_screenshot, view_name)


@tool(
    "execute_code_async",
    "Run Python in a background thread WITHOUT waiting. The code must not touch "
    "documents, the GUI or views directly — the application is not thread-safe and doing "
    "so wedges the app. Hand every document/view write to the GUI thread with the "
    "injected commit(fn) helper: build shapes in the background "
    "(Part.makeBox, base.fuse(...) — the slow part), then "
    "def apply(): obj.Shape = fused; doc.recompute() and commit(apply). "
    "Returns a job_id to poll with get_async_status.",
    {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    category="build", geometry=True, phases=_P("----XX-"),
)
def _execute_code_async(code: str):
    allowed, message = codeguard.gate_execution(code, 'execute_code_async')   # scripts are off by default / asked per call
    if not allowed:
        raise ToolError(message)
    allowed, message = netguard.gate_code(code, 'execute_code_async')
    if not allowed:
        raise ToolError(message)
    res = _engine().execute_code_async(code)
    if not res.get("success"):
        raise ToolError(f"Failed to start async execution: {res.get('error')}")
    return (
        f"Background job started (job_id: {res.get('job_id')}).\n"
        + res.get("message", "Poll get_async_status(job_id) for completion."),
        None,
    )


@tool(
    "execute_code_headless",
    "Run a complete Oscar Python script in an isolated headless engine "
    "process. For geometry work that can crash or block: helical threads "
    "(makeHelix+makePipeShell), lofts/sweeps, many-tool booleans, long "
    "parametric rebuilds. A native crash kills only the helper; open documents "
    "survive. The script runs without GUI: import the API modules yourself, open "
    "documents from disk, save with doc.save()/saveAs(); print progress to "
    "stdout. After it saved a .FCStd that is open here, call reload_document.",
    {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Complete standalone script"},
            "timeout": {"type": "number", "default": 600},
        },
        "required": ["code"],
    },
    category="build", mutating=True, phases=_P("----XX-"),
)
def _execute_code_headless(code: str, timeout: float = 600):
    allowed, message = codeguard.gate_execution(code, 'execute_code_headless')   # scripts are off by default / asked per call
    if not allowed:
        raise ToolError(message)
    allowed, message = netguard.gate_code(code, 'execute_code_headless')
    if not allowed:
        raise ToolError(message)
    res = _engine().execute_code_headless(code, timeout)
    if res.get("success"):
        text = "Headless script finished (exit 0)."
    else:
        reason = res.get("error")
        if not reason and res.get("crashed"):
            reason = f"the headless engine CRASHED (exit status {res.get('returncode')})"
        elif not reason and res.get("returncode") is not None:
            reason = f"script exited with code {res['returncode']}"
        text = f"Headless script FAILED: {reason or 'unknown error'}"
        if res.get("timed_out"):
            text += (" Raise 'timeout' or split the work into smaller steps "
                     "that each finish in time.")
    if res.get("output"):
        text += "\nOutput:\n" + str(res["output"]).rstrip()
    text += (
        "\nNOTE: the headless helper is a SEPARATE process — everything it "
        "created exists only there unless the script saved files to disk. "
        "Nothing appears in this session until you open/import those files "
        "(import_file, or reload_document for an already-open file)."
    )
    if not res.get("success"):
        raise ToolError(text)
    return text, None


@tool(
    "get_async_status",
    "Report state of background jobs started by execute_code_async "
    "(running/done/failed, with error and traceback). Answers even while a "
    "job runs. With no job_id, lists all running jobs and recent history.",
    {"type": "object", "properties": {"job_id": {"type": "string", "default": ""}}},
    category="read", phases=_P("----XX-"),
)
def _get_async_status(job_id: str = ""):
    res = _engine().get_async_status(job_id)
    if not res.get("success"):
        raise ToolError(f"Failed to get async status: {res.get('error')}")
    job = res.get("job")
    if job is None:
        return json.dumps(res.get("jobs", []), indent=1), None
    text = f"Async job {job['id']}: {job.get('state', 'unknown')}"
    if job.get("state") == "running":
        text += " — STILL RUNNING: poll again until done/failed before reporting any result."
    if job.get("error"):
        text += f"\nError: {job['error']}"
    if job.get("traceback"):
        text += f"\n{job['traceback']}"
    return text, None


@tool(
    "get_objects",
    "List the objects of a document as a compact table (name, label, type, valid, "
    "solids, shape type, visibility, volume, bounding box, expressions, sketch DoF). "
    "The workspace state in each user message already names the objects; use this when "
    "you need their state. Origin helpers are hidden. filter keeps rows whose name, "
    "label or type contains the text; limit/offset page through long documents; "
    "get_object gives one object's full properties. "
    f"{_READ_SHOT_DOC}",
    {
        "type": "object",
        "properties": {
            "doc_name": _SCHEMA_DOC_NAME,
            "filter": {"type": "string",
                       "description": "Only rows whose Name, Label or Type contains this text (case-insensitive)"},
            "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500,
                      "description": "Maximum rows returned"},
            "offset": {"type": "integer", "default": 0, "minimum": 0,
                       "description": "Rows to skip (use the offset printed under a truncated table)"},
            "include_screenshot": {"type": "boolean", "default": False},
            "view_name": {"type": "string", "enum": VIEW_NAMES, "default": "Isometric"},
        },
        "required": ["doc_name"],
    },
    category="read", phases=_P("XXXXXXX"),
)
def _get_objects(doc_name: str, include_screenshot: bool = False, view_name: str = "Isometric",
                 filter: str = "", limit: int = 50, offset: int = 0):
    from .cad import serialize as _serialize

    actual = _require_document(doc_name)
    page = _read(_engine()._object_page, actual, filter=filter or "", limit=limit, offset=offset)
    if isinstance(page, dict) and page.get("error"):
        # a failed GUI read is an error, never "the document is empty"
        raise ToolError(str(page["error"]))
    text = _serialize.render_objects_table(actual, page, filter=filter or "", limit=limit, offset=offset)
    return _shot(text, include_screenshot, view_name, actual)


@tool(
    "get_object",
    "Get one object's serialized properties (dimensions, placement, state, "
    "volume, bounding box, expressions, links, and PropertiesList: every property "
    "name that can be edited) — use it to verify a result or see what can be edited. "
    f"{_READ_SHOT_DOC}",
    {
        "type": "object",
        "properties": {
            "doc_name": _SCHEMA_DOC_NAME,
            "obj_name": _SCHEMA_OBJ_NAME,
            "include_screenshot": {"type": "boolean", "default": False},
            "view_name": {"type": "string", "enum": VIEW_NAMES, "default": "Isometric"},
        },
        "required": ["doc_name", "obj_name"],
    },
    category="read", phases=_P("XXXXXXX"),
)
def _get_object(doc_name: str, obj_name: str, include_screenshot: bool = False,
                view_name: str = "Isometric"):
    actual = _require_document(doc_name)
    obj = _read(_engine().get_object, actual, obj_name, detail=True)
    if obj is None:
        raise ToolError(
            f"Object '{obj_name}' not found in document '{actual}' — use the internal "
            "Name (see get_objects), not the Label."
        )
    text = json.dumps(obj, indent=1, default=str)
    return _shot(text, include_screenshot, view_name, actual)


@tool(
    "get_view",
    "Capture a screenshot of the 3D view (the model is visible to you). Views: "
    + ", ".join(VIEW_NAMES)
    + ". doc_name shows that document (default: the active one); focus_object "
    "fits one object, otherwise everything is framed. width/height override the "
    "viewport size (longest edge capped at 1024).",
    {
        "type": "object",
        "properties": {
            "view_name": {"type": "string", "enum": VIEW_NAMES, "default": "Isometric"},
            "doc_name": _SCHEMA_DOC_NAME,
            "width": {"type": "integer"},
            "height": {"type": "integer"},
            "focus_object": _SCHEMA_OBJ_NAME,
        },
    },
    category="read", phases=_P("XXXXXXX"),
)
def _get_view(view_name: str = "Isometric", width: int | None = None, height: int | None = None,
              focus_object: str | None = None, doc_name: str | None = None):
    if doc_name:
        doc_name = _require_document(doc_name)
    shot, reason = _engine()._screenshot_with_reason(
        view_name, width, height, focus_object, doc_name=doc_name)
    if shot:
        return "Screenshot captured.", shot
    raise ToolError(
        f"Cannot capture a screenshot: {reason or 'no 3D view is active'} (for example no "
        "document is open, a TechDraw/Spreadsheet view is in front, or the application has "
        "no GUI). Use get_objects / get_object to inspect the model instead."
    )


@tool(
    "list_documents",
    "List the names of the currently open documents. Names are the internal "
    "document identifiers used by every other tool; run this first when "
    "unsure which document to target.",
    {"type": "object", "properties": {}},
    category="read", phases=_P("XXXXXXX"),
)
def _list_documents():
    try:
        docs = _engine().list_documents()
    except Exception as e:
        raise ToolError(f"Failed to list documents: {e}") from e
    return json.dumps(docs), None


@tool(
    "reload_document",
    "Close and re-open a document from disk so the GUI picks up changes made "
    "outside this process (e.g. by execute_code_headless). Use AFTER the "
    "document's .FCStd file was modified externally. A document with unsaved "
    "changes in the GUI is NOT reloaded (they would be lost) unless "
    "discard_unsaved is true: set that only after the user said the unsaved "
    "edits may be thrown away.",
    {
        "type": "object",
        "properties": {
            "doc_name": _SCHEMA_DOC_NAME,
            "discard_unsaved": {"type": "boolean", "default": False},
        },
        "required": ["doc_name"],
    },
    category="session", mutating=True, phases=_P("----XXX"),
)
def _reload_document(doc_name: str, discard_unsaved: bool = False):
    res = _engine().reload_document(doc_name, discard_unsaved=bool(discard_unsaved))
    if res.get("success"):
        text = f"Document '{res['document_name']}' reloaded from disk."
        if res["document_name"] != res.get("previous_name", res["document_name"]):
            text += (f" It was named '{res['previous_name']}' before; use "
                     f"'{res['document_name']}' from now on.")
        if res.get("rescue_copy"):
            text += (" The in-memory version had unsaved changes that the reload discarded; "
                     f"a copy of it was kept at {res['rescue_copy']}.")
        return _warned(text, res), None
    raise ToolError(f"Failed to reload document: {res.get('error')}")


@tool(
    "import_file",
    "Import a file into a document: STEP/IGES/BREP (solids), STL/OBJ/3MF/PLY "
    "(meshes), DXF/SVG (drawings), or open an FCStd document. Files the user "
    "attaches to the chat arrive in the prompt with their absolute paths — "
    "pass such a path here unchanged. Returns the document name and each "
    "created object's kind, size and position. A mesh can be looked at and "
    "measured but not cut, filleted or fused: set as_solid when the design "
    "must change the part or build on it.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path of the file to import"},
            "doc_name": {
                "type": "string",
                "description": "Target document (default: active document; a new one is created if none is open)",
            },
            "as_solid": {
                "type": "boolean",
                "default": False,
                "description": "STL/OBJ/3MF/PLY only: make each mesh a solid (closed meshes up to "
                               f"{_imported.MAX_SOLID_TRIANGLES:,} triangles) so it can be cut, filleted and fused",
            },
            "include_screenshot": {"type": "boolean", "default": True},
            "view_name": {"type": "string", "enum": VIEW_NAMES, "default": "Isometric"},
        },
        "required": ["path"],
    },
    category="source", geometry=True, phases=_P("-X--X--"),
)
def _import_file(path: str, doc_name: str | None = None, include_screenshot: bool = True,
                 view_name: str = "Isometric", as_solid: bool = False):
    if path.lower().startswith(("http://", "https://")):
        raise ToolError("import_file takes a LOCAL path. For a URL, ask the user "
                        "via download_file (it requests permission and saves to "
                        "~/Downloads/Oscar), then import the returned path.")
    res = _engine().import_file(path, doc_name or "", as_solid=bool(as_solid))
    if not res.get("success"):
        raise ToolError(f"Failed to import file: {res.get('error')}")
    return _shot(_imported_text(path, res), include_screenshot, view_name, res["document_name"])


def _imported_text(path: str, res: dict) -> str:
    """What the model reads of an import: the new objects, each one's kind, size and position, and what a mesh
    cannot do."""
    objects = res.get("objects") or []
    listing = ", ".join(objects[:20]) or "—"
    parts = res.get("parts") or []
    lines = [
        f"Imported '{os.path.basename(path)}' into document '{res['document_name']}' "
        f"({res.get('count', len(objects))} new objects: {listing}).",
        *_imported.lines(parts),
    ]
    lines += [f"Note: {note}." for note in res.get("solid_notes") or []]
    if any(part["kind"] == "mesh" for part in parts) and not res.get("solid_notes"):
        lines.append("A mesh can be looked at and measured, but not cut, filleted or fused. To change it or build on "
                     "it, delete it and import the file again with as_solid=true.")
    return "\n".join(lines)






@tool(
    "get_parts_list",
    "List the parts (.FCStd files) available in the parts library.",
    {"type": "object", "properties": {}},
    category="read", phases=_P("-X--X--"),
)
def _get_parts_list():
    try:
        parts = _engine().parts_list()
    except Exception as e:
        raise ToolError(f"Failed to get parts list: {e}") from e
    if parts:
        return json.dumps(parts, indent=1), None
    return "No parts library installed.", None


@tool(
    "insert_part_from_library",
    f"Insert a .FCStd part from the parts library into the active document. "
    f"Use relative_path as listed by get_parts_list. {_INCLUDE_SHOT_DOC}",
    {
        "type": "object",
        "properties": {
            "relative_path": {"type": "string", "description": "e.g. 'Fasteners/Screw/M3x10.FCStd'"},
            "include_screenshot": {"type": "boolean", "default": True},
            "view_name": {"type": "string", "enum": VIEW_NAMES, "default": "Isometric"},
        },
        "required": ["relative_path"],
    },
    category="build", geometry=True, phases=_P("----X--"),
)
def _insert_part_from_library(relative_path: str, include_screenshot: bool = True,
                              view_name: str = "Isometric"):
    res = _engine().insert_part(relative_path)
    if not res.get("success"):
        raise ToolError(f"Failed to insert part from library: {res.get('error')}")
    text = f"Part inserted from library: {res.get('message', '')}"
    return _shot(text, include_screenshot, view_name)
















# --------------------------------------------------------------------------
# Structured parts: apply a validated FeaturePlan to a document as a native
# spreadsheet-driven PartDesign tree. The plan JSON is CAD-neutral (schema v1
# and v2); the FreeCAD translator lives in oscar/plan/builder.py.

_PLAN_SCHEMA_HINT = (
    "FeaturePlan: schema_version 1 or 2, id, part_class, units='mm', target='freecad', frame (datum planes), "
    "parameters (name → {value, source{kind: user_requirement|datasheet|step_measurement|standard|assumption, "
    "ref} or evidence{op,detail} or assumption}; a derived one is {expr:'plate_w/2'}), features = ordered list of "
    "{id, op, ...} where every dimension is a parameter NAME. v1 ops: sketch (rectangle|circle|slot|profile), pad, "
    "pocket, revolution, hole. v2 adds fillet|chamfer {size, edges: all_outer|vertical|top_outer|bottom_outer|"
    "edges_of_feature:<id>}, thickness {wall, open_faces: top|bottom}, linear_pattern {of, count, spacing, "
    "direction}, polar_pattern {of, count, angle, axis}, mirror {of, plane}, boss {face, profile, length}; a sketch "
    "may take several 'profiles'; 'parts' [{id, features, placement{x,y,z,rx,ry,rz: parameter names}}] builds one "
    "Body per part, never fused, plus 'assembly' {placements:[{name, part, x, y, z}]}. Example v2: "
    '{"schema_version":2,"id":"plate","part_class":"bracket","units":"mm","frame":{"planes":{"base":{"origin":'
    '[0,0,0],"normal":[0,0,1]}}},"parameters":{"plate_w":{"value":60,"source":{"kind":"user_requirement","ref":'
    '"user: 60 mm wide"}},"plate_t":{"value":4,"source":{"kind":"user_requirement","ref":"user: 4 mm thick"}},'
    '"corner_r":{"value":5,"source":{"kind":"assumption","ref":"typical corner radius"}}},"features":[{"id":"base_sk",'
    '"op":"sketch","plane":{"frame_plane":"base"},"profile":{"shape":"rectangle","width":"plate_w","height":'
    '"plate_w"}},{"id":"base_pad","op":"pad","sketch":"base_sk","length":"plate_t"},{"id":"corners","op":"fillet",'
    '"size":"corner_r","edges":"vertical"}]}'
)


@tool(
    "apply_feature_plan",
    f"Build an editable parametric part (a plate, flange, bracket, fixture or any prismatic part) from a "
    f"FeaturePlan: "
    f"creates a Parameters spreadsheet (one aliased row per parameter), a "
    f"PartDesign Body per part, real constrained sketches and pad/pocket/revolution/"
    f"hole/fillet/chamfer/shell/pattern/mirror/boss features — every dimension expression-bound to the "
    f"spreadsheet, so changing a cell reshapes the model. Validation errors name the field and the nearest valid "
    f"spelling: fix the plan and send it again. "
    f"Each feature is checkpointed; on failure the document stays at the "
    f"last good feature and the failing feature id + error come back. "
    f"A previous apply_feature_plan build in the document (its Parameters "
    f"sheet and Body, together with anything added to that Body) is "
    f"replaced; your own sheets and Bodies are never touched - the new "
    f"build then gets its own Parameters001/Body001. "
    f"{_PLAN_SCHEMA_HINT}",
    {
        "type": "object",
        "properties": {
            "doc_name": _SCHEMA_DOC_NAME,
            "plan": {"type": "object", "description": "FeaturePlan object (schema_version 1 or 2)"},
        },
        "required": ["doc_name", "plan"],
    },
    category="build", geometry=True, phases=_P("----X--"),
)
def _apply_feature_plan(doc_name: str, plan: dict):
    def task():
        from .cad.engine import _resolve_doc
        from .plan import builder

        doc = _resolve_doc(doc_name)
        if doc is None:
            return {"success": False, "error": f"document '{doc_name}' is not open"}
        return builder.apply_plan(doc, plan)

    from .cad import dispatch as _d

    result, _ = _d.dispatch_stoppable(task, timeout=600.0, op="plan_apply")
    if not isinstance(result, dict):
        raise ToolError(f"apply_feature_plan failed: {result}")
    if not result.get("success"):
        raise ToolError(toolpipe.build_failure_text(result))
    text = (
        f"Plan applied: {result['feature_count']} features, "
        f"{result['parameter_count']} parameters → spreadsheet "
        f"'{result['spreadsheet']}', body '{result['body']}'."
    )
    for f in result["features"]:
        info = f.get("info") or {}
        extra = ""
        if "fully_constrained" in info:
            extra = (f" (fully constrained: {info['fully_constrained']}, "
                     f"{info['dimensional_expressions']} param expressions)")
        text += f"\n- {f['id']} [{f['op']}] → {f['object']}{extra}"
    return _shot(text, True, "Isometric", doc_name)


@tool(
    "feature_tree_status",
    "Check the feature tree of a document: every object's recompute state and validity, solids, sketch "
    "constraints, the spreadsheet parameters and which are expression-bound. It is the same report as "
    "validate_model; use it after apply_feature_plan (every Body of a multi-part plan is listed) or to inspect a "
    "rebuild failure.",
    {
        "type": "object",
        "properties": {"doc_name": _SCHEMA_DOC_NAME},
        "required": ["doc_name"],
    },
    category="verify", phases=_P("----XXX"),
)
def _feature_tree_status(doc_name: str):
    """The ``validate_model`` report of ``doc_name`` (``tools_verify.feature_tree_status``)."""
    from . import tools_verify

    return tools_verify.feature_tree_status(doc_name)



@tool(
    "reapply_plan_feature",
    f"(Re-)apply ONE feature of a FeaturePlan into a document that was "
    f"built with apply_feature_plan (repair path). The feature must "
    f"not have later features depending on it — those cases re-apply the "
    f"whole plan. Pass the whole (possibly patched) FeaturePlan, schema 1 or 2, exactly as for "
    f"apply_feature_plan; in a multi-part plan the feature is rebuilt inside its own part's Body.",
    {
        "type": "object",
        "properties": {
            "doc_name": _SCHEMA_DOC_NAME,
            "plan": {"type": "object",
                     "description": "the (possibly patched) FeaturePlan"},
            "feature_id": {"type": "string",
                           "description": "id of the feature to apply"},
        },
        "required": ["doc_name", "plan", "feature_id"],
    },
    category="build", geometry=True, phases=_P("----X--"),
)
def _reapply_plan_feature(doc_name: str, plan: dict, feature_id: str):
    def task():
        from .cad.engine import _resolve_doc
        from .plan import builder

        doc = _resolve_doc(doc_name)
        if doc is None:
            return {"success": False,
                    "error": f"document '{doc_name}' is not open"}
        return builder.apply_feature(doc, plan, feature_id)

    from .cad import dispatch as _d

    result, _ = _d.dispatch_stoppable(task, timeout=600.0, op="plan_apply_feature")
    if not isinstance(result, dict):
        raise ToolError(f"reapply_plan_feature failed: {result}")
    if not result.get("success"):
        raise ToolError(toolpipe.build_failure_text(result, feature_id))
    text = f"Feature '{feature_id}' applied → object '{result['object']}'."
    if result.get("parameters_changed"):
        # the plan is the source of truth for the sheet: say what it reset,
        # including a value the user may have edited by hand
        text += (" Spreadsheet parameters set from the plan: "
                 + ", ".join(result["parameters_changed"]) + ".")
    return _shot(text, True, "Isometric", doc_name)


# Download circuit breaker (per turn): after two failed/declined
# attempts the tool refuses immediately. The agent log showed a model
# retrying downloads 9 times after a 403 — each retry burned another
# permission prompt (or its 120 s timeout) while nothing got built.
_DOWNLOAD_FAILURES = 0
_DOWNLOAD_CIRCUIT = ""


def reset_turn_circuits() -> None:
    """New turn → downloads get a fresh budget (called by AgentWorker)."""
    global _DOWNLOAD_FAILURES, _DOWNLOAD_CIRCUIT
    _DOWNLOAD_FAILURES = 0
    _DOWNLOAD_CIRCUIT = ""


_ASK_USER_ATTACH = (
    "ASK THE USER NOW, in your reply, with the URL included: \"I could "
    "not download the file automatically. Please open {url} in your "
    "browser, download it, and ATTACH it to this chat — I will continue "
    "from the attachment.\" Attached CAD/mesh files arrive as absolute "
    "paths you can import_file. Meanwhile, continue every part of the "
    "task that does not need the file."
)

@tool(
    "download_file",
    "Download a file from the internet — the ONLY sanctioned network path. "
    "The USER IS ASKED FIRST (permission dialog with the URL and your "
    "reason); the decision covers the WHOLE turn. Use a URL you KNOW is "
    "the vendor's or a reputable library's — never guess one. After a "
    "failure or decline: at most one retry with a different trusted URL, "
    "then STOP downloading and ask the user to open the page in their "
    "browser, download the file and ATTACH it to the chat (attachments "
    "arrive as import_file paths) — while you continue every part of the "
    "task that does not need the file. Streams to "
    "~/Downloads/Oscar, size-capped at 200 MB.",

    {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http(s) URL"},
            "filename": {"type": "string",
                         "description": "target filename (default from URL)"},
            "reason": {"type": "string",
                       "description": "why this download is needed — shown "
                                      "to the user in the permission dialog"},
        },
        "required": ["url", "reason"],
    },
    category="net", phases=_P("-X-----"),
)
def _download_file(url: str, reason: str, filename: str = ""):
    global _DOWNLOAD_FAILURES, _DOWNLOAD_CIRCUIT
    if _DOWNLOAD_CIRCUIT:
        raise ToolError(_DOWNLOAD_CIRCUIT)
    if not url.lower().startswith(("http://", "https://")):
        raise ToolError(f"download_file: only http(s) URLs, got {url!r}")

    def strike(cause: str) -> str:
        global _DOWNLOAD_FAILURES, _DOWNLOAD_CIRCUIT
        _DOWNLOAD_FAILURES += 1
        message = cause + " " + _ASK_USER_ATTACH.format(url=url)
        if _DOWNLOAD_FAILURES >= 2:
            _DOWNLOAD_CIRCUIT = (
                "Download circuit OPEN for this turn "
                f"({_DOWNLOAD_FAILURES} failed attempts). Do not call "
                "download_file or write download code again this turn. "
                + _ASK_USER_ATTACH.format(url=url))
            return _DOWNLOAD_CIRCUIT
        return message + " At most one retry with a DIFFERENT trusted " \
               "URL is allowed — then ask the user."

    from urllib.parse import urlsplit as _urlsplit

    try:
        _download_host = (_urlsplit(url).hostname or "").lower()
    except ValueError:
        _download_host = ""
    verdict = netguard.request_permission(
        f"Oscar wants to download a file:\n• {url}",
        detail=f"Reason: {reason}", tool="download_file", host=_download_host)   # 'Always allow' = THIS host for THIS tool
    if verdict not in ("allow", "always"):
        raise ToolError(strike("The user DECLINED this download."))
    dest = _download_dir()
    path, error = netguard.download(url, dest, filename)
    if error:
        raise ToolError(strike(f"download_file failed: {error}"))
    size_kb = os.path.getsize(path) // 1024
    return f"Downloaded {url} → {path} ({size_kb} KB). " \
           f"Import it with import_file.", None


@tool(
    "run_fem_analysis",
    "Run the CalculiX solver on an existing Fem::AnalysisPython and return "
    "max von Mises stress (MPa) and max displacement (mm). Prerequisites in "
    "the document: a Part-derived solid; the analysis container; a "
    "Fem::MaterialCommon; a Fem::FemMeshGmsh on the geometry (auto-meshed at "
    "creation); at least one Fem::ConstraintFixed and one "
    "Fem::ConstraintForce/Pressure. The solver blocks all other tools while "
    "it runs.",
    {
        "type": "object",
        "properties": {
            "doc_name": _SCHEMA_DOC_NAME,
            "analysis_name": {"type": "string", "description": "Name of the Fem::AnalysisPython object"},
            "timeout": {"type": "integer", "default": 600},
            "include_screenshot": {"type": "boolean", "default": True},
            "view_name": {"type": "string", "enum": VIEW_NAMES, "default": "Isometric"},
        },
        "required": ["doc_name", "analysis_name"],
    },
    category="fem", mutating=True, geometry=True, phases=_P("----XX-"),
)
def _run_fem_analysis(doc_name: str, analysis_name: str, timeout: int = 600,
                      include_screenshot: bool = True, view_name: str = "Isometric"):
    res = _engine().run_fem(doc_name, analysis_name, timeout)
    if res.get("success"):
        def fmt(v, unit):
            return f"{v:.4g} {unit}" if isinstance(v, (int, float)) else f"unavailable ({unit})"
        text = (
            f"FEM analysis '{analysis_name}' solved. "
            f"max von Mises = {fmt(res.get('max_von_mises_MPa'), 'MPa')}, "
            f"max displacement = {fmt(res.get('max_displacement_mm'), 'mm')} "
            f"({res.get('node_count')} nodes)."
        )
    else:
        raise ToolError(f"FEM analysis '{analysis_name}' failed: {res.get('error')}")
    return _shot(text, include_screenshot, view_name, doc_name)


# --------------------------------------------------------------------------
# Registry API and the run_tool pipeline live in oscar/toolrun.py; re-exported here so that
# ``oscar.tools`` stays the one public entry point (``tools.specs``, ``tools.run_tool`` ...).
# Imported AFTER every ``@tool`` above and BEFORE the other tool modules.

from .toolrun import (  # noqa: E402,F401
    CORE_TOOLS, INTENT_CATEGORIES, INTENT_HIDDEN, LITE_TOOLS, _coerce, _prepare_arguments,
    derive_title, get_tools, is_geometry, is_mutating, mutating_names, run_tool, specs, tool_meta)


# --------------------------------------------------------------------------
# The other tool modules. They do ``from .tools import tool, ToolError,
# current_context`` at their top and register with ``@tool(...)``; this loop runs after every
# ``@tool`` and every registry function above exists, so that import is safe. A module that is
# not delivered yet is tolerated; a missing DEPENDENCY inside one is a real bug and raises.

_TOOL_MODULES = ("tools_flow", "tools_project", "tools_ref", "tools_kit", "tools_verify", "tools_web")


def _import_tool_modules(modules: Iterable[str] = _TOOL_MODULES) -> list[str]:
    """Import the other tool modules so their ``@tool`` registrations run.

    Returns the names that were imported. A module that does not exist (yet) is skipped; a
    ``ModuleNotFoundError`` raised for anything ELSE (a dependency missing inside a delivered
    module) and every other exception propagate: that is a real bug and must not hide behind
    a half-registered tool surface.
    """
    loaded: list[str] = []
    for short in modules:
        full = f"{__package__}.{short}"
        try:
            importlib.import_module(full)
        except ModuleNotFoundError as e:       # module not delivered yet: tolerated
            if e.name != full:                 # a missing DEPENDENCY inside it is a real bug
                raise
            continue
        loaded.append(short)
    return loaded


_import_tool_modules()

