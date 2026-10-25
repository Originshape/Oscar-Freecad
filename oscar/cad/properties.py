"""Apply plain dictionary values onto object properties.

Chat tool arguments arrive as JSON. This module translates the natural JSON
shapes into the engine's property types:

    {"x": 1, "y": 2, "z": 3}                          -> Vector
    {"Base": {...}, "Rotation": {"Axis": …, "Angle":…}} -> Placement (MERGED, see below)
    "Box" (for link properties)                        -> the named object
    [{"object": "Pad", "face": "Face3"}]               -> reference tuples
    "=Params.wall_t*2"                                 -> an expression binding

Placement edits are merges: only the parts the caller supplies change. ``{"Base": {"z": 4}}``
moves the object up and keeps its x, y and rotation; ``{"Rotation": …}`` keeps the position;
``{"x": 1, "y": 2, "z": 3}`` (no ``Base`` / ``Rotation`` key) is a position only;
``{"Yaw": 0, "Pitch": 0, "Roll": 90}`` (degrees) is a rotation only.

A string starting with ``=`` is an expression: ``{"Length": "=Params.wall_t*2"}`` binds the
property (``setExpression``); ``"="`` alone removes the binding. Writing a plain value onto
an expression-bound property still works but is reported in the notes, because the
expression will overwrite it at the next recompute.

Property names starting with ``ViewObject.`` — or a nested ``"ViewObject":
{...}`` map — target the visual object, which lets callers restyle shapes
(``ShapeColor``) through the same path. Without a GUI (console sessions)
view properties are skipped, never an error.

Link properties are recognized by their property type (App::PropertyLink*,
XLink*), so ``"Shapes": ["A", "B"]`` on a MultiFuse resolves exactly like
``"Base": "A"`` on a Cut; an unknown name is an error that names the
available objects. Enumeration properties accept their member names (a case-only miss is
repaired, any other miss lists the members with a ``similar:`` hint).

The apply is ATOMIC: every key is validated and converted first, and nothing is touched when
any of them is wrong (one error lists all problems). If an assignment itself fails, the keys
set before it are restored.
"""

import difflib
from typing import Any

import FreeCAD as App

# Fallback when the property type cannot be queried: properties that hold
# links to other objects; values may be given as the other object's Name
# (or Label) instead of the object itself.
_LINK_KEYS = {
    "Base", "Tool", "Source", "Profile", "Tool1", "Tool2",
    "Sections", "Spine", "Shapes", "Objects", "Links", "Group",
}

PLACEMENT_FORMS = (
    "Placement accepts {\"Base\": {\"x\":…, \"y\":…, \"z\":…}, \"Rotation\": {\"Axis\": {\"x\":…, "
    "\"y\":…, \"z\":…}, \"Angle\": degrees}} (either part may be omitted and keeps its current "
    "value), {\"x\":…, \"y\":…, \"z\":…} for the position only, or {\"Yaw\":…, \"Pitch\":…, "
    "\"Roll\":…} (degrees) for the rotation only"
)

_COORDS = ("x", "y", "z")
_EULER = ("Yaw", "Pitch", "Roll")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _vector(data: Any):
    """A Vector from {"x", "y"(, "z")} (z defaults to 0); None for anything that is not shaped like one."""
    if isinstance(data, dict) and "x" in data and "y" in data:
        return App.Vector(float(data["x"]), float(data["y"]), float(data.get("z", 0)))
    return None


def _base_vector(raw: Any, current) -> Any:
    """``raw`` as a Base position merged over ``current``: a dict may name only some axes."""
    if isinstance(raw, dict) and any(k in raw for k in _COORDS):
        merged = [current.x, current.y, current.z]
        for index, axis in enumerate(_COORDS):
            if axis in raw:
                merged[index] = float(raw[axis])
        return App.Vector(*merged)
    if isinstance(raw, (list, tuple)) and len(raw) == 3 and all(_is_number(v) for v in raw):
        return App.Vector(*[float(v) for v in raw])
    raise ValueError("Base must be {\"x\":…, \"y\":…, \"z\":…} (axes may be omitted) or [x, y, z]")


def _axis(raw: Any):
    if isinstance(raw, (list, tuple)) and len(raw) in (2, 3) and all(_is_number(v) for v in raw):
        return App.Vector(*[float(v) for v in raw], *([0.0] * (3 - len(raw))))
    vector = _vector(raw)
    if vector is None:
        raise ValueError("Rotation Axis must be {\"x\":…, \"y\":…, \"z\":…} or [x, y, z] (z is optional)")
    return vector


def _rotated(axis, angle: float):
    """``App.Rotation(axis, angle)``; a zero axis is only acceptable for a zero angle (it is no rotation at all)."""
    if axis.Length == 0:
        if angle != 0:
            raise ValueError("Rotation Axis must not be the zero vector")
        return App.Rotation()
    return App.Rotation(axis, angle)


def _rotation(raw: Any):
    """``raw`` as an ``App.Rotation``: {Axis, Angle}, {Yaw, Pitch, Roll}, [ax, ay, az, angle]."""
    if isinstance(raw, dict):
        if any(k in raw for k in _EULER):
            yaw, pitch, roll = (float(raw.get(k, 0)) for k in _EULER)
            return App.Rotation(yaw, pitch, roll)
        if "Axis" in raw or "Angle" in raw:
            if "Angle" not in raw:
                raise ValueError("Rotation needs an Angle (degrees) next to its Axis")
            axis = _axis(raw["Axis"]) if "Axis" in raw else App.Vector(0, 0, 1)   # an Angle alone turns about Z
            return _rotated(axis, float(raw["Angle"]))
    elif isinstance(raw, (list, tuple)) and len(raw) == 4 and all(_is_number(v) for v in raw):
        return _rotated(App.Vector(*[float(v) for v in raw[:3]]), float(raw[3]))
    raise ValueError("Rotation must be {\"Axis\":…, \"Angle\": degrees}, {\"Yaw\":…, \"Pitch\":…, "
                     "\"Roll\":…} or [ax, ay, az, angle]")


def _placement(data: Any, current):
    """The Placement ``data`` describes, merged over ``current`` (parts not supplied are kept).

    Raises ValueError listing the accepted forms when ``data`` matches none of them."""
    if isinstance(data, App.Placement):
        return data
    try:
        if isinstance(data, (list, tuple)):
            return App.Placement(_base_vector(data, current.Base), current.Rotation)
        if not isinstance(data, dict) or not data:
            raise ValueError("not an object")
        known = {"Base", "Rotation"}
        if known & set(data):
            extra = sorted(set(data) - known)
            if extra:
                raise ValueError(f"unknown key(s) {', '.join(repr(k) for k in extra)}")
            # null / {} / [] mean "leave that part as it is"
            base = (_base_vector(data["Base"], current.Base)
                    if data.get("Base") not in (None, {}, []) else current.Base)
            rotation = (_rotation(data["Rotation"])
                        if data.get("Rotation") not in (None, {}, []) else current.Rotation)
            return App.Placement(base, rotation)
        if any(k in data for k in _COORDS) and not any(k in data for k in _EULER):
            return App.Placement(_base_vector(data, current.Base), current.Rotation)
        if any(k in data for k in _EULER):
            return App.Placement(current.Base, _rotation(data))
        raise ValueError("no Base, Rotation, x/y/z or Yaw/Pitch/Roll key")
    except (ValueError, TypeError) as e:
        raise ValueError(f"{e}; {PLACEMENT_FORMS}") from e


def _find_object(document, name: str):
    obj = document.getObject(name)
    if obj is None:  # fall back to a label match
        for candidate in document.Objects:
            if candidate.Label == name:
                return candidate
    return obj


def _resolve_link(document, value, key: str = "", strict: bool = True):
    """Object names (or labels) -> objects; lists resolve element-wise.

    An unknown name raises instead of silently linking None, which would
    only surface later as an opaque 'invalid state' after recompute.
    ``strict=False`` (inside LinkSub values such as ["Sketch", ["Face1"]])
    keeps unknown strings — they are sub-element names, not objects.
    """
    if isinstance(value, str):
        obj = _find_object(document, value)
        if obj is None:
            if not strict:
                return value
            names = ", ".join(o.Name for o in document.Objects[:30]) or "none"
            raise ValueError(
                f"'{key or 'link'}' refers to '{value}', which is not an object in "
                f"'{document.Name}' (objects: {names})"
            )
        return obj
    if isinstance(value, dict) and ("object" in value or "object_name" in value):
        return _resolve_link(document, value.get("object", value.get("object_name")), key)
    if isinstance(value, (list, tuple)):
        return [_resolve_link(document, item, key, strict) for item in value]
    return value


def _link_type(target, name: str, key: str) -> str:
    """The property's link type id ('' when it is not a link property)."""
    try:
        type_id = target.getTypeIdOfProperty(name)
    except Exception:
        return "App::PropertyLink" if key in _LINK_KEYS else ""
    if type_id.startswith(("App::PropertyLink", "App::PropertyXLink")):
        return type_id
    return ""


def _references(document, entries):
    if not isinstance(entries, list):
        return entries
    resolved = []
    for entry in entries:
        if not isinstance(entry, dict):
            resolved.append(entry)
            continue
        target = _resolve_link(document, entry.get("object", entry.get("object_name", "")))
        subelements = tuple(
            name for name in (entry.get("face"), entry.get("edge"), entry.get("vertex"))
            if name
        )
        if target is not None:
            resolved.append((target, subelements))
    return resolved


def _similar(name: str, candidates, n: int = 5) -> list[str]:
    return difflib.get_close_matches(name, list(candidates), n=n, cutoff=0.6)


def _enum_members(target, name: str) -> list[str]:
    """The member names of an enumeration property ([] when it is not one)."""
    try:
        if target.getTypeIdOfProperty(name) != "App::PropertyEnumeration":
            return []
        return [str(m) for m in target.getEnumerationsOfProperty(name)]
    except Exception:  # noqa: BLE001
        return []


def _check_enum(name: str, value: Any, members: list[str]) -> Any:
    """``value`` as a valid member (case repaired) or ValueError with a ``similar:`` hint."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"'{name}' must be one of {'|'.join(members)}, got {value!r}")
    if isinstance(value, int):
        if 0 <= value < len(members):
            return members[value]
        raise ValueError(f"'{name}' index {value} is out of range; it must be one of {'|'.join(members)}")
    if value in members:
        return value
    folded = [m for m in members if m.lower() == value.strip().lower()]
    if len(folded) == 1:
        return folded[0]
    close = [m for m in members if m.lower().startswith(value.lower())] if len(value) >= 2 else []
    close = close or [m for m in members if m in _similar(value, members)]
    hint = f" (similar: {', '.join(close)})" if close else ""
    raise ValueError(f"'{name}' must be one of {'|'.join(members)}, got {value!r}{hint}")


def _current_placement(obj):
    try:
        return obj.Placement
    except Exception:  # noqa: BLE001 - objects without a Placement get the identity
        return App.Placement()


class _Op:
    """One validated assignment: ``kind`` is ``"set"`` (a value) or ``"expr"`` (an expression)."""

    __slots__ = ("key", "target", "name", "value", "kind")

    def __init__(self, key, target, name, value, kind):
        self.key, self.target, self.name, self.value, self.kind = key, target, name, value, kind


def _bound_note(target, name: str, key: str) -> str:
    """A note when ``name`` is expression-bound (the expression would overwrite a plain value)."""
    try:
        for path, expression in target.ExpressionEngine or []:
            if path == name or path.startswith(name + "."):
                return (f"'{key}' is bound to the expression '{expression}', which overwrites "
                        f"the value you set at the next recompute; change what the expression "
                        f"reads, or set \"=<new expression>\" (or \"=\" to unbind) instead")
    except Exception:  # noqa: BLE001
        pass
    return ""


def _flatten(properties: dict[str, Any]):
    """``(key, value)`` pairs with the nested ``"ViewObject": {...}`` map expanded."""
    for key, value in properties.items():
        if key == "ViewObject" and isinstance(value, dict):
            for inner, inner_value in value.items():
                yield f"ViewObject.{inner}", inner_value
        else:
            yield key, value


def _plan(obj, properties: dict[str, Any]) -> tuple[list[_Op], list[str], list[str]]:
    """Validate and convert every key without touching ``obj``.

    Returns ``(operations, notes, problems)``; ``problems`` lists every key that cannot be
    applied, each starting ``cannot set property '<key>' on '<object>'``.
    """
    ops: list[_Op] = []
    notes: list[str] = []
    problems: list[str] = []
    for key, value in _flatten(properties):
        if key.startswith("ViewObject."):
            target, name = getattr(obj, "ViewObject", None), key.split(".", 1)[1]
            if target is None:
                continue  # no GUI (console session): appearance does not exist
        else:
            target, name = obj, key
        try:
            op = _plan_one(obj, target, name, key, value, notes)
        except ValueError as e:
            problems.append(f"cannot set property '{key}' on '{obj.Name}': {e}")
            continue
        ops.append(op)
    return ops, notes, problems


def _plan_one(obj, target, name: str, key: str, value: Any, notes: list[str]) -> _Op:
    """Validate one key (raises ValueError with the reason) and return its operation."""
    if isinstance(value, str) and value.startswith("=") and target is obj and not _is_string_property(obj, name):
        expression = value[1:].strip()
        if expression:
            try:
                obj.evalExpression(expression)
            except AttributeError:
                pass  # older builds cannot pre-evaluate: the assignment reports parse errors
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"expression '{expression}' is invalid: {e}") from e
        return _Op(key, obj, name, expression or None, "expr")
    if "." in name and target is obj:
        raise ValueError("only an '=expression' value can be set on a nested property path; "
                         "set the whole property (for Placement: {\"Base\": {\"z\": 4}}) instead")
    try:
        exists = hasattr(target, name)
    except Exception:  # noqa: BLE001 - exotic getters raise something other than AttributeError
        exists = True
    if not exists:
        props = list(getattr(target, "PropertiesList", []) or [])
        close = _similar(name, props)
        raise ValueError(f"no such property{'; similar: ' + ', '.join(close) if close else ''}")
    note = _bound_note(target, name, key) if target is obj else ""
    if note:
        notes.append(note)

    if name == "Placement" and value is not None:
        value = _placement(value, _current_placement(target))
    elif key == "References":
        value = _references(obj.Document, value)
    elif target is obj and isinstance(value, (str, dict, list, tuple)) and (link_type := _link_type(obj, name, key)):
        sub = "Sub" in link_type
        value = _resolve_link(obj.Document, value, key,
                              strict=not (sub and isinstance(value, (list, tuple))))
        if (sub and isinstance(value, list) and len(value) == 2
                and isinstance(value[0], App.DocumentObject) and isinstance(value[1], (list, str))):
            value = tuple(value)  # (object, ["Face1", …])
    elif isinstance(value, dict):
        vector = _vector(value)
        if vector is not None:
            if isinstance(getattr(target, name), App.Vector):
                value = vector
        elif target is obj and name != "Placement" and isinstance(getattr(target, name, None), App.Vector):
            raise ValueError("a vector is {\"x\":…, \"y\":…, \"z\":…}")
    elif isinstance(value, list) and name.endswith("Color") and 3 <= len(value) <= 4:
        value = tuple(float(v) for v in value)
    else:
        members = _enum_members(target, name) if target is obj else []
        if members and value is not None:
            value = _check_enum(name, value, members)
    return _Op(key, target, name, value, "set")


def _is_string_property(obj, name: str) -> bool:
    """True for string-typed properties (Label, Label2, ...), where a leading ``=`` is plain text."""
    try:
        return obj.getTypeIdOfProperty(name) in ("App::PropertyString", "App::PropertyFont")
    except Exception:  # noqa: BLE001
        return False


def _snapshot(op: _Op):
    """The value (or expression) ``op`` is about to replace, for the rollback; None when unknown."""
    try:
        if op.kind == "expr":
            for path, expression in op.target.ExpressionEngine or []:
                if path == op.name:
                    return ("expr", str(expression))
            return ("expr", None)
        return ("value", getattr(op.target, op.name))
    except Exception:  # noqa: BLE001
        return None


def _restore(op: _Op, saved) -> None:
    if saved is None:
        return
    try:
        if saved[0] == "expr":
            op.target.setExpression(op.name, saved[1])
        else:
            setattr(op.target, op.name, saved[1])
    except Exception:  # noqa: BLE001 - best effort: the engine's transaction abort is the backstop
        pass


def apply_properties_report(obj, properties: dict[str, Any], undo_log: list | None = None) -> dict[str, Any]:
    """Apply ``properties`` atomically; returns ``{"applied": [keys], "notes": [text]}``.

    Phase 1 validates and converts every key (unknown property with ``similar:`` names,
    wrong enum member, unknown link target, malformed Placement or expression) and raises ONE
    ValueError listing all problems, with nothing changed. Phase 2 assigns in order; if an
    assignment fails the keys already set are restored and the error names the applied and
    the unapplied keys. ``notes`` explain effects the caller might not expect (a plain value
    written onto an expression-bound property).

    ``undo_log`` (a list) receives, for a SUCCESSFUL call, one entry per assignment recording what it replaced;
    ``restore_properties(undo_log)`` puts it all back later (the engine uses it to roll an edit back when the
    recompute leaves the object invalid). A failed call leaves the log as it found it.
    """
    if not isinstance(properties, dict):
        raise ValueError(
            f"properties must be a JSON object (name -> value), got {type(properties).__name__}"
        )
    ops, notes, problems = _plan(obj, properties)
    if problems:
        if len(problems) == 1:
            raise ValueError(f"{problems[0]}. Nothing was changed.")
        raise ValueError(f"{len(problems)} properties cannot be applied, nothing was changed: "
                         + "; ".join(problems))
    done: list[tuple[_Op, Any]] = []
    for position, op in enumerate(ops):
        saved = _snapshot(op)
        try:
            if op.kind == "expr":
                op.target.setExpression(op.name, op.value)
            else:
                setattr(op.target, op.name, op.value)
        except Exception as e:  # noqa: BLE001
            for earlier, earlier_saved in reversed(done):
                _restore(earlier, earlier_saved)
            applied = [d[0].key for d in done]
            pending = [o.key for o in ops[position + 1:]]
            raise ValueError(
                f"cannot set property '{op.key}' on '{obj.Name}': {e}. Nothing was changed "
                f"(rolled back: {', '.join(applied) or 'none'}; not applied: "
                f"{', '.join([op.key] + pending)})."
            ) from e
        done.append((op, saved))
    if undo_log is not None:
        undo_log.extend(done)
    return {"applied": [op.key for op, _ in done], "notes": notes}


def apply_properties(obj, properties: dict[str, Any], undo_log: list | None = None) -> list[str]:
    """Set each property atomically; returns the names that were applied.

    See ``apply_properties_report`` for the contract (including ``undo_log``); this form drops the notes."""
    return apply_properties_report(obj, properties, undo_log)["applied"]


def restore_properties(undo_log: list) -> None:
    """Undo a successful ``apply_properties(..., undo_log=...)``: newest assignment first, then the log is emptied.
    Best effort: a property that refuses to be reset is left as it is."""
    for op, saved in reversed(list(undo_log)):
        _restore(op, saved)
    undo_log.clear()
