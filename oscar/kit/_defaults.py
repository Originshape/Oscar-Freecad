"""Board-sized defaults for ``kit.spec.default_spec`` (private module of ``oscar.kit``; pure, no FreeCAD).

The default numbers are those of a Pi 4 in ABS. They are right for every board that is about as big as a Pi, but a
Zero 2 W, a Pico, a Nano or an ESP32 dev kit cannot take nine 24 mm vent slots, 12 mm hook beams or four 6 mm hooks: the validator
rejects the default spec of those boards, and a model that asks for ``enclosure_spec(action="defaults")`` would get an
unusable spec. ``board_sized`` shrinks exactly the fields that do not fit (vent count and slot length, hook beam length,
width, thickness and overhang, hook count) and leaves every other field, and every board that already validates, byte for byte
at the default values. When no hook geometry can fit at all the lid becomes a ``friction`` lid (no hooks needed) so the result
ALWAYS validates.

The search is deterministic and ordered by distance from the default (largest count, widest hook, thickest beam, biggest overhang
first), so the same board always gives the same spec. Results are cached per ``(board, material)``;
``reset_cache()`` clears the cache (tests that swap the board library).
"""

import math
import threading
from typing import Any

from . import spec as _spec

_CACHE: "dict[tuple[str, str], dict]" = {}
_LOCK = threading.Lock()
_LOCAL = threading.local()          # re-entrancy guard: validate() may build specs from dicts, which asks for defaults again

_HOOK_COUNTS = (4, 2)
_HOOK_WIDTHS = (6.0, 4.0, 3.0)
_HOOK_THICKNESS = (1.2, 1.0, 0.8)
_HOOK_OVERHANGS = (0.8, 0.6, 0.5, 0.4)
_MIN_BEAM = 5.0                      # hooks.length minimum


def reset_cache() -> None:
    """Forget the cached board-sized defaults."""
    with _LOCK:
        _CACHE.clear()


def _floor1(x: float) -> float:
    """``x`` rounded DOWN to 0.1 mm (never above x: a limit must not be exceeded by rounding)."""
    return math.floor(x * 10.0 + 1e-9) / 10.0


def _errors(d: dict, board: dict, prefix: "str | None" = None) -> list[str]:
    """The validation errors of the spec dict ``d`` (only those starting with ``prefix`` when given)."""
    _ok, errors, _warns = _spec.validate(d, board)
    return [e for e in errors if prefix is None or e.startswith(prefix)]


def _fit_vents(d: dict, board: dict) -> None:
    """Shrink the default vent array until it lies inside the lid (or turn the vents off when not even one slot fits)."""
    v = d["vents"]
    area = _spec.vent_area(d, board, v["part"], v["face"])
    along = area["u1"] - area["u0"]            # angle 0, centre = face centre: slots run along u, the array spans v
    across = area["v1"] - area["v0"]
    sw, pitch = float(v["slot_w"]), float(v["pitch"])
    if along < 4.0 or across < sw:
        v["enabled"] = False
        return
    v["slot_l"] = min(float(v["slot_l"]), max(_floor1(along), 1.0))
    v["count"] = int(max(1, min(int(v["count"]), math.floor((across - sw) / pitch + 1e-9) + 1)))


def _fit_hooks(d: dict, board: dict) -> bool:
    """Find the hook numbers closest to the default that validate; False when none does."""
    h = d["hooks"]
    top = _spec.derived(d, board)
    room = top["z_cavity_top"]["value"] - float(d["lid"]["lip_h"]) - top["z_floor_top"]["value"]
    length = min(float(h["length"]), _floor1(room))
    if length < _MIN_BEAM:
        return False
    h["length"] = length
    for count in _HOOK_COUNTS:
        for w in _HOOK_WIDTHS:
            for t in _HOOK_THICKNESS:
                for overhang in _HOOK_OVERHANGS:
                    if overhang <= float(d["lid"]["fit"]):
                        continue
                    h.update(count=count, w=w, t=t, overhang=overhang)
                    if not _errors(d, board, "hooks"):
                        return True
    return False


def _size(d: dict, board: dict) -> dict:
    if not _errors(d, board):
        return d
    d = _spec.merge(d, {})
    try:
        _fit_vents(d, board)
        if d["lid"]["style"] == "snap_fit" and not _fit_hooks(d, board):
            d["lid"]["style"] = "friction"
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return d
    return d


def board_sized(spec: "_spec.EnclosureSpec") -> "_spec.EnclosureSpec":
    """``spec`` (a plain ``default_spec`` of some board) with the fields adapted so that it validates on that board.

    Returns ``spec`` itself when the board is unknown, the board library is missing, or the defaults already validate. Never raises.
    """
    if getattr(_LOCAL, "busy", False) or spec.board == "rpi4b":
        return spec
    key = (spec.board, spec.material)
    with _LOCK:
        cached = _CACHE.get(key)
    if cached is None:
        _LOCAL.busy = True
        try:
            try:
                board = _spec.load_board(spec.board)
            except (KeyError, LookupError, ValueError, OSError):
                return spec
            d = _spec.to_dict(spec)
            try:
                cached = _size(d, board)
            except Exception:    # noqa: BLE001 - defaults must never fail to build
                return spec
        finally:
            _LOCAL.busy = False
        with _LOCK:
            _CACHE[key] = cached
    if cached == _spec.to_dict(spec):
        return spec
    return _spec._build(_spec.merge(_spec.to_dict(spec), cached))


def adapted_fields(board_id: str, material: str = "ABS") -> "dict[str, Any]":
    """``{dotted path: (contract default, board value)}`` for every field ``board_sized`` changed (what the tools report)."""
    base = _spec.EnclosureSpec()
    base.board, base.material = board_id, material
    sized = board_sized(base)
    old, new = _spec.to_dict(_spec.EnclosureSpec()), _spec.to_dict(sized)
    out: dict[str, Any] = {}

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for k in a:
                walk(a[k], b.get(k), f"{path}.{k}" if path else k)
        elif a != b and path not in ("board", "material"):
            out[path] = (a, b)

    walk(old, new, "")
    return out
