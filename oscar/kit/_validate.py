"""Validation of an EnclosureSpec against its rules (pure, no FreeCAD).

Private module of ``oscar.kit.spec``: ``spec.validate`` is the public entry point and delegates here (lazily, so
``spec`` stays importable on its own). One ``_Rules`` run holds the coerced dict, the board and the accumulated messages
and applies the rules in table order; messages follow ``"<path>: <problem>; <what to do>"``.
"""

import difflib
import math
import re
from typing import Any, Iterable

from . import materials as _materials
from . import spec as _spec
from .spec import (
    DESIGN_PARAMS, FACES, KIT_ID_RE, LID_ORIENTATIONS, LID_STYLES, NAME_RE, PARTS, POCKET_SEAT, SCHEMA_VERSION,
    SHELL_ORIENTATIONS, SHRINK_MODES, STRAIN_WARN_FRACTION, TOL, VENT_PATTERNS, _ENTRY_KINDS, _KIND_WORDS, _LEAF_KINDS,
    _build, _get_path, _n, _template, derived, from_dict, hook_positions, hook_strain, hook_strain_limit, to_dict,
    vent_area, vent_extent, EnclosureSpec)


def _structure_errors(raw: dict) -> list[str]:
    """Unknown keys (with difflib suggestions) and wrongly shaped groups in a raw spec dict."""
    errors: list[str] = []
    template = _template()

    def unknown(path: str, key: str, known: Iterable[str]) -> str:
        known = list(known)
        similar = difflib.get_close_matches(key, known, n=3, cutoff=0.6)
        full = f"{path}.{key}" if path else key
        if similar:
            return f"{full}: unknown key '{key}' (similar: {', '.join(similar)})"
        return f"{full}: unknown key '{key}'; known keys: {', '.join(known)}"

    for key, value in raw.items():
        if key not in template:
            errors.append(unknown("", key, template))
            continue
        tmpl = template[key]
        if isinstance(tmpl, dict):
            if not isinstance(value, dict):
                errors.append(f"{key}: expected an object with the keys {', '.join(tmpl)}; got {value!r}")
                continue
            for k in value:
                if k not in tmpl:
                    errors.append(unknown(key, k, tmpl))
        elif key in _ENTRY_KINDS and isinstance(value, (list, tuple)):
            for i, entry in enumerate(value):
                if isinstance(entry, dict):
                    errors.extend(unknown(f"{key}[{i}]", k, _ENTRY_KINDS[key]) for k in entry if k not in _ENTRY_KINDS[key])
    for key in ("ports",):
        value = raw.get(key)
        if isinstance(value, dict) and isinstance(value.get("overrides"), (list, tuple)):
            for i, entry in enumerate(value["overrides"]):
                if isinstance(entry, dict):
                    errors.extend(unknown(f"ports.overrides[{i}]", k, _ENTRY_KINDS["ports.overrides"])
                                  for k in entry if k not in _ENTRY_KINDS["ports.overrides"])
    return errors


def _ok_kind(kind: str, v: Any) -> bool:
    if kind in ("num", "deg"):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if kind == "int":
        return isinstance(v, int) and not isinstance(v, bool)
    if kind == "bool":
        return isinstance(v, bool)
    if kind in ("str", "enum", "id", "material"):
        return isinstance(v, str)
    if kind == "idlist":
        return isinstance(v, list) and all(isinstance(i, str) for i in v)
    if kind == "center":
        return v is None or (isinstance(v, list) and len(v) == 2 and all(_ok_kind("num", i) for i in v))
    if kind == "rgb":
        return isinstance(v, list) and len(v) == 3 and all(_ok_kind("num", i) for i in v)
    if kind == "objlist":
        return isinstance(v, list)
    return True


def _type_errors(d: dict) -> "tuple[list[str], set[str]]":
    """Type problems of the (coerced) dict and the set of leaf paths that are unusable for the rule checks."""
    errors: list[str] = []
    bad: set[str] = set()
    for path, kind in _LEAF_KINDS.items():
        try:
            v = _get_path(d, path)
        except (KeyError, TypeError):
            continue
        if not _ok_kind(kind, v):
            bad.add(path)
            errors.append(f"{path}: expected {_KIND_WORDS[kind]}, got {v!r}")
            continue
        if kind == "objlist":
            for i, entry in enumerate(v):
                if not isinstance(entry, dict):
                    bad.add(path)
                    errors.append(f"{path}[{i}]: expected an object with the keys {', '.join(_ENTRY_KINDS[path])}; got {entry!r}")
                    continue
                for k, val in entry.items():
                    if k in _ENTRY_KINDS[path] and not _ok_kind(_ENTRY_KINDS[path][k], val):
                        bad.add(path)
                        errors.append(f"{path}[{i}].{k}: expected {_KIND_WORDS[_ENTRY_KINDS[path][k]]}, got {val!r}")
    return errors, bad


class _Rules:
    """One validation run: holds the typed dict, the board and the accumulated messages."""

    def __init__(self, d: dict, board: "dict | None", errors: list[str], warnings: list[str], bad: set[str]) -> None:
        self.d, self.board, self.errors, self.warnings, self.bad = d, board, errors, warnings, bad
        self.spec = _build(d)
        self.mat: "dict | None" = None
        self.dv: "dict | None" = None

    # --- accessors ------------------------------------------------------------------------------------------------
    def num(self, path: str) -> "float | None":
        if path in self.bad:
            return None
        v = _get_path(self.d, path)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    def text(self, path: str) -> "str | None":
        if path in self.bad:
            return None
        v = _get_path(self.d, path)
        return v if isinstance(v, str) else None

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def enum(self, path: str, allowed: Iterable[str], noun: str = "") -> "str | None":
        v = self.text(path)
        allowed = tuple(allowed)
        if v is not None and v not in allowed:
            self.err(f"{path}: {v!r} is not {noun or 'a valid value'}; use one of {', '.join(allowed)}.")
            return None
        return v

    # --- the rules, in table order --------------------------------------------------------------------------------
    def run(self, board_state: str, board_id: str, known_ids: list[str]) -> None:
        name = self.text("name")
        if name is not None and not NAME_RE.match(name):
            self.err(f"name: {name!r} is not a valid name; use 1 to 24 letters, digits or underscores starting with a letter (for example PiCase).")
        kit_id = self.text("kit_id")
        if kit_id is not None and not KIT_ID_RE.match(kit_id):
            self.err(f"kit_id: {kit_id!r} is not a valid id; use 1 to 16 letters, digits or underscores starting with a letter (default enc1).")
        version = self.num("version")
        if version is not None and int(version) != SCHEMA_VERSION:
            self.err(f"version: {int(version)} is not supported; this build reads schema {SCHEMA_VERSION}.")
        self._board(board_state, board_id, known_ids)
        self._material()
        self._thickness()
        self._clearance()
        self._corner()
        self._standoff()
        self._lid()
        self._derive()
        self._hooks()
        self._vents()
        self._ports()
        self._cutouts()
        lift = self.num("lid_lift")
        if lift is not None and not 0 <= lift <= 100:
            self.err(f"lid_lift: {_n(lift)} is outside 0..100 mm; use a value from 0 to 100 (default 0.0).")
        self._orientation_and_colors()

    def _board(self, state: str, board_id: str, detail: list[str]) -> None:
        """Report the board lookup outcome: ``detail`` is the known ids ('unknown'), the data's id ('mismatch') or the reason ('broken')."""
        if "board" in self.bad:
            return
        if state == "empty":
            self.err("board: is empty; use a board id such as 'rpi4b'. Use find_reference_part.")
        elif state == "unknown":
            self.err(f"board: {board_id!r} is not in the board library; known: {', '.join(detail) or 'none'}. Use find_reference_part.")
        elif state == "unavailable":
            self.warn("board: the board library is not available in this build; board-dependent checks were skipped")
        elif state == "broken":
            self.err(f"board: the data of {board_id!r} could not be read ({detail[0] if detail else 'unknown error'}); fix the board file.")
        elif state == "mismatch":
            self.err(f"board: the spec names {board_id!r} but the board data passed is {detail[0] if detail else '?'!r}; pass the data of {board_id!r}.")
        elif self.board is not None and not self.board.get("verified", False):
            self.warn(f"board '{self.board.get('id', board_id)}': dimensions are not yet verified against the drawing")

    def _material(self) -> None:
        mat = self.text("material")
        if mat is None:
            return
        if mat not in _materials.MATERIALS:
            self.err(f"material: {mat!r} is not supported; use one of {', '.join(_materials.MATERIALS)}.")
            return
        self.mat = _materials.get(mat)

    def _thickness(self) -> None:
        mw = self.mat["min_wall"] if self.mat else None
        for path in ("wall_t", "floor_t", "lid_t"):
            x = self.num(path)
            if x is None:
                continue
            if mw is not None and x < mw:
                self.err(f"{path}: {_n(x)} is below the minimum {_n(mw)} mm for {self.spec.material}; use {_n(mw)} or more (default 2.0).")
            elif x > 6:
                self.err(f"{path}: {_n(x)} is above the maximum 6.0 mm; use 6.0 or less (default 2.0).")
            elif path == "wall_t" and x < 1.6:
                self.warn(f"wall_t: {_n(x)} is thin for FDM; 1.6 mm or more prints more reliably.")

    def _clearance(self) -> None:
        x = self.num("clearance")
        if x is not None:
            if x < 0.1:
                self.err(f"clearance: {_n(x)} is below the minimum 0.1 mm; use 0.1 or more (default 0.3).")
            elif x > 2.0:
                self.err(f"clearance: {_n(x)} is above the maximum 2.0 mm; use 2.0 or less (default 0.3).")
            elif x < 0.15:
                self.warn(f"clearance: {_n(x)} is tight for FDM; 0.15 mm or more fits more reliably.")
        sc = self.text("shrink_comp")
        if sc == "scale":
            self.err("shrink_comp: 'scale' is not supported yet; use 'clearance' or 'none'")
        elif sc is not None and sc not in SHRINK_MODES:
            self.err(f"shrink_comp: {sc!r} is not a valid mode; use 'none' or 'clearance' ('scale' is reserved).")

    def _corner(self) -> None:
        r, wall = self.num("corner_r"), self.num("wall_t")
        if r is None:
            return
        if r < 0:
            self.err(f"corner_r: {_n(r)} is negative; use 0 or more (default 3.0).")
            return
        if self.board is not None:
            try:
                limit = min(float(self.board["outline"]["length"]), float(self.board["outline"]["width"])) / 4
            except (KeyError, TypeError, ValueError):
                limit = None
            if limit is not None and r > limit:
                self.err(f"corner_r: {_n(r)} is above the maximum {_n(limit)} mm (a quarter of the board's short edge); use {_n(limit)} or less.")
                return
        if wall is not None and r - wall < 0:
            self.err(f"corner_r: {_n(r)} is smaller than wall_t {_n(wall)}; the cavity corner radius corner_r - wall_t must not be negative; use {_n(wall)} or more, or reduce wall_t.")

    def _standoff(self) -> None:
        h, dia, pilot = self.num("standoff.h"), self.num("standoff.d"), self.num("standoff.pilot_d")
        if h is not None:
            bottom = None
            if self.board is not None:
                try:
                    bottom = float(self.board["height"]["min_bottom_z"])
                except (KeyError, TypeError, ValueError):
                    bottom = None
            need_under = None if bottom is None or bottom >= 0 else 1.0 - bottom
            required = max(2.0, need_under) if need_under is not None else 2.0
            if need_under is not None and h < need_under:
                label = "the board underside"
                for c in (self.board or {}).get("connectors", []):
                    if c.get("box", {}).get("z", [0])[0] == bottom:
                        label = "the microSD socket" if c.get("kind") == "microsd" else f"the '{c.get('id')}' connector"
                        break
                self.err(f"standoff.h: {_n(h)} leaves less than 1.0 mm under the board for {label} (underside z {bottom:g}); use {_n(required)} or more")
            elif h < 2.0:
                self.err(f"standoff.h: {_n(h)} is below the minimum 2.0 mm; use {_n(required)} or more (default 4.0).")
        if dia is not None:
            if dia < 4.0:
                self.err(f"standoff.d: {_n(dia)} is below the minimum 4.0 mm; use 4.0 or more (default 6.0).")
            elif dia > 10:
                self.err(f"standoff.d: {_n(dia)} is above the maximum 10.0 mm; use 10.0 or less (default 6.0).")
            elif self.board is not None:
                pad = (self.board.get("mounting_holes") or {}).get("pad_diameter")
                if isinstance(pad, (int, float)) and dia > pad:
                    self.warn(f"standoff.d: {_n(dia)} is larger than the board's hole pad {_n(float(pad))} mm; the ring could touch components on the underside; use {_n(float(pad))} or less.")
        if pilot is not None:
            if pilot < 1.5:
                self.err(f"standoff.pilot_d: {_n(pilot)} is below the minimum 1.5 mm; use 1.5 or more (default 2.2).")
            elif dia is not None and pilot >= dia - 1.0:
                self.err(f"standoff.pilot_d: {_n(pilot)} is too large for a {_n(dia)} mm standoff; use less than {_n(dia - 1.0)} (default 2.2).")
            elif self.board is not None:
                m = re.match(r"^[Mm](\d+(?:\.\d+)?)", str((self.board.get("mounting_holes") or {}).get("thread", "")))
                if m and pilot > 0.9 * float(m.group(1)):
                    self.warn(f"standoff.pilot_d: {_n(pilot)} is more than 90% of the {m.group(0).upper()} thread diameter; a self-tapping screw may not bite; use {_n(0.9 * float(m.group(1)))} or less.")

    def _lid(self) -> None:
        self.enum("lid.style", LID_STYLES, "a lid style")
        for path, minimum, default in (("lid.gap", 0.5, 1.0), ("lid.lip_h", 2.0, 3.0), ("lid.lip_t", 0.8, 1.2)):
            x = self.num(path)
            if x is not None and x < minimum:
                self.err(f"{path}: {_n(x)} is below the minimum {_n(minimum)} mm; use {_n(minimum)} or more (default {_n(default)}).")
        fit = self.num("lid.fit")
        if fit is not None:
            if fit < 0.1:
                self.err(f"lid.fit: {_n(fit)} is below the minimum 0.1 mm; use 0.1 or more (default 0.25).")
            elif fit > 1.0:
                self.err(f"lid.fit: {_n(fit)} is above the maximum 1.0 mm; use 1.0 or less (default 0.25).")
            elif fit < 0.15:
                self.warn(f"lid.fit: {_n(fit)} is tight for FDM; 0.15 mm or more fits more reliably.")

    def _derive(self) -> None:
        """Derived values for the rules that need the board; stays None when inputs or board are unusable."""
        if self.board is None or self.errors_in_inputs():
            return
        try:
            self.dv = derived(self.spec, self.board)
        except (ValueError, KeyError, TypeError) as exc:
            self.dv = None
            if "finite" in str(exc):     # an absurdly large value (1e308) overflowed a derived dimension
                self.err("spec: a derived dimension is not a finite number; some value is absurdly large; "
                         "use realistic millimetre values (wall_t 2.0, lid.lip_t 1.2, ports.gap 0.3 ...).")

    def errors_in_inputs(self) -> bool:
        """True when any numeric input of the derived formulas is missing or mistyped."""
        needed = {path for path, _u, _n_ in DESIGN_PARAMS.values()}
        return bool(self.bad & needed) or self.mat is None or self.text("shrink_comp") is None

    def _hooks(self) -> None:
        if self.text("lid.style") != "snap_fit":
            return
        count = self.num("hooks.count")
        if count is not None and (int(count) % 2 or not 2 <= count <= 8):
            self.err(f"hooks.count: {int(count)} must be an even number from 2 to 8; use 4 (default).")
        w, t, ov = self.num("hooks.w"), self.num("hooks.t"), self.num("hooks.overhang")
        length, barb = self.num("hooks.length"), self.num("hooks.barb_h")
        wm, ramp, ret = self.num("hooks.window_margin"), self.num("hooks.ramp_deg"), self.num("hooks.retention_deg")
        fit, wall = self.num("lid.fit"), self.num("wall_t")
        if w is not None and w <= 0:
            self.err(f"hooks.w: {_n(w)} must be greater than 0 (default 6.0).")
        if t is not None and t < 0.8:
            self.err(f"hooks.t: {_n(t)} is below the minimum 0.8 mm; use 0.8 or more (default 1.2).")
        if ov is not None:
            if not 0.4 <= ov <= 1.5:
                self.err(f"hooks.overhang: {_n(ov)} is outside 0.4..1.5 mm; use a value in that range (default 0.8).")
            if fit is not None and ov <= fit:
                self.err(f"hooks.overhang: {_n(ov)} does not exceed lid.fit {_n(fit)}; the barb must reach past the gap; use more than {_n(fit)} (default 0.8).")
        if length is not None:
            if length < 5:
                self.err(f"hooks.length: {_n(length)} is below the minimum 5.0 mm; use 5.0 or more (default 12.0).")
            elif self.dv is not None and fit is not None:
                lip_h = self.num("lid.lip_h")
                if lip_h is not None:
                    top, bottom = self.dv["z_cavity_top"]["value"], self.dv["z_floor_top"]["value"]
                    room = top - lip_h - bottom
                    if length > room + TOL:
                        self.err(f"hooks.length: {_n(length)} is longer than the {_n(room)} mm of wall below the skirt; use {_n(room)} or less (default 12.0).")
        if barb is not None and not 0.6 <= barb <= 2.0:
            self.err(f"hooks.barb_h: {_n(barb)} is outside 0.6..2.0 mm; use a value in that range (default 1.0).")
        if wm is not None and not 0.1 <= wm <= 0.4:
            self.err(f"hooks.window_margin: {_n(wm)} is outside 0.1..0.4 mm; the retention check lifts the lid 0.5 mm, so the margin must stay below that (default 0.3).")
        if ramp is not None and not 15 <= ramp <= 60:
            self.err(f"hooks.ramp_deg: {ramp:g} is outside 15..60 degrees; use a value in that range (default 30).")
        if ret is not None:
            if not 30 <= ret <= 80:
                self.err(f"hooks.retention_deg: {ret:g} is outside 30..80 degrees; use a value in that range (default 45).")
            elif ret > 60:
                self.warn(f"hooks.retention_deg: {ret:g} degrees makes the lid hard to open again; a re-openable lid needs 60 degrees or less.")
        if ov is not None and fit is not None and wall is not None and self.mat is not None and ov > fit:
            behind = wall - (ov - fit + POCKET_SEAT)
            if behind < self.mat["min_wall"] - TOL:
                self.err(f"hooks.overhang: the catch pocket ({_n(ov - fit + POCKET_SEAT)} mm deep) leaves {_n(behind)} mm of wall behind it, below the {self.spec.material} minimum {_n(self.mat['min_wall'])} mm; reduce hooks.overhang or increase wall_t.")
        if t is not None and ov is not None and fit is not None and length is not None and length > 0 and ov > fit and self.mat is not None:
            strain = hook_strain(self.spec)
            limit = hook_strain_limit(self.spec)
            if strain > limit:
                self.err(f"hooks.length: strain {strain:.4f} exceeds the FDM limit {limit:.4f} for {self.spec.material} (1.5 * t * (overhang - fit) / length^2); lengthen the beam, thin it (hooks.t) or reduce hooks.overhang.")
            elif strain > STRAIN_WARN_FRACTION * limit:
                self.warn(f"hooks.length: strain {strain:.4f} is {strain / limit:.0%} of the FDM limit {limit:.4f}; consider a longer beam or less overhang.")
        if self.dv is not None and count is not None and count >= 2 and int(count) % 2 == 0 and count <= 8 and w is not None and w > 0:
            try:
                got = len(hook_positions(self.spec, self.board))
            except (ValueError, KeyError, TypeError):
                got = None
            if got is not None and got < int(count):
                self.err(f"hooks.count: only {got} positions are free of port openings on {self.board.get('id')!r}; reduce count or move ports")

    def _vents(self) -> None:
        if self.d["vents"].get("enabled") is not True:
            return
        count, sw, sl, pitch = self.num("vents.count"), self.num("vents.slot_w"), self.num("vents.slot_l"), self.num("vents.pitch")
        part = self.enum("vents.part", PARTS, "a part")
        face = self.enum("vents.face", FACES, "a face")
        self.enum("vents.pattern", VENT_PATTERNS, "a pattern")
        if count is not None and not 1 <= count <= 60:
            self.err(f"vents.count: {int(count)} is outside 1..60; use a count from 1 to 60 (default 9).")
        if sw is not None and sw < 0.5:
            self.err(f"vents.slot_w: {_n(sw)} is below the minimum 0.5 mm; use 0.5 or more (default 2.0).")
        if sl is not None and sl < 0.5:
            self.err(f"vents.slot_l: {_n(sl)} is below the minimum 0.5 mm; use 0.5 or more (default 24.0).")
        if pitch is not None and pitch < 1.0:
            self.err(f"vents.pitch: {_n(pitch)} is below the minimum 1.0 mm; use 1.0 or more (default 4.0).")
        if pitch is not None and sw is not None and self.mat is not None and pitch >= 1.0 and sw >= 0.5:
            web = pitch - sw
            if web < self.mat["min_web"] - TOL:
                self.err(f"vents.pitch: pitch {_n(pitch)} - slot_w {_n(sw)} leaves a {_n(web)} mm web, below the {self.spec.material} minimum {_n(self.mat['min_web'])} mm; use a pitch of {_n(sw + self.mat['min_web'])} or more.")
        if self.dv is None or part is None or face is None or None in (count, sw, sl, pitch):
            return
        if not (1 <= count <= 60 and sw >= 0.5 and sl >= 0.5 and pitch >= 1.0):
            return
        self._vent_fit(part, face, int(count), sw, sl, pitch)

    def _vent_fit(self, part: str, face: str, count: int, sw: float, sl: float, pitch: float) -> None:
        v = self.d["vents"]
        angle = self.num("vents.angle_deg") or 0.0
        area = vent_area(self.spec, self.board, part, face)
        span, length, _c = vent_extent(self.spec)
        word = "slots" if v.get("pattern") == "slots" else "holes"
        rad = math.radians(angle)
        cos, sin = abs(math.cos(rad)), abs(math.sin(rad))
        hu, hv = (length / 2) * cos + (span / 2) * sin, (length / 2) * sin + (span / 2) * cos
        center = v.get("center") if "vents.center" not in self.bad else None
        cu, cv = (area["cu"], area["cv"]) if center is None else (center[0], center[1])
        avail_u = 2 * min(cu - area["u0"], area["u1"] - cu)
        avail_v = 2 * min(cv - area["v0"], area["v1"] - cv)
        if 2 * hu <= avail_u + 1e-9 and 2 * hv <= avail_v + 1e-9:     # hu / hv are half extents, avail_* full widths
            return
        if sin < 1e-9 or cos < 1e-9:    # array axes parallel to the face axes
            across_is_v = sin < 1e-9
            avail_across, avail_along = (avail_v, avail_u) if across_is_v else (avail_u, avail_v)
            if span > avail_across + 1e-9:
                self.err(f"vents: {count} {word} at pitch {pitch:.1f} span {span:.1f} mm but the {part} is only {max(avail_across, 0.0):.1f} mm wide there; reduce count or pitch.")
            if length > avail_along + 1e-9:
                self.err(f"vents.slot_l: {length:.1f} mm is longer than the {max(avail_along, 0.0):.1f} mm available on the {part} {face}; reduce slot_l.")
            return
        self.err(f"vents: the rotated array needs {2 * hu:.1f} x {2 * hv:.1f} mm but only {max(avail_u, 0.0):.1f} x {max(avail_v, 0.0):.1f} mm is available on the {part} {face}; reduce count, pitch or slot_l.")

    def _ports(self) -> None:
        include, exclude = self.d["ports"]["include"], self.d["ports"]["exclude"]
        ids = [str(c.get("id")) for c in (self.board or {}).get("connectors", [])]
        board_id = (self.board or {}).get("id", self.spec.board)

        def check(path: str, value: Any) -> None:
            if not isinstance(value, str) or value in ids or self.board is None:
                return
            similar = difflib.get_close_matches(value, ids, n=1, cutoff=0.5)
            hint = f" (similar: {similar[0]})" if similar else ""
            self.err(f"{path}: {value!r} is not a connector of {board_id!r}{hint}; connectors: {', '.join(ids)}")

        if "ports.include" not in self.bad:
            for i in include:
                check("ports.include", i)
        if "ports.exclude" not in self.bad:
            for i in exclude:
                check("ports.exclude", i)
        if "ports.include" not in self.bad and "ports.exclude" not in self.bad:
            both = [i for i in include if i in exclude]
            if both:
                self.warn(f"ports: {', '.join(repr(i) for i in both)} is in both include and exclude; exclude wins")
        margin, gap = self.num("ports.margin"), self.num("ports.gap")
        if margin is not None:
            if margin < 0:
                self.err(f"ports.margin: {_n(margin)} is negative; use 0 to 2 mm (default 0.4).")
            elif margin > 2:
                self.err(f"ports.margin: {_n(margin)} is above the maximum 2.0 mm; use 2.0 or less (default 0.4).")
            elif margin < 0.2:
                self.warn(f"ports.margin: {_n(margin)} is tight; 0.2 mm or more leaves room for printing tolerance.")
        if gap is not None and gap < 0:
            self.err(f"ports.gap: {_n(gap)} is negative; use 0 or more (default 0.3).")
        if "ports.overrides" not in self.bad:
            for i, entry in enumerate(self.d["ports"]["overrides"]):
                if not entry.get("id"):
                    self.err(f"ports.overrides[{i}]: missing 'id'; name the connector to override, for example {{\"id\": \"hdmi0\", \"margin\": 0.6}}.")
                else:
                    check(f"ports.overrides[{i}].id", entry["id"])
                m = entry.get("margin")
                if isinstance(m, (int, float)) and not isinstance(m, bool) and not 0 <= m <= 2:
                    self.err(f"ports.overrides[{i}].margin: {_n(float(m))} is outside 0..2 mm; use a value from 0 to 2.")

    def _cutouts(self) -> None:
        if "cutouts" in self.bad:
            return
        for i, entry in enumerate(self.d["cutouts"]):
            missing = [k for k in ("part", "face", "u", "v", "w", "h") if k not in entry]
            if missing:
                self.err(f"cutouts[{i}]: missing {', '.join(repr(k) for k in missing)}; a cut-out needs part, face, u, v, w and h.")
            if entry.get("part") is not None and entry["part"] not in PARTS:
                self.err(f"cutouts[{i}].part: {entry['part']!r} is not a part; use 'shell' or 'lid'.")
            if entry.get("face") is not None and entry["face"] not in FACES:
                self.err(f"cutouts[{i}].face: {entry['face']!r} is not a face; use one of {', '.join(FACES)}.")
            for k in ("w", "h"):
                x = entry.get(k)
                if isinstance(x, (int, float)) and not isinstance(x, bool) and x < 0.1:
                    self.err(f"cutouts[{i}].{k}: {_n(float(x))} is below the minimum 0.1 mm; use 0.1 or more.")

    def _orientation_and_colors(self) -> None:
        self.enum("print_orientation.shell", SHELL_ORIENTATIONS, "a shell orientation")
        self.enum("print_orientation.lid", LID_ORIENTATIONS, "a lid orientation")
        for key in ("shell", "lid", "ref"):
            path = f"colors.{key}"
            if path in self.bad:
                continue
            for channel in self.d["colors"][key]:
                if not 0 <= channel <= 1:
                    self.err(f"{path}: channel {channel:g} is outside 0..1; use values from 0 to 1, for example [0.18, 0.45, 0.75].")
                    break


def validate(spec: "EnclosureSpec | dict", board: "dict | None" = None) -> "tuple[bool, list[str], list[str]]":
    """Check ``spec`` against every enclosure-spec rule. Returns ``(ok, errors, warnings)``.

    ``spec`` may be an ``EnclosureSpec`` or a (partial) dict; a dict is merged onto the defaults like ``from_dict``,
    and its unknown keys are reported as errors with ``difflib`` suggestions. ``board`` is the board data of
    ``spec.board``; when None it is loaded through ``refparts.catalog.load_board``. An unknown board id is an
    error that lists the known ids; while the board library does not exist in the build the board-dependent rules
    (corner, standoff, hooks, vents, ports) are skipped with one warning. Messages are
    ``"<path>: <problem>; <what to do>"``. Never raises for bad values. ``ok`` is ``not errors``.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if isinstance(spec, EnclosureSpec):
        typed = to_dict(from_dict(to_dict(spec)))    # normalised exactly like a dict input
    elif isinstance(spec, dict):
        errors.extend(_structure_errors(spec))
        typed = to_dict(from_dict(spec))
    else:
        return False, [f"spec: expected an object, got {type(spec).__name__}; pass the spec as a JSON object."], []
    type_errors, bad = _type_errors(typed)
    errors.extend(type_errors)
    board_id = typed.get("board") if isinstance(typed.get("board"), str) else ""
    state, detail = "given", []
    resolved = board
    if "board" in bad:
        resolved = None
    elif not board_id:
        state, resolved = "empty", None
    elif board is not None:
        if board.get("id") not in (None, board_id):
            state, detail, resolved = "mismatch", [str(board.get("id"))], None
    else:
        try:
            resolved = _spec.load_board(board_id)
            state = "loaded"
        except KeyError:
            state, detail, resolved = "unknown", _spec._known_board_ids(), None
        except LookupError:
            state, resolved = "unavailable", None
        except Exception as exc:    # noqa: BLE001 - an unreadable board file is reported, never raised
            state, detail, resolved = "broken", [f"{type(exc).__name__}: {exc}"], None
    rules = _Rules(typed, resolved, errors, warnings, bad)
    rules.run(state, board_id, detail)
    return not errors, errors, warnings
