"""Marking up the model: the words for where the user drew (the pen next to the paperclip, ``panel_markup``).

The user draws on a picture of the 3D view; the points under the strokes are picked in the model, and the draft
starts with a line naming the part(s) they hit, the side those face and the region they cover, e.g.
``[Marked on the model (see the attached picture): part “Shell”, its top (+Z), around x 12 to 40, y −8 to 15,
z 20 mm.]``. The picture goes with the message as an image. The OriginShape workspace's pen uses the same words, so
Oscar reads a mark the same way wherever it was drawn.

Pure logic, no Qt, no FreeCAD: unit-testable in any interpreter.
"""

import math
from dataclasses import dataclass

PICKED_POINTS = 240   # stroke points picked in the model, at most

# (axis, sign, words) of each side a mark can face
SIDES = (
    (0, 1, "right side (+X)"),
    (0, -1, "left side (−X)"),
    (1, 1, "back (+Y)"),
    (1, -1, "front (−Y)"),
    (2, 1, "top (+Z)"),
    (2, -1, "bottom (−Z)"),
)


@dataclass
class MarkHit:
    """A stroke point that landed on the model: the part's label, the point (global, mm) and the surface normal
    there (None when the picked element has none)."""
    part: str
    point: tuple
    normal: "tuple | None" = None


def sample(points: list, limit: int = PICKED_POINTS) -> list:
    """Every n-th of ``points`` so that at most ``limit`` remain (strokes are dense; picking each point is not)."""
    every = max(1, -(-len(points) // max(1, limit)))
    return points[::every]


def _mm(value: float) -> str:
    """``value`` to a tenth of a millimetre, halves up (as JavaScript's Math.round), with a real minus sign."""
    rounded = math.floor(value * 10 + 0.5) / 10
    text = f"{rounded:.1f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text.replace("-", "−")


def describe(hits: list) -> str:
    """The marks in words: which parts, which side, and the region they cover (mm)."""
    if not hits:
        return "[Marked on the model: see the attached picture.]"
    counts: dict = {}
    for hit in hits:
        counts[hit.part] = counts.get(hit.part, 0) + 1
    parts = sorted(counts, key=lambda name: -counts[name])   # stable: ties keep the order they were hit in
    main = [hit for hit in hits if hit.part == parts[0]]
    facing = ""
    normals = [hit.normal for hit in main if hit.normal is not None]
    if normals:
        average = [sum(n[axis] for n in normals) / len(normals) for axis in range(3)]
        axis, sign, side = max(SIDES, key=lambda candidate: candidate[1] * average[candidate[0]])
        if sign * average[axis] > 0.6:
            facing = f", its {side}"
    region = []
    for index, name in enumerate("xyz"):
        values = [hit.point[index] for hit in main]
        low, high = min(values), max(values)
        region.append(f"{name} {_mm((low + high) / 2)}" if high - low < 0.5 else f"{name} {_mm(low)} to {_mm(high)}")
    quoted = [f"“{name}”" for name in parts]
    named = quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"
    noun = "part" if len(parts) == 1 else "parts"
    return f"[Marked on the model (see the attached picture): {noun} {named}{facing}, around {', '.join(region)} mm.]"
