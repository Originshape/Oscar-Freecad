"""Printable materials known to the enclosure kit (pure, no FreeCAD).

One table drives every material-dependent rule of the kit: the spec validator
(``kit.spec``), the derived clearance (``shrink`` compensation) and the DFM
checks (``kit.checks``). The numbers are conventional FDM rules of thumb, not
material certifications: the Build Summary always
lists physical validation as NOT validated.

``shrink`` is a fraction (0.008 = 0.8 %), ``min_wall`` and ``min_web`` are
millimetres, ``max_strain`` is the injection-moulding strain limit of a
snap-fit beam (dimensionless). FDM prints use ``max_strain * FDM_DERATE``.
"""

import difflib

# The default process is FDM and the hook beams bend across layer interfaces,
# so the strain limit used for hooks is the datasheet limit times this factor.
FDM_DERATE = 0.5

MATERIALS: dict[str, dict[str, float]] = {
    "ABS": {"shrink": 0.008, "min_wall": 1.2, "min_web": 1.2, "max_strain": 0.020},
    "ASA": {"shrink": 0.007, "min_wall": 1.2, "min_web": 1.2, "max_strain": 0.020},
    "PLA": {"shrink": 0.003, "min_wall": 1.0, "min_web": 1.0, "max_strain": 0.015},
    "PETG": {"shrink": 0.004, "min_wall": 1.2, "min_web": 1.2, "max_strain": 0.030},
    "PC": {"shrink": 0.007, "min_wall": 1.2, "min_web": 1.2, "max_strain": 0.025},
    "NYLON": {"shrink": 0.015, "min_wall": 1.2, "min_web": 1.2, "max_strain": 0.050},
}


def names() -> list[str]:
    """The supported material names, in table order."""
    return list(MATERIALS)


def normalize(material: object) -> str:
    """Canonical (upper-case, trimmed) spelling of a material name.

    A non-string value is returned unchanged so validators can report it; the
    result is NOT checked against ``MATERIALS`` (use ``is_known``).
    """
    return material.strip().upper() if isinstance(material, str) else material  # type: ignore[return-value]


def is_known(material: object) -> bool:
    """True when ``material`` (after ``normalize``) is a key of ``MATERIALS``."""
    return isinstance(material, str) and normalize(material) in MATERIALS


def get(material: str) -> dict[str, float]:
    """The property row of ``material`` (a copy).

    Raises ``KeyError`` whose message names the known materials and, when one is
    close, the most similar one (``'ABS '`` and ``'abs'`` resolve; ``'ABZ'`` suggests ``ABS``).
    """
    key = normalize(material)
    if isinstance(key, str) and key in MATERIALS:
        return dict(MATERIALS[key])
    similar = difflib.get_close_matches(str(key), list(MATERIALS), n=1) if isinstance(key, str) else []
    hint = f" (similar: {similar[0]})" if similar else ""
    raise KeyError(f"material {material!r} is not supported{hint}; known: {', '.join(MATERIALS)}")


def shrink_pct(material: str) -> float:
    """Shrinkage of ``material`` in percent (ABS -> 0.8), the unit of the ``shrink_pct`` parameter."""
    return round(get(material)["shrink"] * 100.0, 6)


def hook_strain_limit(material: str, *, molded: bool = False) -> float:
    """Allowed bending strain of a snap-fit beam in ``material``.

    FDM (default): ``max_strain * FDM_DERATE`` (ABS 0.020 * 0.5 = 0.010).
    ``molded=True``: the injection-moulding limit without the derate, shown as an INFO row only.
    """
    limit = get(material)["max_strain"]
    return limit if molded else limit * FDM_DERATE
