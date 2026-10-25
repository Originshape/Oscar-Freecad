"""The enclosure kit: a deterministic, parametric, rebuildable enclosure builder.

Submodules (loaded on first access, so ``import oscar.kit`` stays light and needs no FreeCAD):

* ``spec``       EnclosureSpec, validation, stage hashes, derived parameters   (pure)
* ``materials``  printable materials and the FDM strain derate                 (pure)
* ``params``     the ``Params`` spreadsheet and its alias / unit conventions   (FreeCAD used only inside methods)
* ``checks``     DFM rows (pure) and the kit-specific fit / coverage rows (``fit_rows``, ``spec_coverage``: FreeCAD, GUI thread)
* ``geometry``, ``build``  the stage builders (FreeCAD, GUI thread; ``build.build_stage`` is the one-stage entry the tools use)

``kit.spec`` etc. therefore work as attributes of the module ``execute_code`` receives as ``kit``
(``scripting.register_helpers("kit", kit)``).
"""

import importlib

_SUBMODULES = ("spec", "materials", "params", "checks", "geometry", "build")


def __getattr__(name: str):
    """Import a submodule on first attribute access (PEP 562); a missing optional submodule is an AttributeError."""
    if name in _SUBMODULES:
        full = f"{__name__}.{name}"
        try:
            return importlib.import_module(full)
        except ModuleNotFoundError as exc:
            if exc.name == full:
                raise AttributeError(f"module {__name__!r} has no submodule {name!r} in this build") from None
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULES))
