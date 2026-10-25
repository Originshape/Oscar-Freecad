"""Oscar CAD runtime — the engine behind Oscar and the MCP bridge.

Package layout (all original Oscar code):

- ``dispatch``   — runs document/view work on the GUI thread from any thread
- ``properties`` — maps plain dictionaries onto object properties
- ``serialize``  — objects to JSON-safe dictionaries
- ``views``      — viewport orientation and screenshot capture
- ``scripting``  — scripted execution, background jobs and the commit helper
- ``headless``   — crash-isolated script runs in a separate engine process
- ``engine``     — the CadEngine: the full operation surface used by the chat
                   tools and mirrored by the XML-RPC bridge
- ``server``     — the localhost bridge server

``get_engine()`` returns the shared CadEngine instance.
"""

__all__ = ["CadEngine", "get_engine"]


def __getattr__(name):
    """``CadEngine`` / ``get_engine`` on first use (PEP 562): importing ``oscar.cad.dispatch`` at FreeCAD
    start-up must not load the engine and the FreeCAD modules it needs."""
    if name in __all__:
        from . import engine

        value = getattr(engine, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
