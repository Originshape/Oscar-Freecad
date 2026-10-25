"""The Oscar MCP bridge — localhost access for external AI clients.

External assistants (anything speaking XML-RPC to localhost) get the exact
same engine the built-in Oscar chat uses: every bridge call lands on the
shared CadEngine, which owns GUI-thread routing, the script namespace and
the background-job registry. Default port 9876, loopback only;
``OSCAR_BRIDGE_PORT`` overrides it (a test instance can then run beside
the user's live app, e.g. the GUI smoke test in CI).
"""

import os

import FreeCAD

from .cad import server as _server
from .cad.engine import get_engine

DEFAULT_PORT = 9876
ENV_PORT = "OSCAR_BRIDGE_PORT"


def configured_port() -> int:
    """The port the bridge uses: ``OSCAR_BRIDGE_PORT`` when it is an integer in 1..65535, else 9876.

    Read at call time. A value that is not a valid port is ignored (the default applies) instead of
    stopping the application from starting its bridge."""
    raw = os.environ.get(ENV_PORT, "").strip()
    try:
        port = int(raw)
    except ValueError:
        return DEFAULT_PORT
    return port if 1 <= port <= 65535 else DEFAULT_PORT


def start(port: int | None = None) -> str:
    from .cad import dispatch

    dispatch.init()  # called from GUI contexts (autostart, menu command)
    message = _server.start(get_engine(), port or configured_port())
    FreeCAD.Console.PrintMessage(message + "\n")
    return message


def stop() -> str:
    return _server.stop()


def is_running() -> bool:
    return _server.is_running()


def status_line() -> str:
    if _server.is_running():
        return f"running on port {_server.port()}"
    return f"stopped (port {configured_port()})"
