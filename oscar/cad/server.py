"""Oscar MCP bridge server.

Exposes the CadEngine over XML-RPC on localhost so external AI clients can
drive the application the same way the built-in chat does. The server owns
no CAD state of its own: every call is forwarded to the shared engine, which
routes document/view work onto the GUI thread.

Security posture for v1: loopback only, and only for non-browser clients.
The engine can run arbitrary code (execute_code), so a web page the user
happens to have open must not be able to reach it — neither with a plain
cross-site POST nor via DNS rebinding. Requests are therefore refused when
they carry an Origin header (browsers send one on a cross-site POST;
XML-RPC clients never do), when the Host header is not this loopback
listener (DNS rebinding), when the body is not text/xml (a cross-site
request that skips the CORS preflight cannot be), or when the body is
oversized. There is deliberately no token: external MCP clients are already
deployed without one. Binding to anything wider than localhost is a
decision for a future version with authentication — not silently here.
"""

import os
import re
import socket
import socketserver
import threading
import xmlrpc.client
from urllib.parse import urlsplit
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

DEFAULT_PORT = 9876   # oscar.bridge.configured_port() applies OSCAR_BRIDGE_PORT on top of this

# A script can legitimately be large, but nothing a client sends needs more.
MAX_REQUEST_BYTES = 32 * 1024 * 1024

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

# Characters outside the XML 1.0 Char set. xmlrpc.client only escapes & < >,
# so one stray ESC/NUL in a script's output makes the whole reply ill-formed
# and the client loses the result of a call that already ran.
_XML_ILLEGAL = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def _clean(value):
    """Replace XML-illegal characters in every string of a reply value."""
    if isinstance(value, str):
        return _XML_ILLEGAL.sub("\ufffd", value)
    if isinstance(value, dict):
        return {_clean(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class _Handler(SimpleXMLRPCRequestHandler):
    def _reject(self, code: int, reason: str) -> None:
        self.close_connection = True
        self.send_response(code, reason)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.flush()
        # The request body was never read; closing on unread data makes the
        # kernel send an RST that can wipe the response before the client
        # reads it. Half-close, then drain a bounded amount.
        try:
            self.connection.shutdown(socket.SHUT_WR)
            self.connection.settimeout(0.5)
            left = 1 << 20
            while left > 0:
                chunk = self.connection.recv(65536)
                if not chunk:
                    break
                left -= len(chunk)
        except OSError:
            pass

    def _host_is_this_listener(self) -> bool:
        host = self.headers.get("Host")
        if host is None:
            return True  # HTTP/1.0-style client; browsers always send Host
        try:
            parts = urlsplit("//" + host.strip())
            name, port = parts.hostname, parts.port
        except ValueError:
            return False
        return ((name or "").lower() in _LOOPBACK_HOSTS
                and port == self.server.server_address[1])

    def do_POST(self):
        # Every check runs before the body is read.
        if self.headers.get("Origin") is not None:
            return self._reject(403, "cross-origin requests are not accepted")
        if not self._host_is_this_listener():
            return self._reject(403, "unexpected Host header")
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0]
        if ctype.strip().lower() != "text/xml":
            return self._reject(415, "Content-Type must be text/xml")
        try:
            size = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._reject(411, "Content-Length required")
        if size < 0 or size > MAX_REQUEST_BYTES:
            return self._reject(413, "request too large")
        super().do_POST()


class _ThreadingXMLRPCServer(socketserver.ThreadingMixIn, SimpleXMLRPCServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second process bind a port that is
    # still listening; on POSIX it only skips TIME_WAIT.
    allow_reuse_address = os.name != "nt"

    def _dispatch(self, method, params):
        try:
            return _clean(super()._dispatch(method, params))
        except xmlrpc.client.Fault:
            raise
        except Exception as exc:
            # Same text the stdlib would build, minus illegal characters.
            raise xmlrpc.client.Fault(1, _clean("%s:%s" % (type(exc), exc)))


_server: _ThreadingXMLRPCServer | None = None
_thread: threading.Thread | None = None
_stopper: threading.Thread | None = None


def start(engine, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> str:
    """Start serving the engine; returns a status message."""
    global _server, _thread
    if host not in _LOOPBACK_HOSTS:
        return f"bridge refused to bind {host!r} — loopback only"
    if _stopper is not None:
        # A stop() still releasing the socket would make this bind fail.
        _stopper.join(2.0)
    if _server is not None:
        return f"Oscar MCP bridge already running on {_server.server_address[1]}"
    try:
        server = _ThreadingXMLRPCServer(
            (host, port), requestHandler=_Handler,
            allow_none=True, logRequests=False
        )
    except OSError as e:
        return f"bridge could not bind {host}:{port} — {e}"

    server.register_instance(engine)
    _thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05},
        daemon=True, name="oscar-bridge"
    )
    _thread.start()
    _server = server
    return f"Oscar MCP bridge listening on {host}:{server.server_address[1]}"


def stop() -> str:
    """Stop the server and release the port."""
    global _server, _thread, _stopper
    if _server is None:
        return "bridge was not running"

    server, _thread = _server, None
    _server = None

    def shutdown():
        server.shutdown()
        server.server_close()

    # shutdown() blocks until the accept loop exits; never stall the GUI over
    # it — but give the (50 ms poll) loop a moment so the port is normally
    # free by the time we return, and start() waits for stragglers.
    _stopper = threading.Thread(target=shutdown, daemon=True,
                                name="oscar-bridge-stop")
    _stopper.start()
    _stopper.join(1.0)
    return "bridge stopped"


def is_running() -> bool:
    return _server is not None


def port() -> int:
    return _server.server_address[1] if _server else DEFAULT_PORT
