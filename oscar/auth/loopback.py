"""Browser sign-in: OAuth 2.1 authorization code with PKCE and a loopback redirect (RFC 8252).

``LoopbackSignIn`` binds an HTTP listener on ``127.0.0.1`` (a free port), sends the browser to the server's authorize
page, and accepts exactly ONE ``GET /callback?code=...&state=...``. The state is checked in constant time, and the code
is exchanged for tokens with the PKCE verifier, which never left this process.

Threading:

* ``prepare()`` and ``open()`` are fast and safe on the GUI thread: no network. ``prepare`` takes the authorize
  endpoint from the cached discovery document, or falls back to ``{api_url}/oauth/authorize``, which the API answers
  with the same login and consent pages.
* ``wait(cancelled, timeout)`` blocks, so it runs on a worker (a settings job). It fetches the discovery document
  first (the code exchange needs it anyway), so a server that cannot be reached ends the sign-in at once instead of
  after ``timeout``. It polls ``cancelled`` every 100 ms and closes the listener on every exit path. A listener
  nobody waits for closes itself after ``LISTENER_MAX_S``. All threads are daemons.

The listener serves nothing but ``GET /callback`` (everything else is a 404), answers only to its own ``Host``
(a DNS-rebinding page cannot reach it), and never logs a request line (it carries the code). Once the sign-in has
really finished (the token exchange included), it sends the browser on to the app's result page,
``https://app.originshape.design/device/done?result=...`` (``RESULT_PATH`` on the metadata's ``app_origin``, else on
the API's address), with a 303: the tab ends on the app's address with a clear page, not on ``127.0.0.1`` with the
code in the address bar. Only the outcome word travels (``signed-in``,
``declined``, ...): never the code, the state or an error's text; FreeCAD shows the details itself.
"""

import html
import queue
import re
import socket
import socketserver
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit

from .. import turnlog as _turnlog
from . import config, discovery, pkce, transport
from .transport import (CLIENT_ID, SCOPE, AuthError, SignInCancelled, SignInDeclined, SignInTimedOut, TokenSet,
                        error_code)

CALLBACK_PATH = "/callback"
RESULT_PATH = "/device/done"   # the app's page that says how a FreeCAD sign-in ended
POLL_S = 0.1                 # how often wait() looks at ``cancelled``
PAGE_WAIT_S = 20.0           # how long the browser's callback request waits for the final outcome
LISTENER_MAX_S = 900.0       # a listener nobody waits for closes itself
HANDLER_TIMEOUT_S = 10.0     # a client that connects and sends nothing is dropped
_CODE_RE = re.compile(r"[A-Za-z0-9\-._~+/=]{8,2048}")


def short_hostname() -> str:
    """This computer's name without its domain, for the server's "connected devices" list."""
    try:
        name = socket.gethostname()
    except OSError:
        name = ""
    return clean_device_name(name.split(".")[0])


def clean_device_name(name: object) -> str:
    """A device name safe to put in a URL and to show on a web page: printable, at most 64 characters."""
    text = name if isinstance(name, str) else ""
    text = re.sub(r"[^A-Za-z0-9 _.'()\-]", "", text).strip()[:64].strip()
    return text or "FreeCAD"


# -- pages ------------------------------------------------------------------------------------------------

def _page(title: str, message: str, *, ok: bool) -> bytes:
    accent = "#2e7d4f" if ok else "#b1452c"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer"><title>{html.escape(title)} - Oscar for FreeCAD</title>
<style>
body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#f3f0e8;
color:#22252a;font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
main{{max-width:30rem;margin:1.5rem;padding:2rem 2.25rem;background:#fffdf8;border:1px solid #ddd6c6;
border-radius:10px}}
.brand{{font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:#6d675b}}
h1{{font-size:1.35rem;margin:.45rem 0 .6rem;color:{accent}}}
p{{margin:0}}
@media (prefers-color-scheme:dark){{body{{background:#16181b;color:#e7e5e0}}
main{{background:#1f2226;border-color:#353941}}.brand{{color:#a29d92}}}}
</style></head>
<body><main><div class="brand">OriginShape &middot; Oscar for FreeCAD</div>
<h1>{html.escape(title)}</h1><p>{html.escape(message)}</p></main></body></html>
""".encode("utf-8")


# How a sign-in ended, in the words the app's result page knows (RESULT_PATH).
SIGNED_IN = "signed-in"
PENDING = "pending"          # the code exchange took longer than PAGE_WAIT_S: FreeCAD finishes on its own
ALREADY = "already-used"     # a second answer for a sign-in that had one
ENDED = "ended"              # FreeCAD stopped waiting (closed, timed out)
DECLINED = "declined"
CANCELLED = "cancelled"
MISMATCH = "mismatch"        # the answer belongs to another sign-in
FAILED = "failed"

_NOT_FOUND = (404, _page("Not found", "There is nothing here.", ok=False))

_HEADERS = (
    ("Content-Type", "text/html; charset=utf-8"),
    ("Cache-Control", "no-store"),
    ("Pragma", "no-cache"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
                                "form-action 'none'; frame-ancestors 'none'"),
    ("Connection", "close"),
)


def _failure_result(error: BaseException) -> str:
    if isinstance(error, SignInDeclined):
        return DECLINED
    if isinstance(error, SignInCancelled):
        return CANCELLED
    if isinstance(error, AuthError) and error.code == "state_mismatch":
        return MISMATCH
    return FAILED


# -- the listener -----------------------------------------------------------------------------------------

class _ListenerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False   # on Windows SO_REUSEADDR would let another process bind the same port
    timeout = POLL_S              # handle_request() returns after this when idle, so the loop can stop

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):   # Windows: nobody else may bind this port while we listen
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        # TCPServer's bind, not HTTPServer's: that one calls getfqdn(), a reverse DNS lookup that can stall the GUI.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = host, port


class _CallbackHandler(BaseHTTPRequestHandler):
    owner: "LoopbackSignIn"
    protocol_version = "HTTP/1.0"
    timeout = HANDLER_TIMEOUT_S
    server_version = "Oscar"

    def version_string(self) -> str:
        return self.server_version

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server's signature
        pass   # the request line carries the authorization code: it must never reach a console or a log

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        parts = urlsplit(self.path)
        if parts.path != CALLBACK_PATH or not self._host_ok():
            self._reply(*_NOT_FOUND)
            return
        self._redirect(self.owner.result_url(self.owner._on_callback(_parse_query(parts.query))))

    def do_HEAD(self) -> None:  # noqa: N802
        self._reply(*_NOT_FOUND, head=True)

    def _not_found(self) -> None:
        self._reply(*_NOT_FOUND)

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _not_found

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        port = self.server.server_address[1]
        return not host or host in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _reply(self, status: int, body: bytes, *, head: bool = False, location: str = "") -> None:
        self.close_connection = True
        try:
            self.send_response(status)
            for name, value in _HEADERS:
                self.send_header(name, value)
            if location:
                self.send_header("Location", location)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head:
                self.wfile.write(body)
        except OSError:
            pass   # the browser went away

    def _redirect(self, location: str) -> None:
        """303 to the app's result page. No-referrer keeps this address (with the code) out of the app's request."""
        self._reply(303, b"", location=location)


def _parse_query(query: str) -> dict[str, str]:
    """The callback parameters; a repeated or unparseable parameter marks the answer as malformed."""
    try:
        pairs = parse_qsl(query, keep_blank_values=True, max_num_fields=20)
    except ValueError:
        return {"_malformed": "1"}
    out: dict[str, str] = {}
    for key, value in pairs:
        if key in out:
            return {"_malformed": "1"}
        out[key] = value
    return out


# -- the flow ---------------------------------------------------------------------------------------------

class LoopbackSignIn:
    """One browser sign-in. ``prepare()`` -> ``open()`` -> ``wait(cancelled)`` -> ``TokenSet``; single use."""

    def __init__(self, api_url: str | None = None, *, open_browser: Callable[[str], Any] = webbrowser.open,
                 device_name: str | None = None):
        self.api_url = config.normalise_url(api_url) if api_url else config.api_url()
        self.device_name = clean_device_name(device_name) if device_name else short_hostname()
        self.authorize_url = ""
        self.redirect_uri = ""
        self.port = 0
        self._open_browser = open_browser
        self._verifier = pkce.new_verifier()
        self._state = pkce.new_state()
        self._lock = threading.Lock()
        self._callbacks: "queue.Queue[dict[str, str]]" = queue.Queue()
        self._result = PENDING
        self._page_ready = threading.Event()
        self._closing = threading.Event()
        self._accepting = True
        self._finished = False
        self._thread: threading.Thread | None = None

    def __repr__(self) -> str:
        return f"LoopbackSignIn(api_url={self.api_url!r}, port={self.port})"

    def __enter__(self) -> "LoopbackSignIn":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- GUI thread ---------------------------------------------------------------------------------------

    def prepare(self) -> str:
        """Bind the listener (once) and return the authorize URL. No network; raises ``AuthError`` when this sign-in
        already finished, ``OSError`` when no local port can be bound."""
        with self._lock:
            if self._finished:
                raise AuthError("This sign-in has already finished. Start a new one.", code="finished")
            if self._thread is None:
                owner = self

                class Handler(_CallbackHandler):
                    pass

                Handler.owner = owner
                httpd = _ListenerServer(("127.0.0.1", 0), Handler)
                self.port = httpd.server_address[1]
                self.redirect_uri = f"http://127.0.0.1:{self.port}{CALLBACK_PATH}"
                self.authorize_url = self._authorize_url()
                self._thread = threading.Thread(target=self._serve, args=(httpd, time.monotonic() + LISTENER_MAX_S),
                                                name="oscar-signin-listener", daemon=True)
                self._thread.start()
            return self.authorize_url

    def open(self) -> bool:
        """Open the authorize URL in the browser. False when no browser could be started (show the link instead)."""
        url = self.prepare()
        try:
            opened = self._open_browser(url)
        except Exception:  # noqa: BLE001 - a missing browser is reported as False, never as a crash
            return False
        return opened is not False

    def _authorize_url(self) -> str:
        meta = discovery.cached(self.api_url)
        endpoint = meta.authorization_endpoint if meta is not None else self.api_url + "/oauth/authorize"
        params = urlencode({
            "response_type": "code", "client_id": CLIENT_ID, "redirect_uri": self.redirect_uri, "scope": SCOPE,
            "state": self._state, "code_challenge": pkce.challenge(self._verifier), "code_challenge_method": "S256",
            "device_name": self.device_name,
        })
        return f"{endpoint}{'&' if '?' in endpoint else '?'}{params}"

    def result_url(self, result: str) -> str:
        """The app's page that tells the browser how this sign-in ended: on the app the server names in its metadata
        (``app_origin``, the same address in production), else on the server the sign-in went to."""
        meta = discovery.cached(self.api_url)
        app = (meta.app_origin if meta is not None else "") or self.api_url
        return f"{app.rstrip('/')}{RESULT_PATH}?{urlencode({'result': result})}"

    # -- worker thread ------------------------------------------------------------------------------------

    def wait(self, cancelled: threading.Event | None, timeout: float = 600.0) -> TokenSet:
        """Block until the browser answers, then exchange the code. Returns the ``TokenSet``; raises
        ``SignInCancelled``, ``SignInTimedOut``, ``SignInDeclined``, ``AuthUnavailable`` or ``AuthError``. The
        server's metadata is fetched before waiting, so an unreachable server (or one that is not an OriginShape
        server) raises at once. The listener is closed when this returns, whatever happened."""
        self.prepare()
        cancelled = cancelled if cancelled is not None else threading.Event()
        try:
            discovery.fetch(self.api_url)
            query = self._next_callback(cancelled, timeout)
            tokens = self._redeem(query, cancelled)
        except AuthError as e:
            self._show(_failure_result(e))
            _log(e.code)
            raise
        except BaseException:
            self._show(FAILED)
            _log("error")
            raise
        else:
            self._show(SIGNED_IN)
            _log("ok")
            return tokens
        finally:
            self.close()

    def close(self) -> None:
        """Stop listening (idempotent). Called by ``wait`` on every exit; call it yourself when you never wait."""
        if not self._page_ready.is_set():
            self._show(ENDED)   # before ``_finished``: a late browser request then gets this result, not "pending"
        with self._lock:
            self._finished = True
            thread, self._thread = self._thread, None
        self._closing.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(2.0)

    def _next_callback(self, cancelled: threading.Event, timeout: float) -> dict[str, str]:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if cancelled.is_set():
                raise SignInCancelled("Sign-in was cancelled.")
            try:
                return self._callbacks.get_nowait()   # an answer that arrived just before the listener closed
            except queue.Empty:
                pass
            if self._closing.is_set():
                raise SignInCancelled("The sign-in was closed before the browser answered.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SignInTimedOut("The browser did not finish the sign-in in time. Start it again.")
            try:
                return self._callbacks.get(timeout=min(POLL_S, remaining))
            except queue.Empty:
                continue

    def _redeem(self, query: dict[str, str], cancelled: threading.Event) -> TokenSet:
        if "_malformed" in query:
            raise AuthError("The browser's answer was malformed. Start the sign-in again.", code="bad_response")
        if not pkce.states_match(self._state, query.get("state")):
            raise AuthError("The browser's answer did not match this sign-in, so it was ignored. Start the sign-in "
                            "again.", code="state_mismatch")
        error = query.get("error")
        if error is not None:
            if error == "access_denied":
                if transport.says_suspended({}, query.get("error_description") or ""):
                    raise transport.AccountSuspended(transport.SUSPENDED_MESSAGE)
                raise SignInDeclined("The sign-in was declined in the browser.")
            code = error_code(error) or "authorize_error"
            raise AuthError(f"The sign-in failed in the browser ({code}).", code=code)
        code = query.get("code") or ""
        if not _CODE_RE.fullmatch(code):
            raise AuthError("The browser's answer carried no usable sign-in code.", code="bad_response")
        meta = discovery.fetch(self.api_url)
        reply = transport.request("POST", meta.token_endpoint, data={
            "grant_type": "authorization_code", "client_id": CLIENT_ID, "code": code,
            "redirect_uri": self.redirect_uri, "code_verifier": self._verifier,
        })
        tokens = transport.token_set(reply, code, self._verifier)
        if cancelled.is_set():   # the user gave up while the code was being exchanged: leave no connected device
            transport.revoke_quietly(meta.revocation_endpoint, tokens.refresh_token)
            raise SignInCancelled("Sign-in was cancelled.")
        return tokens

    # -- listener thread ----------------------------------------------------------------------------------

    def _serve(self, httpd: _ListenerServer, deadline: float) -> None:
        try:
            while not self._closing.is_set() and time.monotonic() < deadline:
                httpd.handle_request()
        except Exception:  # noqa: BLE001 - a socket error while closing ends the loop quietly
            pass
        finally:
            try:
                httpd.server_close()
            except OSError:
                pass
            with self._lock:
                self._finished = True
            self._closing.set()   # a wait() still running sees the listener is gone

    def _on_callback(self, query: dict[str, str]) -> str:
        """Runs on a handler thread: hand the FIRST answer to ``wait`` and return the result it decides."""
        with self._lock:
            first, self._accepting = self._accepting, False
            finished = self._finished
        if not first:
            return ALREADY
        if finished:
            return self._result
        self._callbacks.put(query)
        if self._page_ready.wait(PAGE_WAIT_S):
            return self._result
        return PENDING

    def _show(self, result: str) -> None:
        self._result = result
        self._page_ready.set()


def _log(outcome: str) -> None:
    _turnlog.log_event("auth", event="sign_in", method="browser", outcome=outcome)
