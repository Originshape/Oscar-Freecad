"""Transport plumbing for ``oscar.providers``. No oscar imports.

* ``AttemptRunner`` - ONE provider attempt (the ``requests.post`` that waits for headers AND the
  stream parse) in a daemon reader thread that pushes events into an UNBOUNDED queue. A slow
  consumer can therefore never block the reader, and the consumer can poll Stop and emit ticks even
  while the server is silent.
* Cancelling an attempt: set the abort event, shut the raw socket down (this is what wakes a ``recv`` that is
  blocked in the reader; ``Response.close()`` alone does not and can even block its caller until the server
  answers) - for READING ONLY on a TLS socket, see ``shutdown_socket`` - close the response on a helper thread,
  join the reader for at most one second.
* Retry policy helpers: which HTTP statuses are retried, ``Retry-After``, backoff, quota detection and
  the human sentence for an HTTP error. A provider may refine the sentence and the policy per status
  (``ChatLoop._http_error``), as the OriginShape provider does for its account errors.
"""

import datetime
import email.utils
import http.client
import os
import queue
import random
import socket
import ssl
import threading
import time

import requests

try:  # urllib3 is requests' own dependency; guarded so a stripped build still imports
    from urllib3.exceptions import HTTPError as _Urllib3Error
except Exception:  # noqa: BLE001
    _Urllib3Error = ()

class StopRequested(Exception):
    """Raised inside a stream when the user pressed Stop."""


# -- retry policy ---------------------------------------------------------

MAX_ATTEMPTS = 5
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, *range(520, 530)})
BACKOFF_CAP_S = 20.0
RETRY_AFTER_CAP_S = 60.0
RETRY_SLEEP_BUDGET_S = 90.0
EARLY_EOF_BILLED_CHARS = 2000   # an attempt that streamed more than this and then dropped is a billed generation
_QUOTA_WORDS = ("insufficient_quota", "exceeded your current quota", "billing", "credit")

_STATUS_LABEL = {
    400: "bad request", 401: "unauthorized (check the API key)", 402: "payment required",
    403: "forbidden", 404: "not found (check the base URL and the model name)", 408: "request timeout",
    409: "conflict", 413: "request too large", 422: "unprocessable request", 425: "too early",
    429: "rate limited", 500: "server error", 502: "bad gateway", 503: "service unavailable",
    504: "gateway timeout", 529: "overloaded",
}

# Errors raised while talking to the server that retrying cannot fix (bad URL, bad header).
_PERMANENT_REQUEST_ERRORS = (
    requests.exceptions.InvalidURL, requests.exceptions.MissingSchema, requests.exceptions.InvalidSchema,
    requests.exceptions.InvalidHeader, requests.exceptions.URLRequired,
)
_TRANSPORT_ERRORS = (requests.RequestException, OSError, http.client.HTTPException, _Urllib3Error) \
    if _Urllib3Error else (requests.RequestException, OSError, http.client.HTTPException)


class ProviderError(RuntimeError):
    """A provider failure carrying what the retry policy needs.

    ``status`` is the HTTP status (0 for in-stream or transport errors), ``retriable`` says whether
    the same request is worth sending again, ``kind`` is ``"http" | "stream" | "transport" | "auth" | "account"``
    (``auth``: the sign-in is missing or refused; ``account``: the account cannot be used right now, e.g. no
    credits). ``code`` is a short machine reason (``"insufficient_credits"``, ``"signed_out"`` ...; ``""`` when
    there is none), ``data`` what the server said about it (a dict, never a secret) and ``retry_after`` the wait in
    seconds the server asked for (None when it named none): ``ChatLoop`` stops retrying when that wait exceeds what
    the call may still sleep.
    """

    def __init__(self, message: str, *, status: int = 0, retriable: bool = False, kind: str = "", code: str = "",
                 data: dict | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retriable = retriable
        self.kind = kind
        self.code = code
        self.data = data
        self.retry_after = retry_after


def anthropic_base(profile) -> str:
    """Host root of an Anthropic-style endpoint: gateways are often configured with the /v1 (or even the full
    /v1/messages) already on, so ``/v1/messages`` and ``/v1/models`` are appended to what is left."""
    base = (getattr(profile, "base_url", "") or "https://api.anthropic.com").rstrip("/")
    for suffix in ("/v1/messages", "/v1"):
        if base.endswith(suffix):
            return base[: -len(suffix)].rstrip("/")
    return base


def is_quota_error(text: str) -> bool:
    """True when an error body is about quota / billing / credit: retrying cannot help."""
    lower = (text or "").lower()
    return any(word in lower for word in _QUOTA_WORDS)


def retriable_message(text: str) -> bool:
    """True when an in-stream error message names an overload, a rate limit or a broken connection to the model (the
    OriginShape gateway's "connection to the model was interrupted", Anthropic's own ``api_error``), and is not a quota
    problem."""
    lower = (text or "").lower()
    if is_quota_error(lower):
        return False
    return any(word in lower for word in ("overloaded", "rate_limit", "rate limit", "too many requests", "capacity",
                                          "interrupted", "api_error", "internal server error"))


def http_error(status: int, body: str) -> ProviderError:
    """The final error for an HTTP failure: ``"HTTP 429 rate limited: <first 200 chars of the body>"``."""
    label = _STATUS_LABEL.get(status) or ("server error" if status >= 500 else "error")
    quota = is_quota_error(body)
    if quota:
        label = "quota or billing problem"
    snippet = " ".join((body or "").split())[:200]
    message = f"HTTP {status} {label}: {snippet}" if snippet else f"HTTP {status} {label}"
    return ProviderError(message, status=status, retriable=(status in RETRY_STATUS and not quota), kind="http")


def is_transport_error(exc: BaseException) -> bool:
    """True for connection-level failures that are worth retrying (not for a malformed URL or a bug)."""
    if isinstance(exc, _PERMANENT_REQUEST_ERRORS):
        return False
    return isinstance(exc, _TRANSPORT_ERRORS)


def _header(headers, name: str):
    if not headers:
        return None
    try:
        value = headers.get(name)
        if value is not None:
            return value
        lowered = name.lower()
        for key in list(headers):
            if str(key).lower() == lowered:
                return headers[key]
    except Exception:  # noqa: BLE001 - exotic header containers
        return None
    return None


def retry_after_seconds(headers, now: float | None = None) -> float | None:
    """The ``Retry-After`` header as seconds (a number or an HTTP date), clamped to >= 0; ``None`` if absent/invalid."""
    value = _header(headers, "Retry-After")
    if value is None:
        return None
    text = str(value).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    current = datetime.datetime.fromtimestamp(now if now is not None else time.time(), datetime.timezone.utc)
    return max(0.0, (when - current).total_seconds())


def backoff_delay(attempt: int, jitter=random.uniform) -> float:
    """``min(2 ** attempt + jitter(0..1), 20)`` seconds; ``attempt`` is the 0-based number of the attempt that just failed."""
    return min(2.0 ** max(0, attempt) + jitter(0.0, 1.0), BACKOFF_CAP_S)


def retry_delay(attempt: int, headers=None, *, jitter=random.uniform) -> float:
    """Seconds to wait before the next attempt: ``Retry-After`` (capped at 60 s) when sent, else the backoff."""
    advised = retry_after_seconds(headers)
    if advised is not None:
        return min(advised, RETRY_AFTER_CAP_S)
    return backoff_delay(attempt, jitter)


# -- cancelling ------------------------------------------------------------------------------------------


def _shutdown_how(sock) -> int:
    """How to shut ``sock`` down so a ``recv`` blocked on it returns.

    A TLS socket (and anything that is not a plain ``socket.socket``) is shut down for READING only on POSIX:
    with ``SHUT_RDWR`` OpenSSL answers the forced EOF with an alert write that fails with EPIPE, and the resulting
    SIGPIPE kills a process that never ignored it (FreeCAD's embedded Python). A plain TCP socket sends nothing
    on its own, so the full shutdown is safe there (and lets the server notice the client is gone at once).
    Windows has no SIGPIPE and needs the full shutdown to release the blocked ``recv``.
    """
    if os.name == "nt":
        return socket.SHUT_RDWR
    plain = isinstance(sock, socket.socket) and not isinstance(sock, ssl.SSLSocket)
    return socket.SHUT_RDWR if plain else socket.SHUT_RD


def shutdown_socket(response) -> bool:
    """``socket.shutdown(...)`` (``_shutdown_how``) on the socket behind a streaming ``requests`` response; True if
    one was shut down.

    Every step of the attribute chain is guarded: test fakes have no ``raw``. Two chains are tried
    (the buffered file http.client reads from, and urllib3's connection) because the layout differs
    between urllib3 versions.
    """
    raw = getattr(response, "raw", None)
    candidates = [
        getattr(getattr(getattr(getattr(raw, "_fp", None), "fp", None), "raw", None), "_sock", None),
        getattr(getattr(raw, "_connection", None), "sock", None),
    ]
    done = False
    for sock in candidates:
        if sock is None or not hasattr(sock, "shutdown"):
            continue
        try:
            sock.shutdown(_shutdown_how(sock))
            done = True
        except Exception:  # noqa: BLE001 - already closed, not connected, ...
            pass
    return done


def close_quietly(response) -> None:
    """``response.close()`` that never raises (and may be a no-op on fakes)."""
    close = getattr(response, "close", None)
    if close is not None:
        try:
            close()
        except Exception:  # noqa: BLE001
            pass


class AttemptRunner:
    """Runs ``work(runner)`` on a daemon reader thread; events go to ``runner.events`` (an unbounded queue).

    ``work`` pushes ``(kind, payload)`` tuples with ``runner.put``. Whatever it raises is delivered as
    ``("_exc", exception)``; the very last item is always ``("_end", None)``. ``runner.response`` is set
    by ``work`` as soon as there is a response object, so ``cancel`` can reach its socket.
    """

    live_threads = 0
    _live_lock = threading.Lock()

    def __init__(self, work, *, name: str = "oscar-reader"):
        self.events: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.abort = threading.Event()
        self.response = None
        self._work = work
        self._thread = threading.Thread(target=self._main, name=name, daemon=True)

    def put(self, kind: str, payload=None) -> None:
        """Queue one event for the consumer (never blocks)."""
        self.events.put((kind, payload))

    def start(self) -> None:
        """Start the reader thread."""
        with AttemptRunner._live_lock:
            AttemptRunner.live_threads += 1
        self._thread.start()

    def _main(self) -> None:
        try:
            self._work(self)
        except BaseException as e:  # noqa: BLE001 - delivered to the consumer, which decides
            self.put("_exc", e)
        finally:
            with AttemptRunner._live_lock:
                AttemptRunner.live_threads -= 1
            self.put("_end")

    @property
    def alive(self) -> bool:
        """True while the reader thread is running."""
        return self._thread.is_alive()

    def cancel(self, join_s: float = 1.0) -> bool:
        """Abort the attempt: abort event, socket shutdown, close, then join the reader for at most ``join_s``.

        Returns True when the reader ended. The close runs on a helper thread because
        ``Response.close()`` can block its caller for as long as the server stays silent.
        """
        self.abort.set()
        if not self._thread.is_alive():
            return True   # finished on its own: the reader has already closed its response
        response = self.response
        if response is not None:
            shutdown_socket(response)
            threading.Thread(target=close_quietly, args=(response,), daemon=True, name="oscar-closer").start()
        thread = self._thread
        if thread.is_alive() and thread is not threading.current_thread():
            thread.join(max(0.0, join_s))
        return not thread.is_alive()

    @classmethod
    def live_reader_count(cls) -> int:
        """Reader threads currently alive in this process."""
        with cls._live_lock:
            return cls.live_threads
