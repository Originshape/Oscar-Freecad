"""Persistent agent activity log: JSON lines in ``agent.log``.

Where the file lives, in priority order:

1. ``OSCAR_LOG_DIR`` (read at CALL time, so a test can change it between events);
2. under a test run (``unittest`` imported or ``PYTEST_CURRENT_TEST`` set, and no GUI session) and with no
   override: ``<tempdir>/oscar-test-logs-<pid>/agent.log``, so a test can never write into the real log;
3. ``<FreeCAD user app data>/Oscar/logs`` (``~/.local/share/Oscar/logs`` without FreeCAD).

Every tool call, permission decision and turn lifecycle event is appended with a timestamp, so "it stopped
after N seconds" is answerable from disk next time. Every line carries ``ts``, ``kind`` and ``src``
(``"app"``, or ``"test"`` under a test run), then the module-level context (``set_context``: ``turn_id``,
``chat_id``, ``model``), then the event's own fields. The file rotates at 5 MB (``agent.log`` ->
``agent.log.1`` ..., three generations kept).

Payload discipline: argument NAMES and sizes, never full code or API bodies, never keys. The log must be
safe to share; as a safety net ``log_event`` masks values stored under credential-looking field names and
clips very long strings. This module is pure: FreeCAD is imported lazily and only to find the default
directory.
"""

import json
import os
import sys
import tempfile
import threading
import time

ROTATE_BYTES = 5 * 1024 * 1024      # rotate when agent.log reaches this size
KEEP_ROTATED = 3                    # agent.log.1 .. agent.log.N are kept
MAX_VALUE_CHARS = 4000              # longer string values are clipped (a log line is not a transcript)
_REDACT_KEYS = frozenset({
    "api_key", "apikey", "authorization", "access_token", "refresh_token", "id_token", "token",
    "password", "passwd", "secret", "client_secret", "bearer", "cookie", "x-api-key",
    # OriginShape sign-in (oscar.auth): PKCE verifier, device and authorization codes, refresh payloads
    "code_verifier", "device_code", "authorization_code", "refresh", "user_code",
})

_LOCK = threading.Lock()
_CONTEXT: dict = {}
_WARNED = False


# -- location ---------------------------------------------------------------------------------------------

def _gui_up() -> bool:
    """True inside the running GUI application (never auto-redirect a real session's log)."""
    fc = sys.modules.get("FreeCAD")
    try:
        return bool(getattr(fc, "GuiUp", 0))
    except Exception:  # noqa: BLE001
        return False


def under_test() -> bool:
    """True when this process is a test run: ``unittest`` is imported (and no GUI is up) or pytest set
    ``PYTEST_CURRENT_TEST``. Used to redirect the log away from the user's real one."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    return "unittest" in sys.modules and not _gui_up()


def _default_dir() -> str:
    try:
        import FreeCAD

        return os.path.join(FreeCAD.getUserAppDataDir(), "Oscar", "logs")
    except Exception:  # noqa: BLE001 - no FreeCAD (CI, plain python): a per-user fallback
        return os.path.join(os.path.expanduser("~"), ".local", "share", "Oscar", "logs")


def log_dir() -> str:
    """The directory of ``agent.log`` (created). ``OSCAR_LOG_DIR`` wins; a test run without it is
    redirected to ``<tempdir>/oscar-test-logs-<pid>``; otherwise the app-support default."""
    base = os.environ.get("OSCAR_LOG_DIR")
    if not base:
        if under_test():
            base = os.path.join(tempfile.gettempdir(), f"oscar-test-logs-{os.getpid()}")
        else:
            base = _default_dir()
    os.makedirs(base, exist_ok=True)
    return base


def log_path() -> str:
    """Absolute path of the current ``agent.log`` (its directory is created). Resolved on every call."""
    return os.path.join(log_dir(), "agent.log")


# -- context ----------------------------------------------------------------------------------------------

def set_context(**fields) -> None:
    """Merge ``fields`` (``turn_id``, ``chat_id``, ``model``, ...) into every event logged from now on.
    A value of ``None`` removes that key. The context is process-wide, not per thread: the agent sets it at
    turn start and calls ``clear_context`` in its teardown."""
    with _LOCK:
        for key, value in fields.items():
            if value is None:
                _CONTEXT.pop(key, None)
            else:
                _CONTEXT[key] = value


def clear_context() -> None:
    """Forget every field set by ``set_context``."""
    with _LOCK:
        _CONTEXT.clear()


def get_context() -> dict:
    """A copy of the current context (for tests and diagnostics)."""
    with _LOCK:
        return dict(_CONTEXT)


# -- writing ----------------------------------------------------------------------------------------------

def _scrub(value, depth: int = 0):
    """Mask credential-looking keys and clip long strings, recursively (depth-limited)."""
    if isinstance(value, str):
        if len(value) > MAX_VALUE_CHARS:
            return value[:MAX_VALUE_CHARS] + f"...[+{len(value) - MAX_VALUE_CHARS} chars]"
        return value
    if depth >= 4:
        return value
    if isinstance(value, dict):
        return {k: ("[redacted]" if isinstance(k, str) and k.lower() in _REDACT_KEYS and value[k] not in (None, "")
                    else _scrub(v, depth + 1)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v, depth + 1) for v in value]
    return value


def _rotate_if_needed(path: str) -> None:
    """Shift ``agent.log`` -> ``.1`` -> ``.2`` ... once it is ``ROTATE_BYTES`` or larger. Never raises."""
    try:
        if os.path.getsize(path) < ROTATE_BYTES:
            return
    except OSError:
        return
    try:
        for i in range(max(KEEP_ROTATED, 1), 0, -1):
            src = path if i == 1 else f"{path}.{i - 1}"
            if os.path.exists(src):
                os.replace(src, f"{path}.{i}")
    except OSError:
        pass  # another process rotated first, or the file is locked (Windows): keep appending


def _warn_once(message: str) -> None:
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    try:
        sys.stderr.write(f"oscar turnlog: {message}\n")
    except Exception:  # noqa: BLE001
        pass


def log_event(kind: str, **fields) -> None:
    """Append one event line; a failure never breaks the caller.

    The line is ``{"ts", "kind", "src", <context>, <fields>}``: ``fields`` may not override the first three.
    Values under credential-looking names are masked and strings over ``MAX_VALUE_CHARS`` are clipped."""
    try:
        with _LOCK:
            body = {**_CONTEXT, **fields}
            body = {k: v for k, v in body.items() if k not in ("ts", "kind", "src")}
            entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": str(kind),
                     "src": "test" if under_test() else "app", **_scrub(body)}
            line = json.dumps(entry, default=str) + "\n"
            path = log_path()
            _rotate_if_needed(path)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:  # noqa: BLE001 - logging must never raise
        _warn_once(f"{type(e).__name__}: {e}")


# -- reading ----------------------------------------------------------------------------------------------

def _last_lines(path: str, n: int) -> list[bytes]:
    """The last ``n`` non-empty lines of ``path`` as bytes (oldest first); ``[]`` when unreadable."""
    if n <= 0:
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            data = b""
            while pos > 0 and data.count(b"\n") <= n:
                step = min(65536, pos)
                pos -= step
                f.seek(pos)
                data = f.read(step) + data
    except OSError:
        return []
    rows = data.split(b"\n")
    if pos > 0:
        rows = rows[1:]  # the first row is a partial line
    return [r for r in rows if r.strip()][-n:]


def tail(limit: int = 40) -> list[dict]:
    """The most recent ``limit`` events, oldest first (for debugging from the console and for tests).

    Reads the same file ``log_event`` writes (so it also works under the test redirect) and continues into
    the rotated generations when ``agent.log`` alone holds fewer lines. Unparseable lines are skipped."""
    try:
        base = log_path()
    except OSError:
        return []
    names = [base] + [f"{base}.{i}" for i in range(1, max(KEEP_ROTATED, 1) + 1)]
    events: list[dict] = []
    for name in names:
        rows = _last_lines(name, limit - len(events))
        parsed = []
        for row in rows:
            try:
                item = json.loads(row.decode("utf-8", "replace"))
            except ValueError:
                continue
            if isinstance(item, dict):
                parsed.append(item)
        events = parsed + events
        if len(events) >= limit:
            break
    return events[-limit:] if limit > 0 else []
