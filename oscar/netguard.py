"""Human-in-the-loop gate for Oscar's network access.

Oscar runs model-written code (execute_code and friends). When that code
wants the network — downloads, HTTP calls, sockets, package installs —
the user decides FIRST. Two layers:

* static intent analysis (``analyze``/``needs_permission``) runs BEFORE
  any execute_code* tool hands code to the runtime: imports of network
  libraries, fetch calls, pip installs and remote URLs are named back to
  the user verbatim;
* the permission prompt (``request_permission``) is a modal dialog on the
  GUI thread — Allow once / Always allow THIS host for THIS tool (a
  persisted per-(tool, host) grant, revocable in Settings) / Deny, auto-denying after a timeout or in console sessions where no
  human is watching.

``download_file`` is the sanctioned network path: prompted the same way,
streamed off the GUI thread with a size cap into ``default_download_dir()``
(``~/Downloads/Oscar``), so Oscar never needs to hand-roll download
code (which is also how a QThread-in-exec-code could previously take the
app down). ``download_verified`` is that path's engine: it writes through a
temp file, hashes and sniffs what arrived (an HTML page or a git-LFS pointer
is never accepted as a CAD file), and ``extract_zip`` unpacks an archive
without letting a member write outside its folder.
"""

import ast
import contextlib
import hashlib
import itertools
import os
import re
import sys
import threading
import time
import uuid
import zipfile
from collections.abc import Callable, Sequence
from zlib import error as zlib_error
from dataclasses import asdict, dataclass, field
from urllib.parse import unquote, urljoin, urlsplit
from . import fsretry as _fsretry

from .refparts import identity as _identity

NETWORK_IMPORTS = ("requests", "urllib", "urllib3", "httpx", "aiohttp", "http.client", "httplib2",
                   "ftplib", "smtplib", "imaplib", "poplib", "xmlrpc", "xmlrpc.client", "paramiko", "socket", "telnetlib",
                   "websockets", "websocket", "pycurl", "ssl")
# A script can reach the network without importing a library: through a command line tool (the URL may be a variable)
# or through a dynamic import. Both over-approximate on purpose: a false prompt costs one click.
_SHELL_RE = re.compile(r"\b(?:subprocess|os\.system|os\.popen|Popen|check_output|check_call|shlex)\b")
_NET_COMMAND_RE = re.compile(r"\b(?:curl|wget|aria2c|scp|sftp|rsync|ssh|ncat|netcat)\b")
_DYNAMIC_IMPORT_RE = re.compile(r"""(?:__import__|import_module)\(\s*['"]([\w.]+)""")
_IMPORT_LINE_RE = re.compile(r"^\s*(?:import\s+(.+)|from\s+([\w.]+)\s+import\b)")
_URL_RE = re.compile(r"https?://[^\s'\"<>)]+", re.IGNORECASE)
_INSTALL_RE = re.compile(
    r"\b(?:pip3?\s+install|python\d*(?:\.\d+)?\s+-m\s+pip|pip\.main|"
    r"uv\s+pip\s+install|conda\s+install)\b", re.IGNORECASE)
_FETCH_RE = re.compile(
    r"\b(?:urlopen|urlretrieve|requests\.(?:get|post|put|head|request|"
    r"Session|stream)|httpx\.(?:get|post)|aiohttp\.ClientSession|"
    r"socket\.socket|urllib\.request|urllib3\.\w+|PoolManager)\b|"
    r"\b(?:create_connection|getaddrinfo|gethostbyname(?:_ex)?|"
    r"open_connection|start_server|HTTPS?Connection|SMTP(?:_SSL)?|"
    r"FTP(?:_TLS)?)\s*\(")
# exec / eval / compile of a string that mentions the network: the strings can hide an import the line scan cannot see.
_DYNAMIC_EXEC_RE = re.compile(r"\b(?:exec|eval|compile)\s*\(")
# a script that starts another interpreter (python / a shell / node ...) can run anything outside the in-process guard
_INTERPRETER_RE = re.compile(r"\b(?:python[\d.]*|pythonw|sys\.executable|pypy\d*|freecad(?:cmd)?|node(?:js)?|deno|bun|ruby|perl|php|lua|"
                             r"bash|zsh|dash|fish|powershell|pwsh|cmd(?:\.exe)?)\b|\b(?:sh)\s+-c\b|/bin/sh\b")
_MARKER_RE = re.compile(r"\b(?:" + "|".join(re.escape(m) for m in NETWORK_IMPORTS) + r")\b")
_LOCAL = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "::1")
# Modules that reach the network (or start other programs) below what the runtime audit hook can see: Qt's own network
# stack, the raw socket and TLS modules, native calls, and child processes (on FreeCAD's Python 3.11 a multiprocessing
# "spawn" starts its child without any audit event). Importing one, or naming it, is asked about up front.
_GUARD_BYPASS_NAMES = {
    "QtNetwork": "Qt network module", "QtWebSockets": "Qt network module",
    "QtWebEngineCore": "Qt web engine", "QtWebEngineWidgets": "Qt web engine",
    "_socket": "low-level network module", "_ssl": "low-level network module",
    "ctypes": "native code access", "cffi": "native code access",
    "multiprocessing": "child processes", "_posixsubprocess": "child processes", "_winapi": "child processes",
    "ProcessPoolExecutor": "child processes",
}
# Calls that import or load code by name: a computed (non-literal) name hides what is loaded from the scan above.
_IMPORT_CALLS = frozenset({"__import__", "import_module", "find_spec", "module_from_spec", "spec_from_file_location",
                           "load_module", "exec_module", "run_module", "run_path", "load_source"})


def _import_pattern(mod: str) -> "re.Pattern":
    # A statement starts at the line start or after ';' / ':' ("x = 1;
    # import socket", "try: import socket"); the module may sit anywhere in
    # a comma list ("import os, socket"); a dotted module is also reached
    # as "from http import client".
    name = re.escape(mod)
    forms = [rf"import\s+(?:[\w.]+(?:\s+as\s+\w+)?\s*,\s*)*{name}\b",
             rf"from\s+{name}\b"]
    if "." in mod:
        pkg, leaf = mod.rsplit(".", 1)
        forms.append(rf"from\s+{re.escape(pkg)}\s+import\s+[^#]*\b{leaf}\b")
    return re.compile(r"(?:^|[;:])\s*(?:" + "|".join(forms) + ")")


_IMPORT_RES = [(mod, _import_pattern(mod)) for mod in NETWORK_IMPORTS]


def _imported_names(code: str) -> list[str]:
    """Dotted names the code imports, at any depth; [] if it won't parse
    (the line scan in analyze() still runs for such snippets)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.level:
                names.append(node.module)
                names += [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                # `from . import socket` / `from .socket import x`: a bare
                # script has no package to be relative to, so these fail at
                # run time — but the gate over-approximates (a false prompt
                # costs one click), so the names are still checked.
                if node.module:
                    names.append(node.module)
                names += [alias.name for alias in node.names]
    return names


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _bypass_reasons(code: str) -> list[str]:
    """What the import scan cannot judge: a module loaded under a COMPUTED name (``import_module("so" + "cket")``,
    ``__import__(name)``, ``spec.loader.exec_module(m)``) and the modules in ``_GUARD_BYPASS_NAMES``, imported or named
    (``from PySide import QtNetwork``, ``import ctypes``, ``mp = multiprocessing``). [] if the code does not parse."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    reasons: list[str] = []

    def add(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    def check_module(name: str) -> None:
        for part in str(name).split("."):
            if part in _GUARD_BYPASS_NAMES:
                add(f"{_GUARD_BYPASS_NAMES[part]}: {part}")

    for name in _imported_names(code):
        check_module(name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in _IMPORT_CALLS:
            first = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "name"), None)
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                check_module(first.value)
            else:
                try:
                    text = ast.unparse(node)
                except Exception:  # noqa: BLE001 - the reason text is cosmetic
                    text = _call_name(node) + "(...)"
                add(f"dynamic import of a computed module name: {text[:60]}")
        elif isinstance(node, ast.Name) and node.id in _GUARD_BYPASS_NAMES:
            check_module(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in _GUARD_BYPASS_NAMES:
            check_module(node.attr)
    return reasons


DENY_TIMEOUT_S = 120
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024

# UI hook: the panel sets this so "Waiting for your approval…" replaces
# the bare spinner while a permission dialog is open (netguard announces
# waiting=True on show, False on answer).
on_permission_wait = None

# Hook for a remote view of the turn (connect mode shows approvals in the browser): called with
# ``{"id", "op", "title", "text", "detailed", "choices", "state"}`` once when a question is put to the human
# (``state`` "waiting") and once when it is settled ("approved", "declined" or "no_answer", with ``by``:
# "freecad" or "browser"). ``choices`` are the answers a remote view may give (``[{"key", "label", "role"}]``,
# ``Choice.remote``). Questions nobody can be asked (console) are not reported. A failing hook is ignored.
on_approval = None
_APPROVAL_IDS = itertools.count(1)

# Answers from a remote view (``answer_remote``), by approval id. A question accepts one while its dialog is open
# (or about to open); the dialog then clicks that button itself, so the answer goes exactly the way a click does
# (``Choice.on_click`` included) and the dialog closes.
REMOTE_POLL_MS = 200
_REMOTE_LOCK = threading.Lock()
_REMOTE_OPEN: dict[int, frozenset[str]] = {}
_REMOTE_ANSWERS: dict[int, str] = {}


def answer_remote(approval_id: int, key: str) -> bool:
    """Answer an open question from a remote view with one of its remote ``Choice`` keys (any thread). Returns
    whether it was taken: False when the question is no longer open, the key isn't offered remotely, or another
    answer came first. Whoever may answer remotely is the caller's decision (connect mode: a setting)."""
    with _REMOTE_LOCK:
        allowed = _REMOTE_OPEN.get(approval_id)
        if allowed is None or key not in allowed or approval_id in _REMOTE_ANSWERS:
            return False
        _REMOTE_ANSWERS[approval_id] = key
        return True


def _take_remote(approval_id: int) -> str | None:
    with _REMOTE_LOCK:
        return _REMOTE_ANSWERS.pop(approval_id, None)

# Turn-scoped decision cache: one human decision per turn covers the whole
# category. Without this, a model retrying downloads popped one dialog per
# attempt — and when nobody clicked, each burned the full 120 s timeout.
# That cascade turned one 403 into twenty minutes of "thinking".
_TURN_DECISION: dict[str, str] = {}


def reset_turn() -> None:
    """New turn → the user's previous allow/deny no longer applies."""
    _TURN_DECISION.clear()


def _announce_approval(info: dict, state: str) -> None:
    hook = on_approval
    if hook is None:
        return
    try:
        hook({**info, "state": state})
    except Exception:  # noqa: BLE001 - a remote view never breaks the question
        pass


def _approval_state(answer: str, choices: Sequence["Choice"]) -> str:
    role = next((choice.role for choice in choices if choice.key == answer), "")
    if role in ("YesRole", "AcceptRole"):
        return "approved"
    if role in ("NoRole", "RejectRole"):
        return "declined"
    return "no_answer"


def _announce_wait(waiting: bool) -> None:
    hook = on_permission_wait
    if hook is None:
        return
    try:
        hook(waiting)
    except Exception:
        pass


def _network_modules_of(line: str) -> list[str]:
    """The ``NETWORK_IMPORTS`` entries one line of code imports: ``import a, b as c``, ``from a.b import c`` and the
    dynamic ``__import__("a")`` / ``importlib.import_module("a")``. A name matches an entry when it is that module or a
    submodule of it (``urllib.request`` is ``urllib``; ``urllib3`` is not)."""
    names: list[str] = []
    found = _IMPORT_LINE_RE.match(line)
    if found:
        if found.group(2):
            names.append(found.group(2))
        else:
            for part in found.group(1).split(","):
                module = re.match(r"\s*([\w.]+)", part)
                if module:
                    names.append(module.group(1))
    names += _DYNAMIC_IMPORT_RE.findall(line)
    hits: list[str] = []
    for name in names:
        for mod in NETWORK_IMPORTS:
            if (name == mod or name.startswith(mod + ".")) and mod not in hits:
                hits.append(mod)
    return hits


def analyze(code: str) -> dict:
    """Static network-intent analysis of a Python snippet.

    Returns {"network": bool, "reasons": [...], "urls": [...]}. Deliberately
    over-approximates (an import with no call still asks) — a false prompt
    costs one click; a missed download costs trust.
    """
    reasons: list[str] = []
    urls = sorted(set(m.group(0).rstrip(".,)") for m in
                      _URL_RE.finditer(code or "")))
    # The parse sees every import form (any depth, any statement layout); the line scans below still catch what only looks
    # like one (a script written out as a string, a snippet that does not parse) and the dynamic imports.
    for name in _imported_names(code or ""):
        for mod in NETWORK_IMPORTS:
            if (name == mod or name.startswith(mod + ".")) and f"import {mod}" not in reasons:
                reasons.append(f"import {mod}")
    uncommented = []
    for line in (code or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        uncommented.append(line)
        for mod in _network_modules_of(line):
            if f"import {mod}" not in reasons:
                reasons.append(f"import {mod}")
        for mod, pattern in _IMPORT_RES:
            if pattern.search(line) and f"import {mod}" not in reasons:
                reasons.append(f"import {mod}")
        if _FETCH_RE.search(line):
            reasons.append(f"fetch call: {stripped[:60]}")
        if _INSTALL_RE.search(line):
            reasons.append(f"package install: {stripped[:60]}")
        if _DYNAMIC_EXEC_RE.search(line) and (_MARKER_RE.search(line) or _FETCH_RE.search(line) or _NET_COMMAND_RE.search(line)
                                              or _URL_RE.search(line)):
            reasons.append(f"dynamic code with network markers: {stripped[:60]}")
    # name the most specific module back: "from xmlrpc import client" is "import xmlrpc.client", not also "import xmlrpc"
    named = [r[len("import "):] for r in reasons if r.startswith("import ")]
    reasons = [r for r in reasons
               if not (r.startswith("import ") and any(n.startswith(r[len("import "):] + ".") for n in named))]
    reasons += [r for r in _bypass_reasons(code or "") if r not in reasons]
    body = "\n".join(uncommented)
    if _SHELL_RE.search(body) and _NET_COMMAND_RE.search(body):
        reasons.append("network command line tool: " + _NET_COMMAND_RE.search(body).group(0))
    elif _SHELL_RE.search(body) and _INTERPRETER_RE.search(body):
        reasons.append("child interpreter: " + _INTERPRETER_RE.search(body).group(0))
    remote = [u for u in urls
              if not any(loc in u.split("//", 1)[-1] for loc in _LOCAL)]
    if remote:
        reasons.append("remote URLs")
    elif urls:
        reasons.append("local URLs (localhost)")
    return {"network": bool(reasons), "reasons": reasons[:8], "urls": urls[:8]}


def needs_permission(code: str) -> tuple[bool, str]:
    """(needs_prompt, summary) — summary names the intent for the dialog."""
    verdict = analyze(code)
    if not verdict["network"]:
        return False, ""
    lines = ["Oscar's code wants network access:"]
    lines += [f"• {r}" for r in verdict["reasons"]]
    if verdict["urls"]:
        lines.append("URLs: " + ", ".join(verdict["urls"][:4]))
    return True, "\n".join(lines)


# --- per-(tool, host) grants -----------------------------------------------------------------------------------
#
# A persistent grant is ``{tool, host, granted_at}`` in the settings store (``params.get_net_grants``): "this TOOL may talk to
# this HOST". There is no blanket flag any more: the legacy global ``AllowNetwork`` pref grants NOTHING (``legacy_flag`` only
# reports it so the Settings dialog can show it as a revocable legacy entry). Callers that cannot name a host (the script gate,
# ``download_file``) are "unscoped": their "Always allow" is stored visibly as ``{tool: "network", host: "*"}`` and can be
# revoked like any other grant.

UNSCOPED_TOOL = "network"
UNSCOPED_HOST = "*"


def _norm_host(host) -> str:
    return str(host or "").strip().lower().rstrip(".")


def list_grants() -> list[dict]:
    """The persistent grants (copies): ``[{"tool", "host", "granted_at"}]``."""
    from . import params as _params

    return _params.get_net_grants()


def has_grant(tool: str, host: str) -> bool:
    """True when ``tool`` was granted ``host`` for good (exact match; the unscoped pair is ``network`` / ``*``)."""
    host = _norm_host(host)
    return any(g["tool"] == tool and g["host"] == host for g in list_grants())


def add_grant(tool: str, host: str) -> bool:
    """Store a persistent grant (idempotent); True when it was new."""
    from . import params as _params

    return _params.add_net_grant(str(tool), _norm_host(host))


def revoke_grant(tool: str, host: str) -> bool:
    """Remove one grant (and this turn's cached decision for it); True when something was removed."""
    from . import params as _params
    from .turnlog import log_event

    removed = _params.remove_net_grant(str(tool), _norm_host(host))
    _TURN_DECISION.pop(_cache_key(tool, host), None)
    if (tool, _norm_host(host)) == (UNSCOPED_TOOL, UNSCOPED_HOST):
        _TURN_DECISION.pop("download", None)
    log_event("net_grant_revoked", tool=str(tool), host=_norm_host(host), removed=removed)
    return removed


def revoke_all() -> int:
    """Remove every grant, the legacy flag included, and forget this turn's decisions; returns how many entries went."""
    from . import params as _params
    from .turnlog import log_event

    legacy = legacy_flag()
    count = _params.clear_net_grants() + (1 if legacy else 0)
    _params.set_bool(_LEGACY_FLAG, False)
    _TURN_DECISION.clear()
    log_event("net_grant_revoked", tool="*", host="*", removed=count)
    return count


_LEGACY_FLAG = "AllowNetwork"


def legacy_flag() -> bool:
    """True when the old global 'Always allow' pref is still set. It grants NOTHING any more; Settings lists it so it can be revoked."""
    from . import params as _params

    try:
        return bool(_params.get_bool(_LEGACY_FLAG, False))
    except Exception:  # noqa: BLE001
        return False


def _cache_key(tool, host) -> str:
    """Turn-cache key: the legacy ``download`` key for unscoped calls, ``tool|host`` otherwise."""
    if not tool or not host:
        return "download"
    return f"{tool}|{_norm_host(host)}"


def audit_use(tool: str, url: str, **fields) -> None:
    """One audit event per network use (``net_use``): the tool and the HOST only, never the path or query of the URL."""
    from .turnlog import log_event

    try:
        host = _norm_host(urlsplit(url or "").hostname)
    except ValueError:
        host = "?"
    log_event("net_use", tool=str(tool), host=host or "?", **fields)


# --- the approval dialog (shared with codeguard) ------------------------------------------------------------------

# What ``ask_user`` returns when no choice was clicked.
NO_QT = "no_qt"             # Qt cannot be imported: no dialog is possible
NO_HUMAN = "no_human"       # no window a human could see (a console session): no dialog was shown
NO_ANSWER = ""              # the dialog timed out, was closed without a choice, or the GUI thread reached it too late
FAILED = "failed"           # the GUI thread did not run the dialog task in time (or it failed)


@dataclass(frozen=True)
class Choice:
    """One button of an ``ask_user`` dialog."""

    key: str                                      # what ask_user returns when this button is clicked
    label: str
    role: str = "AcceptRole"                      # a QMessageBox button role name: YesRole, AcceptRole, NoRole, RejectRole ...
    on_click: Callable[[], None] | None = None    # runs inside the GUI task right after the click, so a late click still counts
    remote: bool = True                           # a remote view may give this answer (``answer_remote``); False: only here


def _human_can_answer(QtWidgets) -> bool:
    """Is there a human who can see a dialog? A QApplication alone is NOT proof: console test runs may create one
    without any window. Only a real main window (or any shown window) makes the prompt real."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        return False
    try:
        import FreeCADGui as Gui

        return Gui.getMainWindow() is not None
    except Exception:  # noqa: BLE001
        return app.activeWindow() is not None


def ask_user(title: str, text: str, *, choices: Sequence[Choice], default: str, object_name: str, op: str,
             informative: str = "", detailed: str = "", timeout_s: float | None = None) -> str:
    """Ask the human a question in a modal dialog on the GUI thread; returns the ``key`` of the clicked ``Choice``.

    Without a human to ask nothing is shown: ``NO_QT`` / ``NO_HUMAN`` (console sessions), and the caller must refuse.
    The dialog closes itself after ``timeout_s`` (default ``DENY_TIMEOUT_S``) with ``NO_ANSWER``; a request the busy
    GUI thread only reaches after that deadline answers ``NO_ANSWER`` without showing anything, and a GUI thread that
    does not run it at all gives ``FAILED``. ``detailed`` goes into the scrollable "Show Details" area (plain text).
    The panel shows "Waiting for your approval" meanwhile (``on_permission_wait``). The dialog is dispatched with
    ``reentrant=True``: it holds no transaction, so other GUI tasks keep running while it waits.

    A remote view (``on_approval``) may answer with a ``remote`` choice through ``answer_remote`` while the
    question is open: the dialog clicks that button (or, if it isn't showing yet, the answer is taken at once)."""
    try:
        from PySide import QtCore, QtWidgets
    except ImportError:
        return NO_QT
    if not _human_can_answer(QtWidgets):
        return NO_HUMAN
    from .cad import dispatch

    app = QtWidgets.QApplication.instance()
    limit = float(DENY_TIMEOUT_S if timeout_s is None else timeout_s)
    # One deadline for the whole request. The dialog closes itself at it, and a request the GUI thread only reaches
    # afterwards (it was busy) is answered without a dialog: dispatch stops waiting 30 s later, so nobody is shown a
    # question the agent has already given up on.
    deadline = time.monotonic() + limit
    remote = [choice for choice in choices if choice.remote]
    info = {"id": next(_APPROVAL_IDS), "op": op, "title": title, "text": text, "detailed": detailed,
            "choices": [{"key": choice.key, "label": choice.label, "role": choice.role} for choice in remote]}
    by_remote: list[bool] = []

    def take_remote() -> Choice | None:
        key = _take_remote(info["id"])
        return next((choice for choice in remote if choice.key == key), None) if key is not None else None

    def ask() -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return NO_ANSWER
        early = take_remote()   # answered from the browser before the busy GUI thread got here
        if early is not None:
            by_remote.append(True)
            if early.on_click is not None:
                early.on_click()
            return early.key
        # The dialog is created from a queued call, so macOS will not focus-raise it on its own: without explicit
        # parenting, raise and activate below it can sit BEHIND the window while the user watches a spinner and the
        # timeout answers for them.
        parent = app.activeWindow() if app is not None else None
        if parent is None:
            try:
                import FreeCADGui as Gui

                parent = Gui.getMainWindow()
            except Exception:  # noqa: BLE001
                parent = None
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, title, text, QtWidgets.QMessageBox.NoButton, parent)
        box.setObjectName(object_name)   # the rebrand stylesheet (ui_rebrand) and the GUI smoke test find the dialog by this name
        plain = getattr(box, "setTextFormat", None)
        if plain is not None:
            plain(QtCore.Qt.PlainText)   # model-written text is data, never markup
        if informative:
            box.setInformativeText(informative)
        if detailed:
            box.setDetailedText(detailed)
        buttons = {}
        for choice in choices:
            role = getattr(QtWidgets.QMessageBox, choice.role, None)
            if role is None:
                role = QtWidgets.QMessageBox.NoRole
            buttons[choice.key] = box.addButton(choice.label, role)
        if default in buttons:
            box.setDefaultButton(buttons[default])
        box.setModal(True)
        box.show()
        box.raise_()
        box.activateWindow()
        # The caller gives up at the deadline; the dialog must go away with that answer instead of staying on screen
        # (modal, blocking the window) as a question nobody is waiting for any more.
        expired: list[bool] = []

        def expire() -> None:
            expired.append(True)
            getattr(box, "reject", lambda: None)()

        timer = QtCore.QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(expire)
        timer.start(max(1, int(remaining * 1000)))

        def poll_remote() -> None:
            choice = take_remote()
            if choice is not None and choice.key in buttons:
                by_remote.append(True)
                buttons[choice.key].click()   # the same path as a click here: clickedButton, on_click, close

        poller = QtCore.QTimer()
        poller.timeout.connect(poll_remote)
        if remote:
            poller.start(REMOTE_POLL_MS)
        _announce_wait(True)
        try:
            box.exec()
        finally:
            timer.stop()
            poller.stop()
            _announce_wait(False)
        if expired:
            return NO_ANSWER
        clicked = box.clickedButton()
        for choice in choices:
            if clicked is not None and clicked is buttons.get(choice.key):
                if choice.on_click is not None:
                    choice.on_click()
                return choice.key
        return NO_ANSWER

    with _REMOTE_LOCK:
        _REMOTE_OPEN[info["id"]] = frozenset(choice.key for choice in remote)
    _announce_approval(info, "waiting")
    _announce_wait(True)  # in case the dialog never becomes visible
    answer = FAILED
    try:
        result, failed = dispatch.dispatch(ask, timeout=limit + 30.0, op=op, reentrant=True)
        answer = FAILED if failed else (result if isinstance(result, str) else NO_ANSWER)
    finally:
        with _REMOTE_LOCK:
            _REMOTE_OPEN.pop(info["id"], None)
            _REMOTE_ANSWERS.pop(info["id"], None)
        _announce_wait(False)
        _announce_approval({**info, "by": "browser" if by_remote else "freecad"}, _approval_state(answer, choices))
    return answer


def request_permission(summary: str, detail: str = "", *, tool: str = "", host: str = "") -> str:
    """Ask the human. Returns 'allow' | 'always' | 'deny'.

    ``tool`` + ``host`` make the request SCOPED: a stored grant for exactly that pair answers 'allow' without a dialog, the dialog
    offers Allow once (this turn) / Always allow THIS host for THIS tool / Deny, and 'Always' stores ``{tool, host, granted_at}``.
    Another host (a redirect) is another request. Without them the request is unscoped (the script gate, ``download_file``): its
    'Always' stores the visible grant ``network`` / ``*``. The legacy global flag grants nothing.

    Console sessions (no QApplication) and timeouts deny by default —
    when nobody can answer, the answer is no. ``OSCAR_ALLOW_NET=1``
    overrides for unattended runs (documented escape hatch). One decision per turn is cached per (tool, host) (unscoped: one).
    """
    scoped = bool(tool and host)
    g_tool, g_host = (str(tool), _norm_host(host)) if scoped else (UNSCOPED_TOOL, UNSCOPED_HOST)
    cache_key = _cache_key(tool, host) if scoped else "download"
    if os.environ.get("OSCAR_ALLOW_NET") == "1":
        from .turnlog import log_event

        log_event("net_permission", decision="allow_env_override", tool=g_tool, host=g_host)
        return "allow"
    from .turnlog import log_event

    if has_grant(g_tool, g_host):
        log_event("net_permission", decision="allow_saved", tool=g_tool, host=g_host)
        return "allow"
    if cache_key in _TURN_DECISION:
        log_event("net_permission", decision=_TURN_DECISION[cache_key]
                  + "_cached_turn", tool=g_tool, host=g_host)
        return _TURN_DECISION[cache_key]

    def remember_always() -> None:
        # Saved at click time, not by the caller: a click that lands after the agent's wait was abandoned must still
        # be honoured.
        try:
            add_grant(g_tool, g_host)
        except OSError as e:
            # params.SettingsUnavailable: the settings file is locked right now. The click still allows this turn
            # (an exception here would turn it into a deny); the next turn asks again.
            log_event("net_permission", decision="grant_not_saved", tool=g_tool, host=g_host, reason=type(e).__name__)

    answer = ask_user(
        "Oscar — network access", summary, informative=detail,
        choices=(Choice("allow", "Allow once", "YesRole"),
                 # A lasting grant on this computer is only ever given here, never from the browser.
                 Choice("always", f"Always allow {g_host} for {g_tool}" if scoped else "Always allow", "AcceptRole",
                        remember_always, remote=False),
                 Choice("deny", "Deny", "NoRole")),
        default="deny", object_name="oscarPermission", op="net_permission")
    if answer == NO_QT:
        log_event("net_permission", decision="deny_no_qt")
        return "deny"
    if answer == NO_HUMAN:
        log_event("net_permission", decision="deny_console", tool=g_tool, host=g_host)
        _TURN_DECISION[cache_key] = "deny"
        return "deny"
    if answer in ("allow", "always"):
        result = answer
        log_event("net_permission", decision=result, tool=g_tool, host=g_host)
    else:
        result = "deny"
        log_event("net_permission", decision="deny", tool=g_tool, host=g_host,
                  reason={FAILED: "timeout_or_dispatch_failure", NO_ANSWER: "no_answer"}.get(answer, "declined"))
    _TURN_DECISION[cache_key] = result
    return result


def gate_code(code: str, what: str) -> tuple[bool, str]:
    """Pre-execution gate for the execute_code tools.

    Returns (allowed, message). On denial the message is model-facing:
    it says the user declined and what to do instead.
    """
    needs, summary = needs_permission(code)
    if not needs:
        return True, ""
    from .turnlog import log_event

    log_event("net_gate", tool=what, summary=summary[:200])
    verdict = request_permission(
        summary, detail=f"Requested by: {what}")
    if verdict in ("allow", "always"):
        return True, ""
    return False, (
        "The user DECLINED network access for this code"
        + (f" ({', '.join(analyze(code)['reasons'][:3])})."
           if analyze(code)["reasons"] else ".")
        + " Do not retry. Options: ask the user to download the file "
          "themselves and attach it, or use the download_file tool with a "
          "clear reason so the user can approve it explicitly.")



# --- runtime audit hook for execute_code -----------------------

_AUDIT_LOCK = threading.Lock()
_AUDIT_INSTALLED = False
_AUDIT_TLS = threading.local()
_NET_TEXT_RE = re.compile(_NET_COMMAND_RE.pattern + r"|https?://|\bpip3?\s+install\b|\s-m\s+pip\b", re.IGNORECASE)
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1", "")

# Events that start another program (or fork this one). A child process is not covered by the audit hook (it is a different
# interpreter), so ``python3 -c "import socket..."`` would escape the guard: any spawn needs the network grant unless the
# program is one of a few harmless local utilities. Deliberately an allowlist (a false block costs the model one
# 'ask the user' round trip; a missed escape costs trust). Newer Pythons (3.14; not FreeCAD's 3.11) also raise
# ``_posixsubprocess.fork_exec``: after ``subprocess.Popen``, and alone for a multiprocessing "spawn". On Windows
# ``os.startfile`` hands a file or URL to another program (``webbrowser.open`` uses it), so it counts as a spawn.
_SPAWN_EVENTS = ("subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn", "os.exec", "os.fork", "os.forkpty",
                 "_posixsubprocess.fork_exec", "os.startfile", "os.startfile/2")
_LOOKUP_EVENTS = ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr")
_SEND_EVENTS = ("socket.connect", "socket.sendto", "socket.sendmsg")
# The events the guard judges. Everything else returns before any work: ``os.*`` events (listdir, open, remove ...)
# are frequent and never need the stack walk below.
_GUARDED_EVENTS = frozenset(_SEND_EVENTS + _LOOKUP_EVENTS + ("socket.getnameinfo",) + _SPAWN_EVENTS)
_HOOK_PREFIXES = ("socket.", "subprocess.", "os.", "_posixsubprocess.")
# Model-written code is compiled under these file names: "<execute_code:N>" / "<execute_code>" (execute_code),
# "<execute_code_async>" (background jobs) and "<execute_code_headless>" (the headless child).
SCRIPT_FILE_PREFIX = "<execute_code"
_BENIGN_PROGRAMS = frozenset({"echo", "ls", "pwd", "true", "false", "sleep", "date", "uname", "whoami", "cat", "wc", "head", "tail"})
_SHELL_META_RE = re.compile(r"[;&|`$<>()\n\\]")


_SYSTEM_BIN_DIRS = ("/bin", "/usr/bin", "/sbin", "/usr/sbin")


def _spawn_executable(event: str, args) -> str:
    """The path a spawn event will really run, resolved the way the OS would (the ``env`` PATH the call passes, else this process's;
    relative to the call's ``cwd``): '' when it cannot be resolved. Never trusts the program NAME alone - a script can plant an
    executable called ``cat`` in its working directory or at the front of PATH."""
    import shutil

    try:
        env = cwd = None
        if event == "subprocess.Popen":
            argv = args[1] if len(args) > 1 else ()
            exe = args[0] or (argv[0] if isinstance(argv, (list, tuple)) and argv else "")
            cwd = args[2] if len(args) > 2 else None
            env = args[3] if len(args) > 3 else None
        elif event == "os.system":
            parts = str(args[0] if args else "").split()
            exe = parts[0] if parts else ""
        elif event == "os.posix_spawn":
            exe, env = (args[0] if args else ""), (args[2] if len(args) > 2 else None)
        elif event == "os.spawn":
            exe, env = (args[1] if len(args) > 1 else ""), (args[3] if len(args) > 3 else None)
        elif event == "os.exec":
            exe, env = (args[0] if args else ""), (args[2] if len(args) > 2 else None)
        elif event == "_posixsubprocess.fork_exec":
            # (executable candidates, argv, env): the parent already searched PATH, the child runs the first
            # candidate that exists. A relative candidate depends on a cwd the event does not carry: not trusted.
            for candidate in (args[0] if args else None) or ():
                path = os.fsdecode(candidate)
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    return os.path.normpath(path) if os.path.isabs(path) else ""
            return ""
        else:
            return ""
        exe = os.fsdecode(exe)
        if not exe:
            return ""
        if os.sep in exe or (os.altsep and os.altsep in exe):
            return os.path.normpath(os.path.join(os.fsdecode(cwd) if cwd else os.getcwd(), exe))
        search = None
        if env:
            for key in ("PATH", b"PATH"):
                if key in env:
                    search = os.fsdecode(env[key])
                    break
        if search is None:
            search = os.environ.get("PATH", os.defpath)
        return shutil.which(exe, path=search) or ""
    except Exception:  # noqa: BLE001 - unresolvable = not trusted
        return ""


def _is_system_binary(path: str) -> bool:
    """True only for an absolute path (as looked up AND after symlinks) inside a system directory the current user cannot
    write: a file the script could have planted is never one of these."""
    try:
        if not path or not os.path.isabs(path):
            return False
        dirs = _SYSTEM_BIN_DIRS
        if os.name == "nt":
            root = os.environ.get("SystemRoot", r"C:\Windows")
            dirs = (os.path.join(root, "System32"),)
        norm = (lambda p: os.path.normcase(os.path.normpath(p)))
        trusted = {norm(d) for d in dirs}
        real = os.path.realpath(path)
        # the lookup path AND the real file (after symlinks; merged-/usr links such as /bin -> /usr/bin stay inside the set)
        if norm(os.path.dirname(path)) not in trusted or norm(os.path.dirname(real)) not in trusted:
            return False
        return not os.access(os.path.dirname(real), os.W_OK)
    except Exception:  # noqa: BLE001
        return False


def _spawn_text(event: str, args) -> str:
    """The command line of a spawn audit event as one string (for the keyword checks)."""
    try:
        if event == "subprocess.Popen":
            argv = args[1] if len(args) > 1 else ()
            return " ".join(map(str, [args[0]] + (list(argv) if isinstance(argv, (list, tuple)) else [argv])))
        if event == "_posixsubprocess.fork_exec":   # the argv only: the environment (a proxy URL ...) is not the command
            argv = args[1] if len(args) > 1 else ()
            return " ".join(os.fsdecode(a) if isinstance(a, bytes) else str(a) for a in (argv or ()))
        return " ".join(map(str, args[:3]))
    except Exception:  # noqa: BLE001
        return ""


def _spawn_program(event: str, args) -> str:
    """Lower-case base name (no extension) of the program a spawn event starts; '' when it cannot be told (a fork,
    a shell command with metacharacters) - which is never on the allowlist."""
    try:
        if event == "subprocess.Popen":
            argv = args[1] if len(args) > 1 else ()
            exe = args[0] or (argv[0] if isinstance(argv, (list, tuple)) and argv else "")
        elif event == "os.system":
            command = str(args[0] if args else "")
            if _SHELL_META_RE.search(command):
                return ""
            parts = command.split()
            exe = parts[0] if parts else ""
        elif event == "os.posix_spawn":
            exe = args[0] if args else ""
        elif event == "os.spawn":
            exe = args[1] if len(args) > 1 else ""
        elif event == "os.exec":
            exe = args[0] if args else ""
        elif event == "_posixsubprocess.fork_exec":   # the program that really runs, never the argv[0] it is given
            candidates = list((args[0] if args else None) or ())
            exe = _spawn_executable(event, args) or (candidates[0] if candidates else "")
        else:  # os.fork / os.forkpty
            return ""
        name = os.path.basename(os.fsdecode(exe).replace("\\", "/")).lower()
        return name[:-4] if name.endswith(".exe") else name
    except Exception:  # noqa: BLE001
        return ""


def audit_enabled() -> bool:
    """False when ``OSCAR_NET_AUDIT=0`` (the kill switch) is set."""
    return os.environ.get("OSCAR_NET_AUDIT", "1").strip() != "0"


def network_granted() -> bool:
    """True when the user already allowed network for this turn / always (the unscoped grant ``network`` / ``*``) / by the unattended-run override.

    Never prompts (the audit hook fires inside running user code, on the GUI thread, where a modal prompt would deadlock):
    the prompt happens in ``gate_code`` BEFORE the script runs; this reads its recorded answer."""
    if os.environ.get("OSCAR_ALLOW_NET") == "1":
        return True
    try:
        if has_grant(UNSCOPED_TOOL, UNSCOPED_HOST):
            return True
    except Exception:  # noqa: BLE001
        pass
    return _TURN_DECISION.get("download") in ("allow", "always")


def _blocked_message(what: str) -> str:
    return (f"network access blocked: the script tried {what} but the user has not granted network access for this turn. "
            "Do not retry. Use the download_file tool (it asks the user), or ask the user to attach the file.")


def _loopback_host(host) -> bool:
    """True for a host name or address that never leaves this machine (``None``: a local wildcard lookup)."""
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host).strip().lower().rstrip(".")
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def _loopback_address(address) -> bool:
    if isinstance(address, (str, bytes)):          # AF_UNIX path
        return True
    try:
        host = address[0]
    except Exception:  # noqa: BLE001
        return False
    return host is not None and _loopback_host(host)


def _script_code_on_stack() -> bool:
    """True when a frame compiled from model-written code (``SCRIPT_FILE_PREFIX``) is on the calling thread's stack.

    This catches the callbacks a script leaves behind, which run later outside its ``script_guard``: Qt timers and
    slots, FeaturePython ``execute()`` methods, document observers, ``commit`` functions. Walked only for the guarded
    events and for thread starts, never for the frequent ``os.*`` events."""
    try:
        frame = sys._getframe(1)
    except (AttributeError, ValueError):
        return False
    while frame is not None:
        if frame.f_code.co_filename.startswith(SCRIPT_FILE_PREFIX):
            return True
        frame = frame.f_back
    return False


def _guard_wanted_here() -> bool:
    """True when the calling thread runs model-written code: inside ``script_guard``, or with script code on its stack."""
    return bool(getattr(_AUDIT_TLS, "depth", 0)) or _script_code_on_stack()


def _audit(event: str, args) -> None:
    if event not in _GUARDED_EVENTS or not audit_enabled():
        return
    if not _guard_wanted_here():
        return
    what = ""
    if event in _SEND_EVENTS:
        address = args[1] if len(args) > 1 else None
        if event == "socket.sendmsg" and address is None:
            pass        # a connected socket: its connect was judged already
        elif not _loopback_address(address):
            what = "to open a network connection" if event != "socket.sendmsg" else "to send data over the network"
    elif event in _LOOKUP_EVENTS:
        host = args[0] if args else None
        if not _loopback_host(host):
            if isinstance(host, bytes):
                host = host.decode("ascii", "replace")
            what = f"to resolve the host name {str(host).strip().lower()!r}"
    elif event == "socket.getnameinfo":
        if not _loopback_address(args[0] if args else None):
            what = "to look up the name of a network address"
    elif event in _SPAWN_EVENTS:
        text = _spawn_text(event, args)
        if _NET_TEXT_RE.search(text) or _INSTALL_RE.search(text):
            what = "to start a network command line tool"
        else:
            program = _spawn_program(event, args)
            if program not in _BENIGN_PROGRAMS or not _is_system_binary(_spawn_executable(event, args)):
                what = (f"to start {('the program ' + repr(program)) if program else 'a child process'} (a child process runs "
                        "outside this network guard and could reach the network, e.g. a spawned interpreter)")
    if what and not network_granted():
        raise PermissionError(_blocked_message(what))


def _audit_hook(event, args):
    if not event.startswith(_HOOK_PREFIXES):
        return
    try:
        _audit(event, args)
    except PermissionError:
        raise
    except Exception:  # noqa: BLE001 - a bug in the guard must never break unrelated code
        return


def _guarded_callable(fn: Callable) -> Callable:
    def run_guarded(*args, **kwargs):
        with script_guard():
            return fn(*args, **kwargs)

    return run_guarded


def _guarded_starter(original: Callable) -> Callable:
    """A ``_thread`` starter whose thread function runs inside ``script_guard`` when script code starts it."""
    def start(function, *args, **kwargs):
        try:
            if audit_enabled() and _guard_wanted_here():
                function = _guarded_callable(function)
        except Exception:  # noqa: BLE001 - starting a thread must never fail because of the guard
            pass
        return original(function, *args, **kwargs)

    start.oscar_guard = True   # type: ignore[attr-defined]
    start.__wrapped__ = original   # type: ignore[attr-defined]
    return start


def _wrap_thread_starts() -> None:
    """Threads that model-written code starts inherit its guard. FreeCAD's Python 3.11 raises no audit event when a
    thread starts, so the hook alone cannot see them: ``threading.Thread.start`` and the ``_thread`` starters are
    wrapped once, process-wide. ``threading`` keeps its own reference to the low-level starter, so a Thread is wrapped
    once only. The wrappers act only when the starting code is script code, and never raise."""
    import _thread

    original = threading.Thread.start
    if not getattr(original, "oscar_guard", False):
        def start(self, *args, **kwargs):
            try:
                if not getattr(self, "_oscar_guarded", False) and audit_enabled() and _guard_wanted_here():
                    self._oscar_guarded = True
                    self.run = _guarded_callable(self.run)
            except Exception:  # noqa: BLE001 - starting a thread must never fail because of the guard
                pass
            return original(self, *args, **kwargs)

        start.oscar_guard = True   # type: ignore[attr-defined]
        start.__wrapped__ = original   # type: ignore[attr-defined]
        threading.Thread.start = start
    for name in ("start_new_thread", "start_new", "start_joinable_thread"):
        starter = getattr(_thread, name, None)
        if starter is not None and not getattr(starter, "oscar_guard", False):
            setattr(_thread, name, _guarded_starter(starter))


def install_audit_hook() -> bool:
    """Install the process-wide audit hook and the thread-start wrappers once. Both are inert for code that is not
    model-written (outside ``script_guard`` and without script code on the stack)."""
    global _AUDIT_INSTALLED
    with _AUDIT_LOCK:
        if _AUDIT_INSTALLED:
            return False
        sys.addaudithook(_audit_hook)
        _wrap_thread_starts()
        _AUDIT_INSTALLED = True
        return True


@contextlib.contextmanager
def script_guard():
    """Guard the CALLING thread while it executes model-written code: socket connects and sends, DNS lookups, network
    command line tools and other programs raise ``PermissionError`` (model-facing text) unless ``network_granted()``.
    Re-entrant. Threads the guarded code starts inherit the guard; a callback compiled from script code is judged
    wherever it runs later (``_script_code_on_stack``). Other code on other threads (FreeCAD's own, the MCP bridge, the
    provider requests) is never affected. Disabled by ``OSCAR_NET_AUDIT=0``."""
    if not audit_enabled():
        yield
        return
    install_audit_hook()
    _AUDIT_TLS.depth = getattr(_AUDIT_TLS, "depth", 0) + 1
    try:
        yield
    finally:
        _AUDIT_TLS.depth -= 1


# --- downloads ---------------------------------------------------------------

_BROWSER_HEADERS = {
    # A plain python-requests UA is bot-filtered by many file hosts (GrabCAD etc. answer 403). A browser-like UA fixes most.
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/129.0.0.0 Safari/537.36",
    "Accept": "*/*",
}
_GITHUB_BLOB_RE = re.compile(r"^/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$")
_CD_STAR_RE = re.compile(r"filename\*\s*=\s*([^']*)'[^']*'([^;]+)", re.IGNORECASE)
_CD_PLAIN_RE = re.compile(r'filename\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^;]+))', re.IGNORECASE)
# Extensions a mislabelled download may carry; they are replaced (not kept) when the content turns out to be something else.
_REPLACEABLE_EXTENSIONS = (_identity.CAD_EXTENSIONS
                           | {".pdf", ".zip", ".html", ".htm", ".php", ".txt", ".bin", ".asp", ".aspx", ".jsp", ".cgi", ".download", ".dat"})
_FILENAME_MAX = 120


def default_download_dir() -> str:
    """Where downloads land: ``$OSCAR_DOWNLOAD_DIR`` (read at CALL time, ``~`` expanded) or ``~/Downloads/Oscar``.

    The directory is not created here; ``download_verified`` creates it when it writes.
    """
    env = os.environ.get("OSCAR_DOWNLOAD_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(os.path.expanduser("~"), "Downloads", "Oscar")


@dataclass
class DownloadResult:
    """Outcome of ``download_verified``. ``error == ""`` means success and ``path`` is the finished file.

    On failure ``path`` is empty and no file is left behind, but whatever was learned is kept for the message: ``bytes`` received,
    their ``sha256``, the ``detected_type`` (``identity.KNOWN_TYPES``) and the ``final_url`` after redirects.
    """

    path: str = ""
    bytes: int = 0
    sha256: str = ""
    detected_type: str = ""
    final_url: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """True when the download succeeded (a file at ``path``, no error)."""
        return not self.error and bool(self.path)

    def as_dict(self) -> dict:
        """The six fields as a plain dict (JSON-safe)."""
        return asdict(self)


def github_raw_url(url: str) -> str:
    """Rewrite a GitHub *blob* page URL to its raw file URL; every other URL is returned unchanged.

    ``https://github.com/<owner>/<repo>/blob/<ref>/<path>`` (query and fragment dropped, so ``?raw=true`` and ``#L1`` vanish)
    becomes ``https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>``. A blob URL serves an HTML page, never the file.
    """
    try:
        parts = urlsplit(url or "")
        hostname = parts.hostname or ""
    except ValueError:   # e.g. a "[" in the host that never closes: nothing to rewrite (check_public_url refuses it)
        return url
    if hostname.lower() in ("github.com", "www.github.com"):
        m = _GITHUB_BLOB_RE.match(parts.path)
        if m:
            owner, repo, ref, rest = m.groups()
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{rest}"
    return url


def _header(headers, name: str):
    """Case-insensitive header lookup that also works on a plain dict (requests gives a CaseInsensitiveDict, fakes a dict)."""
    if headers is None:
        return None
    low = name.lower()
    try:
        for key, value in headers.items():
            if str(key).lower() == low:
                return value
    except Exception:
        return None
    return None


def filename_from_content_disposition(value: str) -> str:
    """The file name a ``Content-Disposition`` header proposes (``filename*=UTF-8''...`` wins over ``filename=``), or ``""``.

    Only the last path component is returned (a header may not choose a directory).
    """
    if not value:
        return ""
    name = ""
    star = _CD_STAR_RE.search(value)
    if star:
        charset = (star.group(1) or "utf-8").strip() or "utf-8"
        try:
            name = unquote(star.group(2).strip().strip('"'), encoding=charset, errors="replace")
        except LookupError:
            name = unquote(star.group(2).strip().strip('"'), errors="replace")
    if not name:
        plain = _CD_PLAIN_RE.search(value)
        if plain:
            name = (plain.group(1) if plain.group(1) is not None else plain.group(2) or "").strip()
            name = re.sub(r'\\(["\\])', r"\1", name)      # only \" and \\ are quoted pairs; a Windows path keeps its backslashes
    return re.split(r"[\\/]", name)[-1].strip()


# Windows reserved device names are special with or without an extension ("CON", "nul.txt", "COM1.stl") and open the
# device, not a file.
_WIN_RESERVED_RE = re.compile(r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", re.IGNORECASE)


def _filename_from_url(url: str) -> str:
    """Last path component of the URL's PATH (never its query or fragment, so ``part.stl?dl=1`` is ``part.stl``),
    percent-decoded; a decoded ``%5C`` or ``%2F`` splits like a separator, so a hostile name cannot smuggle a directory in.
    ``""`` for a URL without a usable path (or one that does not parse)."""
    try:
        path = unquote(urlsplit(url or "").path)
    except ValueError:   # e.g. a "[" in the host that never closes
        return ""
    parts = [p for p in re.split(r"[\\/]", path) if p]
    return parts[-1] if parts else ""


def _sanitize_filename(name: str) -> str:
    """Letters, digits, ``.``, ``_`` and ``-`` only; no leading or trailing dots, no dot runs; at most 120 characters with the
    extension kept; never a Windows device name. Always sanitised, never rejected: the result passes ``_unsafe_filename``."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name or "")
    # _unsafe_filename refuses ".." (it is for names the model supplies), so a harmless "bracket..v2.stl" is collapsed first
    name = re.sub(r"\.{2,}", ".", name).strip(".")
    if len(name) > _FILENAME_MAX:
        stem, ext = os.path.splitext(name)
        ext = ext[:16]
        # the cut can land on a "." that would meet the extension's and make a ".." again
        name = stem[:max(1, _FILENAME_MAX - len(ext))].rstrip(".") + ext
    if _WIN_RESERVED_RE.match(name):
        name = "_" + name
    return name or "download"


def _derived_filename(url: str) -> str:
    """The file name a download takes from its URL when nothing better is known: ``_sanitize_filename`` of the path's last
    component (``download`` when there is none)."""
    return _sanitize_filename(_filename_from_url(url))


def _unsafe_filename(name: str) -> bool:
    """A model-supplied filename must be a single plain path component on every OS we ship for: no separators of either
    style, drive letters, streams, control characters or device names."""
    return bool(
        not name or name.startswith(".") or ".." in name
        or re.search(r'[\x00-\x1f<>:"/\\|?*]', name)
        or name != name.rstrip(" .")   # Windows silently drops these
        or _WIN_RESERVED_RE.match(name))


def _inside(path: str, folder: str) -> bool:
    """Does ``path`` still land in ``folder`` once links are resolved?"""
    folder = os.path.normcase(os.path.realpath(folder))
    path = os.path.normcase(os.path.realpath(path))
    try:
        return os.path.commonpath([folder, path]) == folder
    except ValueError:   # different drives
        return False


def _host(url: str) -> str:
    try:
        return (urlsplit(url or "").hostname or "").lower()
    except ValueError:
        return ""


def _fix_extension(name: str, detected: str) -> str:
    """Give a derived file name the extension its CONTENT calls for (``pi4`` + STEP -> ``pi4.step``; ``x.php`` + STL -> ``x.stl``)."""
    canonical = _identity.canonical_extension(detected)
    if not canonical:
        return name
    stem, ext = os.path.splitext(name)
    if _identity.extension_matches(ext, detected):
        return name
    if not ext or ext.lower() in _REPLACEABLE_EXTENSIONS:
        return _sanitize_filename(stem + canonical)
    return _sanitize_filename(name + canonical)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _destination(dest_dir: str, name: str, digest: str) -> tuple[str, bool]:
    """(path, already_there). Never overwrites different content: a taken name gets ``-<8 hex of the sha256>`` before the extension."""
    path = os.path.join(dest_dir, name)
    if not os.path.lexists(path):
        return path, False
    if os.path.isfile(path) and _file_sha256(path) == digest:
        return path, True
    stem, ext = os.path.splitext(name)
    for n in range(1000):
        candidate = os.path.join(dest_dir, f"{stem}-{digest[:8]}{'' if n == 0 else '-' + str(n)}{ext}")
        if not os.path.lexists(candidate):
            return candidate, False
        if os.path.isfile(candidate) and _file_sha256(candidate) == digest:
            return candidate, True
    raise OSError(f"no free file name for {name!r} in {dest_dir!r}")


# --- SSRF guard ---------------------------------------------------------------

MAX_REDIRECTS = 5
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_PRIVATE_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home.arpa", ".intranet", ".corp")


class _Refused(Exception):
    """A URL (or a redirect hop) the download guard will not fetch; the text is model-facing."""


def _ip_refusal(addr) -> str:
    """Why ``addr`` (an ip string) is not a public internet address, or ''."""
    import ipaddress

    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return ""
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    if str(ip) == "169.254.169.254" or str(ip) == "fd00:ec2::254":
        return "a cloud metadata address"
    if ip.is_loopback:
        return "a loopback address"
    if ip.is_link_local:
        return "a link-local address"
    if ip.is_private:
        return "a private-network address"
    if ip.is_multicast or ip.is_unspecified or ip.is_reserved or not ip.is_global:
        return "a non-public address"
    return ""


def _resolve_public(url: str) -> tuple[str, list[str]]:
    """``(refusal, addresses)``: '' + the validated public addresses ``url``'s host resolves to (in resolver order), or a
    model-facing refusal + ``[]``. The addresses are what ``pinned_dns`` connects to, so the name is resolved ONCE: a
    DNS-rebinding host cannot pass validation with a public answer and then connect to an internal one."""
    import socket

    try:
        parts = urlsplit(url or "")
        host = (parts.hostname or "").strip().lower().rstrip(".")
    except ValueError:
        return f"download refused: malformed URL {url!r}", []
    if parts.scheme.lower() not in ("http", "https") or not host:
        return f"download refused: only http(s) URLs with a host are supported, got {url!r}", []
    refused = "download refused: '{}' is {} (internal addresses are never fetched); use a public vendor URL or ask the user to attach the file"
    if host == "localhost" or host.endswith(_PRIVATE_SUFFIXES) or "." not in host and ":" not in host and not host.isdigit():
        return refused.format(host, "an internal host name"), []
    literal = host
    if ":" not in host:
        try:  # decimal / hex / octal / short forms such as 2130706433 or 0x7f.1
            literal = socket.inet_ntoa(socket.inet_aton(host))
        except (OSError, ValueError):
            literal = host
    why = _ip_refusal(literal)
    if why:
        return refused.format(host, why), []
    if literal != host or _is_ip_literal(host):
        return "", [literal]        # an IP literal needs no resolution (and nothing to pin)
    try:
        infos = socket.getaddrinfo(host, None)
    except (OSError, UnicodeError):
        return "", []
    addresses: list[str] = []
    for info in infos:
        address = str(info[4][0])
        why = _ip_refusal(address)
        if why:
            return refused.format(host, f"{why} (it resolves to {address})"), []
        if address not in addresses:
            addresses.append(address)
    return "", addresses


def _is_ip_literal(host: str) -> bool:
    import ipaddress

    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        return True
    except ValueError:
        return False


def check_public_url(url: str) -> str:
    """'' when ``url`` may be fetched, else a model-facing refusal.

    http(s) only; refuses loopback, link-local, private (RFC1918 / ULA / CGNAT), metadata, ``.local`` style names and any
    host name that RESOLVES to such an address (checked again for every redirect hop). A name that does not resolve is let
    through (the real request then fails by itself). Applies before any allowlist decision.
    """
    why, addresses = _resolve_public(url)
    _VALIDATED.last = None if why else (_host(url), list(addresses))
    return why


_VALIDATED = threading.local()      # the addresses ``check_public_url`` just validated (read once by ``_take_validated``)


def take_validated(host: str) -> list[str]:
    """The addresses the LAST ``check_public_url`` call on this thread validated for ``host`` ([] when it was for another
    host, was refused, or the check was replaced - e.g. patched in a test)."""
    last = getattr(_VALIDATED, "last", None)
    _VALIDATED.last = None
    return list(last[1]) if last and last[0] == host else []


_take_validated = take_validated    # the private name the module's own code used first


# --- DNS pinning --------------------------------------------------------------------------------------------------
#
# ``check_public_url`` validates the addresses a name resolves to, but ``requests`` would resolve the name AGAIN when it
# connects; a DNS-rebinding host can answer the first lookup with a public address and the second with an internal one.
# ``pinned_dns`` makes the connection use the addresses that were validated: a thread-scoped override of
# ``urllib3.util.connection.create_connection`` (installed once, inert for every thread / host without a pin). TLS still
# verifies the certificate against the ORIGINAL host name (urllib3 passes it as ``server_hostname``).

_PIN_LOCK = threading.Lock()
_PIN_INSTALLED = False
_PIN_TLS = threading.local()


def _install_pin_hook() -> bool:
    """Wrap ``urllib3.util.connection.create_connection`` once; False when it is already wrapped or urllib3 is absent."""
    global _PIN_INSTALLED
    with _PIN_LOCK:
        if _PIN_INSTALLED:
            return False
        try:
            from urllib3.util import connection as _connection
        except Exception:  # noqa: BLE001 - no urllib3: nothing resolves behind our back
            return False
        original = _connection.create_connection

        def pinned_create_connection(address, *args, **kwargs):
            pins = getattr(_PIN_TLS, "pins", None)
            host = str(address[0]).strip().lower().rstrip(".") if address else ""
            if pins and host in pins:
                last: Exception | None = None
                for ip in pins[host]:
                    try:
                        return original((ip, address[1]), *args, **kwargs)
                    except OSError as exc:
                        last = exc
                raise last if last is not None else OSError(f"no validated address for {host}")
            return original(address, *args, **kwargs)

        pinned_create_connection.__wrapped__ = original   # type: ignore[attr-defined]
        _connection.create_connection = pinned_create_connection
        _PIN_INSTALLED = True
        return True


@contextlib.contextmanager
def pinned_dns(host: str, addresses):
    """Connect ``host`` only to ``addresses`` (the validated, public ones) for requests made on THIS thread inside the block.
    No-op for an empty address list or an IP literal. Re-entrant (the previous pins are restored)."""
    host = (host or "").strip().lower().rstrip(".")
    if not host or not addresses or _is_ip_literal(host) or os.environ.get("OSCAR_DNS_PIN", "1").strip() == "0":
        yield
        return
    _install_pin_hook()
    previous = getattr(_PIN_TLS, "pins", None)
    pins = dict(previous or {})
    pins[host] = list(addresses)
    _PIN_TLS.pins = pins
    try:
        yield
    finally:
        _PIN_TLS.pins = previous


def _close(resp) -> None:
    try:
        resp.close()
    except Exception:  # noqa: BLE001 - a response without close() (or one that fails to) is not worth failing for
        pass


def _fetch_checked(requests, url, *, timeout, headers, allowed, should_stop=None):
    """``requests.get`` that follows at most ``MAX_REDIRECTS`` redirects by hand, validating every hop (SSRF + allowlist).

    Returns the final streaming response; raises ``_Refused`` (model-facing text) when a hop is not allowed.
    """
    hops = 0
    while True:
        why = check_public_url(url)
        addresses = [] if why else _take_validated(_host(url))   # resolved ONCE: connect to exactly these (pinned_dns)
        if why:
            raise _Refused(why if hops == 0 else why.replace("download refused:", "download refused: a redirect led to a URL and"))
        if allowed is not None and _host(url) not in allowed:
            if hops == 0:
                raise _Refused(f"download refused: host '{_host(url)}' is not on the allowlist ({', '.join(sorted(allowed))}); "
                               "use an official vendor URL or ask the user to attach the file")
            raise _Refused(f"download refused: the URL redirected to '{_host(url)}', which is not on the allowlist "
                           f"({', '.join(sorted(allowed))}); ask the user to attach the file")
        with pinned_dns(_host(url), addresses):     # connect to the addresses just validated, never resolve again
            resp = requests.get(url, stream=True, timeout=timeout, headers=dict(headers), allow_redirects=False)
        code = getattr(resp, "status_code", None)
        location = _header(getattr(resp, "headers", None), "Location")
        if code in _REDIRECT_CODES and location:
            _close(resp)
            hops += 1
            if hops > MAX_REDIRECTS:
                raise _Refused(f"download refused: more than {MAX_REDIRECTS} redirects")
            url = urljoin(url, str(location))
            continue
        reported = getattr(resp, "url", None)
        if isinstance(reported, str) and reported and reported != url:
            try:
                why = check_public_url(reported) or (
                    "" if allowed is None or _host(reported) in allowed else "allowlist")
            except Exception:  # noqa: BLE001
                why = ""
            if why == "allowlist":
                _close(resp)
                raise _Refused(f"download refused: the URL redirected to '{_host(reported)}', which is not on the allowlist "
                               f"({', '.join(sorted(allowed))}); ask the user to attach the file")
            if why:
                _close(resp)
                raise _Refused(why)
        return resp


def _http_failure(url: str, exc) -> str:
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if code in (401, 403):
        return (
            f"HTTP {code} — the site blocks automated downloads. Do "
            f"NOT retry this or other URLs from it. Ask the user: "
            f"\"I could not download {url} automatically (HTTP {code}). "
            f"Please open the page in your browser, download the file, "
            f"and attach it to this chat — I will continue from the "
            f"attachment.\" Meanwhile, continue with any work that "
            f"does not need the file.")
    return f"download failed: {type(exc).__name__}: {exc}"


def download_verified(url: str, dest_dir: str, filename: str = "", max_bytes: int = MAX_DOWNLOAD_BYTES,
                      timeout: float = 60.0, *, expect=None, allowed_hosts=None, should_stop=None) -> DownloadResult:
    """Stream ``url`` into ``dest_dir`` safely and say what arrived. Off the GUI thread; never raises.

    * the URL is rewritten from a GitHub *blob* page to the raw file (``github_raw_url``) and must be http(s);
    * the body goes to ``<name>.tmp-<pid>-<rand>`` in ``dest_dir`` (sha256 computed while streaming) and is renamed with
      ``os.replace`` only after every check passed, so no reader ever sees a partial file and a failure leaves nothing behind
      (zero-byte, truncated, over-cap, HTML, git-LFS pointer, wrong type, I/O error, ``should_stop``);
    * the file name is ``filename`` (rejected unless it is ONE plain name: no separator of either style, ``..``, drive letter,
      control character, Windows device name, and no link under it that leads out of ``dest_dir``), else the
      ``Content-Disposition`` name, else the last URL path segment (query and fragment stripped, always sanitised, never rejected); a derived name gets the extension its content calls for (a
      ``pi4`` that is a STEP file becomes ``pi4.step``). A name already taken by DIFFERENT bytes is never overwritten: the file
      is saved as ``<stem>-<8 hex of sha256><ext>``; identical bytes reuse the existing file;
    * the content is sniffed (``refparts.identity.sniff_file``): an HTML page, a git-LFS pointer and an empty body are always
      errors (with a "use the raw URL" hint); ``expect`` (a type name or an iterable of them, e.g. ``("step", "zip")``) makes
      every other type an error too, while without it any other content is accepted and reported in ``detected_type``;
    * ``allowed_hosts`` (host names, exact match) restricts both the URL and the final URL after redirects;
    * a size cap (``Content-Length`` checked up front, the stream aborted past ``max_bytes``) and a short read against
      ``Content-Length`` are errors; ``should_stop()`` is polled between chunks.

    The text of an HTTP 401/403 failure and of the size errors is unchanged from the legacy ``download``.
    """
    import requests

    result = DownloadResult(final_url=url or "")
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        result.error = f"only http(s) URLs are supported, got {url!r}"
        return result
    url = github_raw_url(url)
    result.final_url = url
    explicit = bool(filename)
    if explicit and (_unsafe_filename(filename) or not _inside(os.path.join(dest_dir, filename), dest_dir)):
        # a name that is not one plain component, or a link planted under that name that leads out of the folder
        result.error = f"unsafe filename {filename!r}"
        return result
    allowed = {str(h).lower() for h in allowed_hosts} if allowed_hosts else None
    why = check_public_url(url)
    if why:
        result.error = why
        return result
    if allowed is not None and _host(url) not in allowed:
        result.error = (f"download refused: host '{_host(url)}' is not on the allowlist ({', '.join(sorted(allowed))}); "
                        "use an official vendor URL or ask the user to attach the file")
        return result
    expected_types = ((expect,) if isinstance(expect, str) else tuple(expect)) if expect else ()
    expected_types = tuple(str(t).lower() for t in expected_types)

    tmp = ""
    try:
        os.makedirs(dest_dir, exist_ok=True)
        resp0 = _fetch_checked(requests, url, timeout=(15, timeout), headers=_BROWSER_HEADERS, allowed=allowed)
        with resp0 as resp:
            resp.raise_for_status()
            reported = getattr(resp, "url", None)
            result.final_url = reported if isinstance(reported, str) and reported else url
            headers = getattr(resp, "headers", None)
            declared = None
            length = _header(headers, "Content-Length")
            if length not in (None, ""):
                try:
                    declared = int(length)
                except (TypeError, ValueError):
                    declared = None
            if declared is not None and declared > max_bytes:
                result.error = f"download too large: {declared / 1e6:.1f} MB > {max_bytes / 1e6:.0f} MB cap"
                return result
            encoded = bool(_header(headers, "Content-Encoding"))
            content_type = str(_header(headers, "Content-Type") or "").split(";")[0].strip().lower()
            name = filename if explicit else _sanitize_filename(
                filename_from_content_disposition(str(_header(headers, "Content-Disposition") or ""))
                or _filename_from_url(result.final_url))
            tmp = os.path.join(dest_dir, f"{name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
            digest = hashlib.sha256()
            written = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if should_stop is not None and should_stop():
                        result.error = "download stopped"
                        return result
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        result.error = f"download exceeded {max_bytes / 1e6:.0f} MB cap — aborted"
                        return result
                    digest.update(chunk)
                    fh.write(chunk)
                fh.flush()
                os.fsync(fh.fileno())
        result.bytes, result.sha256 = written, digest.hexdigest()
        if declared is not None and not encoded and written != declared:
            result.error = (f"download incomplete: received {written} of {declared} bytes; the connection was cut, "
                            "retry once or ask the user to attach the file")
            return result
        detected = _identity.sniff_file(tmp)
        if detected == _identity.UNKNOWN and content_type == "text/html":
            detected = _identity.HTML
        result.detected_type = detected
        hint = _identity.rejection_hint(detected)
        if hint:
            result.error = f"download failed: {hint}"
            return result
        if expected_types and detected not in expected_types:
            wanted = " or ".join(_identity.describe_type(t) for t in expected_types)
            result.error = (f"download failed: expected {wanted} but received {_identity.describe_type(detected)} "
                            f"({written} bytes); use the direct file URL or ask the user to attach the file")
            return result
        final_name = name if explicit else _fix_extension(name, detected)
        if _unsafe_filename(final_name) or not _inside(os.path.join(dest_dir, final_name), dest_dir):
            result.error = f"unsafe filename {final_name!r}"
            return result
        target, existed = _destination(dest_dir, final_name, result.sha256)
        if not existed:
            _fsretry.replace(tmp, target)
            tmp = ""
        result.path = target
        return result
    except _Refused as exc:
        result.error = str(exc)
        return result
    except requests.RequestException as exc:
        result.error = _http_failure(url, exc)
        return result
    except OSError as exc:
        result.error = f"download failed: cannot write into {dest_dir!r}: {exc}"
        return result
    except Exception as exc:  # noqa: BLE001 - a download must never take the agent thread down; report it as a failure
        result.error = f"download failed: {type(exc).__name__}: {exc}"
        return result
    finally:
        if tmp and os.path.lexists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        if result.error:
            result.path = ""


def download(url: str, dest_dir: str, filename: str = "",
             max_bytes: int = MAX_DOWNLOAD_BYTES,
             timeout: float = 60.0) -> tuple[str, str]:
    """Streamed download off the GUI thread. Returns (path, error).

    Legacy tuple form of ``download_verified`` (same checks, same file naming, same atomic write); use that function when
    the hash, the detected type or the final URL matter.
    """
    result = download_verified(url, dest_dir, filename, max_bytes=max_bytes, timeout=timeout)
    return result.path, result.error


# --- zip extraction ----------------------------------------------------------

MAX_ZIP_MEMBERS = 2000


@dataclass
class ExtractResult:
    """Outcome of ``extract_zip``: ``files`` (absolute paths written), ``cad_files`` (the CAD / mesh subset, sorted),
    ``skipped`` (``"<member>: <reason>"`` lines), ``total_bytes`` written and ``error`` (``""`` on success; nothing is left
    behind on failure)."""

    dest_dir: str = ""
    files: list = field(default_factory=list)
    cad_files: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    total_bytes: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        """True when the archive was extracted."""
        return not self.error


def _safe_member_parts(name: str) -> list[str] | None:
    """Path parts of an archive member name, or None when it could escape the destination (absolute, drive letter, ``..``, NUL, control)."""
    if not name or any(ord(ch) < 32 for ch in name):
        return None
    norm = name.replace("\\", "/")
    if norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
        return None
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None
    return parts


def extract_zip(zip_path: str, dest_dir: str, *, max_total_bytes: int = MAX_DOWNLOAD_BYTES,
                max_members: int = MAX_ZIP_MEMBERS) -> ExtractResult:
    """Extract a ZIP into ``dest_dir`` without letting any member write outside it. Never raises.

    The whole archive is refused (nothing written) when any member name is absolute, has a drive letter, contains ``..`` or a
    control character (zip-slip), when it has more than ``max_members`` members, or when its declared uncompressed size exceeds
    ``max_total_bytes``; the cap is enforced again on the bytes actually written (a header can lie) and a bad CRC aborts, in
    both cases rolling back every file this call created. Symbolic links and encrypted members are skipped and reported in
    ``skipped``; ``__MACOSX/`` and ``.DS_Store`` entries are ignored; permissions are not restored. Files are written through a
    temp name and renamed (use a fresh directory: a file of the same name is replaced). ``cad_files`` lists the members whose
    extension is a CAD / mesh format.
    """
    result = ExtractResult(dest_dir=os.path.abspath(dest_dir))
    created: list[str] = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            infos = zf.infolist()
            if len(infos) > max_members:
                result.error = f"zip refused: {len(infos)} members exceed the limit of {max_members}"
                return result
            declared = sum(i.file_size for i in infos)
            if declared > max_total_bytes:
                result.error = (f"zip refused: it would expand to {declared / 1e6:.0f} MB, over the {max_total_bytes / 1e6:.0f} MB cap "
                                "(possible zip bomb)")
                return result
            plan = []
            for info in infos:
                parts = _safe_member_parts(info.filename)
                if parts is None:
                    result.error = f"zip refused: unsafe member path {info.filename!r} (it would write outside the destination)"
                    return result
                plan.append((info, parts))
            os.makedirs(result.dest_dir, exist_ok=True)
            root = os.path.realpath(result.dest_dir)
            for info, parts in plan:
                if info.is_dir() or not parts or parts[0] == "__MACOSX" or parts[-1] == ".DS_Store":
                    continue
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    result.skipped.append(f"{info.filename}: symbolic link not extracted")
                    continue
                if info.flag_bits & 0x1:
                    result.skipped.append(f"{info.filename}: encrypted member not extracted")
                    continue
                target = os.path.realpath(os.path.join(root, *parts))
                if os.path.commonpath([root, target]) != root:
                    result.error = f"zip refused: member {info.filename!r} resolves outside the destination"
                    break
                os.makedirs(os.path.dirname(target), exist_ok=True)
                tmp = f"{target}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
                try:
                    with zf.open(info) as src, open(tmp, "wb") as out:
                        for chunk in iter(lambda: src.read(1 << 16), b""):
                            result.total_bytes += len(chunk)
                            if result.total_bytes > max_total_bytes:
                                raise OverflowError
                            out.write(chunk)
                    _fsretry.replace(tmp, target)
                except OverflowError:
                    result.error = f"zip refused: extracted data exceeds the {max_total_bytes / 1e6:.0f} MB cap (possible zip bomb)"
                    break
                finally:
                    if os.path.lexists(tmp):
                        os.remove(tmp)
                created.append(target)
                result.files.append(os.path.join(result.dest_dir, *parts))
    except zipfile.BadZipFile as exc:
        result.error = f"zip refused: not a valid ZIP archive ({exc})"
    except (OSError, RuntimeError, EOFError, ValueError, zlib_error) as exc:
        result.error = f"zip extraction failed: {type(exc).__name__}: {exc}"
    if result.error:
        for path in created:
            try:
                os.remove(path)
            except OSError:
                pass
        result.files, result.cad_files, result.total_bytes = [], [], 0
        return result
    result.cad_files = sorted(f for f in result.files if os.path.splitext(f)[1].lower() in _identity.CAD_EXTENSIONS)
    return result
