"""Crash-isolated script execution.

Geometry kernels can fault on pathological operations (helical sweeps,
many-tool booleans). Inside the interactive process such a fault would take
the whole application down with every unsaved document; here the script runs
in a throwaway headless engine process instead — a native crash kills only
that helper, and the caller gets the exit status and printed output back.

Scripts must be self-contained: import the API modules themselves, open
documents from disk, save with doc.save()/saveAs(). Nothing from the
interactive session crosses over.

How the helper is launched (the facts the design rests on, all verified on the bundled
``freecadcmd``): ``-c`` takes at most ~400 characters, so the script itself goes to a file and a tiny
constant bootstrap file runs it; a script that raises or calls ``sys.exit`` loses everything it printed
unless the streams are line-buffered, so the bootstrap switches them to line buffering first (the output
printed before a native crash survives); a Finder-launched app has no locale, which leaves the helper's streams
ASCII (PYTHONUTF8 is ignored by the engine) so printing "O10" raised UnicodeEncodeError, so the bootstrap also
makes both streams UTF-8 (``errors="replace"``); the script is compiled under the name ``<execute_code_headless>``
so tracebacks and error messages name the failing script line; a native crash shows up as a NEGATIVE return
code (``-11`` = SIGSEGV).

Network. The helper runs only the script, so the bootstrap guards the WHOLE process (every thread) with an audit
hook unless the user granted the network for this turn (``OSCAR_HEADLESS_NET=1``, set from
``netguard.network_granted()``) or the audit kill switch ``OSCAR_NET_AUDIT=0`` is set: connections and sends to
other hosts, host name lookups and starting any other program raise ``PermissionError("network access blocked:
...")``.
"""

import json
import math
import os
import re
import signal
import subprocess
from pathlib import Path
from typing import Any
from .. import fsretry as _fsretry

# FreeCAD's console binary, as the official builds name it (FreeCADCmd.exe on Windows).
_CONSOLE_NAMES = ("freecadcmd", "FreeCADCmd", "freecadcmd.exe", "FreeCADCmd.exe")

SCRIPT_NAME = "<execute_code_headless>"   # the compile() filename of the helper's script
_ENV_SCRIPT = "OSCAR_HEADLESS_SCRIPT"     # path of the script the bootstrap runs
_ENV_NET = "OSCAR_HEADLESS_NET"           # "1": the network was granted for this turn; anything else: the helper blocks it
_ENV_AUDIT = "OSCAR_NET_AUDIT"            # netguard's kill switch ("0"), honoured by the helper too
_ERR_MARK = "@@OSCAR-HEADLESS-ERROR@@ "   # prefix of the one JSON line the bootstrap writes to stderr on an exception

# communicate() raises OverflowError for timeouts beyond ~2.1e6 s, and a model passing 99999999 means "no limit":
# clamp instead of failing the call.
_MAX_TIMEOUT = 86400.0

# Startup noise the caller does not need in the returned output. Every pattern is ANCHORED on purpose: the script's
# own prints ("FreeCAD objects created", "Volume ... (95.5%) of target", "LGPL part library loaded") come back untouched.
_NOISE = tuple(re.compile(pattern) for pattern in (
    r"^\s*FreeCAD \d",                                   # the version banner, recognised by its shape
    r"^\s*(\(C\)|\u00a9).*\b2001\b",                    # copyright banner
    r"^\s*FreeCAD is free and open-source",              # licence banner
    r"^\s*(Importing project files|Postprocessing)",
    # recompute progress: "(37 %)" fragments and "Recompute......", often glued together on one tab-separated line
    r"^\s*((\(\d+\s*%\)|Recompute\.*)\s*)+$",
))
# macOS prints this on every start when the 3Dconnexion driver is not installed: a stderr line the model must never see.
_NOISE_DRIVER = ("Failed to open library", "3Dconnexion")

# The bootstrap runs INSIDE the helper. It must stay standalone (no oscar import) and short. It is a %-template: a
# literal percent sign must be written twice. The network hook mirrors netguard's runtime guard for a process that
# runs nothing but the script: every thread is covered, and any other program it starts is refused.
_BOOTSTRAP = '''\
import json, os, sys, traceback
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
def _network_guard():
    # a closure: the script runs next to this code in __main__, and must not switch the guard off by rebinding names
    local = ("localhost", "127.0.0.1", "::1", "")
    spawn = frozenset(("subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn", "os.exec", "os.fork",
                       "os.forkpty", "_posixsubprocess.fork_exec", "_winapi.CreateProcess", "os.startfile",
                       "os.startfile/2"))
    prefixes = ("socket.", "subprocess.", "os.", "_posixsubprocess.", "_winapi.")

    def local_host(host):
        if host is None:
            return True
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        host = str(host).strip().lower().rstrip(".")
        return host in local or host.startswith("127.")

    def local_address(address):
        if isinstance(address, (str, bytes)):
            return True
        try:
            return address[0] is not None and local_host(address[0])
        except Exception:
            return False

    def guard(event, args):
        if not event.startswith(prefixes):
            return
        what = ""
        if event in ("socket.connect", "socket.sendto", "socket.sendmsg"):
            address = args[1] if len(args) > 1 else None
            if not (event == "socket.sendmsg" and address is None) and not local_address(address):
                what = "to open a network connection"
        elif event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
            if not local_host(args[0] if args else None):
                what = "to resolve a host name"
        elif event == "socket.getnameinfo":
            if not local_address(args[0] if args else None):
                what = "to look up the name of a network address"
        elif event in spawn:
            what = "to start another program (it would run outside this guard)"
        if what:
            raise PermissionError("network access blocked: the headless script tried " + what + " but the user has "
                                  "not granted network access for this turn. Do not retry. Use the download_file tool "
                                  "(it asks the user), or ask the user to attach the file.")
    return guard
if os.environ.get(%(net)r) != "1" and os.environ.get(%(audit)r, "1").strip() != "0":
    sys.addaudithook(_network_guard())
del _network_guard
_path = os.environ[%(env)r]
_name = %(name)r
try:
    with open(_path, encoding="utf-8") as _f:
        _code = compile(_f.read(), _name, "exec")
    exec(_code, {"__name__": "__main__", "__file__": _path})
except SystemExit:
    raise
except BaseException as _e:
    _frames = [fr for fr in traceback.extract_tb(_e.__traceback__) if fr.filename == _name]
    try:
        _line = getattr(_e, "lineno", None) if isinstance(_e, SyntaxError) and _e.filename == _name else None
        _info = {"type": type(_e).__name__, "mro": [c.__name__ for c in type(_e).__mro__], "message": str(_e),
                 "lines": [fr.lineno for fr in _frames], "names": [fr.name for fr in _frames],
                 "syntax_line": _line, "name": getattr(_e, "name", None)}
        sys.stdout.flush()
        sys.stderr.write(%(mark)r + json.dumps(_info) + "\\n")
        sys.stderr.flush()
    except Exception:
        pass
    sys.exit(1)
''' % {"env": _ENV_SCRIPT, "name": SCRIPT_NAME, "mark": _ERR_MARK, "net": _ENV_NET, "audit": _ENV_AUDIT}

_BOOT_FILE = "_oscar_headless_boot.py"


def find_console() -> list[str] | None:
    """Locate the headless engine binary: bundle first, then PATH. ``None`` when there is none."""
    try:
        import FreeCAD

        # macOS keeps the binary under <home>/bin while getResourceDir() is
        # the share/ directory, so both are searched.
        bases = (FreeCAD.getResourceDir(), FreeCAD.getHomePath())
    except Exception:
        bases = ()
    for base in bases:
        for name in _CONSOLE_NAMES:
            candidate = Path(base) / "bin" / name
            if candidate.is_file():
                return [str(candidate)]
    import shutil

    for name in _CONSOLE_NAMES:
        found = shutil.which(name)
        if found:
            return [found]
    return None


def _script_dir() -> str:
    """The directory scripts are written to: ``paths.cache_dir("headless")`` (honours OSCAR_CACHE_DIR at
    call time), with the documented default when the paths module is unavailable."""
    try:
        from ..paths import cache_dir

        return cache_dir("headless")
    except Exception:
        base = os.environ.get("OSCAR_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache",
                                                                         "oscar")
        path = os.path.join(base, "headless")
        os.makedirs(path, exist_ok=True)
        return path


def _write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.tmp-{os.getpid()}-{os.urandom(3).hex()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
    _fsretry.replace(tmp, path)


def _boot_path(directory: str) -> str:
    """The bootstrap file (rewritten only when its content changed, so concurrent runs never see a partial file)."""
    path = os.path.join(directory, _BOOT_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() == _BOOTSTRAP:
                return path
    except OSError:
        pass
    _write_atomic(path, _BOOTSTRAP)
    return path


def _network_granted() -> bool:
    """``netguard.network_granted()``, imported lazily (this module must import without the rest of Oscar); False when
    it cannot be asked."""
    try:
        from .. import netguard

        return bool(netguard.network_granted())
    except Exception:  # noqa: BLE001 - no answer means no grant
        return False


def run(code: str, timeout: float = 600.0, *, allow_network: bool | None = None) -> dict[str, Any]:
    """Run ``code`` in a fresh headless process and report the outcome. A failed script, a missing engine or an
    unwritable script directory are results, never exceptions; only an unexpected internal error or an interrupt while
    waiting propagates, and then the engine process tree has been killed first.

    ``allow_network`` (default: ``netguard.network_granted()`` at call time) lets the helper reach the network and
    start other programs; otherwise its whole-process guard refuses them with "network access blocked".

    Returns ``{"success": bool, "returncode": int, "crashed": bool, "output": str}`` (stdout + stderr, startup
    noise removed; whatever the script printed before a failure or a native crash is kept) plus, when it failed,
    ``"error"``: a model-readable line derived from the exit status (``killed by signal 11 (SIGSEGV)`` for a crash;
    for an uncaught exception the failing script line, the exception and a hint for known traps, in the same shape
    as ``execute_code`` errors). A timeout returns ``{"success": False, "timed_out": True, "crashed": False,
    "error": ..., "output": ...}``; a missing engine or a script that cannot be written returns
    ``{"success": False, "error": ...}``."""
    if not math.isfinite(timeout) or timeout <= 0:
        return {"success": False, "error": "timeout must be a positive finite number"}
    timeout = min(timeout, _MAX_TIMEOUT)
    argv = find_console()
    if argv is None:
        return {"success": False, "error": "no headless engine binary found"}

    script = ""
    try:
        directory = _script_dir()
        script = os.path.join(directory, f"headless_{os.getpid()}_{os.urandom(4).hex()}.py")
        _write_atomic(script, code)
        boot = _boot_path(directory)
    except OSError as e:
        _remove(script)
        return {"success": False, "error": f"could not prepare the headless script: {type(e).__name__}: {e}"}
    argv = argv + ["-c", f"exec(open({boot!r}, encoding='utf-8').read())"]
    if allow_network is None:
        allow_network = _network_granted()
    env = dict(os.environ)
    env[_ENV_SCRIPT] = script
    env[_ENV_NET] = "1" if allow_network else "0"
    try:
        return _run_process(argv, timeout, env=env, code=code)
    except OSError as e:
        return {"success": False, "error": f"could not start the headless engine: {type(e).__name__}: {e}"}
    finally:
        _remove(script)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the engine AND whatever it spawned (a script's own Popen children would otherwise outlive the timeout,
    and on Windows hold the pipes open)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=10)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001 - best effort, the plain kill below is the fallback
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _run_process(argv: list[str], timeout: float, env: dict | None = None, code: str = "") -> dict[str, Any]:
    """Run the engine process and build the result dict ``run`` documents.

    The engine gets its own process group (POSIX) so a timeout kills the whole tree. ``subprocess.run`` killed the
    child on ANY failure while waiting; a bare Popen does not, so any unexpected error or interrupt between the start
    and a finished ``communicate()`` kills the engine before it propagates (otherwise it would run on unseen, and
    could finish its job later over a retry's files)."""
    extra = {} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", env=env, **extra,
    )
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired as e:   # an escaped grandchild holds the pipe
                stdout, stderr = e.stdout, e.stderr
            return {
                "success": False,
                "error": f"timed out after {timeout:g}s (partial output preserved); the helper process was killed",
                "output": (_clean(stdout) + "\n" + _clean(stderr)).strip(),
                "timed_out": True,
                "crashed": False,
            }
    except BaseException:
        _kill_tree(proc)
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        for pipe in (proc.stdout, proc.stderr):
            try:
                pipe.close()
            except Exception:  # noqa: BLE001
                pass
        raise
    rc = proc.returncode
    stderr, info = _split_error(stderr)
    out: dict[str, Any] = {
        "success": rc == 0,
        "returncode": rc,
        "crashed": _is_crash(rc),
        "output": (_clean(stdout) + "\n" + _clean(stderr)).strip(),
    }
    if rc != 0:
        out["error"] = _failure_text(code, rc, info, bool(out["output"]))
    return out


def _remove(path: str) -> None:
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


def _is_crash(returncode: int) -> bool:
    """A native crash: a negative return code (killed by a signal, POSIX) or an NTSTATUS error code (Windows,
    ``0xC0000005`` = access violation)."""
    return returncode < 0 or returncode >= 0xC0000000


def _signal_text(returncode: int) -> str:
    if returncode < 0:
        number = -returncode
        try:
            return f"signal {number} ({signal.Signals(number).name})"
        except ValueError:
            return f"signal {number}"
    return f"status 0x{returncode:08X}"


def _failure_text(code: str, returncode: int, info: dict | None, had_output: bool) -> str:
    """The ``error`` text of a non-zero exit, derived from the return code and, for an uncaught exception, from the
    structured report the bootstrap wrote (formatted like an ``execute_code`` error)."""
    if _is_crash(returncode):
        text = (f"the headless helper CRASHED (killed by {_signal_text(returncode)}): the geometry kernel faulted on "
                "this operation. Your session is unaffected. Simplify the operation (smaller counts, no helical "
                "sweeps, fewer boolean tools), validate each shape with isValid() before the next step, or split "
                "the work into smaller scripts")
        if not had_output:
            text += "; nothing was printed before the crash"
        return text
    if info is not None:
        return _format_exception(code, info)
    return (f"the headless script exited with status {returncode} (no exception was reported; a script that calls "
            "sys.exit() with a non-zero status fails the step)")


def _format_exception(code: str, info: dict) -> str:
    """The shared ``execute_code`` error formatting (line, source line, call chain, hint) for an exception reported by
    the helper; falls back to a plain ``Type: message`` when the formatter cannot be imported."""
    lines = [int(n) for n in (info.get("lines") or []) if isinstance(n, int)]
    if info.get("syntax_line") and not lines:
        lines = [int(info["syntax_line"])]
    try:
        from .scripting import format_script_error

        return format_script_error(code, str(info.get("type", "Error")), str(info.get("message", "")), lines,
                                   mro=[str(n) for n in (info.get("mro") or [])], missing=info.get("name"),
                                   names=[str(n) for n in (info.get("names") or [])])
    except Exception:  # noqa: BLE001 - scripting needs the GUI modules; a plain message is still useful
        text = f"{info.get('type', 'Error')}: {info.get('message', '')}"
        src = code.splitlines()
        if lines and 0 < lines[-1] <= len(src):
            text += f"\n  at script line {lines[-1]}: {src[lines[-1] - 1].strip()}"
        return text


def _split_error(stderr: str | bytes | None) -> tuple[str, dict | None]:
    """Remove the bootstrap's JSON error line from ``stderr``; returns ``(remaining stderr, report | None)``."""
    if not stderr:
        return "", None
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    keep: list[str] = []
    info: dict | None = None
    for line in stderr.split("\n"):
        if line.startswith(_ERR_MARK):
            try:
                data = json.loads(line[len(_ERR_MARK):])
                if isinstance(data, dict):
                    info = data
            except ValueError:
                pass
            continue
        keep.append(line)
    return "\n".join(keep), info


def _is_noise(line: str) -> bool:
    if any(pattern.match(line) for pattern in _NOISE):
        return True
    return all(marker in line for marker in _NOISE_DRIVER)


def _clean(output: str | bytes | None) -> str:
    """``output`` without blank lines and engine startup noise (a bytes value is decoded first)."""
    if not output:
        return ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return "\n".join(line for line in output.splitlines() if line.strip() and not _is_noise(line))
