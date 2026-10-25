"""Run work on the GUI thread.

Documents, the 3D scenegraph and every view API belong to the main (GUI)
thread. Oscar's agent, the bridge server and background jobs live on other
threads, so everything they ask for crosses this dispatcher:

    result, failed = dispatch(callable, timeout=60, op="create_object")

The callable is queued and a queued-connected Qt signal wakes the GUI thread,
which drains the queue. Results come back through a per-call slot, so a
timeout in one call can never poison another.

Two shortcuts run work directly, without the queue:

- console sessions (``freecadcmd`` / no QApplication): no GUI thread exists,
  so the caller thread is the safest thread there is;
- a caller that already IS the GUI thread: queueing would deadlock it.

Errors never propagate as exceptions across the boundary — the dispatcher
returns them, mirroring the (result | error-dict) convention the engine and
bridge use.

Cancellation. The GUI thread cannot cancel a task that is already running, but
a task that has NOT started when its caller gives up (timeout, or the optional
``should_stop`` poll turning true) is marked ABANDONED and never starts, so a
timed-out hook or script that was queued behind a long task does not run later
on a document the caller has already moved on from. A task that HAS started when the timeout expires is given
``_RUNNING_GRACE`` seconds more to deliver its real result (a task about to succeed is not reported as a failure,
which would make the model redo it); only then is the caller told it is STILL RUNNING.

SystemExit and KeyboardInterrupt are errors too, never exceptions: raised inside a Qt slot they would terminate the
whole application, unsaved documents and all, so a script's ``sys.exit()`` comes back as an error result.

One task at a time. A nested event loop inside a running task (``Gui.updateGui()``, ``processEvents()``, FreeCAD's
recompute progress bar, a modal dialog) delivers the wake signal again. That nested drain returns at once and leaves
the queue to the outer loop (or to the re-wake after an inline task), so a bridge request or a background job's
``commit`` never runs inside a script's open transaction, where the script's rollback would delete its work. A task
dispatched with ``reentrant=True`` (a permission dialog, which holds no transaction) does not block the queue: other
callers keep running while it waits for the user.
"""

import queue
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

import FreeCAD
from PySide import QtCore, QtWidgets

_POLL_S = 0.25   # slice of the wait: how often ``should_stop`` is consulted


# How long a task that has already STARTED is given beyond its timeout to deliver its real result
# before the caller is told it is still running (work that is about to succeed must not be reported as
# a failure: the model would retry it and the work would happen twice).
_RUNNING_GRACE = 5.0


class _Request:
    """One queued task. ``started`` / ``abandoned`` are decided under ``lock`` so a task either
    runs to completion or never starts: never both. ``reentrant`` tasks let the queue drain while they run
    (see the module docstring)."""

    __slots__ = ("fn", "done", "result", "failed", "started", "abandoned", "lock", "reentrant")

    def __init__(self, fn, reentrant: bool = False):
        self.fn = fn
        self.done = threading.Event()
        self.result = None
        self.failed = False
        self.started = False
        self.abandoned = False
        self.lock = threading.Lock()
        self.reentrant = bool(reentrant)


# How many non-reentrant tasks are running on THIS thread (the GUI thread in the application). Thread-local because
# re-entrancy happens on one thread, and the tests drain the queue from several stand-in GUI threads at once.
_TLS = threading.local()


def _depth() -> int:
    return getattr(_TLS, "depth", 0)


def _enter(reentrant: bool) -> None:
    if not reentrant:
        _TLS.depth = _depth() + 1


def _leave(reentrant: bool) -> None:
    if not reentrant:
        _TLS.depth = max(0, _depth() - 1)


class _Waker(QtCore.QObject):
    """Cross-thread wake-up for the GUI thread's queue drain."""

    rouse = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.rouse.connect(self._drain, QtCore.Qt.QueuedConnection)

    def _drain(self):
        drain_queue()


_waker: _Waker | None = None
_pending: "queue.Queue[_Request]" = queue.Queue()


def init() -> None:
    """Create the wake bridge. MUST run on the GUI thread.

    A QObject carries its creating thread's affinity, so a waker built on a
    worker thread would deliver its own signal back to that worker — never
    to the GUI loop. The application's GUI entry points (workbench init,
    panel creation, bridge start) call this before any cross-thread work.
    """
    global _waker
    if _waker is None:
        _waker = _Waker()


def shutdown() -> None:
    """Drop the wake bridge (application teardown; ``init`` creates a new one)."""
    global _waker
    _waker = None


def drain_queue() -> None:
    """Execute every queued request that is still wanted. Runs on the GUI thread.

    A request whose caller already gave up (``abandoned``) is dropped without running. Called from a nested event
    loop while a non-reentrant task runs on this thread, it returns at once and leaves the queue to the outer loop."""
    if _depth():
        return
    while True:
        try:
            request = _pending.get_nowait()
        except queue.Empty:
            return
        with request.lock:
            if request.abandoned:
                request.done.set()
                continue
            request.started = True
        _enter(request.reentrant)
        try:
            request.result = request.fn()
        except BaseException as e:  # noqa: BLE001
            # BaseException: a SystemExit / KeyboardInterrupt escaping this Qt slot would take the whole
            # application (unsaved documents and all) down with it; a script's sys.exit() comes back as an
            # error result instead, and the loop keeps draining.
            FreeCAD.Console.PrintError(
                f"Oscar: GUI task failed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc(limit=4)}"
            )
            request.result = f"{type(e).__name__}: {e}"
            request.failed = True
        finally:
            _leave(request.reentrant)
            request.done.set()


def _inline(fn, reentrant: bool = False):
    _enter(reentrant)
    try:
        return fn(), False
    except BaseException as e:  # noqa: BLE001 - see drain_queue
        FreeCAD.Console.PrintError(
            f"Oscar: inline task failed: {type(e).__name__}: {e}\n"
            f"{traceback.format_exc(limit=4)}"
        )
        return f"{type(e).__name__}: {e}", True
    finally:
        _leave(reentrant)
        _rewake()


def _rewake() -> None:
    """After an inline task on the GUI thread: a request queued meanwhile was refused by the nested drain, so wake
    the event loop again to run it. Never raises (the inline result must reach its caller)."""
    if _depth() or _waker is None or _pending.empty():
        return
    try:
        _wake()
    except Exception:  # noqa: BLE001 - a waker whose Qt object is gone: the next dispatch wakes the loop anyway
        pass


def _run_inline() -> bool:
    """True when ``fn`` can run on the calling thread: a console session or the GUI thread itself."""
    app = QtWidgets.QApplication.instance()
    return app is None or QtCore.QThread.currentThread() == app.thread()


def _wake() -> None:
    """Ask the GUI thread to drain the queue."""
    _waker.rouse.emit()


def _stopped_result() -> dict:
    return {"success": False, "error": "stopped", "stopped": True}


def _should_stop(should_stop: Callable[[], bool] | None) -> bool:
    if should_stop is None:
        return False
    try:
        return bool(should_stop())
    except Exception:
        return False


def bound_should_stop() -> Callable[[], bool] | None:
    """The Stop signal of the tool context bound to the CALLING thread.

    ``None`` when no agent turn runs on this thread (a call from the GUI thread, a bridge handler or a test
    that binds no context never polls). ``tools.run_tool`` runs inside ``toolctx.use_context``, so every engine
    method a tool handler calls inherits the turn's Stop button through this function (the context var is
    per-thread: a task already handed to the GUI thread keeps the callable captured here)."""
    try:
        from ..toolctx import current_context

        return current_context().should_stop
    except Exception:  # noqa: BLE001 - the Stop plumbing must never break a call
        return None


def failure_text(result, default: str = "the GUI thread did not answer") -> str:
    """Model-readable text of a failed dispatch result: the ``error`` of a timeout / stopped dict, a string
    as it is (``"<Type>: <message>"`` of a task that raised), else ``default``."""
    if isinstance(result, dict):
        text = result.get("error")
        return str(text) if text else default
    return str(result) if result not in (None, "") else default


def _timeout_result(op: str, timeout: float, started: bool) -> dict:
    """The model-readable timeout result. The first sentence is the stable text callers match on
    (``did not complete within``); the second says whether the task is still running (it started and
    cannot be cancelled: do NOT re-run it) or was cancelled before it started (nothing ran)."""
    text = f"'{op}' did not complete within {timeout:g}s (the GUI thread stayed busy)"
    if started:
        text += (". It is STILL RUNNING on the GUI thread and cannot be cancelled: do not re-run it; wait, then "
                 "poll get_objects to see its effect, or split the work into smaller steps")
    else:
        text += ". It never started and has been cancelled, so nothing ran; retry with a smaller step"
    return {"success": False, "error": text, "timed_out": True, "still_running": started}


def _wait(request: _Request, timeout: float | None, op: str, should_stop):
    """Wait for ``request`` in 0.25 s slices; give up on timeout or when ``should_stop()`` turns true."""
    deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
    while True:
        if request.done.is_set():
            return request.result, request.failed
        if deadline is None:
            step = _POLL_S
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            step = min(_POLL_S, remaining)
        if request.done.wait(step):
            return request.result, request.failed
        if _should_stop(should_stop):
            with request.lock:
                if request.done.is_set():
                    return request.result, request.failed
                request.abandoned = True   # a task that has not started never will; one that has keeps running
            return _stopped_result(), True
    with request.lock:
        if request.done.is_set():
            return request.result, request.failed
        started = request.started
        if not started:
            request.abandoned = True
    if started:
        # It had started when the time ran out: give it a short grace to deliver its real outcome (Stop
        # still cuts the wait short) rather than report a failure for work that is about to succeed.
        grace_end = time.monotonic() + _RUNNING_GRACE
        while True:
            remaining = grace_end - time.monotonic()
            if remaining <= 0:
                break
            if request.done.wait(min(_POLL_S, remaining)):
                return request.result, request.failed
            if _should_stop(should_stop):
                return _stopped_result(), True
        if request.done.is_set():
            return request.result, request.failed
    return _timeout_result(op, timeout or 0.0, started), True


def dispatch(fn, timeout: float = 60.0, op: str = "gui task", *,
             should_stop: Callable[[], bool] | None = None, reentrant: bool = False):
    """Run ``fn`` on the GUI thread; return ``(result, failed)``.

    Inline (console sessions, or the caller already is the GUI thread): ``fn`` runs at once, unless
    ``should_stop()`` is already true, which returns ``({"success": False, "error": "stopped",
    "stopped": True}, True)`` without running it. Queued: the caller waits in 0.25 s slices; a timeout
    returns ``({"success": False, "error": "'<op>' did not complete within <t>s (the GUI thread stayed
    busy) ...", "timed_out": True, "still_running": bool}, True)`` (a task that had started gets ``_RUNNING_GRACE`` more
    seconds to deliver its real result first) and ``should_stop()`` turning true returns
    the ``stopped`` result above. In both cases a task that has not started is abandoned and never runs;
    a task already running keeps running (it cannot be cancelled) and ``still_running`` says so.
    ``fn`` exceptions come back as ``("<Type>: <message>", True)``.

    ``reentrant=True`` is for a task that only waits for the user (a permission dialog): it holds no
    transaction, so queued tasks may run inside its event loop. Every other task blocks the queue while it runs."""
    if _run_inline():
        if _should_stop(should_stop):
            return _stopped_result(), True
        return _inline(fn, reentrant)

    init()
    request = _Request(fn, reentrant)
    _pending.put(request)
    _wake()
    return _wait(request, timeout, op, should_stop)


def dispatch_stoppable(fn, timeout: float = 60.0, op: str = "gui task"):
    """``dispatch`` with the bound tool context's Stop signal (``bound_should_stop``): the CadEngine methods that
    can queue behind a long GUI task (reload, create / edit / delete, screenshots) use
    it, so Stop abandons work that has not started instead of leaving the worker thread blocked for up to the
    timeout (600 s for ``apply_feature_plan``). Same return value as ``dispatch``."""
    return dispatch(fn, timeout, op, should_stop=bound_should_stop())


def run_stoppable(fn: Callable[[], Any], should_stop: Callable[[], bool] | None = None) -> Any:
    """Run blocking NON-GUI work (a worker subprocess call) so that Stop can abandon it; return ``fn()``.

    ``fn`` runs on a daemon helper thread while the caller waits in 0.25 s slices. When ``should_stop`` (default:
    ``bound_should_stop()``) turns true the caller gets ``{"error": "stopped"}`` at once and the helper is left to finish
    on its own (its result is discarded; a worker has its own timeout), so a ten-minute operation never keeps the
    agent thread from honouring the Stop button. An exception inside ``fn`` is re-raised in the caller."""
    poll = should_stop if should_stop is not None else bound_should_stop()
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            box["error"] = e

    helper = threading.Thread(target=work, name="oscar-stoppable", daemon=True)
    helper.start()
    while True:
        helper.join(_POLL_S)
        if not helper.is_alive():
            break
        if _should_stop(poll):
            return {"error": "stopped"}
    if "error" in box:
        raise box["error"]
    return box.get("value")
