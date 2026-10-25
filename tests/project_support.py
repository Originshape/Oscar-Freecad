"""Shared helpers of the Project test modules (test_project*.py). Contains no test case on purpose.

* ``make_project`` / ``record`` build a Project in a temp folder and feed it tool results (no panel, no real snapshots).
* ``FakeGui`` replaces ``oscar.project_gui.load_dispatch`` with a dispatcher that runs tasks on its OWN thread, like
  the real GUI thread, so the lock-discipline tests can observe what the GUI thread can do while a hook is waiting.
* ``FakeDoc`` imitates the one FreeCAD behaviour the save code depends on (``saveCopy`` appends ``.FCStd``).
"""

import os
import queue
import shutil
import sys
import tempfile
import threading
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

from support.fakes import tiny_png_snapshot  # noqa: E402,F401  (re-export for the test modules)

from oscar import project_gui as gui  # noqa: E402
from oscar.project import Project  # noqa: E402

CHAT_ID = "20261001-000000-abc123"
TURN = "T20261001-000001-0001"


def temp_root(prefix: str = "proj_") -> str:
    """A fresh empty directory in the test sandbox."""
    return isolation.fresh_dir(prefix)


def make_project(root: str | None = None, title: str = "Pi 4 housing", chat_id: str = CHAT_ID, *,
                 snapshot_fn=tiny_png_snapshot, create: bool = False, **kw) -> Project:
    """A Project under ``root`` (default: a new temp folder) that is lazy unless ``create=True``."""
    return Project.for_chat(chat_id, title, projects_root=root or temp_root(), snapshot_fn=snapshot_fn,
                            create=create, **kw)


def record(project: Project, tool: str = "execute_code", *, title: str = "Make a box", args=None, text: str = "ok",
           ok: bool = True, doc: str = "", delta=None, health=None, mutating: bool = True, geometry: bool = True,
           turn_id: str = TURN, round_: int = 0, call_id: str = "c1", dur_s: float = 0.1):
    """``project.on_tool_result`` with sensible defaults (a geometry step that added ``Box``)."""
    return project.on_tool_result(
        turn_id=turn_id, round=round_, call_id=call_id, tool=tool, title=title,
        args={"code": "print(1)"} if args is None else args, text=text, ok=ok, dur_s=dur_s, doc=doc,
        delta={"added": ["Box"], "removed": [], "changed": []} if delta is None else delta, health=health,
        mutating=mutating, geometry=geometry)


def read_lines(project: Project) -> list[str]:
    """The raw lines of steps.jsonl (``[]`` when there is none)."""
    try:
        with open(project.path("steps.jsonl"), encoding="utf-8") as f:
            return f.read().split("\n")[:-1]
    except OSError:
        return []


class FakeGui:
    """A dispatcher with a real second thread: ``dispatch(fn, timeout, op, should_stop=None)`` runs ``fn`` on that
    thread and waits for it, exactly like ``cad.dispatch.dispatch`` does for the GUI thread.

    ``before_task`` (optional) is called ON the GUI thread right before each task, while the calling worker is
    blocked inside ``dispatch``; its return value is kept in ``observations``. ``ops`` lists the ``op`` names seen;
    ``delay`` makes every task sleep first (a busy GUI)."""

    def __init__(self, before_task=None, delay: float = 0.0):
        self.before_task = before_task
        self.delay = delay
        self.ops: list[str] = []
        self.observations: list = []
        self._queue: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="fake-gui", daemon=True)
        self._thread.start()
        self.thread_id = self._thread.ident
        self._saved = None

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, box, done = item
            try:
                if self.before_task is not None:
                    self.observations.append(self.before_task())
                if self.delay:
                    threading.Event().wait(self.delay)
                box["result"] = fn()
            except BaseException as exc:  # noqa: BLE001 - reported like dispatch does
                box["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                done.set()

    def dispatch(self, fn, timeout: float = 60.0, op: str = "gui task", should_stop=None):
        """``(result, failed)`` like ``cad.dispatch.dispatch``."""
        self.ops.append(op)
        box: dict = {}
        done = threading.Event()
        self._queue.put((fn, box, done))
        if not done.wait(timeout):
            return {"success": False, "error": f"'{op}' did not complete within {timeout:g}s (the GUI thread stayed busy)"}, True
        if "error" in box:
            return {"success": False, "error": box["error"]}, True
        return box["result"], False

    def install(self) -> "FakeGui":
        """Make ``project_gui.gui_call`` use this dispatcher until ``uninstall``."""
        self._saved = gui.load_dispatch
        gui.load_dispatch = lambda: self.dispatch
        return self

    def uninstall(self) -> None:
        if self._saved is not None:
            gui.load_dispatch = self._saved
            self._saved = None
        self._queue.put(None)


class FakeDoc:
    """The save-relevant part of an ``App.Document``: ``saveCopy(path)`` appends ``.FCStd`` when the name does not
    end in it (the verified FreeCAD behaviour), returns None, never raises and creates missing parent folders."""

    def __init__(self, name: str = "Fake", *, write: bool = True, garbage: bool = False):
        self.Name = name
        self.Objects: list = []
        self.write = write
        self.garbage = garbage
        self.saved_to: list[str] = []

    def saveCopy(self, path: str):  # noqa: N802 - FreeCAD API name
        if not path.endswith(".FCStd"):
            path += ".FCStd"
        self.saved_to.append(path)
        if not self.write:
            return None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if self.garbage:
            with open(path, "wb") as f:
                f.write(b"not a zip")
            return None
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("Document.xml", f"<Document name='{self.Name}'/>")
        return None


def close_docs(names) -> None:
    """Close the FreeCAD documents ``names`` (missing ones are ignored)."""
    import FreeCAD as App

    for name in list(names):
        if name in App.listDocuments():
            try:
                App.closeDocument(name)
            except Exception:  # noqa: BLE001
                pass


def rmtree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def new_temp_file(directory: str, name: str, data: bytes = b"data") -> str:
    path = os.path.join(directory, name)
    os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def sandbox_dir() -> str:
    return tempfile.mkdtemp(prefix="p_", dir=isolation.sandbox_root())
