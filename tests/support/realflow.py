"""tests/support/realflow.py - a REAL Build Flow under test: real ``tools.run_tool``, real Project, real FreeCAD documents.

The only fakes are the model (``ScriptedProvider`` / ``ScriptedServer``), the 3D-view snapshotter (a 1x1 PNG) and the
clock (``ManualClock``: the watchdog and the heartbeat read it, so a "five minute" reasoning round costs no time).
``RealFlowCase`` is the base of the acceptance tests (invariants, provider matrix, trace replay, compaction):

* every test gets its own documents (closed again in cleanup), project folder, offline network (the permission dialog
  is denied, ``requests.get`` raises) and no ``OSCAR_REFPART_LOCAL_DIR`` unless the test sets ``self.use_mirror()``;
* ``make_worker`` builds an ``AgentWorker`` with that setup; ``turn`` runs one turn and returns the ``TurnResult``;
* ``section`` / ``check_structure`` read the app-generated Build Summary the way the invariant tests do.

Importing this module imports ``FreeCAD`` (test modules that use it already run under ``freecadcmd``).
"""

import copy
import hashlib
import os
import re
import unittest
import uuid
from unittest import mock

import FreeCAD as App

from oscar import agent, netguard, params
from oscar.models import ModelCaps
from oscar.project import REFERENCE_STATUSES
from oscar.providers import Message
from support import isolation
from support.fakes import temp_project
from support.gaps import known_gap  # noqa: F401 - re-exported for the tests
from support.scripted_llm import ScriptedProvider, call, calls
from support.signal_recorder import SignalRecorder

MIRROR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "refparts", "mirror")
SUMMARY_HEADINGS = ("### Model tree", "### Requirements", "### Parameters", "### Provenance", "### Validated",
                    "### NOT validated")
PI_PROMPT = "Build a housing for a Raspberry Pi 4 with a snap-fit lid in ABS."


class ManualClock:
    """An injectable monotonic clock: ``clock()`` is the time, ``advance(seconds)`` moves it."""

    def __init__(self, start: float = 1000.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def scripted_profile(base_url: str = "http://127.0.0.1:9/v1", **extra) -> "params.Profile":
    return params.Profile(name="t9", provider="openai_compat", model="scripted", base_url=base_url, **extra)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# the base is aliased so ``test_suite_guard`` does not mistake this support module for an undiscovered test file
_UnitBase = unittest.TestCase


class RealFlowCase(_UnitBase):
    """Base class: see the module docstring."""

    def setUp(self):
        self.docs: list[str] = []
        self.addCleanup(self.close_docs)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("OSCAR_REFPART_LOCAL_DIR", None)
        os.environ.pop("OSCAR_ALLOW_NET", None)
        os.environ["OSCAR_DOWNLOAD_DIR"] = isolation.fresh_dir("rf_dl_")
        for patcher in (mock.patch.object(netguard, "request_permission", return_value="deny"),
                        mock.patch("requests.get", side_effect=AssertionError("unexpected network access"))):
            patcher.start()
            self.addCleanup(patcher.stop)
        try:
            from oscar.refparts import catalog, fetch

            fetch.forget_all()
            # Migration (integrator, wave 1): the shipped catalog now lists STEP sources for the Pi 4 (community) and
            # the Pi 5 (official). With the dialog denied that is ``status=declined``, not ``status=nominal``; these
            # flow tests describe "a part without a STEP source" (the offline mirror is found by file name, not by
            # URL), so the copy handed to the code under test has the sources stripped. A test that needs a listed
            # URL patches ``catalog._catalog`` itself (test_invariants declined download).
            shipped = copy.deepcopy(catalog._catalog())
            for entry in shipped.get("entries", []):
                entry["step_urls"] = []
            patcher = mock.patch.object(catalog, "_catalog", return_value=shipped)
            patcher.start()
            self.addCleanup(patcher.stop)
        except Exception:  # noqa: BLE001 - a missing optional module must not break an unrelated test
            pass
        netguard.reset_turn()
        self.clock = ManualClock()
        self.provider = None
        self.worker = None
        self.rec = None
        self.history: list[Message] = []

    # -- documents and projects -------------------------------------------------------------------------------

    def close_docs(self):
        for name in self.docs:
            if name in App.listDocuments():
                App.closeDocument(name)

    def doc_name(self, prefix: str = "T9") -> str:
        """A unique document name; the document is closed again in cleanup (it is created by the model's
        ``create_document`` call, or by ``open_doc``)."""
        name = f"{prefix}{uuid.uuid4().hex[:6]}"
        self.docs.append(name)
        return name

    def open_doc(self, prefix: str = "T9Open") -> str:
        """A document that exists BEFORE the turn (the user's own)."""
        name = self.doc_name(prefix)
        App.newDocument(name)
        return name

    def new_project(self, title: str = "Test project", chat_id: str | None = None, *, create: bool = True):
        """A REAL Project in a fresh folder (``create=False``: the directory appears lazily, like in the product)."""
        return temp_project(chat_id=chat_id or f"2026100{uuid.uuid4().hex[:1]}-101010-{uuid.uuid4().hex[:6]}", title=title,
                            create=create)

    def use_mirror(self) -> None:
        """Point ``OSCAR_REFPART_LOCAL_DIR`` at the Pi 4-like STEP fixture (an offline "download")."""
        os.environ["OSCAR_REFPART_LOCAL_DIR"] = os.path.abspath(MIRROR)

    # -- script helpers ---------------------------------------------------------------------------------------

    @staticmethod
    def box(doc: str, name: str = "Plate", **props):
        kw = dict(doc_name=doc, obj_type="Part::Box", obj_name=name, include_screenshot=False)
        if props:
            kw["obj_properties"] = props
        return call("create_object", **kw)

    @staticmethod
    def code(doc: str, body: str, *, header: str = "import FreeCAD as App\nimport Part\n") -> str:
        """A script that starts with ``d = App.getDocument(<doc>)``."""
        return f"{header}d = App.getDocument('{doc}')\n{body}\n"

    # -- running ----------------------------------------------------------------------------------------------

    def make_worker(self, prompt, script=None, *, project=None, provider=None, profile=None, caps=None,
                    history=None, resolve_caps=False, **kw):
        """An ``AgentWorker`` over a scripted provider (or ``provider``), the injected clock and the real tools. ``caps``
        defaults to a plain ``ModelCaps()``; ``resolve_caps=True`` lets the worker resolve them from the profile
        (``Profile.caps_override`` applies), the way the product does."""
        self.history = history if history is not None else [Message(role="user", content=prompt)]
        self.provider = provider if provider is not None else ScriptedProvider(script or [], advance=self.clock.advance)
        kw.setdefault("clock", self.clock)
        self.worker = agent.AgentWorker(self.history, provider=self.provider, profile=profile or scripted_profile(),
                                        caps=None if resolve_caps else (caps if caps is not None else ModelCaps()), project=project, **kw)
        self.rec = SignalRecorder(self.worker)
        return self.worker

    def turn(self, prompt, script=None, **kw):
        """Build a worker and run one turn; the ``TurnResult`` (the worker is ``self.worker``)."""
        return self.make_worker(prompt, script, **kw).run_turn()

    def notes(self, tag: str) -> list:
        return [m for m in self.history if getattr(m, "synthetic", False) and m.tag == tag]

    # -- reading the Build Summary ----------------------------------------------------------------------------

    @staticmethod
    def section(text: str, heading: str) -> str:
        """The body of ``### <heading>`` of the Build Summary (up to the next heading), or ``""``."""
        m = re.search(rf"^### {re.escape(heading)}\s*$(.*?)(?=^#{{2,3}} |\Z)", text, re.S | re.M)
        return m.group(1) if m else ""

    def check_structure(self, result, project) -> None:
        """The app-owned parts of the invariants that hold for EVERY finished work turn that has a Project."""
        text = result.text
        self.assertIn("## Build Summary", text)
        for heading in SUMMARY_HEADINGS:
            self.assertIn(heading, text, heading)
        prov = project.read_provenance()
        ref = prov["reference"]
        self.assertIn(ref["status"], REFERENCE_STATUSES)
        self.assertIn(f"reference status: {ref['status']}", text)
        if ref["status"] in ("none", "catalog-nominal"):
            self.assertIn("STEP-based collision check", self.section(text, "NOT validated"))
        if ref["status"] != "none":
            entry = next(e for e in prov["entries"] if e["id"] == ref["entry"])
            self.assertTrue(entry["source"])
            self.assertIn(entry["status"], ("obtained", "declined", "failed", "nominal", "attached"))
            if entry["status"] in ("obtained", "attached"):
                self.assertEqual(len(entry["sha256"]), 64)
                # the reference entry (the import) may name the file only through the download entry with the same hash
                holders = [e for e in prov["entries"] if e["sha256"] == entry["sha256"] and e["local_path"]]
                self.assertTrue(holders, "no entry records where the sourced file is stored")
                self.assertTrue(all(os.path.isfile(project.path(e["local_path"])) for e in holders))


def frozen(obj):
    """A deep copy (scripts are consumed by providers: reuse needs a fresh copy per run)."""
    return copy.deepcopy(obj)
