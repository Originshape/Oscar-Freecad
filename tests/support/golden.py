"""tests/support/golden.py - one shared run of the golden Pi 4 scenario.

``GoldenRun`` builds the sandbox (closed documents, UndoMode 1 like the GUI, offline network, the Pi-4-like STEP as
an offline mirror or not, a REAL Project with the fake snapshotter, the injected clock) and plays the ``golden_pi4``
persona through the real ``AgentWorker`` + real ``tools.run_tool`` once. The test classes read ``run.result``,
``run.project``, ``run.rec`` ... in as many ``test_*`` methods as they like (one module-level run per variant).

Importing this module imports ``FreeCAD`` (test modules already run under ``freecadcmd``).
"""

import time
import unittest

import FreeCAD as App

from support import personas
from support.fakes import tiny_png_snapshot  # noqa: F401 - the snapshotter every Project here uses
from support.realflow import RealFlowCase

DOC = "PiCase"
_UnitBase = unittest.TestCase


_FlowBase = RealFlowCase   # aliased so test_suite_guard does not mistake this support module for an undiscovered test file


class _Sandbox(_FlowBase):
    """A RealFlowCase that is driven by hand (``setUp`` / ``doCleanups``) instead of by the runner."""

    def runTest(self):  # pragma: no cover - required by TestCase(methodName) only
        pass


class GoldenRun:
    """The scripted Pi 4 build, played once. ``nominal`` runs it without the offline STEP mirror."""

    def __init__(self, *, nominal: bool = False, lid_lift: float | None = None, chat_id: str = "20261002-101010-90d0e1",
                 prefix=(), caps=None, config=None, persist: bool = False):
        """``prefix``: rounds a weak model wastes BEFORE the golden script starts (plan-only replies, bad JSON,
        a reasoning flood); ``caps`` / ``config``: passed to the worker; ``persist``: save the chat after every
        round through ``chats.persist_round`` (``self.session`` is the session dict)."""
        self.prefix = list(prefix)
        self.caps = caps
        self.config = config
        self.persist = persist
        self.session = None
        self.nominal = nominal
        self.lid_lift = lid_lift
        self.chat_id = chat_id
        self.case = _Sandbox()
        self.result = None
        self.project = None
        self.rec = None
        self.wall_s = 0.0
        self.doc_created = ""
        self.script = None
        self.worker = None

    def start(self) -> "GoldenRun":
        case = self.case
        case.setUp()
        for name in list(App.listDocuments()):   # hygiene: a leaked document would make newDocument return PiCase1
            App.closeDocument(name)
        case.docs.append(DOC)
        if not self.nominal:
            case.use_mirror()
        self.project = case.new_project("Pi 4 housing", chat_id=self.chat_id)
        self.script = personas.build("golden_pi4", nominal=self.nominal, lid_lift=self.lid_lift)
        self._undo_mode_on_new_documents()
        kw = {}
        if self.caps is not None:
            kw["caps"] = self.caps
        if self.config is not None:
            kw["config"] = self.config
        if self.persist:
            from oscar import chats

            self.session = chats.new_session("Pi 4 housing")
            kw["on_round"] = lambda history: chats.persist_round(self.session)
        t0 = time.monotonic()
        try:
            worker = case.make_worker(personas.prompt_for("golden_pi4"), self.prefix + self.script, project=self.project, **kw)
            if self.session is not None:
                self.session["history"] = case.history
            self.worker = worker
            self.result = worker.run_turn()
        finally:
            self.wall_s = time.monotonic() - t0
            self._restore_undo_mode()
        self.rec = case.rec
        return self

    def stop(self) -> None:
        self.case.doCleanups()

    # -- UndoMode: freecadcmd documents default to 0, the GUI to 1; the rollback paths must run as in the GUI -------

    def _undo_mode_on_new_documents(self) -> None:
        original = App.newDocument
        self._original_new_document = original

        def new_document(*args, **kwargs):
            doc = original(*args, **kwargs)
            try:
                doc.UndoMode = 1
            except Exception:  # noqa: BLE001 - a property that cannot be set must not break the scenario
                pass
            return doc

        App.newDocument = new_document

    def _restore_undo_mode(self) -> None:
        App.newDocument = self._original_new_document

    # -- convenience --------------------------------------------------------------------------------------------

    @property
    def doc(self):
        return App.getDocument(DOC)

    def steps(self):
        return self.project.read_steps()
