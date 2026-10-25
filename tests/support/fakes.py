"""tests/support/fakes.py - small fakes shared by the flow tests."""

import base64
import os
import unittest
from dataclasses import dataclass, field

# a valid 1x1 PNG
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def tiny_png_snapshot(doc_name: str, abs_path: str, width: int, height: int):
    """Drop-in for ``Project(snapshot_fn=...)``: writes a 1x1 PNG, returns ``(True, "")``."""
    with open(abs_path, "wb") as f:
        f.write(_PNG_1X1)
    return True, ""


@dataclass
class FakeValidation:
    """ValidationResult look-alike (oscar.tools_verify.ValidationResult has the same fields)."""
    ok: bool = True
    doc: str = ""
    health: dict = field(default_factory=lambda: {"valid": True, "counts": {"objects": 1, "solids": 1, "errors": 0, "warnings": 0},
                                                  "findings": [], "rows": [], "delta": None})
    gate_errors: list = field(default_factory=list)    # health errors that count for the FinishGate
    fit_rows: list = field(default_factory=list)
    fit_ran: bool = False
    fail_rows: list = field(default_factory=list)
    dfm: list = field(default_factory=list)
    coverage: list = field(default_factory=list)
    ran: list = field(default_factory=lambda: ["health"])
    not_validated: list = field(default_factory=lambda: ["step_collision", "physical"])
    text: str = "[health] ok"
    after_step: int = 0


def fake_validation(ok: bool = True, fail_rows=()):
    v = FakeValidation(ok=ok and not fail_rows, fail_rows=list(fail_rows))
    if fail_rows:
        v.fit_ran = True
        v.fit_rows = list(fail_rows)
    if not ok:
        v.health = dict(v.health, valid=False)
        v.gate_errors = [{"severity": "error", "code": "FRAGMENTS", "object": "Shell",
                          "message": "Shell: 3 solids, expected 1; fuse or delete the floating pieces"}]
    return v


def legacy_provider(rounds, *, strict: bool = True):
    """A ``ScriptedProvider`` from the ``(kind, payload)`` tuples of ``test_oscar_regressions.ScriptedProvider``
    (kinds ``text``, ``cutoff``, ``nameless``, anything else = a list of tool calls)."""
    from support.scripted_llm import ScriptedProvider, from_legacy

    return ScriptedProvider([from_legacy(kind, payload) for kind, payload in rounds], strict=strict)


def temp_project(*, chat_id: str = "c0ffee", title: str = "Test project", snapshot_fn=tiny_png_snapshot,
                 create: bool = True, root: str | None = None, clock=None):
    """A REAL ``oscar.project.Project`` in a fresh directory inside the test sandbox, with the fake PNG
    snapshotter (no GUI). Skips the calling test when ``oscar.project`` does not exist yet (a later stage)."""
    try:
        from oscar.project import Project
    except ModuleNotFoundError as e:
        if e.name not in ("oscar.project", "project"):
            raise
        raise unittest.SkipTest("oscar.project is not available yet") from e
    from support import isolation

    root = root or os.path.join(isolation.fresh_dir("proj_"), "project")
    kwargs = {"chat_id": chat_id, "title": title, "snapshot_fn": snapshot_fn, "create": create}
    if clock is not None:
        kwargs["clock"] = clock
    return Project(root, **kwargs)
