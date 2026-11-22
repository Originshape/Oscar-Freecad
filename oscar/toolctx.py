"""Per-call context shared by the agent loop, tools.run_tool and the Project / Ledger hooks.

PURE PYTHON: no FreeCAD, no Qt, no oscar imports, so every module may import it.
The agent sets the context (``use_context``) around ``tools.run_tool``; run_tool and
tool handlers read it with ``current_context()``. A ContextVar is per-thread: code that
the handler hands to ``dispatch.dispatch`` runs on the GUI thread and must capture what
it needs from the context BEFORE dispatching.
"""

import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


def _never() -> bool:
    return False


@dataclass
class ToolOutcome:
    """What run_tool learned about one call (filled by run_tool, read by the agent)."""

    name: str = ""
    call_id: str = ""
    title: str = ""
    ok: bool = True                 # what the MODEL sees: not is_error (False when the handler failed OR the health check found new errors)
    handler_ok: bool = True         # the handler itself succeeded (health findings do NOT flip it); Project, the ledger, the dependents rule and the breaker's `repeat` rule use THIS
    health_failed: bool = False     # the call ran but left NEW health errors (is_error was set by the health suffix)
    text: str = ""                  # final text returned to the model (after hooks and clipping)
    dur_s: float = 0.0
    doc: str = ""
    mutating: bool = False
    geometry: bool = False
    delta: dict = field(default_factory=dict)       # {"added": [...], "removed": [...], "changed": [...]}
    health: dict | None = None                       # summary {"valid", "errors", "warnings", "line", "findings" (<= 5)} or None; the full report is ctx.state["last_health"][doc]
    new_errors: list = field(default_factory=list)   # health error lines introduced by this call
    step_index: int = -1             # Project StepRecord.index; -1 = not recorded
    snapshot_path: str = ""          # ABSOLUTE path of the app-captured PNG, "" = none
    snapshot_note: str = ""
    checkpoint: str = ""
    script_path: str = ""
    image_b64: str = ""              # screenshot handed to the model, may be ""
    args: dict = field(default_factory=dict)         # arguments AFTER ctx.state["args_override"] (build_enclosure: the resolved stage)
    extra: dict = field(default_factory=dict)        # handler facts moved from ctx.state["extra"] by run_tool: reference_result, param_names, ...


@dataclass
class ToolContext:
    """The per-call context bound with ``use_context`` (see the module docstring for who writes what)."""
    project: Any = None              # oscar.project.Project | None
    ledger: Any = None               # oscar.flow.Ledger | None
    turn_id: str = ""
    round: int = 0
    call_id: str = ""
    phase: str = ""
    intent: str = ""
    step_title: str = ""             # filled by run_tool from the universal ``step_title`` argument
    outcome: ToolOutcome | None = None   # filled by run_tool
    state: dict = field(default_factory=dict)  # per-TURN scratch: the agent creates ONE dict per turn and passes the SAME dict to every per-call ToolContext
    should_stop: Callable[[], bool] = _never

    @property
    def active(self) -> bool:
        """True when run_tool should run its hooks (a Project or a Ledger is bound)."""
        return self.project is not None or self.ledger is not None


_CTX: "contextvars.ContextVar[ToolContext | None]" = contextvars.ContextVar(
    "oscar_tool_ctx", default=None)


def current_context() -> ToolContext:
    """The bound context, or a fresh inert one (``active`` is False)."""
    return _CTX.get() or ToolContext()


@contextlib.contextmanager
def use_context(ctx: ToolContext) -> Iterator[ToolContext]:
    """Bind ``ctx`` for the current thread / task for the duration of the ``with`` block; the previous binding is restored on exit, also on error."""
    token = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(token)
