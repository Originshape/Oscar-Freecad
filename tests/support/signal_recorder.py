"""tests/support/signal_recorder.py - record AgentWorker signals in emission order.

Works without a Qt event loop: the worker is run synchronously (``worker.run_turn()``), and
signals connected to plain Python callables are delivered directly on the emitting thread.
Slots use FIXED arity (PySide matches slot arity to the signal).
"""

import json

ARITY = {
    "text_delta": 1, "reasoning_delta": 1, "tool_started": 1, "tool_finished": 3, "turn_finished": 2,
    "notice": 1, "flow_paused": 1, "step_started": 1, "step_finished": 1, "plan_changed": 1,
    "progress": 1, "round_reset": 1, "artifacts_changed": 1, "turn_outcome": 2,
}


class SignalRecorder:
    def __init__(self, worker):
        self.events: list[tuple[str, tuple]] = []
        for name, arity in ARITY.items():
            signal = getattr(worker, name, None)
            if signal is not None:
                signal.connect(self._slot(name, arity))

    def _slot(self, name, arity):
        ev = self.events
        if arity == 1:
            return lambda a: ev.append((name, (a,)))
        if arity == 2:
            return lambda a, b: ev.append((name, (a, b)))
        return lambda a, b, c: ev.append((name, (a, b, c)))

    def names(self) -> list[str]:
        return [n for n, _ in self.events]

    def of(self, name: str) -> list[tuple]:
        return [a for n, a in self.events if n == name]

    def json_of(self, name: str) -> list:
        return [json.loads(a[0]) for a in self.of(name)]

    def count(self, name: str) -> int:
        """How many times ``name`` was emitted."""
        return sum(1 for n, _ in self.events if n == name)

    def index_of(self, name: str, start: int = 0) -> int:
        """Position of the first ``name`` at or after ``start`` in emission order, ``-1`` when absent."""
        return next((i for i, (n, _) in enumerate(self.events) if i >= start and n == name), -1)

    def ordered(self, *names: str) -> bool:
        """True when the FIRST emissions of ``names`` occur in exactly this order (all must be present)."""
        positions = [self.index_of(n) for n in names]
        return all(p >= 0 for p in positions) and positions == sorted(positions) and len(set(positions)) == len(positions)

    def assert_ordered(self, *names: str) -> None:
        """Fail with the emission order when ``ordered(*names)`` is false (invariant checks such as ``turn_outcome``
        arriving before ``turn_finished``)."""
        assert self.ordered(*names), f"expected {list(names)} in this order; emitted: {self.names()}"

    def text(self) -> str:
        """All ``text_delta`` payloads concatenated."""
        return "".join(a[0] for a in self.of("text_delta"))
