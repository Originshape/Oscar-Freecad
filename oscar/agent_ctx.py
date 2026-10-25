"""Context wiring of the agent loop (private module of ``oscar.agent``).

``ContextWiring`` is a mixin of ``AgentWorker``. It owns everything the loop needs to decide WHAT goes into a
provider request, apart from the conversation itself:

* the static system prompt of a turn (``prompt.build_system_prompt`` with the turn-start intent's tool names, the
  lite switch of the model caps and the playbooks of the intent), built once per turn;
* the tool specs of a round (``tools.specs(phases=flow.visible_phases(ledger), intent=..., lite=...)``) whose
  visibility only GROWS during a turn (a union with every name already offered, so an escalation from ``other`` to
  ``generic_part`` can never hide ``fem`` tools that were visible) and over a build (a follow-up turn of a
  chat that already built something is offered every phase at once);
* the derived request view (``context.build_request_view``) built from the append-only history, with the pinned task
  memory made from state the agent already holds (ledger, the project's on-disk records, the last health reports):
  NO GUI dispatch and no per-round document reads;
* the floor rule (a context window that cannot hold the system prompt, the tools and the newest rounds:
  ultra-lite first, then a visible refusal);
* the round-0 plan line next to ``<workspace_state>``;
* the health baseline of the turn (``ctx.state["baseline"]``) and the default validator (``tools_verify``).

Nothing here touches ``App.*`` except ``_take_baseline``, which goes through ``cad.health.snapshot_all_via_dispatch``
(the one documented GUI round trip of a turn start). Everything the mixin needs from the worker is a plain attribute
set by ``AgentWorker._begin_turn`` (see ``init_wiring``).
"""

import importlib
import re
from collections.abc import Mapping
from typing import Any, Callable

from . import codeguard as _codeguard
from . import context as _context
from . import prompt as _prompt
from . import toolpipe as _toolpipe
from . import tools as _tools
from .flow import visible_phases
from .flow_intent import is_goahead
from .turnlog import log_event

__all__ = ["ContextWiring", "ContextTooSmall", "default_validator", "playbooks_for", "inject_plan_line"]

_PARTS_REQUEST = re.compile(
    r"\b(screws?|nuts?|bolts?|washers?|bearings?|fasteners?|standard parts?|parts? librar(?:y|ies)|library)\b", re.I)
_PATH_LINE = re.compile(r"^[ \t]*path:[ \t]*(\S.*?)[ \t]*$", re.M)
_FILE_IN_RESULT = re.compile(
    r"((?:/|[A-Za-z]:[\\/])[^\s'\"<>|]+\.(?:step|stp|stl|obj|iges|igs|3mf|fcstd|brep|ply|dxf|svg))", re.I)
_FILE_TOOLS = frozenset({"download_file", "source_reference_part", "import_file", "import_reference_part"})
_PLAN_MARK = "PLAN (intent "
_DECISION_KINDS = frozenset({"decision"})


class ContextTooSmall(RuntimeError):
    """Raised by ``_request_for_round`` when even the ultra-lite request cannot fit the model's context window."""


# intents whose prismatic parts are better built from a structured (schema 2) plan
_STRUCTURED_INTENTS = ("bracket", "fixture", "generic_part")


def playbooks_for(intent: str, request: str, names: "set[str] | frozenset[str]") -> tuple[str, ...]:
    """The prompt playbooks that are ACTIVE for a turn: fem for fem, the parts
    library when the request names standard parts and the library tools are offered, and structured_part for a
    bracket / fixture / generic_part turn that is offered ``apply_feature_plan``. Pure; no playbook otherwise."""
    books: list[str] = []
    if intent == "fem":
        books.append("fem")
    if (intent in ("generic_part", "modify", "other") and "insert_part_from_library" in names
            and _PARTS_REQUEST.search(request or "")):
        books.append("parts_library")
    if intent in _STRUCTURED_INTENTS and "apply_feature_plan" in names:
        books.append("structured_part")
    return tuple(books)


def inject_plan_line(content: str, plan: str) -> str:
    """``content`` with ``plan`` placed at the end of its ``<workspace_state>`` block (a block is created when there is
    none), so everything that strips the block (``loopcore.real_user_request``, the context's typed-text cut) strips
    the plan with it. Idempotent: a message that already carries a plan line is returned unchanged."""
    content = content or ""
    if not plan or _PLAN_MARK in content:
        return content
    close = content.rfind("</workspace_state>")
    if close >= 0:
        return f"{content[:close].rstrip()}\n\n{plan}\n{content[close:]}"
    return f"{content}\n\n<workspace_state>\n{plan}\n</workspace_state>"


def default_validator() -> "Callable[[str | None, Any], Any] | None":
    """The real validator ``(doc, ctx) -> ValidationResult | None`` or ``None`` while ``tools_verify`` is not installed.

    Without a document name (no geometry document was touched this turn) it returns ``None``: no validation claim,
    never a guess at "the active document" (a user with two documents open must not have the wrong one judged).
    """
    try:
        tools_verify = importlib.import_module(f"{__package__}.tools_verify")
    except ModuleNotFoundError as e:
        if e.name not in (f"{__package__}.tools_verify", "tools_verify"):
            raise
        return None
    run = getattr(tools_verify, "run_validation", None)
    if not callable(run):
        return None

    def validator(doc, ctx):
        if not doc:
            return None
        return run(doc, ctx=ctx)

    return validator


class _LazyScriptPaths(Mapping):
    """``{call id: saved script path}`` read from the Project's step records on first use only (the context view asks
    for it only when it elides old scripts, so a normal round never parses the journal)."""

    def __init__(self, project: Any):
        self._project = project
        self._paths: "dict[str, str] | None" = None

    def _load(self) -> "dict[str, str]":
        if self._paths is None:
            try:
                reader = getattr(self._project, "read_steps", None)
                self._paths = _context.script_paths_from_steps(reader() if callable(reader) else ())
            except Exception:  # noqa: BLE001 - a missing path only makes the elision marker shorter
                self._paths = {}
        return self._paths

    def __bool__(self) -> bool:      # a truthiness test must not trigger the load
        return True

    def __getitem__(self, key):
        return self._load()[key]

    def __iter__(self):
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())


def _read(project: Any, name: str, default: Any) -> Any:
    """``project.<name>()`` or ``default`` when the project has no such reader or it fails (test stubs, disk errors)."""
    reader = getattr(project, name, None)
    if not callable(reader):
        return default
    try:
        value = reader()
    except Exception:  # noqa: BLE001
        return default
    return default if value is None else value


class ContextWiring:
    """Mixin of ``AgentWorker``: prompt, tool specs, request view, memory, floor rule, baseline, validator."""

    # -- state -----------------------------------------------------------------------------------------------

    def init_wiring(self) -> None:
        """Reset the per-turn wiring state (called by ``_begin_turn`` before anything else of this mixin)."""
        self._ultra = False
        self._offered: set[str] = set()
        self._usage_prompt: int | None = None       # measured prompt tokens of the last answered request
        self._sent_estimate: int | None = None      # the uncalibrated estimate of THAT request (calibration pair)
        self._pending_sent: int | None = None       # the estimate of the request in flight
        self._files_scanned = 0
        self._files_seen: list[str] = []
        self._steered: list[str] = []               # what the user typed while the turn ran (kept in the pinned memory)
        self._baseline_taken = False
        self._all_phases = False
        self._lite = self._lite_wanted()
        self._playbooks: tuple[str, ...] = ()

    def _lite_wanted(self) -> bool:
        """``models.lite_prompt_wanted(caps)``: small window, text tool mode or no parallel calls (False without caps)."""
        caps = getattr(self, "caps", None)
        if caps is None:
            return False
        try:
            from . import models as _models

            return bool(_models.lite_prompt_wanted(caps))
        except ModuleNotFoundError as e:
            if e.name not in (f"{__package__}.models", "models"):
                raise
        except Exception:  # noqa: BLE001
            pass
        return False

    # -- system prompt ------------------------------------------------------------------------------------------

    def _turn_tool_names(self, intent: str) -> frozenset[str]:
        """Names the system prompt may mention: the turn-start intent's categories, NOT the phases (so the prompt is
        constant for the whole turn); lite narrows them to the lite set. The script tools are left out while scripts
        are switched off (``codeguard.hidden_tools``), so the prompt drops the sentences about them."""
        try:
            return frozenset(s.name for s in _tools.specs(intent=intent if self._flow() else None, lite=self._lite,
                                                          exclude=_codeguard.hidden_tools()))
        except Exception:  # noqa: BLE001
            return frozenset()

    def _build_turn_prompt(self) -> str:
        """The static system prompt of this turn (computed once; ``prompt.build_system_prompt`` depends only on
        ``(lite, playbooks, tool_names)``)."""
        intent = self._ledger.intent if self._flow() else "other"
        names = self._turn_tool_names(intent)
        self._playbooks = playbooks_for(intent, self._request or "", names)
        return _prompt.build_system_prompt(self.caps, lite=self._lite, playbooks=self._playbooks,
                                           tool_names=names or None)

    # -- tools per round ------------------------------------------------------------------------------------------

    def _note_prior_build(self) -> None:
        """Turn start: a chat that already built something (a carried plan, or tool calls in the earlier history) is a
        build in progress, so its follow-up turns (``add a fan hole on the side``) are offered EVERY phase from the first
        round: visibility only grows over a build, and a new ledger must not shrink what the build already had."""
        carried = getattr(self, "_carried", None)
        history = self.history[:-1] if self.history else []
        self._all_phases = bool((carried is not None and carried.steps) or any(m.tool_calls for m in history))

    def _round_specs(self) -> list:
        """The tool specs offered this round. Visibility is cumulative: ``flow.visible_phases(ledger)`` already only
        grows (``max_phase``), and the union with every name offered earlier in the turn keeps an intent escalation
        from shrinking the list. Without flow enforcement (kill switch) every registered tool is offered. The script
        tools are never offered while scripts are switched off (``codeguard.hidden_tools``)."""
        flow_on = self._flow()
        ledger = self._ledger
        hidden = _codeguard.hidden_tools()
        if self._ultra:
            names = [n for n in _context.ULTRA_LITE_TOOLS if n in self._offered]
            return _tools.specs(names=names, lite=True, exclude=hidden)
        specs = _tools.specs(phases=visible_phases(ledger) if flow_on and not self._all_phases else None,
                             intent=ledger.intent if flow_on else None, lite=self._lite, exclude=hidden)
        now = {s.name for s in specs}
        if not now.issuperset(self._offered):
            specs = _tools.specs(names=self._offered | now, lite=self._lite, exclude=hidden)
            now = {s.name for s in specs}
        self._offered |= now
        return specs

    # -- pinned memory --------------------------------------------------------------------------------------------

    def _scan_files(self) -> list[str]:
        """Attachment and download paths worth remembering: the attachment list, ``path:`` lines of attachment blocks
        and file paths named by the results of the file tools (scanned incrementally, once per message)."""
        history = self.history
        for index in range(self._files_scanned, len(history)):
            message = history[index]
            if message.role == "user" and not message.tool_results and not getattr(message, "synthetic", False):
                for path in _PATH_LINE.findall(message.content or ""):
                    self._remember_file(path)
            for result in message.tool_results or ():
                if result.name in _FILE_TOOLS and not result.is_error:
                    for path in _FILE_IN_RESULT.findall(result.content or ""):
                        self._remember_file(path)
        self._files_scanned = len(history)
        return list(self._files_seen)

    def _remember_file(self, path: str) -> None:
        path = path.strip().strip("'\"")
        if path and path not in self._files_seen and len(self._files_seen) < 12:
            self._files_seen.append(path)

    def _object_rows(self) -> list[dict]:
        """Object rows for the memory from ``ctx.state["last_health"]`` (what ``run_tool`` collected; ``[]`` before the
        first geometry call; the opening ``<workspace_state>`` already lists the pre-existing objects)."""
        rows: list[dict] = []
        last = self._state.get("last_health")
        for doc, report in (last.items() if isinstance(last, dict) else ()):
            for row in (report.get("rows") if isinstance(report, dict) else None) or ():
                if isinstance(row, dict) and row.get("name") and not row.get("consumed"):
                    rows.append({"name": row.get("name"), "type": row.get("type", ""), "solids": row.get("solids") or 0,
                                 "volume": row.get("volume"), "visible": row.get("visible"),
                                 "valid": row.get("valid"), "doc": doc})
        return rows[:80]

    def _memory_inputs(self) -> "_context.MemoryInputs":
        """``context.MemoryInputs`` from state the worker already holds: no GUI dispatch."""
        ledger = self._ledger
        live = bool(self._flow() and self._live and ledger.steps)
        ledger_dict = None
        if live:
            try:
                ledger_dict = ledger.to_dict()
            except Exception:  # noqa: BLE001
                ledger_dict = None
        request = (ledger.request if live and ledger.request else "") or self._turn_request()
        carried = getattr(self, "_carried", None)
        if not live and carried is not None and carried.request and is_goahead(request):
            # "continue" / "yes" after a FINISHED build is a bare go-ahead, not a request: the memory keeps the task's own
            request = carried.request
        project = self.project
        params_rows: list = []
        reference = None
        decisions: list[str] = []
        root = ""
        if project is not None:
            root = getattr(project, "root", "") or ""
            params_rows = list((_read(project, "read_params", {}) or {}).get("params") or [])
            provenance = _read(project, "read_provenance", {}) or {}
            reference = provenance.get("reference") if isinstance(provenance, dict) else None
            if isinstance(reference, dict):
                entry = reference.get("entry")
                if isinstance(entry, str) and entry:
                    found = next((e for e in provenance.get("entries") or [] if isinstance(e, dict) and e.get("id") == entry), None)
                    reference = dict(reference, entry=found or {})
            for note in _read(project, "read_notes", []) or ():
                if isinstance(note, dict) and note.get("kind") in _DECISION_KINDS and note.get("text"):
                    decisions.append(str(note["text"]))
        decisions.extend(f"user, mid-turn: {text}" for text in self._steered[-6:])
        for waiver in (ledger_dict or {}).get("waivers") or ():
            if isinstance(waiver, dict) and waiver.get("what"):
                decisions.append(f"waived {waiver['what']}: {waiver.get('reason', '')}".strip())
        files = list(dict.fromkeys([*[str(p) for p in getattr(self, "_attachments", ()) or ()], *self._scan_files()]))
        return _context.MemoryInputs(
            request=request, ledger=ledger_dict, params=params_rows, objects=self._object_rows(), reference=reference,
            files=files, decisions=decisions, project_dir=root, turn_index=1)

    def _pinned_memory(self, max_chars: int, specs: list) -> str:
        """The pinned task memory text (never raises: an unreadable input only shortens it)."""
        try:
            return _context.pinned_memory(self._memory_inputs(), max_chars=max_chars,
                                          tool_names=[s.name for s in specs])
        except Exception as e:  # noqa: BLE001
            log_event("context_error", where="memory", error=f"{type(e).__name__}: {e}"[:200])
            return ""

    # -- the request of a round ------------------------------------------------------------------------------------

    def _view_options(self, specs: list, memory: str) -> dict:
        return dict(memory=memory, caps=self.caps, system=self._system, tools=specs,
                    usage_prompt_tokens=self._usage_prompt, sent_estimate=self._sent_estimate,
                    script_paths=_LazyScriptPaths(self.project) if self.project is not None else None)

    def _request_for_round(self) -> "tuple[list, list]":
        """``(view, specs)`` of the next provider call.

        The view is derived from the stored history (never mutated); ``stats`` are logged as ``context_view``. When
        the floor rule fires the request degrades to the ultra-lite tool set and a 1200 character memory (once per
        turn); when even that cannot fit, ``ContextTooSmall`` carries the visible refusal text.
        """
        specs = self._round_specs()
        memory_chars = _context.ULTRA_LITE_MEMORY_CHARS if self._ultra else _context.DEFAULT_MEMORY_CHARS
        try:
            view, stats = _context.build_request_view(self.history, **self._view_options(
                specs, self._pinned_memory(memory_chars, specs)))
            if stats.get("too_small") and not self._ultra:
                self._degrade_to_ultra_lite(stats)
                specs = self._round_specs()
                view, stats = _context.build_request_view(self.history, **self._view_options(
                    specs, self._pinned_memory(_context.ULTRA_LITE_MEMORY_CHARS, specs)))
        except Exception as e:  # noqa: BLE001 - a defect in context management must never end a turn
            log_event("context_error", where="view", error=f"{type(e).__name__}: {e}"[:200])
            view, stats = list(self.history), {"too_small": False, "raw_after": 0, "fallback": True}
        if stats.get("too_small"):
            raise ContextTooSmall(_context.too_small_notice(int(stats.get("window") or 0)))
        self._pending_sent = stats.get("raw_after")
        log_event("context_view", **{k: v for k, v in stats.items() if k != "alternation_problems"},
                  messages=len(view), tools=len(specs), ultra=self._ultra)
        return view, specs

    def _degrade_to_ultra_lite(self, stats: dict) -> None:
        """The floor rule's first step: the 8 most useful tools, the lite prompt, a 1200 character memory."""
        self._ultra = True
        self._lite = True
        self._state["lite"] = True
        names = [n for n in _context.ULTRA_LITE_TOOLS if n in self._offered]
        self._system = _prompt.build_system_prompt(self.caps, lite=True, tool_names=names or None)
        log_event("context_degraded", window=stats.get("window"), tools=len(names))
        self.notice.emit(f"The context window of this model ({stats.get('window')} tokens) is small: Oscar switched to a "
                         "reduced tool set and a shorter prompt for this turn.")

    def _note_usage(self, reply: Any) -> None:
        """Remember the measured prompt tokens of the answered request next to the estimate of what was sent."""
        info = getattr(reply, "stop_info", None)
        usage = info.get("usage") if isinstance(info, dict) else None
        tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if isinstance(tokens, (int, float)) and not isinstance(tokens, bool) and tokens > 0 and self._pending_sent:
            self._usage_prompt, self._sent_estimate = int(tokens), int(self._pending_sent)

    # -- plan line, baseline ------------------------------------------------------------------------------------------

    def _attach_plan_line(self) -> None:
        """Round 0 of a live work turn: the seeded plan (``Ledger.render_compact``) is appended ONCE to the newest user
        message, inside its ``<workspace_state>`` block, stored in the history and never rewritten afterwards."""
        ledger = self._ledger
        if not (self._flow() and self._live and ledger.steps and self.history):
            return
        last = self.history[-1]
        if last.role != "user" or last.tool_results:
            return
        try:
            last.content = inject_plan_line(last.content or "", ledger.render_compact())
        except Exception as e:  # noqa: BLE001
            log_event("context_error", where="plan_line", error=f"{type(e).__name__}: {e}"[:200])

    def _take_baseline(self) -> None:
        """``ctx.state["baseline"] = {doc: [(code, object), ...]}``: the health findings known before the turn changed
        anything, so the finish gate judges only what the turn created or changed. One cheap hash-keyed
        snapshot through the dispatcher (10 s budget, Stop-aware). On any failure the baseline is ``{}``: a document with
        no recorded baseline counts every one of its errors, the conservative direction (it can only block, never excuse)."""
        if self._baseline_taken:
            return
        self._baseline_taken = True
        baseline: dict[str, list] = {}
        try:
            from .cad import health as _health
            from .toolctx import use_context

            with use_context(self._make_ctx()):
                snapshots = _health.snapshot_all_via_dispatch(timeout=10.0)
            for doc, snap in (snapshots or {}).items():
                baseline[str(doc)] = [tuple(key) for key in getattr(snap, "finding_keys", None) or ()]
        except Exception as e:  # noqa: BLE001 - never blocks the turn
            log_event("context_error", where="baseline", error=f"{type(e).__name__}: {e}"[:200])
        self._state["baseline"] = baseline
        self._seed_touched()

    def _seed_touched(self) -> None:
        """``ctx.state["touched"]``: the objects the MODEL built or changed in earlier turns of this chat (the Project's
        step records), so the gate never excuses them as "pre-existing". The baseline above is what the turn found on
        entry; without this a broken ``Lid`` left by an unverified turn became part of the next turn's baseline and the
        next turn could end ``validated`` over it ("continue" after a failed validation is the usual way). What the
        USER left broken is not in any step record and stays excused. Never raises."""
        project = self.project
        reader = getattr(project, "read_steps", None)
        if not callable(reader):
            return
        try:
            records = list(reader() or ())
        except Exception as e:  # noqa: BLE001
            log_event("context_error", where="touched_seed", error=f"{type(e).__name__}: {e}"[:200])
            return
        touched = self._state.setdefault("touched", {})
        for record in records:
            doc = str(getattr(record, "doc", "") or "")
            delta = getattr(record, "delta", None)
            if not doc or getattr(record, "superseded", False) or not isinstance(delta, dict):
                continue
            names = _toolpipe.delta_names(delta)
            if names:
                touched.setdefault(doc, set()).update(names)

    def _ensure_baseline(self, name: str) -> None:
        """Lazy baseline for a turn that was not a work turn at its start: taken right before its first geometry call
        (the escalation case), so the user's pre-existing broken objects are still excused."""
        if self._baseline_taken or not self._flow():
            return
        entry = _tools.tool_meta(name) if hasattr(_tools, "tool_meta") else None
        if entry is not None and getattr(entry, "geometry", False):
            self._take_baseline()
