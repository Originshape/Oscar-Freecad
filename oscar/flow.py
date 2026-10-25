"""Oscar's build flow: the plan ledger, the phase machine and the finish gate (pure Python).

The model-dependent, prompt-hoped agent flow is replaced by state the APP owns and checks:

* ``Ledger``: the plan (requirements ``R1..Rn`` and steps ``S1..Sn``) with evidence recorded from tool results,
  seeded deterministically from the typed request (``classify_intent`` -> ``extract_requirements`` ->
  ``seed_template``). The model refines it through ``plan_set`` / ``plan_update``; statuses are derived by code.
* the phase machine (``derive_phase`` / ``visible_phases``): the phase is a function of the ledger, never declared;
* ``finish_gate``: the checklist a reply without tool calls must pass before the turn may end,
* ``format_governor``, ``Watchdog`` / ``FailureBreaker`` (``flow_guard``) and ``AgentConfig``: the runtime limits.

No FreeCAD, no Qt, no ``oscar.tools`` / ``oscar.project`` imports: everything takes plain data and duck-typed
objects (a tool outcome is anything with the ``oscar.toolctx.ToolOutcome`` attributes). The request analysis
lives in ``flow_intent``, the guards in ``flow_guard`` and the outcome texts, governor line and ``AgentConfig`` in
``flow_run``; their public names are re-exported here, so callers import everything from ``oscar.flow``.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .flow_guard import (BreakerVerdict, FailureBreaker, Watchdog, WatchdogConfig,  # noqa: F401  (re-exported)
                         WATCHDOG_REASONS)
from .flow_intent import (INTENTS, KEYWORD_RULES, WORK, classify_intent, clauses_of,  # noqa: F401  (re-exported)
                          extract_requirements, is_continuation, needs_reference)

__all__ = [
    "PHASES", "INTENTS", "WORK", "OUTCOMES", "STEP_STATUSES", "TERMINAL", "NOT_REQUESTED",
    "NOT_VERIFIED_HEADER", "NOT_VERIFIED_FOOTER", "NO_CHANGES_TEXT", "ENCLOSURE_REQUIRED_PARAMS",
    "Requirement", "LedgerStep", "Ledger", "LedgerError",
    "classify_intent", "extract_requirements", "needs_reference", "is_continuation", "seed_template",
    "derive_phase", "phase_exit_met", "visible_phases", "TurnState", "GateVerdict", "finish_gate",
    "format_governor", "health_label", "phase_guidance", "tool_category", "AgentConfig",
    "Watchdog", "WatchdogConfig", "FailureBreaker", "BreakerVerdict",
    "compose_unverified", "compose_capped", "compose_needs_help", "compose_failed",
]

PHASES = ("INTAKE", "SOURCE", "PLAN", "PARAMS", "BUILD", "VERIFY", "DELIVER")
_PHASE_INDEX = {p: i for i, p in enumerate(PHASES)}
OUTCOMES = ("answered", "validated", "unverified", "failed", "capped", "stopped", "needs_help", "needs_input")

STEP_STATUSES = ("pending", "running", "done", "blocked", "skipped", "changed")
REQ_STATUSES = ("pending", "done", "blocked", "skipped", "changed")
TERMINAL = frozenset({"done", "blocked", "skipped", "changed"})
_NOTE_REQUIRED = frozenset({"blocked", "skipped", "changed"})

REVERTED_NOTE = "reverted"
NOT_REQUESTED = "not requested in the brief"
NO_STEP_NOTE = "no STEP file: nominal board data used"

# `required_params` of the enclosure template: S4 is complete when the Params sheet holds all of them (and the spec is ok).
ENCLOSURE_REQUIRED_PARAMS = ("wall_t", "floor_t", "lid_t", "clearance", "standoff_h", "lid_gap", "lid_lift",
                             "board_len", "board_w", "pcb_t")

# Tool name -> registry category. The Ledger is pure and must not import oscar.tools, so it carries
# the one fact it needs from the tool matrix; test_agent_loop (it needs FreeCAD) cross-checks it against the live registry.
_TOOL_CATEGORY = {
    "list_documents": "read", "get_objects": "read", "get_object": "read", "get_view": "read",
    "get_parts_list": "read", "get_async_status": "read",
    "plan_set": "plan", "plan_update": "plan", "record_note": "plan", "enclosure_spec": "plan",
    "create_document": "session", "reload_document": "session",
    "import_file": "source", "find_reference_part": "source", "source_reference_part": "source",
    "import_reference_part": "source", "measure_reference_part": "source",
    "download_file": "net", "web_search": "net", "fetch_url": "net",
    "insert_part_from_library": "build", "create_object": "build", "edit_object": "build",
    "delete_object": "build", "execute_code": "build", "execute_code_async": "build",
    "execute_code_headless": "build", "apply_feature_plan": "build", "reapply_plan_feature": "build",
    "build_enclosure": "build", "add_cutout": "build", "add_vents": "build",
    "feature_tree_status": "verify", "validate_model": "verify", "check_fit": "verify",
    "run_fem_analysis": "fem",
    "set_parameters": "params", "get_parameters": "params", "bind_expression": "params", "set_parameter": "params",
    "save_document": "deliver", "export_model": "deliver", "export_rebuild_script": "deliver",
}


def tool_category(name: str) -> str:
    """The registry category of a tool by name (``""`` for an unknown tool)."""
    return _TOOL_CATEGORY.get(name or "", "")


class LedgerError(ValueError):
    """A plan edit the app rejects; the message is model-facing (``<tool>: <what>; <what to do>``)."""


@dataclass
class Requirement:
    """One thing the user asked for. ``status`` / ``covered_by`` / ``evidence`` are DERIVED by the Ledger from its steps."""

    id: str
    text: str
    status: str = "pending"
    covered_by: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    note: str = ""


@dataclass
class LedgerStep:
    """One plan step. ``covers`` (requirement ids) is the ONLY stored step <-> requirement relation."""

    id: str
    title: str
    phase: str
    acceptance: str = ""
    status: str = "pending"
    note: str = ""
    owner: str = "model"                   # "model" | "app" (app-owned steps cannot be set by plan_update)
    required: bool = True
    covers: list[str] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    stages_done: list[str] = field(default_factory=list)
    hint_tools: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    claimed: bool = False                  # done only because the model said so (no recorded work)
    waived_reason: str = ""                # set when the APP skipped the step (a waiver by "app")
    template: bool = True


# ---------------------------------------------------------------------------------------------------
# Template plans

_BUILD_TOOLS = ["create_object", "execute_code", "edit_object"]


def _step(sid: str, phase: str, title: str, acceptance: str, *, owner: str = "model", required: bool = True,
          stages: Sequence[str] = (), hint: Sequence[str] = (), done: bool = False) -> LedgerStep:
    step = LedgerStep(id=sid, title=title, phase=phase, acceptance=acceptance, owner=owner, required=required,
                      stages=list(stages), hint_tools=list(hint))
    if done:
        step.status = "done"
    elif not required:
        step.status, step.note = "skipped", NOT_REQUESTED
    return step


def _kw(name: str, text: str) -> bool:
    return bool(KEYWORD_RULES[name].search(text or ""))


def seed_template(intent: str, requirements: Sequence[dict], needs_reference: bool, *,
                  request: str = "") -> list[LedgerStep]:
    """The template steps of an intent: fixed ids and titles, ``owner=app`` steps completed by code.

    Steps whose requirement did not fire (or whose keyword rule does not match the request) are kept with
    ``required=False``, status ``skipped`` and the note ``not requested in the brief`` so the plan strip shows
    them greyed. ``request`` (the typed text) feeds the keyword rules of the non-enclosure templates; when empty
    the requirement texts are used. Requirement -> step coverage (``covers``) is applied by the caller
    (``Ledger.from_request``) from the requirement dicts. Intents without a template return ``[]``.
    """
    text = request or " ".join(str(r.get("text", "")) for r in requirements)

    def fired(sid: str) -> bool:
        return any(sid in (r.get("covers") or ()) for r in requirements)

    if intent == "enclosure":
        return [
            _step("S1", "INTAKE", "Capture requirements and assumptions", "requirements R1..Rn recorded",
                  owner="app", done=True),
            _step("S2", "SOURCE", "Source the reference part",
                  "reference status recorded (catalog-nominal, step-imported, user-attached) or an explicit decline",
                  required=needs_reference, hint=["find_reference_part", "source_reference_part"]),
            _step("S3", "SOURCE", "Import and measure the reference",
                  "Ref_<ID> exists in the document and dimensions are measured, or skipped because no STEP exists",
                  required=needs_reference, stages=["import", "measure"],
                  hint=["import_reference_part", "measure_reference_part"]),
            _step("S4", "PARAMS", "Parameters table and spec",
                  "Params sheet holds every driving dimension, each with a source; spec validates",
                  stages=["params"], hint=["build_enclosure", "set_parameters", "enclosure_spec"]),
            _step("S5", "BUILD", "Bottom shell: floor, walls, standoffs",
                  "Shell is ONE valid solid; standoffs line up with the board holes",
                  stages=["shell", "standoffs"], hint=["build_enclosure"]),
            _step("S6", "BUILD", "Port cutouts for every connector", "every wall connector has an opening",
                  required=fired("S6"), stages=["ports"], hint=["build_enclosure", "add_cutout"]),
            _step("S7", "BUILD", "Lid with lip", "Lid is ONE valid solid", stages=["lid"], hint=["build_enclosure"]),
            _step("S8", "BUILD", "Air vent slots", "vent array cut with a minimum web",
                  required=fired("S8"), stages=["vents"], hint=["build_enclosure", "add_vents"]),
            _step("S9", "BUILD", "Snap hooks and catch windows", "hooks engage the windows",
                  required=fired("S9"), stages=["hooks"], hint=["build_enclosure"]),
            _step("S10", "BUILD", "Assembly with lid_lift parameter",
                  "parts stay separate; lid_lift drives the lid position", stages=["assembly"],
                  hint=["build_enclosure"]),
            _step("S11", "VERIFY", "Validate model and check fit", "health valid and zero FAIL fit rows", owner="app"),
            _step("S12", "DELIVER", "Save project, export, notes", "NOTES.md, rebuild.py, exports present", owner="app"),
        ]
    if intent == "fixture":
        return [
            _step("S1", "INTAKE", "Capture requirements and assumptions", "requirements recorded", owner="app", done=True),
            _step("S2", "SOURCE", "Source the reference part", "reference status recorded or an explicit decline",
                  required=needs_reference, hint=["find_reference_part", "source_reference_part"]),
            _step("S3", "SOURCE", "Import and measure the reference", "reference imported and measured, or skipped",
                  required=needs_reference, stages=["import", "measure"],
                  hint=["import_reference_part", "measure_reference_part"]),
            _step("S4", "PARAMS", "Parameters table", "driving dimensions are named parameters", required=False,
                  hint=["set_parameters"]),
            _step("S5", "BUILD", "Base and nest pocket for the part", "the part sits in a pocket of the base",
                  hint=_BUILD_TOOLS),
            _step("S6", "BUILD", "Locating and clamping features", "locating / clamping features exist",
                  required=_kw("fixture_clamp", text), hint=_BUILD_TOOLS),
            _step("S7", "VERIFY", "Validate model and check fit", "health valid and zero FAIL fit rows", owner="app"),
            _step("S8", "DELIVER", "Save project, export, notes", "notes and exports present", owner="app"),
        ]
    if intent == "bracket":
        return [
            _step("S1", "INTAKE", "Capture requirements and assumptions", "requirements recorded", owner="app", done=True),
            _step("S2", "PARAMS", "Parameters table", "driving dimensions are named parameters", required=False,
                  hint=["set_parameters"]),
            _step("S3", "BUILD", "Base body and profile", "the bracket body exists as a valid solid", hint=_BUILD_TOOLS),
            _step("S4", "BUILD", "Holes and slots", "the holes and slots exist",
                  required=_kw("bracket_holes", text), hint=["execute_code", "edit_object", "create_object"]),
            _step("S5", "BUILD", "Fillets and finishing features", "finishing features exist", required=False,
                  hint=["edit_object", "execute_code"]),
            _step("S6", "VERIFY", "Validate model", "health valid", owner="app"),
            _step("S7", "DELIVER", "Save project, export, notes", "notes and exports present", owner="app"),
        ]
    if intent == "generic_part":
        return [
            _step("S1", "INTAKE", "Capture requirements and assumptions", "requirements recorded", owner="app", done=True),
            _step("S2", "PARAMS", "Parameters table", "driving dimensions are named parameters", required=False,
                  hint=["set_parameters"]),
            _step("S3", "BUILD", "Model the part", "the requested geometry exists as valid solids", hint=_BUILD_TOOLS),
            _step("S4", "VERIFY", "Validate model", "health valid", owner="app"),
            _step("S5", "DELIVER", "Save project, export, notes", "notes and exports present", owner="app"),
        ]
    if intent == "modify":
        return [
            _step("S1", "BUILD", "Apply the requested change", "the model reflects the requested change",
                  hint=["edit_object", "execute_code", "create_object"]),
            _step("S2", "VERIFY", "Validate model", "health valid", owner="app"),
            _step("S3", "DELIVER", "Save project", "project saved", owner="app"),
        ]
    if intent == "fem":
        return [
            _step("S1", "BUILD", "Set up the analysis (material, mesh, constraints)", "the analysis is fully set up",
                  hint=["create_object", "execute_code"]),
            _step("S2", "BUILD", "Run the solver", "the solver ran and results exist", hint=["run_fem_analysis"]),
            _step("S3", "VERIFY", "Validate model", "health valid", owner="app"),
            _step("S4", "DELIVER", "Save project", "project saved", owner="app"),
        ]
    if intent == "deliver":
        return [_step("S1", "DELIVER", "Produce the requested files", "the requested files exist",
                      hint=["export_model", "save_document"])]
    return []


# ---------------------------------------------------------------------------------------------------
# Ledger

_VALIDATION_FIELDS = ("ok", "doc", "health", "gate_errors", "fit_rows", "fit_ran", "fail_rows", "dfm", "coverage",
                      "ran", "not_validated", "text", "after_step")


def _validation_dict(validation) -> dict | None:
    """A ValidationResult / dict / FakeValidation as a plain dict (``None`` stays ``None``)."""
    if validation is None:
        return None
    if isinstance(validation, dict):
        return dict(validation)
    return {k: getattr(validation, k) for k in _VALIDATION_FIELDS if hasattr(validation, k)}


def _g(obj, name: str, default=None):
    """Attribute of an outcome-like object (or key of a dict), with a default."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _id_list(value) -> list[str]:
    """A list of ids from what a model sent as ``covers``: a list, or one string such as ``"R1"`` / ``"R1, R2"`` / ``"R1 R2"``."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in re.split(r"[\s,;]+", value) if part]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


class Ledger:
    """The plan of one task: requirements, steps, evidence, waivers and the facts the finish gate needs.

    Every mutator bumps ``revision`` and ``updated`` when something changed (the agent emits ``plan_changed`` on
    a new revision) and re-derives ``phase``, ``max_phase`` and every requirement's ``covered_by`` / ``status``
    (the relations live only in ``LedgerStep.covers``). Not thread-safe: the agent thread owns it; other threads
    take ``to_dict()`` snapshots.
    """

    VERSION = 1

    def __init__(self, intent: str = "other", request: str = "", *, now: Callable[[], str] | None = None):
        """An empty ledger (no steps, phase INTAKE); use ``from_request`` / ``from_dict`` for a seeded one."""
        self._now = now or _now_text
        self.version = self.VERSION
        self.intent = intent
        self.phase = "INTAKE"
        self.max_phase = "INTAKE"
        self.request = request
        self.needs_reference = False
        self.reference_hint = ""
        self.reference_status = "none"
        self.requirements: list[Requirement] = []
        self.steps: list[LedgerStep] = []
        self.waivers: list[dict] = []
        self.round = 0
        self.round_cap = 30
        self.revision = 0
        self.updated = self._now()
        self.validated_after_step = 0
        self.last_geometry_step = 0
        self.health_ok = True
        self.open_problems: list[str] = []
        self.spec_ok = False
        self.param_names: list[str] = []
        self._now_stamp = self.updated

    # -- construction -------------------------------------------------------------------------------
    @classmethod
    def from_request(cls, text: str, *, intent: str | None = None, attachments: Sequence[str] = (),
                     now: Callable[[], str] | None = None) -> "Ledger":
        """The seeded ledger of a typed request (INTAKE): intent, requirements ``R1..Rn`` and the template steps.

        ``text`` must be what the user typed (5.2 input rule). A request that needs no workspace change
        (``question`` / ``inspect`` / ``other``) yields a Ledger with NO steps, so callers never need a None check.
        """
        intent = intent or classify_intent(text, attachments=attachments)
        ledger = cls(intent, (text or "").strip()[:4000], now=now)
        ledger._seed(intent, attachments)
        ledger._refresh()
        return ledger

    def _seed(self, intent: str, attachments: Sequence[str] = ()) -> None:
        text = self.request
        reqs = extract_requirements(text, intent)
        needs, hint = needs_reference(text, intent, attachments=attachments)
        self.needs_reference, self.reference_hint = needs, hint
        steps = seed_template(intent, reqs, needs, request=text)
        by_id = {s.id: s for s in steps}
        fallback = next((s for s in steps if s.required and s.owner == "model"), steps[0] if steps else None)
        self.requirements = []
        for r in reqs:
            self.requirements.append(Requirement(id=r["id"], text=r["text"]))
            targets = [by_id[c] for c in r.get("covers", ()) if c in by_id] or ([fallback] if fallback else [])
            for step in targets:
                if r["id"] not in step.covers:
                    step.covers.append(r["id"])
        self.steps = steps
        self.intent = intent

    @classmethod
    def from_dict(cls, d: dict) -> "Ledger":
        """Rebuild a Ledger from ``to_dict()`` output (tolerant: missing keys default, unknown keys are ignored)."""
        d = d if isinstance(d, dict) else {}
        ledger = cls(str(d.get("intent") or "other"), str(d.get("request") or ""))
        for name, default in (("needs_reference", False), ("reference_hint", ""), ("reference_status", "none"),
                              ("round", 0), ("round_cap", 30), ("revision", 0), ("updated", ledger.updated),
                              ("validated_after_step", 0), ("last_geometry_step", 0), ("health_ok", True),
                              ("spec_ok", False)):
            setattr(ledger, name, d.get(name, default))
        try:
            ledger.version = int(d.get("version") or cls.VERSION)
        except (TypeError, ValueError):
            ledger.version = cls.VERSION
        ledger.open_problems = [str(x) for x in d.get("open_problems") or []]
        ledger.param_names = [str(x) for x in d.get("param_names") or []]
        ledger.waivers = [dict(w) for w in d.get("waivers") or [] if isinstance(w, dict)]
        step_fields = set(LedgerStep.__dataclass_fields__)
        for raw in d.get("steps") or []:
            if isinstance(raw, dict) and raw.get("id"):
                data = {k: v for k, v in raw.items() if k in step_fields}
                data.setdefault("title", str(raw.get("id")))
                data.setdefault("phase", "BUILD")
                for key in ("covers", "stages", "stages_done", "hint_tools", "evidence"):
                    data[key] = [dict(e) if isinstance(e, dict) else e for e in data.get(key) or []]
                ledger.steps.append(LedgerStep(**data))
        for raw in d.get("requirements") or []:
            if isinstance(raw, dict) and raw.get("id"):
                ledger.requirements.append(Requirement(id=str(raw["id"]), text=str(raw.get("text", ""))))
        ledger.max_phase = d.get("max_phase") if d.get("max_phase") in _PHASE_INDEX else "INTAKE"
        ledger._refresh()
        return ledger

    def to_dict(self) -> dict:
        """The ledger as JSON (plus the additive ``param_names``); derived fields are fresh."""
        self._refresh()
        return {
            "version": self.version, "intent": self.intent, "phase": self.phase, "max_phase": self.max_phase,
            "request": self.request, "needs_reference": self.needs_reference, "reference_hint": self.reference_hint,
            "reference_status": self.reference_status,
            "requirements": [
                {"id": r.id, "text": r.text, "status": r.status, "covered_by": list(r.covered_by),
                 "evidence": [dict(e) for e in r.evidence], "note": r.note} for r in self.requirements],
            "steps": [
                {"id": s.id, "title": s.title, "phase": s.phase, "acceptance": s.acceptance, "status": s.status,
                 "note": s.note, "owner": s.owner, "required": s.required, "covers": list(s.covers),
                 "stages": list(s.stages), "stages_done": list(s.stages_done), "hint_tools": list(s.hint_tools),
                 "claimed": s.claimed, "waived_reason": s.waived_reason, "template": s.template,
                 "evidence": [dict(e) for e in s.evidence]} for s in self.steps],
            "waivers": [dict(w) for w in self.waivers],
            "round": self.round, "round_cap": self.round_cap, "revision": self.revision, "updated": self.updated,
            "validated_after_step": self.validated_after_step, "last_geometry_step": self.last_geometry_step,
            "health_ok": self.health_ok, "open_problems": list(self.open_problems), "spec_ok": self.spec_ok,
            "param_names": list(self.param_names),
        }

    def to_json(self) -> str:
        """``json.dumps(to_dict(), ensure_ascii=False)``."""
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    # -- derived state --------------------------------------------------------------------------------
    def _refresh(self) -> None:
        self.phase = derive_phase(self)
        if _PHASE_INDEX[self.phase] > _PHASE_INDEX.get(self.max_phase, 0) and self.steps:
            self.max_phase = self.phase
        covering: dict[str, list[LedgerStep]] = {}
        for step in self.steps:
            for rid in step.covers:
                covering.setdefault(rid, []).append(step)
        for req in self.requirements:
            steps = covering.get(req.id, [])
            req.covered_by = [s.id for s in steps]
            req.status, req.note = self._derive_requirement(req, steps)
            seen, evidence = set(), []
            for s in steps:
                for e in s.evidence:
                    idx = e.get("step_index", -1)
                    if isinstance(idx, int) and idx >= 0 and (s.id, idx) not in seen:
                        seen.add((s.id, idx))
                        evidence.append({"step": s.id, "step_index": idx, "kind": e.get("kind", "auto")})
            req.evidence = evidence[-8:]

    def _derive_requirement(self, req: Requirement, steps: list[LedgerStep]) -> tuple[str, str]:
        waiver = next((w for w in reversed(self.waivers) if w.get("what") == req.id and w.get("status")), None)
        if waiver:
            return str(waiver["status"]), str(waiver.get("reason", ""))
        if not steps:
            return "pending", ""
        # A step the model merely CLAIMED done (no auto / confirmed evidence) does not make a requirement done (I2/I3).
        statuses = ["pending" if (s.status == "done" and s.claimed and not self._backed(s)) else s.status for s in steps]
        if all(st == "done" for st in statuses):
            return "done", ""
        blocked = next((s for s in steps if s.status == "blocked"), None)
        if blocked is not None:
            return "blocked", blocked.note
        if all(st in TERMINAL for st in statuses):
            changed = next((s for s in steps if s.status == "changed"), None)
            if changed is not None:
                return "changed", changed.note
            if "done" in statuses:
                return "done", ""
            skipped = next((s for s in steps if s.status == "skipped"), steps[0])
            return "skipped", skipped.note
        return "pending", ""

    def _bump(self) -> None:
        self.revision += 1
        self.updated = self._now()
        self._refresh()

    # -- queries ---------------------------------------------------------------------------------------
    def step(self, step_id: str) -> LedgerStep | None:
        """The step with this id (``None`` when absent)."""
        return next((s for s in self.steps if s.id == step_id), None)

    def requirement(self, req_id: str) -> Requirement | None:
        """The requirement with this id (``None`` when absent)."""
        return next((r for r in self.requirements if r.id == req_id), None)

    def current_step(self) -> LedgerStep | None:
        """The ``running`` step, else the first non-terminal REQUIRED step in ledger order (``None`` when none)."""
        running = next((s for s in self.steps if s.status == "running"), None)
        if running is not None:
            return running
        return next((s for s in self.steps if s.required and s.status not in TERMINAL), None)

    def pending_required(self) -> list[LedgerStep]:
        """Non-terminal required steps the MODEL owns (app-owned steps are completed by code and excluded)."""
        return [s for s in self.steps if s.required and s.owner == "model" and s.status not in TERMINAL]

    def has_open_required_steps(self) -> bool:
        """Any non-terminal required step, app-owned included."""
        return any(s.required and s.status not in TERMINAL for s in self.steps)

    def is_work_turn(self) -> bool:
        """``has_open_required_steps() or intent in WORK`` (state based, language independent)."""
        return self.has_open_required_steps() or self.intent in WORK

    def _app_blocked(self, step: LedgerStep) -> bool:
        return step.status == "blocked" and any(e.get("app_blocked") for e in step.evidence)

    def has_blocked_with_note(self) -> bool:
        """A required model-owned step the MODEL blocked / skipped with a reason: it legitimately needs the user.

        Steps the app skipped itself (a waiver, e.g. nominal board data) or blocked itself do not count.
        """
        return any(s.required and s.owner == "model" and s.status in ("blocked", "skipped") and s.note.strip()
                   and not s.waived_reason and not self._app_blocked(s) for s in self.steps)

    def claimed_only_steps(self) -> list[LedgerStep]:
        """Required steps that are ``done`` only because the model said so: ``claimed`` and no ``auto`` / ``confirmed`` evidence."""
        return [s for s in self.steps if s.required and s.status == "done" and s.claimed and not self._backed(s)]

    @staticmethod
    def _backed(step: LedgerStep) -> bool:
        return any(e.get("kind") in ("auto", "confirmed") for e in step.evidence)

    def is_waived(self, what: str) -> bool:
        """Whether ``what`` (a step id or a fit-row id) carries a waiver with a reason."""
        return any(w.get("what") == what and str(w.get("reason", "")).strip() for w in self.waivers)

    def counts(self) -> dict:
        """``{"done", "total", "pending", "blocked", "skipped"}`` over the REQUIRED steps (+ ``optional``: non-required steps)."""
        req = [s for s in self.steps if s.required]
        return {"done": sum(s.status == "done" for s in req), "total": len(req),
                "pending": sum(s.status in ("pending", "running") for s in req),
                "blocked": sum(s.status == "blocked" for s in req),
                "skipped": sum(s.status == "skipped" for s in req),
                "optional": len(self.steps) - len(req)}

    # -- rendering --------------------------------------------------------------------------------------
    _ICONS = {"done": "[x]", "running": "[~]", "pending": "[ ]", "blocked": "[!]", "skipped": "[-]", "changed": "[=]"}

    def render_compact(self, max_lines: int = 16) -> str:
        """The compact plan text (also the pinned task memory): header, one line per step, requirements.

        Lines beyond ``max_lines`` collapse to ``... +N more`` (already-finished steps are dropped first).
        """
        self._refresh()
        c = self.counts()
        cur = self.current_step()
        head = f"PLAN (intent {self.intent}, phase {self.phase}, {c['done']}/{c['total']} required done)"
        lines = []
        for s in self.steps:
            line = f"{self._ICONS.get(s.status, '[ ]')} {s.id} {s.title}"
            if s.status in ("done", "skipped") and s.note:
                line += f" - {s.note}"
            elif s.status in ("blocked", "changed"):
                line += f" - {s.status}: {s.note}" if s.note else f" - {s.status}"
            if cur is not None and s is cur:
                line += "   <- now"
            lines.append((s, line))
        reqs = ""
        if self.requirements:
            parts = []
            for r in self.requirements:
                if r.status in ("skipped", "changed", "blocked") and r.note:
                    detail = r.note.replace(" in the brief", "")[:40]
                else:
                    detail = ",".join(r.covered_by) or "no step"
                parts.append(f"{r.id} {r.status} ({detail})")
            reqs = "REQUIREMENTS: " + " | ".join(parts)
        redo = self.reverted_steps()
        budget = max(1, max_lines - 1 - (1 if reqs else 0) - (1 if redo else 0))
        shown = lines
        hidden = 0
        if len(lines) > budget:
            keep = max(0, budget - 1)                      # one line is the "... +N more" marker
            shown = lines[:]
            for item in list(shown):                       # finished steps go first, oldest first
                if len(shown) <= keep:
                    break
                if item[0].status in TERMINAL and item[0] is not cur:
                    shown.remove(item)
            while len(shown) > keep:                       # then the oldest line that is not the current step
                shown.pop(next((i for i, it in enumerate(shown) if it[0] is not cur), 0))
            hidden = len(lines) - len(shown)
        out = [head]
        if redo:
            out.append("NOTE: the model was reverted; these steps are open again and must be redone: "
                       + ", ".join(s.id for s in redo))
        if hidden:
            out.append(f"... +{hidden} more")
        out.extend(line for _s, line in shown)
        if reqs:
            out.append(reqs)
        return "\n".join(out)

    def resume_prompt(self) -> str:
        """The message the panel's ``continue:`` anchor sends: ``Continue the task. Plan status: ... Next: S7 "..."``."""
        c = self.counts()
        cur = self.current_step()
        text = f"Continue the task. Plan status: {c['done']}/{c['total']} required steps done."
        if cur is not None:
            text += f' Next: {cur.id} "{cur.title}".'
        return text

    # -- mutations ----------------------------------------------------------------------------------------
    def _evidence(self, outcome, kind: str = "auto") -> dict:
        delta = _g(outcome, "delta", None) or {}
        added = delta.get("added", []) if isinstance(delta, dict) else []
        entry = {"step_index": int(_g(outcome, "step_index", -1) if _g(outcome, "step_index", -1) is not None else -1),
                 "tool": str(_g(outcome, "name", "")), "title": str(_g(outcome, "title", ""))[:80],
                 "ts": self._now(), "objects": [str(a) for a in list(added)[:6]], "kind": kind}
        if _g(outcome, "ok", True) is False:
            entry["ok"] = False          # the call ran but left health errors: not confirmable until a later call fixes them
        return entry

    @staticmethod
    def _add_evidence(step: LedgerStep, entry: dict) -> None:
        step.evidence.append(entry)
        if len(step.evidence) > 40:
            step.evidence = step.evidence[-40:]

    def _open_for_evidence(self, s: LedgerStep) -> bool:
        return s.status not in TERMINAL or (s.status == "done" and s.claimed and not self._backed(s))

    def _demote_running(self, keep: LedgerStep) -> None:
        for other in self.steps:
            if other is not keep and other.status == "running":
                other.status = "pending"

    def _set_running(self, step: LedgerStep) -> None:
        if step.status == "pending":
            step.status = "running"
        if step.status == "running":
            self._demote_running(step)

    def _pick_step(self, tool: str, category: str, *, hint_only: bool = False) -> LedgerStep | None:
        """The ONE step a successful geometry call attaches to (5.5 rules 10, 2a, 2b, 2c).

        ``hint_only`` (a call that changes no geometry, such as ``enclosure_spec``): only a step whose ``hint_tools``
        name the tool is credited, so a spec call can never stand in for modelling work by phase matching.
        """
        cands = [s for s in self.steps if s.owner == "model" and s.required and self._open_for_evidence(s)]
        if not hint_only:
            fix = next((s for s in cands if s.id.startswith("F")), None)
            if fix is not None:
                return fix
        hit = next((s for s in cands if tool in s.hint_tools), None)
        if hit is not None or hint_only:
            return hit
        phases = {"source": ("SOURCE",), "params": ("PARAMS",), "build": ("BUILD",),
                  "fem": ("BUILD",), "session": ("BUILD",)}.get(category)
        if category == "plan":
            phases = tuple(sorted(("PARAMS", "BUILD"), key=lambda p: min(
                (i for i, s in enumerate(self.steps) if s.phase == p and s in cands), default=999)))[:1] or ("BUILD",)
        if phases:
            hit = next((s for s in cands if s.phase in phases), None)
            if hit is not None:
                return hit
        return next((s for s in cands if s.phase == "BUILD"), None)

    def _stage_done(self, step: LedgerStep, stage: str) -> bool:
        """Record a finished stage; completes the step when every stage is done. Returns True when anything changed."""
        changed = False
        if stage not in step.stages_done:
            step.stages_done.append(stage)
            changed = True
        if step.stages and all(s in step.stages_done for s in step.stages) and step.status != "done":
            step.status, step.claimed, changed = "done", False, True
        return changed

    def on_tool_result(self, outcome) -> bool:
        """Auto-tick from one recorded tool outcome; returns True when anything changed.

        Only calls whose HANDLER succeeded tick. A successful geometry call attaches evidence to ONE step
        (an open ``F`` fix step first, then the first step whose ``hint_tools`` names the tool, then by
        category phase, then the first BUILD step), and ``build_enclosure(stage=X)`` records stage ``X`` on every
        step that lists it. Reference tools set ``reference_status``; parameter tools complete the parameters step;
        a successful ``deliver`` tool attaches evidence to the DELIVER step. The first successful geometry call of a ``question`` / ``inspect`` / ``other`` turn
        escalates the ledger to ``generic_part`` BEFORE evidence is attached.
        """
        if not _g(outcome, "handler_ok", True):
            return False
        name = str(_g(outcome, "name", "") or "")
        category = str(_g(outcome, "category", "") or "") or tool_category(name)
        geometry = bool(_g(outcome, "geometry", False))
        args = _g(outcome, "args", None) or {}
        extra = _g(outcome, "extra", None) or {}
        changed = False
        if geometry and self.intent in ("question", "inspect", "other"):
            self.escalate("generic_part")
            changed = True
        if geometry:
            idx = _g(outcome, "step_index", -1)
            new_last = idx if isinstance(idx, int) and idx >= 0 else self.last_geometry_step + 1
            if new_last > self.last_geometry_step:
                self.last_geometry_step = new_last
                changed = True
        stage = str(args.get("stage") or "") if name == "build_enclosure" else ""
        if stage and stage != "next":
            changed |= self._tick_stage(outcome, stage)
        elif geometry or name in ("enclosure_spec", "set_parameters"):
            step = self._pick_step(name, category, hint_only=not geometry)
            if step is not None:
                self._add_evidence(step, self._evidence(outcome))
                if step.note == REVERTED_NOTE:
                    step.note = ""                         # redone after a revert
                self._set_running(step)
                changed = True
        if name == "source_reference_part":
            changed |= self._on_source(outcome, extra)
        elif name == "import_reference_part":
            changed |= self._on_import(outcome, args)
        elif name == "measure_reference_part":
            changed |= self._on_measure(outcome)
        if name in ("set_parameters", "enclosure_spec") or (name == "build_enclosure" and stage == "params"):
            changed |= self._on_params(extra)
        if category == "deliver":
            changed |= self._on_deliver(outcome)
        if changed:
            self._bump()
        return changed

    def _tick_stage(self, outcome, stage: str) -> bool:
        changed = False
        home = {"reference": "params"}.get(stage, stage)
        targets = [s for s in self.steps if stage in s.stages or (home != stage and home in s.stages)]
        for step in targets:
            if step.status == "done" and not step.claimed:
                if stage in step.stages and stage not in step.stages_done:
                    step.stages_done.append(stage)
                    changed = True
                continue
            self._add_evidence(step, self._evidence(outcome))
            if stage in step.stages:
                self._stage_done(step, stage)
            if step.status in ("pending", "skipped", "blocked"):
                step.status = "running"
                step.note = "" if step.status == "running" and step.waived_reason == "" else step.note
            if step.status == "running":
                self._demote_running(step)
            step.claimed = False if step.status == "done" else step.claimed
            changed = True
        if not targets:
            step = self._pick_step("build_enclosure", "build")
            if step is not None:
                self._add_evidence(step, self._evidence(outcome))
                self._set_running(step)
                changed = True
        return changed

    def _first_open(self, phase: str, owner: str | None = None) -> LedgerStep | None:
        return next((s for s in self.steps if s.phase == phase and s.required and s.status not in TERMINAL
                     and (owner is None or s.owner == owner)), None)

    def _on_source(self, outcome, extra: dict) -> bool:
        result = extra.get("reference_result") if isinstance(extra, dict) else None
        if not isinstance(result, dict):
            return False
        status = str(result.get("status", "")).lower()
        if status == "nominal":
            self.set_reference_status("catalog-nominal", "catalog-nominal")
            self._skip_import_step(NO_STEP_NOTE)
            return True
        if status == "obtained":
            sha = str(result.get("sha256", ""))[:12]
            self._finish_source_step(f"STEP file obtained (sha256 {sha})" if sha else "STEP file obtained", outcome)
            return True
        if status in ("declined", "failed"):
            if result.get("has_board") or result.get("board") or result.get("reference_status") == "catalog-nominal":
                self.set_reference_status("catalog-nominal", f"download {status}: nominal data used")
                self._skip_import_step(f"download {status}: nominal data used")
                return True
            s2 = next((s for s in self.steps if s.phase == "SOURCE" and s.owner == "model"
                       and s.hint_tools[:1] == ["find_reference_part"]), None)
            if s2 is not None and s2.status not in ("done", "skipped"):
                s2.status = "blocked"
                s2.note = str(result.get("reason") or f"the reference part could not be {status}: no board data in the catalog")[:300]
                self._add_evidence(s2, {"kind": "auto", "tool": "source_reference_part", "step_index": -1,
                                        "ts": self._now(), "objects": [], "title": s2.note[:80], "app_blocked": True})
                return True
        return False

    def _source_step(self) -> LedgerStep | None:
        return next((s for s in self.steps if s.phase == "SOURCE" and s.owner == "model"
                     and s.hint_tools[:1] == ["find_reference_part"]), None)

    def _finish_source_step(self, note: str, outcome=None) -> None:
        s2 = self._source_step()
        if s2 is not None and s2.status != "done":
            if outcome is not None:
                self._add_evidence(s2, self._evidence(outcome))
            s2.status, s2.note, s2.claimed = "done", note, False

    def _skip_import_step(self, note: str) -> None:
        s3 = next((s for s in self.steps if s.phase == "SOURCE" and s.owner == "model"
                   and "import" in s.stages and s.status not in ("done", "skipped")), None)
        if s3 is not None:
            s3.status, s3.note, s3.waived_reason, s3.claimed = "skipped", note, note, False
            self.waivers.append({"what": s3.id, "reason": note, "by": "app"})

    def _on_import(self, outcome, args: dict) -> bool:
        status = "user-attached" if args.get("path") else "step-imported"
        if self.reference_status in ("none", "catalog-nominal"):
            self.set_reference_status(status, "")
        for s in self.steps:
            if "import" in s.stages and s.phase == "SOURCE" and self._open_for_evidence(s):
                self._add_evidence(s, self._evidence(outcome))
                self._stage_done(s, "import")
                if s.status == "pending":
                    s.status = "running"
        return True

    def _on_measure(self, outcome) -> bool:
        if self.reference_status == "step-imported":
            self.reference_status = "step-measured"
        for s in self.steps:
            if "measure" in s.stages and s.phase == "SOURCE" and self._open_for_evidence(s):
                self._add_evidence(s, self._evidence(outcome))
                self._stage_done(s, "measure")
                if s.status == "pending":
                    s.status = "running"
        return True

    def _on_params(self, extra: dict) -> bool:
        changed = False
        if isinstance(extra, dict):
            names = extra.get("param_names")
            if isinstance(names, (list, tuple)) and list(names) != self.param_names:
                self.param_names = [str(n) for n in names]
                changed = True
            if extra.get("spec_ok") and not self.spec_ok:
                self.spec_ok = True
                changed = True
        if self.intent == "enclosure":
            s4 = next((s for s in self.steps if s.phase == "PARAMS" and s.owner == "model" and s.required
                       and s.status not in TERMINAL), None)
            if s4 is not None and self.spec_ok and all(n in self.param_names for n in ENCLOSURE_REQUIRED_PARAMS):
                s4.status, s4.claimed = "done", False
                s4.note = s4.note or "parameters and spec recorded"
                changed = True
        return changed

    def _on_deliver(self, outcome) -> bool:
        step = next((s for s in self.steps if s.phase == "DELIVER" and s.required and s.status not in TERMINAL), None)
        if step is None:
            return False
        self._add_evidence(step, self._evidence(outcome))
        if step.owner == "model":
            step.status, step.claimed = "done", False
        return True

    def set_reference_status(self, status: str, note: str = "") -> None:
        """Record the reference status; any status other than ``none`` completes the source step (``S2``)."""
        if status == self.reference_status and not note:
            return
        self.reference_status = status
        if status != "none":
            self._finish_source_step(note or status)
        self._bump()

    def escalate(self, intent: str = "generic_part") -> None:
        """A ``question`` / ``inspect`` / ``other`` ledger becomes ``intent`` and seeds that template (idempotent)."""
        if self.intent not in ("question", "inspect", "other") or intent not in WORK:
            return
        self._seed(intent)
        self._bump()

    def archive_waiver(self, what: str, reason: str, by: str) -> None:
        """Record a waiver (a step or fit-row id the app or the user decided to leave open) with its reason."""
        self.waivers.append({"what": what, "reason": reason, "by": by})
        self._bump()

    def confirm_evidenced_steps(self) -> list[str]:
        """Confirm recorded work at finish; returns the ids of the steps it completed (5.7 step 1).

        Every non-terminal REQUIRED model-owned step with at least one ``auto`` evidence entry, no unresolved
        error on its newest evidence and (for staged steps) every stage done becomes ``done`` with an evidence
        entry ``kind="confirmed"`` and note ``confirmed from recorded work``. A step the model marked ``done``
        itself (``claimed``) is confirmed only through ``auto`` evidence of its own: a claim alone never is.
        """
        confirmed = []
        for s in self.steps:
            if not (s.required and s.owner == "model") or s.status in TERMINAL:
                continue
            auto = [e for e in s.evidence if e.get("kind") == "auto" and not e.get("app_blocked")]
            if not auto or auto[-1].get("ok") is False:
                continue
            if s.stages and not all(st in s.stages_done for st in s.stages):
                continue
            last = auto[-1]
            self._add_evidence(s, {"step_index": last.get("step_index", -1), "tool": last.get("tool", ""),
                                   "title": "confirmed from recorded work", "ts": self._now(), "objects": [],
                                   "kind": "confirmed"})
            s.status, s.claimed = "done", False
            s.note = (s.note if s.note != REVERTED_NOTE else "") or "confirmed from recorded work"
            confirmed.append(s.id)
        if confirmed:
            self._bump()
        return confirmed

    def apply_validation(self, validation) -> None:
        """Fold a validation result into the ledger.

        Sets ``health_ok`` (no gate errors), ``validated_after_step`` and ``open_problems``. A passing result
        completes the app-owned VERIFY step(s) and closes an open ``F`` fix step; a failing one calls
        ``regress`` (idempotent while a fix step is open). ``None`` (no validator, or nothing to validate)
        completes the VERIFY step as ``skipped`` with the note ``automatic validation unavailable``: the turn
        makes no validation claim.
        """
        v = _validation_dict(validation)
        verify = [s for s in self.steps if s.phase == "VERIFY" and s.owner == "app" and s.required]
        if v is None:
            for s in verify:
                if s.status not in TERMINAL:
                    s.status, s.note = "skipped", "automatic validation unavailable"
                    s.waived_reason = s.note
            self.open_problems = []
            self._bump()
            return
        gate_errors = list(v.get("gate_errors") or [])
        fails = [r for r in (v.get("fail_rows") or []) if not self.is_waived(str(r.get("id", "")) if isinstance(r, dict) else str(r))]
        ok = not gate_errors and not fails
        self.health_ok = not gate_errors
        try:
            self.validated_after_step = int(v.get("after_step") or 0)
        except (TypeError, ValueError):
            self.validated_after_step = 0
        problems = []
        for e in gate_errors[:6]:
            problems.append(str(e.get("message") or e) if isinstance(e, dict) else str(e))
        for r in fails[:6]:
            rid = r.get("id") if isinstance(r, dict) else r
            detail = (r.get("detail") or r.get("message") or "") if isinstance(r, dict) else ""
            problems.append(f"FAIL {rid}: {detail}".strip().rstrip(":"))
        self.open_problems = problems
        if ok:
            for s in verify:
                if s.status != "done":
                    s.status, s.note = "done", "validation passed"
                    self._add_evidence(s, {"step_index": self.validated_after_step, "tool": "validation",
                                           "title": "validation passed", "ts": self._now(), "objects": [],
                                           "kind": "auto"})
            for s in self.steps:
                if s.id.startswith("F") and s.status not in TERMINAL:
                    s.status, s.claimed = "done", False
                    s.note = s.note or "validation passed"
                    self._add_evidence(s, {"step_index": self.validated_after_step, "tool": "validation",
                                           "title": "validation passed", "ts": self._now(), "objects": [],
                                           "kind": "confirmed"})
            self._bump()
        else:
            self.regress(problems[0] if problems else "validation failed")

    def mark_delivered(self, project_bound: bool = True) -> None:
        """Complete the app-owned DELIVER step(s); without a bound project they are ``skipped`` (``no project bound``)."""
        changed = False
        for s in self.steps:
            if s.phase == "DELIVER" and s.owner == "app" and s.required and s.status not in TERMINAL:
                if project_bound:
                    s.status, s.note = "done", s.note or "project saved"
                else:
                    s.status, s.note, s.waived_reason = "skipped", "no project bound", "no project bound"
                changed = True
        if changed:
            self._bump()

    def regress(self, reason: str) -> None:
        """A failed validation re-opens BUILD: VERIFY goes back to ``pending`` and a model-owned required BUILD step
        ``F<n>`` "Fix validation findings: <reason>" is inserted BEFORE it (``max_phase`` is never lowered).

        Idempotent while an ``F`` step is still open: it updates that step's note instead of inserting another.
        """
        reason = " ".join((reason or "validation failed").split())
        for s in self.steps:
            if s.phase == "VERIFY" and s.owner == "app" and s.required and s.status in TERMINAL:
                s.status, s.note = "pending", ""
        open_fix = next((s for s in self.steps if s.id.startswith("F") and s.status not in TERMINAL), None)
        if open_fix is not None:
            open_fix.note = reason[:300]
        else:
            n = 1 + max([int(s.id[1:]) for s in self.steps if s.id.startswith("F") and s.id[1:].isdigit()] or [0])
            fix = LedgerStep(id=f"F{n}", title=f"Fix validation findings: {reason[:80]}", phase="BUILD",
                             acceptance="validation passes: health valid and no FAIL fit rows", note=reason[:300],
                             hint_tools=["execute_code", "edit_object", "set_parameter"], template=False)
            pos = next((i for i, s in enumerate(self.steps) if s.phase == "VERIFY" and s.owner == "app"), len(self.steps))
            self.steps.insert(pos, fix)
        self._bump()

    def on_revert(self, restored_step: int | None = None) -> None:
        """The model was restored to an earlier state (a turn revert or a rewind to project step ``restored_step``).

        Pure bookkeeping, JSON round-trips. Every step with recorded evidence AFTER the restored point (``step_index`` greater
        than ``restored_step``; ``None`` = the start, so any recorded work) goes back to ``pending`` with the note ``reverted``
        and loses that evidence (and its stage ticks, claim and ``running`` flag); with ``None`` a step that is only CLAIMED
        done (no evidence) reverts too. Requirements and the phase are re-derived, ``last_geometry_step`` /
        ``validated_after_step`` are clamped to the restored point, the open problems cleared, and ``revision`` is bumped
        (always, so the panel redraws the plan). ``max_phase`` never shrinks: the visible tools only grow. Idempotent.
        """
        point = 0 if restored_step is None else max(0, int(restored_step))
        for s in self.steps:
            after = [e for e in s.evidence if isinstance(e.get("step_index"), int) and e["step_index"] > point]
            claimed_only = restored_step is None and s.status == "done" and s.claimed and not s.evidence
            if not after and not claimed_only:
                continue
            s.evidence = [e for e in s.evidence if e not in after]
            s.status, s.note, s.claimed, s.waived_reason = "pending", REVERTED_NOTE, False, ""
            s.stages_done = []
        self.last_geometry_step = min(self.last_geometry_step, point)
        if self.validated_after_step > point:
            self.validated_after_step = 0
        self.open_problems = []
        self.health_ok = True
        self._bump()

    def reverted_steps(self) -> list[LedgerStep]:
        """The steps ``on_revert`` reopened and nothing has redone yet."""
        return [s for s in self.steps if s.status == "pending" and s.note == REVERTED_NOTE]

    # -- plan edits by the model ---------------------------------------------------------------------------
    def plan_ids_text(self) -> str:
        """``S1..S12, F1, R1..R9``: the ids a model may use (for the unknown-id message)."""
        nums = [int(s.id[1:]) for s in self.steps if s.id[:1] == "S" and s.id[1:].isdigit()]
        parts = []
        if nums:
            parts.append(f"S{min(nums)}..S{max(nums)}" if min(nums) != max(nums) else f"S{nums[0]}")
        parts.extend(s.id for s in self.steps if not (s.id[:1] == "S" and s.id[1:].isdigit()))
        rn = [int(r.id[1:]) for r in self.requirements if r.id[:1] == "R" and r.id[1:].isdigit()]
        if rn:
            parts.append(f"R{min(rn)}..R{max(rn)}" if min(rn) != max(rn) else f"R{rn[0]}")
        return ", ".join(parts)

    def plan_update(self, item_id: str, status: str, note: str = "", evidence: str = "", *, by: str = "model") -> str:
        """Set the status of a step (``S..`` / ``F..``) or, for ``blocked`` / ``skipped`` / ``changed``, a requirement.

        Raises ``LedgerError`` with model-facing messages. ``done`` without app evidence stores
        ``claimed=True`` plus an evidence entry ``kind="claim"``; the VERIFY step and the finish gate stay the
        arbiter. Returns a confirmation text.
        """
        item_id = str(item_id or "").strip()
        status = str(status or "").strip().lower()
        note = " ".join(str(note or "").split())[:500]
        evidence = " ".join(str(evidence or "").split())[:500]
        step = self.step(item_id)
        req = None if step is not None else self.requirement(item_id)
        if step is None and req is None:
            raise LedgerError(f"plan_update: '{item_id}' is not in the plan. Plan ids: {self.plan_ids_text()}.")
        if status not in STEP_STATUSES:
            raise LedgerError(f"plan_update: status must be one of {'|'.join(STEP_STATUSES)}, got '{status}'.")
        if status in _NOTE_REQUIRED and not note:
            raise LedgerError("plan_update: 'note' (the reason) is required when status is 'blocked', 'skipped' or 'changed'.")
        if req is not None:
            return self._update_requirement(req, status, note, by)
        if step.owner == "app" and by == "model":
            if step.phase == "VERIFY":
                raise LedgerError(f'plan_update: {step.id} "{step.title}" is completed by the app after a passing validation; '
                                  "do not set it. Run validate_model / check_fit instead.")
            raise LedgerError(f'plan_update: {step.id} "{step.title}" is completed by the app when the turn finishes; do not set it.')
        was_done = step.status == "done"
        step.status = status
        if note or status in ("pending", "running", "done"):
            step.note = note or step.note
        if status in ("pending", "running"):
            step.claimed = False
            if status == "running":
                self._demote_running(step)
        if status == "done":
            if self._backed(step):
                step.claimed = False
            else:
                step.claimed = True
                if not was_done or evidence or note:
                    self._add_evidence(step, {"kind": "claim", "text": evidence or note or "marked done by the model",
                                              "ts": self._now(), "step_index": -1, "tool": "plan_update"})
        elif status in ("blocked", "skipped", "changed"):
            step.claimed = False
        self._bump()
        suffix = " (claimed: no recorded work behind it yet)" if status == "done" and step.claimed else ""
        return f'{step.id} "{step.title}" -> {status}{suffix}'

    def _update_requirement(self, req: Requirement, status: str, note: str, by: str) -> str:
        if status not in _NOTE_REQUIRED:
            raise LedgerError(f"plan_update: requirement '{req.id}' follows its steps ({', '.join(req.covered_by) or 'none'}); "
                              "update the step instead. On a requirement only blocked, skipped or changed (with a note) can be set.")
        self.waivers = [w for w in self.waivers if not (w.get("what") == req.id and w.get("status"))]
        self.waivers.append({"what": req.id, "reason": note, "by": by, "status": status})
        self._bump()
        return f"{req.id} -> {status}: {note}"

    def plan_set(self, requirements: list[dict] | None = None, steps: list[dict] | None = None, *,
                 replace: bool = False) -> list[str]:
        """Edit the plan: merge by default, ``replace=True`` retires the model-owned template steps.

        ``requirements``: ``[{id?, text, covers?}]`` (``covers`` = step ids; setting it replaces the requirement's
        coverage). ``steps``: ``[{id?, title, phase?, acceptance?, covers?}]`` (``covers`` = requirement ids).
        An id that exists updates that item, any other id (or none) adds a new one with a fresh id; a step's phase
        must be one of ``PHASES``. Everything is validated before anything is applied (a rejected call changes
        nothing). Returns warnings (ignored ids, protected app-owned steps).
        """
        requirements = list(requirements or [])
        steps = list(steps or [])
        warnings: list[str] = []
        if len(steps) > 40 or len(requirements) > 30:
            raise LedgerError("plan_set: too many items; send at most 40 steps and 30 requirements.")
        next_s = 1 + max([int(s.id[1:]) for s in self.steps if s.id[:1] == "S" and s.id[1:].isdigit()] or [0])
        for item in steps:
            if not isinstance(item, dict) or not str(item.get("title") or "").strip() and not item.get("id"):
                raise LedgerError("plan_set: every step needs a 'title'.")
        # -- validation first
        probe_s = next_s
        for item in steps:
            sid = str(item.get("id") or "")
            exists = self.step(sid) is not None
            if not exists and not str(item.get("title") or "").strip():
                raise LedgerError(f"plan_set: step '{sid}' is not in the plan; give a 'title' to add it. Plan ids: {self.plan_ids_text()}.")
            phase = item.get("phase")
            if phase is not None and str(phase) not in _PHASE_INDEX:
                shown = sid if exists else f"S{probe_s}"
                raise LedgerError(f"plan_set: step '{shown}' has phase '{phase}'; use one of {'|'.join(PHASES)}.")
            if not exists:
                probe_s += 1
        for item in requirements:
            if not isinstance(item, dict) or not (str(item.get("text") or "").strip() or self.requirement(str(item.get("id") or ""))):
                raise LedgerError("plan_set: every new requirement needs a 'text'.")
        # -- apply
        retired: list[LedgerStep] = []
        if replace:
            for s in self.steps:
                if s.owner == "model" and s.status not in TERMINAL:
                    s.status, s.note, s.required, s.claimed = "changed", "replaced by the model's plan", False, False
                    retired.append(s)
        step_map: dict[str, str] = {}
        req_map: dict[str, str] = {}
        next_r = 1 + max([int(r.id[1:]) for r in self.requirements if r.id[:1] == "R" and r.id[1:].isdigit()] or [0])
        req_targets: list[Requirement] = []        # the requirement each item of ``requirements`` ended up as (id-less items included)
        step_targets: list[LedgerStep] = []        # likewise for ``steps``
        for item in requirements:
            rid = str(item.get("id") or "")
            req = self.requirement(rid)
            text = " ".join(str(item.get("text") or "").split())[:300]
            if req is None:
                new_id = f"R{next_r}"
                next_r += 1
                req = Requirement(id=new_id, text=text)
                self.requirements.append(req)
                if rid:
                    req_map[rid] = new_id
            elif text:
                req.text = text
            req_targets.append(req)
        for item in steps:
            sid = str(item.get("id") or "")
            step = self.step(sid) if sid and not (replace and self.step(sid) in retired) else None
            title = " ".join(str(item.get("title") or "").split())[:120]
            if step is None:
                new_id = f"S{next_s}"
                next_s += 1
                step = LedgerStep(id=new_id, title=title, phase=str(item.get("phase") or "BUILD"),
                                  acceptance=" ".join(str(item.get("acceptance") or "").split())[:300], template=False)
                self._insert_step(step)
                if sid:
                    step_map[sid] = new_id
            elif step.owner == "app":
                warnings.append(f"{step.id} is completed by the app; only its 'covers' can be edited.")
            else:
                if title:
                    step.title = title
                if item.get("phase"):
                    step.phase = str(item["phase"])
                if item.get("acceptance"):
                    step.acceptance = " ".join(str(item["acceptance"]).split())[:300]
            step_targets.append(step)
        valid_r = {r.id for r in self.requirements}
        valid_s = {s.id for s in self.steps}
        for item, step in zip(steps, step_targets):
            if "covers" not in item:
                continue
            wanted = []
            for rid in _id_list(item.get("covers")):
                rid = req_map.get(str(rid), str(rid))
                if rid in valid_r:
                    wanted.append(rid)
                else:
                    warnings.append(f"requirement '{rid}' is not in the plan (ignored).")
            step.covers = wanted
        for item, target in zip(requirements, req_targets):
            if "covers" not in item:
                continue
            rid = target.id
            wanted_steps = []
            for sid in _id_list(item.get("covers")):
                sid = step_map.get(str(sid), str(sid))
                if sid in valid_s:
                    wanted_steps.append(sid)
                else:
                    warnings.append(f"step '{sid}' is not in the plan (ignored).")
            for s in self.steps:
                if rid in s.covers and s.id not in wanted_steps:
                    s.covers.remove(rid)
                if s.id in wanted_steps and rid not in s.covers:
                    s.covers.append(rid)
        for req in self.requirements:        # a new requirement nobody covers is verified at VERIFY
            if not any(req.id in s.covers for s in self.steps):
                verify = next((s for s in self.steps if s.phase == "VERIFY" and s.owner == "app"), None)
                if verify is not None:
                    verify.covers.append(req.id)
        self._bump()
        return warnings

    def _insert_step(self, step: LedgerStep) -> None:
        """Insert in phase order: after the last step whose phase is not later than the new step's phase."""
        idx = _PHASE_INDEX.get(step.phase, 4)
        pos = 0
        for i, s in enumerate(self.steps):
            if _PHASE_INDEX.get(s.phase, 4) <= idx:
                pos = i + 1
        self.steps.insert(pos, step)


# ---------------------------------------------------------------------------------------------------
# Phase machine


def derive_phase(ledger: "Ledger", state: "TurnState | None" = None) -> str:
    """THE phase definition: the phase of ``ledger.current_step()``; ``INTAKE`` without steps; ``DELIVER`` when
    every required step is terminal. The model cannot set it; ``state`` is accepted for symmetry and unused."""
    if not ledger.steps:
        return "INTAKE"
    cur = ledger.current_step()
    return cur.phase if cur is not None and cur.phase in _PHASE_INDEX else "DELIVER"


def phase_exit_met(ledger: "Ledger", phase: str, state: "TurnState | None" = None) -> tuple[bool, str]:
    """``(met, reason when not)``: the LEDGER part of the exit criterion of a phase.

    Every required step of the phase is terminal; INTAKE: requirements and steps exist; PLAN: every requirement
    has a covering step. Never looks at ``spec_ok`` or health (that is the gate's job): a model that cannot
    produce a valid spec must never stay stuck in a phase.
    """
    ledger._refresh()
    if phase == "INTAKE":
        ok = bool(ledger.requirements and ledger.steps)
        return ok, "" if ok else "no requirements or steps are recorded yet"
    if phase == "PLAN":
        missing = [r.id for r in ledger.requirements if not r.covered_by]
        if missing:
            return False, f"requirements without a covering step: {', '.join(missing)}"
    open_steps = [s.id for s in ledger.steps if s.phase == phase and s.required and s.status not in TERMINAL]
    if open_steps:
        return False, f"required steps still open: {', '.join(open_steps)}"
    return True, ""


def visible_phases(ledger: "Ledger") -> frozenset[str]:
    """The phases whose tools are offered this round: MONOTONIC (they only grow during a task).

    No steps (question / inspect / other, an empty plan): EVERY phase, so no intent decision can hide
    ``create_object`` / ``execute_code`` from a model that needs them. Otherwise every phase up to and including
    ``ledger.max_phase``, plus BUILD once ``max_phase`` reached PLAN.
    """
    if ledger is None or not ledger.steps:
        return frozenset(PHASES)
    idx = _PHASE_INDEX.get(ledger.max_phase, 0)
    phases = set(PHASES[: idx + 1])
    if idx >= _PHASE_INDEX["PLAN"]:
        phases.add("BUILD")
    return frozenset(phases)


# ---------------------------------------------------------------------------------------------------
# FinishGate


@dataclass
class TurnState:
    """What the agent has seen so far this turn; the finish gate reads it (it never mutates it)."""

    is_work_turn: bool = False
    geometry_ok: int = 0                 # geometry-tool calls this turn whose HANDLER succeeded
    deliver_ok: int = 0                  # successful category == "deliver" calls this turn
    open_errors: list[str] = field(default_factory=list)   # "<tool>: <first line>" of unresolved errors of the last tool round
    last_script_fail: str = ""           # failure line of the last successful execute_code output, "" if none or superseded
    validation: dict | None = None       # ValidationResult-like dict
    last_geometry_step: int = 0
    gate_nudges: dict[str, int] = field(default_factory=dict)   # reason code -> count so far
    nudges_total: int = 0


@dataclass
class GateVerdict:
    """The outcome of ``finish_gate``: pass, or what is missing and how to push the model on."""

    ok: bool
    codes: list[str] = field(default_factory=list)       # failing rule codes in rule order
    reasons: list[str] = field(default_factory=list)     # human lines (also the NOT VERIFIED bullets)
    nudge: str = ""                                      # message to inject ("" when ok)
    tool_choice: str | None = None                       # None | "<tool name>" | "required" (second NO_WORK nudge)
    outcome_hint: str = ""                               # "" | "needs_input" | "claims"
    exhausted: bool = False                              # budget spent: finish with unverified / failed


GATE_CODES = ("NO_WORK", "DELIVERABLE_MISSING", "OPEN_ERRORS", "HEALTH_FAIL", "SCRIPT_FAIL", "FIT_FAIL",
              "STEPS_PENDING", "CLAIM_ONLY", "REFERENCE_UNRESOLVED")


def _step_label(s: LedgerStep) -> str:
    return f'{s.id} "{s.title}"'


def _pending_detail(s: LedgerStep) -> str:
    missing = [st for st in s.stages if st not in s.stages_done]
    auto = [e for e in s.evidence if e.get("kind") == "auto"]
    if auto and auto[-1].get("ok") is False:
        return "has work that left errors (fix them first)"
    if s.stages and s.stages_done and missing:
        return f"has stages still to build: {', '.join(missing)}"
    if auto:
        return "has recorded work but is not finished"
    return "has no recorded work"


def finish_gate(ledger: "Ledger", state: TurnState, cfg: "AgentConfig", *, reply_nonempty: bool = True,
                offered: Iterable[str] | None = None) -> GateVerdict:
    """Evaluate the finish rules for a reply without tool calls (5.7 steps 3-4). Pure: counters are the agent's.

    ``state.gate_nudges`` / ``nudges_total`` say how often each code was nudged already; the verdict is
    ``exhausted`` when ``cfg.gate_nudges_max`` nudges were used or a failing code was nudged twice. ``reply_nonempty``
    lets a work turn whose model blocked a step with a reason and made no change end ``needs_input`` instead of
    being nudged; ``offered`` (tool names offered this round) restricts the forced tool of the second ``NO_WORK`` nudge.
    """
    work = bool(state.is_work_turn)
    if not work:
        return GateVerdict(ok=True)
    intent = ledger.intent
    produced = state.geometry_ok + state.deliver_ok
    if produced == 0 and reply_nonempty and ledger.has_blocked_with_note():
        return GateVerdict(ok=True, outcome_hint="needs_input")
    codes: list[str] = []
    reasons: list[str] = []
    bullets: list[str] = []
    cur = ledger.current_step()
    now = _step_label(cur) if cur is not None else "the next open step"
    validation = state.validation if isinstance(state.validation, dict) else None

    def add(code: str, reason: str, bullet: str) -> None:
        codes.append(code)
        reasons.append(reason)
        bullets.append(f"- {code}: {bullet}")

    owes_work = bool(ledger.pending_required() or ledger.claimed_only_steps()
                     or max(state.last_geometry_step, ledger.last_geometry_step) == 0)
    if (state.geometry_ok == 0 and intent != "deliver" and (intent in WORK or ledger.has_open_required_steps())
            and owes_work):
        add("NO_WORK", "the model made no change in the document although the request needs geometry",
            f"you made no change in the model; this request needs geometry (next: {now})")
    if intent == "deliver" and state.deliver_ok == 0:
        text = "the requested file was not produced (no successful export_model / save_document)"
        add("DELIVERABLE_MISSING", text, text)
    if state.open_errors:
        listing = "; ".join(state.open_errors[:3])
        add("OPEN_ERRORS", f"unresolved errors from the last tool calls: {listing}",
            f"unresolved errors from your last tool calls: {listing}")
    if validation is not None and validation.get("gate_errors"):
        msgs = []
        for e in list(validation["gate_errors"])[:3]:
            msgs.append(str(e.get("message") or e) if isinstance(e, dict) else str(e))
        text = "; ".join(msgs)
        add("HEALTH_FAIL", text, text)
    if state.last_script_fail:
        add("SCRIPT_FAIL", f"the last script printed a failure line: '{state.last_script_fail}'",
            f"your last script printed a failure line: '{state.last_script_fail}'. Treat it as a failed check and fix the cause")
    if validation is not None:
        fails = [r for r in (validation.get("fail_rows") or [])
                 if not ledger.is_waived(str(r.get("id", "")) if isinstance(r, dict) else str(r))]
        if fails:
            ids = ", ".join(str(r.get("id") if isinstance(r, dict) else r) for r in fails[:6])
            add("FIT_FAIL", f"check_fit reports FAIL: {ids}",
                f"check_fit reports FAIL: {ids}. Fix the geometry or the parameter (set_parameter) and run check_fit again")
    pending = ledger.pending_required()
    if pending and not (intent == "deliver" and "DELIVERABLE_MISSING" in codes):
        listing = "; ".join(f"{_step_label(s)} {_pending_detail(s)}" for s in pending[:6])
        more = f" (+{len(pending) - 6} more)" if len(pending) > 6 else ""
        add("STEPS_PENDING", listing + more, f"{listing}{more}.")
    claimed = ledger.claimed_only_steps()
    if claimed:
        ids = ", ".join(s.id for s in claimed)
        add("CLAIM_ONLY", f"{ids} claimed without recorded work",
            f'{ids} were marked done without any recorded work; do the work with tools, or call '
            'plan_update(id, "skipped" or "blocked", note="<reason>")')
    if ledger.needs_reference and ledger.reference_status == "none":
        s2 = next((s for s in ledger.steps if s.phase == "SOURCE" and s.hint_tools[:1] == ["find_reference_part"]), None)
        if s2 is not None and (s2.status not in TERMINAL or ledger._app_blocked(s2)):
            add("REFERENCE_UNRESOLVED", "the reference part has no recorded status",
                "the reference part has no recorded status; call source_reference_part, or plan_update S2 with a reason")
    if not codes:
        return GateVerdict(ok=True)
    nudged = state.gate_nudges or {}
    exhausted = state.nudges_total >= cfg.gate_nudges_max or any(nudged.get(c, 0) >= 2 for c in codes)
    verdict = GateVerdict(ok=False, codes=codes, reasons=reasons, exhausted=exhausted)
    if exhausted:
        verdict.outcome_hint = "claims" if "CLAIM_ONLY" in codes else ""
        return verdict
    step_codes = {"NO_WORK", "STEPS_PENDING", "CLAIM_ONLY", "REFERENCE_UNRESOLVED"}
    lines = ["[finish-gate] You stopped, but the app found unfinished work:", *bullets]
    if cur is not None and step_codes.intersection(codes):
        lines.append(
            f"Continue with {cur.id} now by calling the tools. If the geometry for a step already exists, call "
            'plan_update(id, "done", note="<where it is>"); if a step is impossible call plan_update(id, "blocked" or '
            '"skipped", note="<reason>"). Do not write a summary yet.')
    else:
        lines.append("Fix the items above by calling the tools. Do not write a summary yet.")
    verdict.nudge = "\n".join(lines)
    if "NO_WORK" in codes and nudged.get("NO_WORK", 0) >= 1:
        names = set(offered) if offered is not None else None
        hints = list(cur.hint_tools) if cur is not None else []
        verdict.tool_choice = next((t for t in hints if names is None or t in names), "required")
    return verdict


# ---------------------------------------------------------------------------------------------------
# Outcome texts, governor line, configuration: private module ``flow_run`` (kept apart to keep this file small)
from .flow_run import (NO_CHANGES_TEXT, NOT_VERIFIED_FOOTER, NOT_VERIFIED_HEADER, AgentConfig,  # noqa: E402,F401
                       compose_capped, compose_failed, compose_needs_help, compose_unverified, format_governor,
                       health_label, phase_guidance)
