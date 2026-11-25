"""§16 user clarification and approval contract (master spec v1.2): pure decision logic.

NO FreeCAD imports — this module decides *when* to ask, builds the rule-4 question
payload, persists answers and resolves a trigger headlessly. Rendering a dialog and
persisting sheet cells are the integration's job (the pipeline hooks in later).

Owner policy (spec §16.1, verbatim core rule): "Oscar must measure the scan using
FreeCAD before asking the user for dimensions. Asking users to manually provide every
dimension is not the default workflow."  Every trigger below fires only when the
ANSWER would change construction; a recorded answer is reused and never re-asked
(spec §16.1 rule 5); a basis-``none`` value is deferred, never invented (§15.2).

State schema (all keys optional; the integration fills them from the survey):
``units``           the ``recon.units.analyse_units`` report (``status`` at minimum)
``scan``            the ``recon.quality.scan_quality`` report (``readiness.verdict``)
``holes``           survey hole records: ``id``/``parameter``, ``through`` (True/False/None),
                    ``depth_basis``, ``depth``, ``diameter``, ``confidence``, ``tolerance``
``classification``  survey classification + ``confidence`` (0..1) + ``alternatives`` names
``deltas``          proposed-vs-measured rows: ``parameter``, ``measured``, ``proposed``,
                    ``tolerance``, ``confidence``, ``basis``, ``critical``, ``wear``,
                    ``evidence``
``datums``          the ``recon.datums.propose_datums`` report (``ambiguities`` list)
``routes``          candidate routes: ``name``, ``native_parametric``, ``coverage_estimate``
"""

import json
import os
import time

from .units import GATE_STATUSES, POSSIBLE_MM_INCH_MISMATCH, SCALE_AMBIGUOUS

# The D11 ask-only-when list (spec §16.1 rule 3), in owner order.
RULE_UNITS = "units_undeterminable"
RULE_INTERPRETATION = "ambiguous_interpretation"
RULE_WORN = "worn_feature"
RULE_THROUGH_BLIND = "through_blind_uncertain"
RULE_DATUMS = "base_direction_symmetry_ambiguous"
RULE_ROUTES = "materially_different_routes"
RULE_TOLERANCE = "out_of_tolerance_proposal"
RULE_CONFIDENCE = "low_confidence_critical"
RULES = (RULE_UNITS, RULE_INTERPRETATION, RULE_WORN, RULE_THROUGH_BLIND, RULE_DATUMS,
         RULE_ROUTES, RULE_TOLERANCE, RULE_CONFIDENCE)

RULE_TEXT = {
    RULE_UNITS: "file units cannot be reliably determined",
    RULE_INTERPRETATION: "multiple geometric interpretations are plausible",
    RULE_WORN: "a feature is worn/damaged/incomplete and its intended dimension cannot "
               "be established from the measurement alone",
    RULE_THROUGH_BLIND: "through-hole versus blind-hole intent cannot be determined",
    RULE_DATUMS: "base face, extrusion direction or symmetry intent is ambiguous",
    RULE_ROUTES: "multiple reconstruction routes have materially different outcomes",
    RULE_TOLERANCE: "a proposed dimension exceeds the declared tolerance",
    RULE_CONFIDENCE: "a critical measurement has insufficient confidence",
}

# The rule-4 three-choice contract (spec §16.1 rule 4, §16.2 row 9).
CHOICE_ACCEPT = "accept"
CHOICE_RETAIN = "retain_measured"
CHOICE_OTHER = "other_value"
CHOICES = (CHOICE_ACCEPT, CHOICE_RETAIN, CHOICE_OTHER)

CLASSIFICATION_CONFIDENCE_MIN = 0.5   # below this + alternatives = ambiguous interpretation
CRITICAL_CONFIDENCE_MIN = 0.5         # critical measurements must reach this confidence
ROUTE_COVERAGE_GAP = 0.2              # candidate coverage estimates differing by >= this
NUMERIC_EPS = 1e-9                    # a proposal that equals the measurement changes nothing

CONFIDENCE_WORDS = {"high": 0.9, "medium": 0.6, "low": 0.3}

# §18.3: approval authorizes a declared fallback; it never upgrades the coverage class
# or the outcome ceiling.
OUTCOME_CEILING_FALLBACK = "BEST AVAILABLE APPROXIMATION"
SCOPE_FALLBACK_ROUTE = "fallback_route"

UNIT_FACTORS_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "inch": 25.4, "in": 25.4}


# ---------------------------------------------------------------------------
# helpers


def _confidence_number(confidence):
    """A uniform 0..1 float for numeric or word confidence; None when absent."""
    if confidence is None:
        return None
    if isinstance(confidence, bool):
        return None
    if isinstance(confidence, (int, float)):
        return float(confidence)
    word = CONFIDENCE_WORDS.get(str(confidence).strip().lower())
    return word if word is not None else None


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
        else None


def _lines(*items):
    """Flatten the given evidence items into plain strings (dicts render as key: value)."""
    out = []
    for item in items:
        if not item:
            continue
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.extend(f"{k}: {v}" for k, v in item.items() if v not in (None, "", [], ()))
        elif isinstance(item, (list, tuple)):
            out.extend(str(i) for i in item if i not in (None, "", [], ()))
        else:
            out.append(str(item))
    return out


def _fmt(value):
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_fmt(v)}" for k, v in value.items())
    return str(value)


def unit_factor(unit):
    """mm-per-unit factor of a confirmed unit answer (``"inch"`` → 25.4); None if unknown."""
    return UNIT_FACTORS_MM.get(str(unit or "").strip().lower())


def _trigger(rule, parameter, *, reason, evidence, threshold, actual, measured=None,
             proposal=None, tolerance=None, confidence=None, extra=None):
    record = {"rule": rule, "parameter": parameter, "reason": reason,
              "evidence": _lines(evidence), "threshold": threshold, "actual": actual,
              "measured": measured, "proposal": proposal, "tolerance": tolerance,
              "confidence": confidence}
    if extra:
        record.update(extra)
    return record


# ---------------------------------------------------------------------------
# trigger detection


def approval_triggers(state):
    """Trigger records for every D11 rule the state's evidence fails (spec §16.3).

    A trigger fires ONLY when the user's answer would change construction — a failing
    rule with no construction-changing question (e.g. a basis-``none`` feature, which
    §15.2 defers rather than asks about) produces no record. Records come back in the
    owner's rule order (``RULES``), input order within a rule."""
    state = state or {}
    out = []

    # 1. units undeterminable — the mm/inch gate statuses are exactly the ones the
    #    units module says need user confirmation (§16.2 row 6).
    units_state = state.get("units") or {}
    status = units_state.get("status")
    if status in GATE_STATUSES:
        dims = units_state.get("dims_source") or units_state.get("dims_mm") or []
        out.append(_trigger(
            RULE_UNITS, "units",
            reason="units undeterminable from the file and its dimensions alone",
            evidence=[units_state.get("source_unit_note"), units_state.get("evidence")],
            threshold="a non-gating unit status (UNIT_KNOWN / SCALE_PLAUSIBLE)",
            actual=status,
            measured={"status": status, "dims": dims},
            extra={"dims": dims, "unit_status": status}))

    # 2. ambiguous interpretation — classification confidence low AND named alternatives.
    cls = state.get("classification") or {}
    cls_conf = _confidence_number(cls.get("confidence"))
    alternatives = [str(a) for a in (cls.get("alternatives") or []) if a]
    if cls_conf is not None and cls_conf < CLASSIFICATION_CONFIDENCE_MIN and alternatives:
        out.append(_trigger(
            RULE_INTERPRETATION, "classification",
            reason=f"classification confidence {cls_conf:.2f} below "
                   f"{CLASSIFICATION_CONFIDENCE_MIN} with plausible alternatives",
            evidence=[f"alternatives: {', '.join(alternatives)}", cls.get("evidence")],
            threshold=f"classification confidence >= {CLASSIFICATION_CONFIDENCE_MIN}",
            actual=round(cls_conf, 4),
            measured=cls.get("class"),
            proposal=alternatives[0],          # the runner-up interpretation, offered as the switch
            confidence=cls_conf))

    # 3. worn/incomplete feature WITH an inferred proposal — accept vs retain changes B.
    #    basis none (nothing establishable) defers instead (§15.2, U7).
    for delta in state.get("deltas") or []:
        wear = delta.get("wear")
        measured = _number(delta.get("measured"))
        proposed = _number(delta.get("proposed"))
        if not wear or delta.get("basis") != "inferred" or proposed is None:
            continue
        if measured is None or abs(proposed - measured) <= NUMERIC_EPS:
            continue    # nothing to choose between — the answer would not change anything
        out.append(_trigger(
            RULE_WORN, delta.get("parameter") or "?",
            reason="the feature is worn/out-of-round, so the measured value is not the "
                   "design intent; a standard size is PROPOSED, never silently applied",
            evidence=[_lines(f"{wear.get('kind', 'wear')}: "
                             f"{_fmt(wear.get('magnitude_mm', wear.get('magnitude')))}"
                             if isinstance(wear, dict) else wear),
                      delta.get("evidence")],
            threshold="measured value is the design intent (no wear evidence)",
            actual=wear if isinstance(wear, dict) else {"wear": wear},
            measured=measured, proposal=proposed,
            tolerance=delta.get("tolerance"),
            confidence=_confidence_number(delta.get("confidence")),
            extra={"proposal_basis": "inferred"}))

    # 4. through-vs-blind — an undetermined `through` or a blind depth that rests on the
    #    wall extent alone (no floor plane seen) is exactly the near-through blind case.
    for hole in state.get("holes") or []:
        through = hole.get("through")
        basis = str(hole.get("depth_basis") or "")
        if through is True:
            continue
        if through is not None and "no floor plane seen" not in basis:
            continue    # blind with a seen floor — established, no question
        param = hole.get("parameter") or hole.get("id") or "hole"
        out.append(_trigger(
            RULE_THROUGH_BLIND, param,
            reason="the bore wall stops without a floor plane, so through-vs-blind "
                   "intent cannot be determined from the scan",
            evidence=[f"depth_basis: {basis or 'not established'}",
                      f"through: {_fmt(through) if through is not None else 'undetermined'}",
                      (state.get("scan") or {}).get("readiness", {}).get("reasons")],
            threshold="a floor plane closing the bore (blind) or a second exit face (through)",
            actual=basis or "no depth evidence",
            measured={"diameter": hole.get("diameter"), "depth": hole.get("depth"),
                      "depth_basis": basis or None, "through": through},
            proposal={"kind": "blind", "depth": hole.get("depth")} if through is False else None,
            tolerance=hole.get("tolerance"),
            confidence=_confidence_number(hole.get("confidence"))))

    # 5. base/direction/symmetry ambiguity — the datums proposal's own ambiguity flags.
    datums = state.get("datums") or {}
    ambiguities = [str(a) for a in (datums.get("ambiguities") or []) if a]
    if ambiguities:
        out.append(_trigger(
            RULE_DATUMS, "datums",
            reason="the 3-2-1 datum proposal carries ambiguity flags, so base face / "
                   "extrusion direction / symmetry intent is not established",
            evidence=ambiguities,
            threshold="an unambiguous 3-2-1 datum frame",
            actual=f"{len(ambiguities)} ambiguity flag(s)",
            measured="datum proposal (largest plane / dominant axis)"))

    # 6. materially different routes — a faceted fallback beside a parametric route
    #    (editability differs), or coverage estimates far apart.
    routes = [r for r in (state.get("routes") or []) if isinstance(r, dict)]
    if len(routes) >= 2:
        parametric = {bool(r.get("native_parametric")) for r in routes}
        coverages = [_number(r.get("coverage_estimate")) for r in routes]
        coverages = [c for c in coverages if c is not None]
        gap = (max(coverages) - min(coverages)) if len(coverages) == len(routes) else None
        mixed = parametric == {True, False}
        if mixed or (gap is not None and gap >= ROUTE_COVERAGE_GAP):
            extra = {"authorization_scope": SCOPE_FALLBACK_ROUTE} if mixed else {}
            out.append(_trigger(
                RULE_ROUTES, "route",
                reason="candidate routes differ materially (native parametric vs "
                       "faceted fallback and/or coverage estimates far apart)",
                evidence=[f"{r.get('name', '?')}: native_parametric="
                          f"{bool(r.get('native_parametric'))}, coverage="
                          f"{_fmt(r.get('coverage_estimate'))}" for r in routes],
                threshold=f"candidates equivalent (same parametric-ness, coverage gap "
                          f"< {ROUTE_COVERAGE_GAP})",
                actual=f"parametric-ness mixed: {sorted(str(p) for p in parametric)}"
                       if mixed else f"coverage gap {gap:.2f}",
                measured={"candidates": [r.get("name") for r in routes]},
                extra=extra))

    for delta in state.get("deltas") or []:
        param = delta.get("parameter") or "?"
        measured = _number(delta.get("measured"))
        proposed = _number(delta.get("proposed"))
        tolerance = _number(delta.get("tolerance"))

        # 7. out-of-tolerance proposal — |proposed - measured| beyond the declared tolerance.
        if None not in (measured, proposed, tolerance):
            deviation = abs(proposed - measured)
            if deviation > tolerance + NUMERIC_EPS:
                out.append(_trigger(
                    RULE_TOLERANCE, param,
                    reason=f"the proposal deviates {deviation:.4g} from the measurement, "
                           f"beyond the declared tolerance {tolerance:.4g}",
                    evidence=delta.get("evidence"),
                    threshold=tolerance, actual=round(deviation, 6),
                    measured=measured, proposal=proposed, tolerance=tolerance,
                    confidence=_confidence_number(delta.get("confidence")),
                    extra={"proposal_basis": delta.get("basis")}))

        # 8. low-confidence critical measurement — the user's confirmed value would
        #    replace a measurement Oscar itself distrusts.
        conf = _confidence_number(delta.get("confidence"))
        if delta.get("critical") and conf is not None and conf < CRITICAL_CONFIDENCE_MIN:
            out.append(_trigger(
                RULE_CONFIDENCE, param,
                reason=f"a critical measurement carries confidence {conf:.2f}, below "
                       f"{CRITICAL_CONFIDENCE_MIN}",
                evidence=delta.get("evidence"),
                threshold=f"critical-measurement confidence >= {CRITICAL_CONFIDENCE_MIN}",
                actual=round(conf, 4),
                measured=measured, proposal=proposed,
                tolerance=delta.get("tolerance"), confidence=conf))

    return out


def pending_questions(triggers):
    """The questions actually worth asking: the FIRST trigger per parameter (spec §16.1
    rule 5, "group related questions when practical") — later rules on the same parameter
    stay in the log but never produce a second dialog."""
    seen, out = set(), []
    for trigger in triggers:
        key = trigger.get("parameter")
        if key in seen:
            continue
        seen.add(key)
        out.append(trigger)
    return out


# ---------------------------------------------------------------------------
# the rule-4 question payload


def _interpretations(dims):
    """All unit readings of the source dimensions, normalized per idiom (spec §16.2)."""
    out = []
    for unit, factor in (("mm", 1.0), ("cm", 10.0), ("inch", 25.4)):
        values = [round(d / factor, 4) for d in dims]
        out.append({"unit": unit, "dims": values,
                    "label": f"{unit} — " + " × ".join(f"{v:g}" for v in values) + f" {unit}"})
    return out


def _unit_recommendation(unit_status):
    if unit_status == POSSIBLE_MM_INCH_MISMATCH:
        return "inch — the dimensions sit exactly on the inch fraction grid, not the mm grid"
    if unit_status == SCALE_AMBIGUOUS:
        return "mm — FreeCAD's native unit; both grids fit, the evidence cannot separate them"
    return "mm"


def build_question(trigger):
    """The rule-4 payload: measured, proposal, why clarification is needed, tolerance,
    confidence, evidence and the accept / retain-measured / other-value options."""
    trigger = trigger or {}
    parameter = trigger.get("parameter")
    rule = trigger.get("rule")
    measured, proposal = trigger.get("measured"), trigger.get("proposal")
    payload = {
        "parameter": parameter,
        "rule": rule,
        "title": f"Confirm {parameter}",
        "reason": f"{RULE_TEXT.get(rule, rule)} "
                  f"(needs {trigger.get('threshold')}; actual {trigger.get('actual')})",
        "measured": measured,
        "proposal": proposal,
        "deviation": None,
        "tolerance": trigger.get("tolerance"),
        "confidence": trigger.get("confidence"),
        "proposal_basis": trigger.get("proposal_basis"),
        "evidence": list(trigger.get("evidence") or []),
        "options": [],
    }
    m_num, p_num = _number(measured), _number(proposal)
    if m_num is not None and p_num is not None:
        payload["deviation"] = round(p_num - m_num, 6)
    if rule == RULE_UNITS:
        dims = trigger.get("dims") or []
        payload["interpretations"] = _interpretations(dims)
        payload["recommendation"] = _unit_recommendation(trigger.get("unit_status"))
        payload["options"] = [{"key": i["unit"], "label": i["label"]} for i in
                              payload["interpretations"]]
        payload["options"].append({"key": CHOICE_OTHER, "label": "Provide another unit"})
        return payload
    options = []
    if proposal is not None:
        options.append({"key": CHOICE_ACCEPT,
                        "label": f"Accept the proposal ({_fmt(proposal)})"})
    if measured is not None:
        options.append({"key": CHOICE_RETAIN,
                        "label": f"Retain the measured value ({_fmt(measured)})"})
    options.append({"key": CHOICE_OTHER, "label": "Provide another value"})
    payload["options"] = options
    if trigger.get("authorization_scope"):
        payload["authorization_scope"] = trigger["authorization_scope"]
    return payload


# ---------------------------------------------------------------------------
# answers


def row_from_delta(delta):
    """The §15 F–K row data of a proposed-vs-measured delta BEFORE any user answer:
    basis ``none`` defers (never invented); an inferred/other proposal is status
    ``proposed`` (never silently applied); nothing here touches the B value."""
    delta = delta or {}
    measured, proposed = _number(delta.get("measured")), _number(delta.get("proposed"))
    basis = delta.get("basis") or ("none" if measured is None else "measured")
    deviation = round(proposed - measured, 6) if None not in (measured, proposed) else None
    status = "deferred" if basis == "none" else "proposed"
    return {"parameter": delta.get("parameter") or "?", "measured": measured,
            "deviation": deviation, "tolerance": delta.get("tolerance"),
            "confidence": delta.get("confidence"), "basis": basis, "status": status}


def answer_outcome(row, answer):
    """Apply a recorded answer to a parameter row → the correct F/B/J/K outcome.

    accept → B = proposal; retain_measured → B = measured; other_value → B = the given
    value.  In all three the user supplied or confirmed the value, so J = ``user`` and
    K = ``approved`` (§15.2: approved ONLY via a recorded approval event)."""
    row, answer = row or {}, answer or {}
    parameter = row.get("parameter") or "?"
    choice = answer.get("choice")
    measured, proposal = _number(row.get("measured")), row.get("proposal")
    if proposal is None:
        proposal = row.get("proposed")
    disposition = None
    if choice == CHOICE_ACCEPT:
        if proposal is None:
            raise ValueError(f"{parameter}: nothing to accept — no proposal on record")
        value = proposal
        disposition = "accepted"
    elif choice == CHOICE_RETAIN:
        if measured is None:
            raise ValueError(f"{parameter}: no measured value to retain")
        value = measured
        disposition = "rejected" if proposal is not None else "confirmed"
    elif choice == CHOICE_OTHER:
        value = answer.get("value")
        if value is None:
            raise ValueError(f"{parameter}: 'other_value' requires a value")
        disposition = "superseded" if proposal is not None else "confirmed"
    else:
        raise ValueError(f"{parameter}: unknown answer choice {choice!r}")
    value_num = _number(value)
    deviation = round(value_num - measured, 6) \
        if value_num is not None and measured is not None else None
    return {"parameter": parameter, "value": value, "measured": measured,
            "deviation": deviation, "tolerance": row.get("tolerance"),
            "confidence": row.get("confidence"), "basis": "user", "status": "approved",
            "proposal_disposition": disposition}


def _safe_id(text):
    safe = "".join(c if c.isalnum() or c in "-_." else "_"
                   for c in str(text or ""))[:80].strip("_")
    return safe or "default"


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class AnswerStore:
    """Answers persisted per (project, parameter) in ``<directory>/<project>/
    reconstruction.json`` (spec §15.2: approved answers keyed by parameter id, reused
    across revisions without re-asking). One project's file is never read for another:
    project (and therefore chat — one Project folder per chat) isolation is the file
    layout itself. The directory is injectable; the JSON is written atomically."""

    def __init__(self, directory, project_id):
        self.directory = str(directory)
        self.project_id = str(project_id)
        self.path = os.path.join(self.directory, _safe_id(self.project_id),
                                 "reconstruction.json")
        self._data = None

    # -- persistence

    def _load(self):
        if self._data is None:
            data = {"schema": 1, "project": self.project_id, "answers": {}, "triggers": []}
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    data.update(loaded)
                    data.setdefault("answers", {})
                    data.setdefault("triggers", [])
            except (OSError, ValueError):
                pass          # absent or corrupt: start empty, never guess, never crash
            self._data = data
        return self._data

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = f"{self.path}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, ensure_ascii=False, indent=1, default=str)
            fh.flush()
        os.replace(tmp, self.path)

    # -- the API

    def record(self, parameter_id, answer, trigger=None):
        """Persist a user answer (and log its trigger record, spec §16.3). Returns the
        stored record. Raises OSError when the disk refuses — the caller decides; a
        failed record must not silently claim persistence."""
        answer = answer or {}
        record = {"parameter": str(parameter_id),
                  "choice": answer.get("choice"),
                  "value": answer.get("value"),
                  "rule": (trigger or {}).get("rule"),
                  "authorization_scope": (trigger or {}).get("authorization_scope"),
                  "ts": _now()}
        data = self._load()
        data["answers"][str(parameter_id)] = record
        if trigger:
            self.log_trigger(trigger)
        self._save()
        return dict(record)

    def log_trigger(self, record):
        """Append a trigger record to the log (§16.3: every question justified by
        recorded uncertainty — and none for auto-proceed paths)."""
        self._load()["triggers"].append(dict(record or {}))
        self._save()

    def lookup(self, parameter_id):
        """The stored answer record for the parameter, or None (never raises)."""
        found = (self._load()["answers"] or {}).get(str(parameter_id))
        return dict(found) if found else None

    def has_answer(self, parameter_id) -> bool:
        return str(parameter_id) in (self._load()["answers"] or {})


# ---------------------------------------------------------------------------
# headless decision


def decide(trigger, store, human_available=False):
    """Resolve one trigger: a recorded answer ALWAYS wins (never re-ask, §16.1 rule 5);
    without a human the result is the ``needs_input`` denial (mirroring netguard's
    NO_HUMAN semantics — the caller must refuse, and no fallback is authorized);
    otherwise the question payload for the dialog."""
    trigger = trigger or {}
    parameter = trigger.get("parameter")
    answer = store.lookup(parameter) if store is not None else None
    if answer is not None:
        result = {"decision": "answered", "parameter": parameter, "answer": answer,
                  "reused": True}
        if (trigger.get("authorization_scope") == SCOPE_FALLBACK_ROUTE
                and answer.get("choice") == CHOICE_ACCEPT):
            # §18.3 / §19: approval authorizes the declared fallback ONLY — as data for
            # the pipeline. It never upgrades the coverage class or outcome ceiling.
            result.update({"authorized": True, "authorization_only": True,
                           "outcome_ceiling": OUTCOME_CEILING_FALLBACK,
                           "note": "approval is authorization only; the faceted coverage "
                                   "class and the outcome ceiling are unchanged"})
        return result
    question = build_question(trigger)
    if not human_available:
        return {"decision": "needs_input", "parameter": parameter,
                "reason": "no_human", "question": question,
                "authorizes_fallback": False, "authorizes_repair": False,
                "note": "no human to ask (console session): the dependent operation is "
                        "denied; this result is not successful reconstruction and never "
                        "authorizes fallback geometry"}
    return {"decision": "question", "parameter": parameter, "question": question}
