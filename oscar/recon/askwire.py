# SPDX-License-Identifier: Apache-2.0
"""The §16 ↔ ask_user bridge: ``recon.approve`` questions become decision cards, structured
card answers become §16 answer records.

Pure decision code (no FreeCAD): the reconstruction flow calls ``resolve_units`` with an ``ask``
callable — in the panel that is ``tools_ask.ask_user`` itself (the worker thread blocks on the
registry while the user answers the card); in tests it is a lambda. Everything that makes a
question ``likeliest`` is decided by approve.py's own recommendation ordering; a card never
invents an option, and CHOICE_OTHER becomes the card's free-text field."""

from .approve import (CHOICE_ACCEPT, CHOICE_OTHER, CHOICE_RETAIN, RULE_INTERPRETATION, RULE_UNITS,
                      approval_triggers, build_question, decide)

MAX_CARD_OPTIONS = 4


def card_from_question(question: dict) -> dict:
    """One §16 question payload → one ``ask_user`` card payload (validated by askuser)."""
    question = question or {}
    subtitle_bits = [str(question.get("reason") or "").strip()]
    if str(question.get("recommendation") or "").strip():
        subtitle_bits.append(f"recommended: {str(question['recommendation']).strip()}")
    for key in ("measured", "proposal", "tolerance", "confidence"):
        if question.get(key) is not None:
            subtitle_bits.append(f"{key}: {question[key]}")
    evidence = [e for e in (question.get("evidence") or []) if e][:1]
    if evidence:
        subtitle_bits.append(f"evidence: {evidence[0]}")
    recommendation = str(question.get("recommendation") or "")
    default_key = CHOICE_RETAIN if question.get("rule") == RULE_INTERPRETATION else ""
    options, marked = [], False
    for option in (question.get("options") or [])[:MAX_CARD_OPTIONS]:
        if option.get("key") == CHOICE_OTHER:
            continue                      # the free-text field plays this role on the card
        key = str(option.get("key") or "")
        if default_key:
            likeliest = key == default_key                  # the MEASURED winner is the default
        else:
            likeliest = (not marked) and (not recommendation or recommendation.startswith(key))
        options.append({"label": str(option.get("label") or key or "?"), "likeliest": likeliest})
        marked = marked or likeliest
    if not options:
        options = [{"label": "Proceed as proposed", "likeliest": True},
                   {"label": "Stop and explain", "likeliest": False}]
    return {"title": str(question.get("title") or "Confirm Oscar's proposal"),
            "subtitle": " | ".join(b for b in subtitle_bits if b)[:400],
            "options": options[:MAX_CARD_OPTIONS],
            "allow_free_text": True}


def answer_from_structured(structured: dict, question: dict) -> dict:
    """An ask_user structured answer → the §16 record ``approve.answer_outcome`` expects."""
    structured = structured or {}
    index = structured.get("index")
    if isinstance(index, int) and index >= 0:
        options = [o for o in (question.get("options") or []) if o.get("key") != CHOICE_OTHER]
        if 0 <= index < len(options):
            return {"choice": options[index].get("key"), "value": None}
    return {"choice": CHOICE_OTHER, "value": structured.get("value") or structured.get("answer")}


def resolve_units(units_report: dict, *, ask, store, parameter: str = "units"):
    """Ask (or reuse) the §16 units decision for a GATE-status units report.

    ``ask(card) -> structured`` is the interactive seam (None ⇒ headless: the honest needs_input
    denial, which never authorizes anything). Returns ``(unit_value_or_None, decision_dict)`` —
    the unit value feeds ``reconstruct``'s ``confirm_units``/``user_unit``; None keeps the
    existing non-verified gate exactly as today."""
    report = dict(units_report or {})
    if report.get("status") not in ("POSSIBLE_MM_INCH_MISMATCH", "SCALE_AMBIGUOUS"):
        return None, {"decision": "not_required", "parameter": parameter}
    triggers = approval_triggers({"units": report})
    if not triggers:
        return None, {"decision": "not_required", "parameter": parameter}
    trigger = triggers[0]
    outcome = decide(trigger, store, human_available=ask is not None)
    if outcome["decision"] == "answered":
        return _unit_of(outcome["answer"].get("choice"), outcome["answer"].get("value")), outcome
    if outcome["decision"] != "question":
        return None, outcome                      # needs_input: honest, nothing applied
    question = build_question(trigger)
    card = card_from_question(question)
    try:
        structured = ask(card)
    except Exception:   # noqa: BLE001 - a failed card is a denied question, never a fallback
        return None, {"decision": "ask_failed", "parameter": parameter}
    answer = answer_from_structured(structured, question)
    if store is not None:
        store.record(parameter, answer, trigger)
    return _unit_of(answer.get("choice"), answer.get("value")), {"decision": "answered",
                                                                 "parameter": parameter,
                                                                 "answer": answer, "reused": False}


_ROUTE_TO_CLASS = {"prismatic_extrusion": "prismatic", "rotational": "rotational"}


def resolve_interpretation(classification: dict, *, ask, store, parameter: str = "classification"):
    """Ask (or reuse) the §16 ambiguous-interpretation decision.

    The MEASURED classification is the likeliest option (retain); the named alternative is the
    proposal. Returns ``(preferred_class_or_None, decision_dict)`` — ``None`` keeps the measured
    winner exactly as today; a class name ("prismatic" / "rotational") overrides the route for
    THIS reconstruction only. Headless / no ask / a failed card: ``(None, ...)`` — never a
    silent route switch."""
    alternatives = [str(a) for a in (classification or {}).get("alternatives") or [] if a]
    triggers = approval_triggers({"classification": dict(classification or {})})
    if not triggers:
        return None, {"decision": "not_required", "parameter": parameter}
    trigger = triggers[0]
    outcome = decide(trigger, store, human_available=ask is not None)
    if outcome["decision"] == "answered":
        return _class_of(outcome["answer"], alternatives), outcome
    if outcome["decision"] != "question":
        return None, outcome
    question = build_question(trigger)
    card = card_from_question(question)
    try:
        structured = ask(card)
    except Exception:   # noqa: BLE001 - a failed card never switches the route
        return None, {"decision": "ask_failed", "parameter": parameter}
    answer = answer_from_structured(structured, question)
    if store is not None:
        store.record(parameter, answer, trigger)
    return _class_of(answer, alternatives), {"decision": "answered", "parameter": parameter,
                                             "answer": answer, "reused": False}


def _class_of(answer, alternatives) -> "str | None":
    """The route class a §16 answer selects: accept = the named alternative (when it maps to a
    route class), retain or anything else = None — the measured winner stands, never a guess."""
    if (answer or {}).get("choice") == CHOICE_ACCEPT and alternatives:
        return _ROUTE_TO_CLASS.get(alternatives[0])
    return None


def _unit_of(choice, value):
    """The unit value for reconstruct: mm/inch by choice key; a typed answer is passed through
    for the units module to interpret (it reports what it cannot honour)."""
    if choice in ("mm", "cm", "inch"):
        return choice
    if choice == CHOICE_OTHER and isinstance(value, str) and value.strip():
        return value.strip()[:24]
    return None
