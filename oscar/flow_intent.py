"""Deterministic request analysis of the build flow (private module of ``oscar.flow``): intent, requirements, reference need.

PURE PYTHON (no FreeCAD, no Qt). Everything here is a function of the text the USER TYPED (and, for the
CAD-attachment signal, attachment paths): no model is consulted. The classifier is an ordered list of regular
expressions; the first match wins and a miss costs only gate strictness, never tool access.

``oscar.flow`` re-exports the public names.
"""

import re
from typing import Sequence

__all__ = ["INTENTS", "WORK", "ENCLOSURE_RULES", "classify_intent", "extract_requirements", "needs_reference",
           "is_continuation", "is_goahead", "answers_open_question", "clauses_of", "CONTINUE_RE", "REQUEST_FRAME", "KEYWORD_RULES"]

INTENTS = ("enclosure", "fixture", "bracket", "generic_part", "modify", "fem",
           "inspect", "deliver", "question", "other")
WORK = frozenset({"enclosure", "fixture", "bracket", "generic_part", "modify", "fem"})

_BUILD_VERBS = r"(?:build|create|make|design|draw|add|cut|extrude|modify|change)"
_BUILD_VERB_RE = re.compile(rf"\b{_BUILD_VERBS}\b")

_INSPECT_START = re.compile(r"^(?:what(?:'s| is| are)?|list|show(?: me)?|describe|inspect|summari[sz]e|count)\b")
_INSPECT_NOUNS = re.compile(r"\b(?:documents?|models?|objects?|trees?|parts?|dimensions?|volumes?)\b")

# A pure VIEW request: a view verb applied to something that already exists ("the / this / my / it"), never to "a <part>".
_VIEW_REQUEST = re.compile(
    r"^(?:(?:please|can you|could you|would you|will you)\s+)*(?:show(?: me)?|let me see|let me look at|view|display|"
    r"look at|zoom (?:in|out)(?: on| to)?|pan to|fit (?:the )?view)\s+(?:to\s+)?(?:the|this|that|my|these|those|it|them|all)\b")

REQUEST_FRAME = re.compile(
    r"\b(?:is it possible|are you able|do you know how|would it be possible|do you think you can|can you|"
    r"could you|would you|will you|please|i need|i want|i'd like|let'?s|how about)\b")
_QUESTION_WORDS = re.compile(r"^(?:what|why|how|which|when|where|who|explain|difference)\b")
_QUESTION_AUX = re.compile(r"^(?:is|are|does|do|should)\b")

_FEM = re.compile(r"\b(?:fem|finite[- ]element|von ?mises|static (?:analysis|stress)|stress analysis|calculix)\b")
_DELIVER_START = re.compile(r"^(?:please )?(?:export|save|send)\b")
_PRINT_START = re.compile(r"^(?:please )?(?:3d[- ]?print|print)\b")
_DELIVER_NOUNS = re.compile(r"\b(?:stl|step|stp|3mf|fcstd|file|project|document|model|it)\b")

# Multilingual enclosure words: latin / cyrillic ones take word boundaries, CJK ones cannot (every ideograph is a word character).
_ENCLOSURE_WORDS = r"\b(?:carcasa|caja|geh(?:ä|a)use|bo(?:î|i)tier|custodia|behuizing|obudowa|корпус)\b|外壳|外殼|筐体|ケース|케이스"
_ENCLOSURE = re.compile(
    r"\b(?:enclosure|housing|casing|case|shell|box)\b.*\b(?:for|around|to (?:hold|house|fit|contain))\b"
    r"|\b(?:enclosure|housing)\b|snap[- ]?fit|" + _ENCLOSURE_WORDS)
_BOARD_HINT = re.compile(r"\b(?:raspberry\s*pi|rpi\s*\d|arduino|esp32|esp8266|pico|jetson|beaglebone|pcb)")
_BOARD_BUILD_VERB = re.compile(r"\b(?:build|make|design|create|construye|erstelle|fais)\b")
_FIXTURE = re.compile(r"\b(?:fixture|jig|cradle|holder|clamping (?:block|plate))\b")
_BRACKET = re.compile(r"\b(?:bracket|mounting plate|mount|stand|l[- ]bracket)\b")
_MODIFY = re.compile(
    r"\b(?:modify|change|edit|update|adjust|resize|move|rotate|rename|replace|increase|decrease|widen|thicken|"
    r"shorten|lengthen|fix|redesign|re-design|rework|revise|remodel|reshape|refine|enlarge|shrink)\b"
    r"|\bmake (?:the|it|them|this)\b")
_GENERIC = re.compile(
    r"\b(?:build|create|make|model|design|draw|sketch|extrude|revolve|flange|gear|shaft|plate|bolt|bearing|pulley|"
    r"part|cylinder|cube|sphere|fillet|chamfer|pocket|pad|box|boxes|hole|holes|slot|slots|drill|bore|countersink|"
    r"counterbore|vent|vents|cutout|cutouts|rib|ribs|boss|thread|loft|sweep|add|cut|put|attach)\b")
_NUMBER_UNIT = re.compile(r"\d+(?:[.,]\d+)?\s?(?:(?:mm|cm|in|deg)(?![a-z])|\"|°)")
_CAD_PATH = re.compile(r"\.(?:step|stp|iges|igs|brep|brp|fcstd|stl|obj|3mf|ply|off|dxf|dwg|sat|x_t)\s*$", re.I)


# The classifier and the requirement extraction look at the first MAX_REQUEST_CHARS characters only (the Ledger keeps the
# same prefix as its `request`): the keyword rules contain `.*` patterns, and a huge paste must never make them quadratic.
MAX_REQUEST_CHARS = 4000


def _clip(text) -> str:
    return (text if isinstance(text, str) else "")[:MAX_REQUEST_CHARS]


def _norm(text: str) -> str:
    return " ".join(_clip(text).lower().split())


def classify_intent(text: str, *, attachments: Sequence[str] = ()) -> str:
    """The intent of a typed request: one of ``INTENTS`` (the first matching rule wins, none -> ``other``).

    ``text`` must be what the user TYPED (never the stored message with notes, ``<workspace_state>`` or
    attachment blocks); attachment PATHS enter only through ``attachments`` (a CAD / mesh path marks a work
    request in a language the keyword rules do not cover).
    """
    t = _norm(text)
    if not t:
        return "other"
    if _INSPECT_START.match(t) and _INSPECT_NOUNS.search(t) and not _BUILD_VERB_RE.search(t):
        return "inspect"
    if _VIEW_REQUEST.match(t) and not _BUILD_VERB_RE.search(t):
        return "inspect"
    if t.endswith("?") and len(t) <= 160 and (
            _QUESTION_WORDS.match(t) or (_QUESTION_AUX.match(t) and not REQUEST_FRAME.search(t))):
        return "question"
    if _FEM.search(t):
        return "fem"
    if _DELIVER_START.match(t) and _DELIVER_NOUNS.search(t) and not _BUILD_VERB_RE.search(t):
        return "deliver"
    if _PRINT_START.match(t) and not _BUILD_VERB_RE.search(t):
        return "deliver"   # '3D print this lid' = prepare/export for printing, not a new design
    if _ENCLOSURE.search(t) or (_BOARD_HINT.search(t) and _BOARD_BUILD_VERB.search(t)):
        return "enclosure"
    if _FIXTURE.search(t):
        return "fixture"
    if _BRACKET.search(t):
        return "bracket"
    if _MODIFY.search(t):
        return "modify"
    if _GENERIC.search(t):
        return "generic_part"
    if _NUMBER_UNIT.search(t) or any(_CAD_PATH.search(str(p)) for p in attachments or ()):
        return "generic_part"
    return "other"


# -- reference need ------------------------------------------------------------------------------------

_PART_HINT = re.compile(
    r"\b(?:raspberry\s*pi|rpi\s*\d?\w*|pi\s*(?:zero|[0-9])|arduino|esp32|esp8266|pico|jetson|beaglebone|stm32|teensy|"
    r"nema\s*\d+|motor|sensor|display|screen|lcd|oled|pcb|board|module|battery|connector)", re.I)
_SOURCE_WORDS = re.compile(r"\b(?:step|stp|datasheet|cad file|download|source it)\b", re.I)
_STEP_BY_STEP = re.compile(r"step[- ]by[- ]step", re.I)
_HINT_PREP = re.compile(r"\b(?:for|around|to hold|to fit|to house)\b", re.I)
_HINT_END = re.compile(r"[.,;\n]|\s(?:with|and)\s", re.I)


def needs_reference(text: str, intent: str, *, attachments: Sequence[str] = ()) -> tuple[bool, str]:
    """``(needs, hint)``: whether the request is about a real part the design wraps, and a short name for it.

    True only for the intents ``enclosure`` and ``fixture`` when the text names a part (board, motor, sensor,
    ...), asks to source / download a file, or a CAD / mesh file is attached. The hint is the text after
    ``for|around|to hold|to fit|to house`` (cut at the first clause end, at most 60 characters), preferring the
    occurrence that actually names a part, else the first part word. Returns ``(False, "")`` when no reference is needed.
    """
    if intent not in ("enclosure", "fixture"):
        return False, ""
    raw = _clip(text)
    scrubbed = _STEP_BY_STEP.sub(" ", raw)
    part = _PART_HINT.search(scrubbed)
    attached = any(_CAD_PATH.search(str(p)) for p in attachments or ())
    if not (part or _SOURCE_WORDS.search(scrubbed) or attached):
        return False, ""
    hint = ""
    candidates = []
    for m in _HINT_PREP.finditer(raw):
        rest = raw[m.end():].strip()
        end = _HINT_END.search(rest)
        candidates.append((rest[: end.start()] if end else rest).strip())
    for c in candidates:
        if c and _PART_HINT.search(c):
            hint = c
            break
    if not hint:
        hint = next((c for c in candidates if c), "")
    if not hint and part:
        hint = part.group(0)
    return True, hint[:60].strip()


# -- continuation --------------------------------------------------------------------------------------

CONTINUE_RE = re.compile(
    r"^\s*(?:continue|continu\w*|go on|go ahead|proceed|resume|keep going|carry on|do it|yes|ok(?:ay)?|sigue|seguir|weiter|"
    r"poursuis\w*|avanti|продолж\w*|继续|続け\w*|계속)\b", re.I)
_RESUME_PREFIX = "Continue the task. Plan status:"


def is_goahead(text: str) -> bool:
    """True for a bare go-ahead: at most 3 words, starting with a go-ahead word, not a question ("continue", "yes", "do it").

    It says nothing about WHAT to continue: ``is_continuation`` needs an open plan on top, and the pinned memory uses
    this to keep the task's own request when the user answers a finished build with "continue".
    """
    raw = (text or "").strip()
    return bool(raw) and not raw.endswith("?") and len(raw.split()) <= 3 and bool(CONTINUE_RE.match(raw))


_PLEASANTRY = re.compile(
    r"^\s*(?:thanks|thank you|thx|ty|cool|nice|great|perfect|awesome|good|fine|got it|wow|lol|hello|hi|hey|bye)\b", re.I)
# Words of a NEW order: a reply that carries one is a new request, not the answer to the model's question.
_NEW_ORDER = re.compile(
    r"\b(?:build|create|make|design|draw|sketch|add|cut|drill|extrude|export|save|send|delete|remove|rebuild|"
    r"simulate|replace|change|modify|resize|move|rotate)\b", re.I)
_ANSWER_MAX_WORDS = 40


def answers_open_question(text: str, prev, attachments: Sequence[str] = ()) -> bool:
    """True when ``text`` (and / or ``attachments``) plausibly ANSWERS what the previous turn asked the user.

    The previous turn ended waiting for the user when ``prev`` has a required step the model blocked / skipped with a
    note (``Ledger.has_blocked_with_note``, the ``needs_input`` outcome: "please attach the STEP file"). The reply is then
    an answer, not a new request, when it carries an attachment, or is a short statement (at most 40 words) that is
    neither a question, a pleasantry nor a new order ("add a fan hole", "export it"): "ABS, 2 mm walls" and "I attached
    the STEP file" resume the plan, "thanks" and "make a gear" do not. Without such a step it is always False.
    """
    try:
        if not prev.has_blocked_with_note():
            return False
    except AttributeError:
        return False
    raw = (text or "").strip()
    if not raw:
        return bool(attachments)
    if raw.endswith("?") or len(raw.split()) > _ANSWER_MAX_WORDS:
        return False
    return not (_PLEASANTRY.match(raw) or _NEW_ORDER.search(raw))


def is_continuation(text: str, prev, *, attachments: Sequence[str] = ()) -> bool:
    """True iff ``text`` resumes the plan of ``prev`` (a Ledger with open required steps).

    Requires ``prev`` with at least one non-terminal required step and either the exact ``prev.resume_prompt()``
    (what the panel's ``continue:`` anchor sends), a typed text of at most 3 words that starts with a go-ahead word, or
    (``answers_open_question``) the answer to a question the previous turn ended with. A question (text ending in ``?``)
    is never a go-ahead, and ``classify_intent == "other"`` is NOT enough: "thanks", "ok then?" or "hello?" in the
    middle of a build must not resume building. ``attachments`` are the paths attached to the message.
    """
    if prev is None:
        return False
    try:
        if not prev.has_open_required_steps():
            return False
    except AttributeError:
        return False
    raw = (text or "").strip()
    if raw:
        try:
            if raw == prev.resume_prompt().strip():
                return True
        except AttributeError:
            pass
        if raw.startswith(_RESUME_PREFIX) or is_goahead(raw):
            return True
    return answers_open_question(raw, prev, attachments)


# -- requirements --------------------------------------------------------------------------------------

_SPLIT = re.compile(r"\n+|;|,(?=\s|$)|\.(?=\s|$)")
_AND = re.compile(r"\s+and\s+", re.I)

# (id of the rule, regex, requirement text, covered step). Rules fire in THIS order; word patterns are plural-safe.
ENCLOSURE_RULES = (
    ("shell", re.compile(r"\b(?:shells?|housings?|enclosures?|cases?|box(?:es)?|bottom)\b", re.I),
     "A bottom shell that holds the board", "S5"),
    ("ports", re.compile(r"\b(?:cutouts?|ports?|i/?o|connectors?|openings?|usb|hdmi|ethernet|audio|sd)\b", re.I),
     "Cutouts for every I/O port", "S6"),
    ("lid", re.compile(r"\b(?:lids?|covers?)\b|snap", re.I), "A lid", "S7"),
    ("vents", re.compile(r"\b(?:vents?|air|cooling|slots?|ventilat\w*)\b", re.I), "Air vent slots", "S8"),
    ("snap", re.compile(r"snap", re.I), "Snap-fit retention of the lid", "S9"),
    ("material", re.compile(r"\b(?:abs|asa|pla|petg|nylon|polycarbonate)\b", re.I), "Material: {material}", "S4"),
    ("source", re.compile(r"\b(?:step|stp|source|download|datasheet|cad file)\b", re.I),
     "Source the reference part", "S2,S3"),
)
_ENCLOSURE_ALWAYS = (("Every driving dimension is a named parameter", "S4"),
                     ("Delivered as a rebuildable project (notes, rebuild script, exports)", "S12"))

# first required BUILD step (or first step of any phase) that carries the user's own clauses, per intent
_FIRST_BUILD = {"fixture": "S5", "bracket": "S3", "generic_part": "S3", "modify": "S1", "fem": "S1",
                "deliver": "S1"}
_DELIVERED_STEP = {"fixture": "S8", "bracket": "S7", "generic_part": "S5", "modify": "S3", "fem": "S4"}

# Keyword rules that make an otherwise optional template step required (shared with ``seed_template``).
KEYWORD_RULES = {
    "fixture_clamp": re.compile(r"\b(?:clamp|locat|pin|stop|fence|screw|bolt)", re.I),
    "bracket_holes": re.compile(r"\b(?:hole|slot|drill|bolt|screw|mount|bore)", re.I),
}

_MAX_REQUIREMENTS = 12


def clauses_of(text: str) -> list[str]:
    """The clauses of a request: split on ``,`` ``;`` sentence-final ``.`` newlines and `` and ``, whitespace collapsed.

    A decimal point or comma inside a number never splits, and `` and `` splits only when both sides keep at
    least two words ("Import and analyze the attached file" stays one clause).
    """
    out = []
    for piece in _SPLIT.split(text or ""):
        parts = [piece]
        halves = _AND.split(piece)
        if len(halves) > 1:
            parts, buf = [], ""
            for half in halves:
                buf = f"{buf} and {half}" if buf else half
                if len(buf.split()) >= 2:
                    parts.append(buf)
                    buf = ""
            if buf:
                if parts:
                    parts[-1] = f"{parts[-1]} and {buf}"
                else:
                    parts.append(buf)
        for part in parts:
            part = " ".join(part.split()).strip(" .,:;-")
            if part:
                out.append(part)
    return out


def _sentence(text: str, limit: int = 160) -> str:
    text = text[:1].upper() + text[1:]
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def extract_requirements(text: str, intent: str) -> list[dict]:
    """The requirements of a typed request: ``[{"id": "R1", "text": ..., "covers": ["S5"]}]``.

    ``enclosure``: the rule table fires in order (shell, ports, lid, vents, snap, material, source), then two
    always-added requirements (parameters, deliverables), then up to 12 requirements in total from the
    remaining clauses (12..160 characters, not matched by a rule), covered by the VERIFY step ``S11``.
    ``fixture`` / ``bracket`` / ``generic_part`` / ``modify`` / ``fem``: one requirement per clause in
    the user's own words plus a final "Delivered as a rebuildable project"; ``deliver``: one per clause;
    ``inspect`` / ``question`` / ``other``: none. ``covers`` lists template step ids (``seed_template`` applies
    them); the Ledger derives every requirement status from the steps.
    """
    text = _clip(text)
    if intent == "enclosure":
        return _enclosure_requirements(text)
    if intent == "deliver":
        parts = clauses_of(text)
        return _numbered([(_sentence(c), [_FIRST_BUILD["deliver"]]) for c in parts[:_MAX_REQUIREMENTS]]
                         or [(_sentence((text or "").strip() or "Produce the requested files"), [_FIRST_BUILD["deliver"]])])
    if intent not in WORK:
        return []
    first = _FIRST_BUILD[intent]
    rows = [(_sentence(c), [first]) for c in clauses_of(text)[: _MAX_REQUIREMENTS - 1]]
    if not rows:
        rows = [(_sentence((text or "").strip() or "Complete the request"), [first])]
    rows.append(("Delivered as a rebuildable project", [_DELIVERED_STEP[intent]]))
    return _numbered(rows)


def _numbered(rows: list[tuple[str, list[str]]]) -> list[dict]:
    return [{"id": f"R{i}", "text": text, "covers": covers} for i, (text, covers) in enumerate(rows, 1)]


def _enclosure_requirements(text: str) -> list[dict]:
    rows: list[tuple[str, list[str]]] = []
    for _name, rx, label, covers in ENCLOSURE_RULES:
        m = rx.search(text)
        if not m:
            continue
        if "{material}" in label:
            label = label.format(material=m.group(0).upper())
        rows.append((label, covers.split(",")))
    rows.extend((label, [covers]) for label, covers in _ENCLOSURE_ALWAYS)
    for clause in clauses_of(text):
        if len(rows) >= _MAX_REQUIREMENTS:
            break
        if not 12 <= len(clause) <= 160 or any(rx.search(clause) for _n, rx, _l, _c in ENCLOSURE_RULES):
            continue
        rows.append((_sentence(clause), ["S11"]))
    return _numbered(rows[:_MAX_REQUIREMENTS])
