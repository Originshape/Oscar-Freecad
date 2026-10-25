"""Sampled star ratings of Oscar turns (pure: no FreeCAD, no Qt).

Now and then, after a substantial turn, the panel asks "How did Oscar do on this one?" with five stars. A rating
the user chooses to send goes to the OriginShape feedback endpoint (``POST /v1/feedback``, kind ``rating``) with
what it rates: the prompt, Oscar's answer, the steps Oscar took (tool, title, whether it worked, how long it took)
and, if the user leaves it ticked, a picture of the model. Nothing is sent unless the user presses Send.

When to ask (``should_ask``):

* only a **substantial** turn that answered: a long prompt, several tool steps, or a change to the model;
* never twice within ``COOLDOWN_S``, and at most ``DAILY_CAP`` times a day;
* and then only by chance (``CHANCE``), so the ratings are a sample, not a reaction to every turn.

The steps are the actions Oscar took, as the panel lists them. The model's own reasoning text (the panel's
thought rows) is not part of a rating.
"""

from __future__ import annotations

import html
import re
import secrets
from dataclasses import dataclass, field
from typing import Any

from . import context_guard

MIN_PROMPT_CHARS = 200          # a prompt this long is substantial on its own
MIN_STEPS = 3                   # ... as are this many tool steps
COOLDOWN_S = 20 * 60            # at least this long between two questions
DAILY_CAP = 3                   # at most this many questions a day
CHANCE = 0.35                   # of the turns that qualify, this share is asked about
PREF_LAST = "RatingLastAsked"   # params int: when the last question was shown (epoch seconds)
PREF_DAY = "RatingAskedDay"     # params string: the day DAILY_CAP counts (YYYY-MM-DD, local)
PREF_COUNT = "RatingAskedCount" # params int: questions shown that day
PROMPT_LIMIT, RESPONSE_LIMIT, STEPS_LIMIT = 20_000, 100_000, 200
TOOL_LIMIT, TITLE_LIMIT, COMMENT_LIMIT = 80, 300, 10_000
SHARED_TEXT = ("Sends this prompt, Oscar’s answer, the steps Oscar took and, if ticked, a picture of the model to "
               "OriginShape, to help improve Oscar.")

# Card states (``Block.detail`` of a RATING block).
OFFERED, SENDING, SENT, FAILED, DISMISSED = "", "sending", "sent", "failed", "dismissed"
SENT_WITHOUT_PICTURE = "sent-no-picture"   # sent, but the picture the user ticked couldn't be taken
# A card's links carry a random key, not its block number: a link the model writes can't match a real card.
_KEY = re.compile(r"[0-9a-f]{16}")


def new_key() -> str:
    """The random key of a new card's links."""
    return secrets.token_hex(8)


@dataclass
class TurnFacts:
    """What the panel knows of a finished turn."""

    prompt: str
    response: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    mutated: bool = False
    ok: bool = True
    outcome: str = ""


def substantial(facts: TurnFacts) -> bool:
    """A turn worth asking about: it answered, and it was a real piece of work."""
    if not facts.ok or not facts.response.strip() or not facts.prompt.strip():
        return False
    return len(facts.prompt.strip()) >= MIN_PROMPT_CHARS or len(facts.steps) >= MIN_STEPS or facts.mutated


def should_ask(facts: TurnFacts, *, now: float, today: str, last_asked: float, asked_day: str, asked_count: int,
               roll: float) -> bool:
    """Whether to show the rating card after this turn. ``roll`` is a random number in [0, 1)."""
    if not substantial(facts):
        return False
    if last_asked and now - last_asked < COOLDOWN_S:
        return False
    if asked_day == today and asked_count >= DAILY_CAP:
        return False
    return roll < CHANCE


def asked(today: str, asked_day: str, asked_count: int) -> tuple[str, int]:
    """The day and count to store once the card is shown."""
    return today, (asked_count + 1 if asked_day == today else 1)


def step_record(name: str, title: str, ok: bool | None, seconds: float | None) -> dict[str, Any]:
    """One step as a rating carries it."""
    record: dict[str, Any] = {"tool": (name or "tool")[:TOOL_LIMIT],
                              "title": context_guard.redact(title or "")[:TITLE_LIMIT]}
    if ok is not None:
        record["ok"] = bool(ok)
    if seconds:
        record["seconds"] = round(max(0.0, float(seconds)), 1)
    return record


def payload(facts: TurnFacts, stars: int, *, comment: str = "", image_png: str = "",
            context: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``POST /v1/feedback`` body of a rating (``image_png``: base64 PNG, or empty for none). Credentials in
    the texts (a key pasted into the prompt, a token a script printed) are redacted, as for the model."""
    if not 1 <= int(stars) <= 5:
        raise ValueError("stars must be 1 to 5")
    body: dict[str, Any] = {
        "kind": "rating",
        "stars": int(stars),
        "prompt": context_guard.redact(facts.prompt.strip())[:PROMPT_LIMIT],
        "response": context_guard.redact(facts.response.strip())[:RESPONSE_LIMIT],
        "steps": list(facts.steps[:STEPS_LIMIT]),
        "context": {"sampled": True, "outcome": facts.outcome or "answered", "changed_model": facts.mutated,
                    **(context or {})},
    }
    if comment.strip():
        body["text"] = context_guard.redact(comment.strip())[:COMMENT_LIMIT]
    if image_png:
        body["image_png"] = image_png
    return body


def card_html(key: str, state: str, *, muted: str = "#888") -> str:
    """The rating card under a turn (``key``: its links' ``new_key``): five star links and Not now; then what
    happened."""
    style = f"color:{muted};font-size:small;"
    if state == SENT:
        return f"<div style='{style}margin:4px 0 4px 4px;'>Thanks: your rating was sent.</div>"
    if state == SENT_WITHOUT_PICTURE:
        return (f"<div style='{style}margin:4px 0 4px 4px;'>Thanks: your rating was sent, without the picture "
                "(the 3D view couldn’t be captured small enough).</div>")
    if state == DISMISSED:
        return ""
    if state == SENDING:
        return f"<div style='{style}margin:4px 0 4px 4px;'>Sending your rating…</div>"
    stars = " ".join(
        f"<a href='oscar://rate/{html.escape(key)}/{n}' title='{n} of 5' "
        f"style='text-decoration:none;font-size:large;color:#d9a21b;'>☆</a>"
        for n in range(1, 6)
    )
    note = (f"<div style='{style}'>The rating couldn’t be sent. Pick the stars to try again.</div>"
            if state == FAILED else "")
    return (
        "<table cellpadding='8' style='border:1px solid rgba(128,128,128,0.35);border-radius:10px;"
        "margin:4px 0 4px 4px;'><tr><td>"
        "<div><b>How did Oscar do on this one?</b></div>"
        f"<div style='margin-top:2px;'>{stars}"
        f" &nbsp; <a href='oscar://rate/{html.escape(key)}/later' style='{style}'>Not now</a></div>"
        f"{note}</td></tr></table>"
    )


def parse_anchor(target: str) -> tuple[str, int | None] | None:
    """``oscar://rate/<key>/<1-5|later>`` -> ``(key, stars)`` (``None`` stars: Not now); anything else -> None."""
    prefix = "oscar://rate/"
    if not target.startswith(prefix):
        return None
    parts = target[len(prefix):].strip("/").split("/")
    if len(parts) != 2 or not _KEY.fullmatch(parts[0]):
        return None
    if parts[1] == "later":
        return parts[0], None
    if parts[1] in {"1", "2", "3", "4", "5"}:
        return parts[0], int(parts[1])
    return None
