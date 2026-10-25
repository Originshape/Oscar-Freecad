"""Pure presentation logic of the Oscar panel: markdown, plan strip, step rows, heartbeat, badges.

PURE PYTHON: no Qt, no FreeCAD, no oscar imports, so every function here is unit-testable
without a GUI (``tests/test_panel_render.py``). ``panel.py`` keeps the Qt widgets (``Block`` /
``Step`` / ``OscarPanel``) and delegates every piece of HTML it can to this module.

What lives here
- ``md_to_html``: the markdown subset the chat renders (code fences, pipe tables, bullet and
  ordered lists, headings, bold / italic, links, bare URLs and absolute paths).
- ``TurnView`` / ``StepView``: the model of ONE turn built from the worker's JSON signal payloads
  (``plan_changed``, ``step_started``, ``step_finished``, ``progress``,
  ``round_reset``, ``artifacts_changed``, ``turn_outcome``). All ``on_*`` methods accept the JSON
  string or the decoded dict, never raise on malformed input, and are GUI-thread only.
- ``plan_strip_html``, ``step_row_html``, ``heartbeat_text``, ``summary_card_html``,
  ``artifacts_strip_html``, ``outcome_badge``: stateless renderers.
- anchor helpers (``parse_anchor``, ``parse_artifact_anchor``) and
  ``classify_file_link`` (which file links may be opened and which are only revealed).
- ``header_text`` (the turn card header, shared by ``TurnView`` and the panel), ``summary_block_html`` (the
  Build Summary card with its action links), ``make_turn_meta`` / ``restore_card_html`` / ``restore_thumbs``
  (the compact per-turn records a chat persists as ``blocks_meta`` and the card that shows them again) and
  ``unique_call_id`` (provider call ids are not unique across turns, thumbnails and rows are keyed by it).
- the OriginShape account: ``account_badge`` / ``account_menu`` (the header badge and its menu, from the account
  state the panel keeps; an Oscar older than the server's minimum shows "Update Oscar"), ``account_card`` /
  ``account_card_html`` (the ``ACCOUNT`` cards: ``onboarding``, ``signed_out``, ``waitlisted``, ``no_credits``,
  ``outdated``), ``account_problem_card`` (which card a failed turn's
  ``account_problem`` payload calls for) and ``format_credits``.

Every function that returns HTML escapes all payload text; colours arrive as parameters.
"""

import html
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import quote

__all__ = [
    "ACCOUNT_CARD_KINDS", "ACCOUNT_TARGETS", "DETAIL_LIMIT", "StepView", "TurnView", "account_badge", "account_card",
    "account_card_html", "account_menu", "account_problem_card", "artifacts_strip_html", "classify_file_link",
    "clip_text", "format_credits", "header_text", "heartbeat_text", "make_turn_meta", "md_to_html", "outcome_badge",
    "outcome_color", "parse_anchor", "parse_artifact_anchor", "plan_strip_html", "restore_card_html",
    "restore_thumbs", "split_build_summary", "step_row_html", "summary_block_html", "summary_card_html",
    "thumb_url", "unique_call_id",
]

DETAIL_LIMIT = 12000   # characters of a tool result / script shown when a step row is expanded
_SUMMARY_LIMIT = 20000  # characters of the Build Summary the card renders

_OK = "#3aa76d"
_WARN = "#d9902b"
_BAD = "#e06c5a"
_LIVE_PHASE_ACTIVITIES = ("thinking", "writing", "tool_args")
_NO_NOTE = "not a geometry tool"   # the snapshot_note of every non-geometry step: expected, so never shown

# ---------------------------------------------------------------------------
# small coercion helpers (payloads come from JSON: any field may be missing or of the wrong type)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _int(value: Any, default: int = 0) -> int:
    return int(_num(value, float(default)))


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _flag(value: Any, default: bool = False) -> bool:
    return default if value is None else bool(value)


def _css(value: str) -> str:
    """A colour value that is safe inside a single-quoted ``style`` attribute."""
    return re.sub(r"[^#\w(),.%\s-]", "", str(value))


def _esc(value: Any) -> str:
    return html.escape(_text(value))


def _load(payload: Any) -> dict | None:
    """Decode a JSON signal payload (``str`` or already a ``dict``); ``None`` when it is neither."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (str, bytes)):
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            return None
        return data if isinstance(data, dict) else None
    return None


def clip_text(text: str, limit: int = DETAIL_LIMIT) -> str:
    """``text`` cut to ``limit`` characters with ONE trailing ellipsis line (never two: a text the
    producer already ended with ``…`` does not get a second one)."""
    text = _text(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    return cut if cut.endswith("…") else cut + "\n…"


def _fmt_secs(seconds: float) -> str:
    """``12 s`` / ``2 min 05 s`` (heartbeat style)."""
    total = max(0, int(seconds))
    if total >= 60:
        return f"{total // 60} min {total % 60:02d} s"
    return f"{total} s"


def _fmt_dur(seconds: float) -> str:
    """``1.8s`` / ``1m05s`` (step rows)."""
    seconds = max(0.0, seconds)
    if seconds >= 60:
        return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"
    return f"{seconds:.1f}s"


def _fmt_chars(count: int) -> str:
    return f"{count / 1000:.1f}k chars" if count >= 1000 else f"{count} chars"


# ---------------------------------------------------------------------------
# Markdown-lite -> HTML

_OPEN, _CLOSE = "\ue000", "\ue001"   # private-use marks that fence the fragments _inline stashes away
_FENCE = re.compile(r"^\s*```")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_TRIPLE_INLINE = re.compile(r"```([^`\n]+)```")
# Emphasis needs a word boundary on the outside and no space just inside, so `2*3*4 mm`, `a * b * c`
# and `x**2 + y**2` are arithmetic, not markup.
_BOLD = re.compile(r"(?<![\w*])\*\*(?=[^\s*])([^*\n]*?[^\s*])\*\*(?![\w*])")
_ITALIC = re.compile(r"(?<![\w*])\*(?=[^\s*])([^*\n]*?[^\s*])\*(?![\w*])")
_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_RULE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
# Bounded label / target: an unbounded ``[^\]]+`` rescans to the end of the text for every ``[`` (quadratic on a long
# run of unclosed brackets, which froze the GUI thread for seconds).
_LINK = re.compile(r"\[([^\]\n]{1,200})\]\(((?:https?|file|oscar)://[^)\s]{1,2000})\)")
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_ENTITY_STOP = r"(?!&(?:lt|gt|quot|#x27);)"   # md_to_html escapes first: `<>"'` arrive as entities
_BARE_URL = re.compile(r"(?<![\w/@='\">])((?:https?|file)://(?:" + _ENTITY_STOP + r"[^\s" + _OPEN + r"])+)")
_POSIX_PATH = re.compile(r"(?<![\w/.~:@=\-])((?:/[\w.@+%-]+){2,})")
_WIN_PATH = re.compile(r"(?<![\w])([A-Za-z]:\\(?:[\w.@+%-]+\\)*[\w.@+%-]+)")
_PLACEHOLDER = re.compile(_OPEN + r"(\d+)" + _CLOSE)
_CODE_STYLE = "background:rgba(128,128,128,.18);padding:0 2px;"


def _anchor(href: str, label: str) -> str:
    return f"<a href='{href}'>{label}</a>"


def _split_trailing(url: str) -> tuple[str, str]:
    """Split the sentence punctuation (and an unbalanced ``)``) off the end of a bare URL / path."""
    tail = ""
    while url:
        last = url[-1]
        if last in ".,;:!?*":
            tail, url = last + tail, url[:-1]
        elif last == ")" and url.count("(") < url.count(")"):
            tail, url = last + tail, url[:-1]
        else:
            break
    return url, tail


def _file_href(path: str) -> str:
    """``file://`` URL of an absolute POSIX path or a ``C:\\...`` path."""
    if re.match(r"[A-Za-z]:\\", path):
        return "file:///" + quote(path.replace("\\", "/"), safe="/:@+-._~%")
    return "file://" + quote(path, safe="/:@+-._~%")


# Paths inside an Oscar project (``project_paths.project_path_re``, handed over at start-up because this module
# stays pure): group 1 is the project folder, group 2 the rest. None: paths are shown as they are.
_PROJECT_BARE: "re.Pattern[str] | None" = None
_PROJECT_QUOTED: "re.Pattern[str] | None" = None
PROJECT_LABEL = "Oscar project"


def set_project_paths(pattern: "re.Pattern[str] | None") -> None:
    """Show paths matching ``pattern`` as ``Oscar project[/rest]`` links (``project_paths.project_path_re()``)."""
    global _PROJECT_BARE, _PROJECT_QUOTED
    _PROJECT_BARE = pattern
    _PROJECT_QUOTED = re.compile(f"`{pattern.pattern}`") if pattern is not None else None


def project_label(rest: str) -> str:
    """``Oscar project`` plus the part below the project folder (``Oscar project/exports``)."""
    rest = rest.replace("\\", "/").strip("/")
    return f"{PROJECT_LABEL}/{rest}" if rest else PROJECT_LABEL


def _inline(text: str) -> str:
    """Inline markup of one (already HTML-escaped) text run. Code spans, links, bare URLs and paths
    are stashed first so emphasis never reaches inside them. A path inside an Oscar project is shown as
    ``Oscar project/<rest>`` (a link to it), never with the full folder."""
    stash: list[str] = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return f"{_OPEN}{len(stash) - 1}{_CLOSE}"

    def project(match: "re.Match[str]", quoted: bool) -> str:
        whole = match.group(1) + match.group(2)
        target, tail = (whole, "") if quoted else _split_trailing(whole)
        rest = target[len(match.group(1)):]
        return keep(_anchor(_file_href(target), _esc(project_label(rest)))) + tail

    if _PROJECT_BARE is not None and _PROJECT_QUOTED is not None:
        text = _PROJECT_QUOTED.sub(lambda m: project(m, True), text)
        text = _PROJECT_BARE.sub(lambda m: project(m, False), text)

    def url(match: "re.Match[str]") -> str:
        target, tail = _split_trailing(match.group(1))
        return keep(_anchor(target, target)) + tail

    def path(match: "re.Match[str]") -> str:
        target, tail = _split_trailing(match.group(1))
        return keep(_anchor(_file_href(target), target)) + tail

    text = _TRIPLE_INLINE.sub(lambda m: keep(f"<code style='{_CODE_STYLE}'>{m.group(1)}</code>"), text)
    text = _INLINE_CODE.sub(lambda m: keep(f"<code style='{_CODE_STYLE}'>{m.group(1)}</code>"), text)
    text = _LINK.sub(lambda m: keep(_anchor(m.group(2), m.group(1))), text)
    text = _BARE_URL.sub(url, text)
    text = _WIN_PATH.sub(path, text)
    text = _POSIX_PATH.sub(path, text)
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    return _PLACEHOLDER.sub(lambda m: stash[int(m.group(1))], text)


def _table_html(rows: list[str]) -> str:
    # drop the |---|---| separator row; it is alignment metadata
    rows = [row for row in rows if not _TABLE_SEP.match(row)]
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    out = [
        "<table cellpadding='3' cellspacing='0' "
        "style='border-collapse:collapse;margin:4px 0;'>"
    ]
    for index, row in enumerate(cells):
        tag = "th" if index == 0 else "td"
        out.append(
            "<tr>"
            + "".join(
                f"<{tag} style='border:1px solid palette(mid);padding:2px 6px;'>"
                f"{_inline(cell)}</{tag}>"
                for cell in row
            )
            + "</tr>"
        )
    out.append("</table>")
    return "".join(out)


def md_to_html(text: str) -> str:
    """Render the chat's markdown subset to Qt rich-text HTML.

    Contract: input is untrusted text, ALL of it is HTML-escaped first, so the result contains only
    the tags generated here. Supported: fenced code blocks (protected before any line handling, an
    unterminated fence while streaming renders as code to the end), pipe tables, ``-`` / ``*``
    bullets, ``1.`` ordered lists (``<ol start=N>``), ``#``..``####`` headings, rules, inline code,
    ``**bold**`` / ``*italic*`` (not inside words or arithmetic such as ``2*3*4``), ``[text](url)``
    links for http(s) / file / es, bare URLs and absolute file paths (anchors). Never raises.
    """
    text = _text(text).replace(_OPEN, "").replace(_CLOSE, "").replace("\r\n", "\n").replace("\r", "\n")
    lines = html.escape(text).split("\n")
    out: list[str] = []
    list_kind = ""
    i = 0

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            out.append(f"</{list_kind}>")
            list_kind = ""

    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line) and line.count("```") == 1:   # a one-line ```code``` span is inline, not a fence
            body: list[str] = []
            i += 1
            while i < len(lines) and not _FENCE.match(lines[i]):
                body.append(lines[i])
                i += 1
            i += 1   # the closing fence (absent when the block is still streaming)
            close_list()
            out.append(f"<pre><code>{chr(10).join(body).rstrip()}</code></pre>")
            continue
        # Pipe tables: header row, |---|---| separator, then body rows.
        if _TABLE_ROW.match(line) and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            rows = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and _TABLE_ROW.match(lines[i]):
                rows.append(lines[i])
                i += 1
            close_list()
            out.append(_table_html(rows))
            continue
        bullet = _BULLET.match(line)
        numbered = None if bullet else _NUMBERED.match(line)
        if bullet or numbered:
            kind = "ul" if bullet else "ol"
            if list_kind != kind:
                close_list()
                if bullet:
                    out.append("<ul style='margin:2px 0 2px 14px;padding:0;'>")
                else:
                    first = int(numbered.group(1))
                    start = f" start='{first}'" if first != 1 else ""
                    out.append(f"<ol{start} style='margin:2px 0 2px 14px;padding:0;'>")
                list_kind = kind
            content = bullet.group(1) if bullet else numbered.group(2)
            out.append(f"<li>{_inline(content)}</li>")
            i += 1
            continue
        close_list()
        heading = _HEADING.match(line)
        if heading:
            level = min(len(heading.group(1)) + 2, 5)
            out.append(f"<h{level} style='margin:6px 0 2px;'>{_inline(heading.group(2))}</h{level}>")
        elif _RULE.match(line):
            out.append("<hr>")
        elif line.strip():
            out.append(f"<p style='margin:2px 0;'>{_inline(line)}</p>")
        i += 1
    close_list()
    return "".join(out)


# ---------------------------------------------------------------------------
# anchors


def parse_anchor(target: str) -> tuple[str, list[str]]:
    """``"show:12:3"`` -> ``("show", ["12", "3"])``; ``"plan:toggle"`` -> ``("plan", ["toggle"])``;
    a target without ``:`` -> ``(target, [])``. URL-like targets (``oscar://``, ``http(s)://``) keep the
    scheme as the verb and the rest as ONE argument."""
    target = _text(target)
    verb, sep, rest = target.partition(":")
    if not sep:
        return target, []
    if rest.startswith("//"):
        return verb, [rest[2:]]
    return verb, rest.split(":")



def parse_artifact_anchor(target: str) -> tuple[str, str]:
    """(action, arg) from ``oscar://artifact/<action>[/<arg>]``; ``("", "")`` for any other target.
    ``arg`` is URL-unquoted (a file name)."""
    from urllib.parse import unquote

    prefix = "oscar://artifact/"
    if not _text(target).startswith(prefix):
        return "", ""
    action, _, arg = target[len(prefix):].partition("/")
    return action, unquote(arg)


_OPENABLE_EXT = frozenset((
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".pdf", ".txt", ".md", ".csv", ".json", ".log",
    ".step", ".stp", ".stl", ".obj", ".3mf", ".brep", ".iges", ".igs", ".fcstd", ".dxf",
))


# macOS bundles are DIRECTORIES that the desktop launches or installs when asked to open them
_BUNDLE_EXT = frozenset((".app", ".workflow", ".action", ".pkg", ".mpkg", ".prefpane", ".saver", ".appex", ".xpc",
                         ".kext", ".bundle", ".plugin", ".wdgt", ".scptd", ".framework"))


def classify_file_link(path: str, is_dir: bool = False) -> str:
    """``"open"`` for a path whose extension is a plain document / image / CAD format, else
    ``"reveal"``: a link in model text must never launch a script or an application bundle (``.py``,
    ``.sh``, ``.command``, ``.app``, ``.exe`` ...), the panel shows its folder instead. With ``is_dir=True``
    (the caller checked ``os.path.isdir``) an ordinary folder is ``"open"`` but a macOS bundle directory
    (``Calculator.app``, ``x.workflow``, ``x.pkg`` ...) is ``"reveal"``, because opening one would run it."""
    name = _text(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()
    dot = name.rfind(".")
    ext = name[dot:] if dot > 0 else ""
    if is_dir:
        return "reveal" if ext in _BUNDLE_EXT else "open"
    return "open" if ext in _OPENABLE_EXT else "reveal"


def thumb_url(call_id: str) -> str:
    """The ``QTextDocument`` resource name of a step thumbnail (the panel registers the decoded image
    under exactly this URL after each ``setHtml``)."""
    return "es-img://step/" + quote(_text(call_id), safe="")


# ---------------------------------------------------------------------------
# outcome badge, Build Summary

_BADGES = {
    "validated": ("✓ validated", "ok"),
    "answered": ("", "ok"),
    "unverified": ("⚠ unverified", "warn"),
    "capped": ("⚠ capped", "warn"),
    "needs_input": ("? needs input", "neutral"),
    "failed": ("✗ failed", "error"),
    "needs_help": ("✗ needs help", "error"),
    "stopped": ("■ stopped", "error"),
}


def outcome_badge(outcome: str) -> tuple[str, str]:
    """``(label, kind)`` of a turn outcome (``flow.OUTCOMES``). ``kind`` is ``ok`` / ``warn`` /
    ``error`` / ``neutral``. The label is what follows `` · `` after the legacy header text; it is
    empty for ``answered`` (a plain answer carries no badge) and for an unknown / empty outcome."""
    return _BADGES.get(_text(outcome), ("", "neutral"))


def outcome_color(kind: str, muted: str = "#888888") -> str:
    """The CSS colour of a badge ``kind`` (``muted`` for ``neutral`` and unknown kinds)."""
    return {"ok": _OK, "warn": _WARN, "error": _BAD}.get(kind, _css(muted))


_SUMMARY_HEADING = re.compile(r"^## Build Summary[ \t]*$", re.M)


def split_build_summary(text: str) -> tuple[str, str]:
    """``(answer, summary)``: ``text`` cut at the first line equal to ``## Build Summary``.
    ``answer`` is the head without trailing blank lines; ``summary`` starts at the heading (``""``
    when the text has none, then ``answer`` is the whole text)."""
    text = _text(text)
    match = _SUMMARY_HEADING.search(text)
    if match is None:
        return text, ""
    return text[:match.start()].rstrip(), text[match.start():].strip()


def summary_card_html(markdown: str) -> str:
    """The Build Summary block (``## Build Summary`` ...) as a bordered card; ``""`` for a blank
    input. The markdown goes through ``md_to_html`` so tables, headings and bullets render; the text is
    clipped at 20000 characters."""
    markdown = _text(markdown).strip()
    if not markdown:
        return ""
    body = md_to_html(clip_text(markdown, _SUMMARY_LIMIT))
    return (
        "<div style='border:1px solid rgba(128,128,128,0.28);border-radius:8px;"
        f"padding:4px 10px;margin:6px 0 2px 4px;'>{body}</div>"
    )


def header_text(*, thought_s: float | None, steps: int, total_s: float, failed: bool = False,
                outcome: str = "") -> str:
    """The text of a FINISHED turn card header, one definition for ``TurnView`` and the panel.

    Legacy formats (pinned by gui_smoke): ``✻ Ran 3 steps · 14.0s``, ``✻ Thought 2.2s · Ran 2 steps · 14.0s``,
    ``✻ Thought 2.2s``. ``thought_s`` is ``None`` when the turn had no reasoning round. Nothing ran: ``✻ Done``
    when the outcome is ``answered`` / ``validated`` / unset, ``✻ Ended`` for every other outcome (a turn that
    ended ``unverified`` or ``failed`` is never called done). `` · ⚠ failed steps`` follows when a step failed,
    then the outcome badge (``outcome_badge``) appended AFTER the legacy text."""
    plural = "s" if steps != 1 else ""
    parts: list[str] = []
    if thought_s is not None and steps:
        parts += [f"Thought {thought_s:.1f}s", f"Ran {steps} step{plural}", f"{max(0.0, total_s):.1f}s"]
    elif thought_s is not None:
        parts.append(f"Thought {thought_s:.1f}s")
    elif steps:
        parts += [f"Ran {steps} step{plural}", f"{max(0.0, total_s):.1f}s"]
    if parts:
        head = "✻ " + " · ".join(parts)
    else:
        head = "✻ Done" if _text(outcome) in ("", "answered", "validated") else "✻ Ended"
    if failed:
        head += " · ⚠ failed steps"
    label, _kind = outcome_badge(outcome)
    return f"{head} · {label}" if label else head


def summary_block_html(markdown: str, actions: "list[tuple[str, str]] | None" = None, *,
                       muted: str = "#888888") -> str:
    """The Build Summary card (``summary_card_html``) followed by a row of action links.

    ``actions`` is a list of ``(anchor target, label)``; the only one the panel uses is ``("continue:",
    "Continue")`` after the round cap. ``""`` when there is neither a summary nor an action. Labels are
    escaped; targets are restricted to the anchors the panel itself generates (no quote or space)."""
    card = summary_card_html(markdown)
    links = []
    for target, label in actions or []:
        target = _text(target)
        if not target or re.search(r"[\s'\"<>]", target):
            continue
        links.append(f"<a href='{target}' style='text-decoration:none;'>▶ {_esc(label)}</a>")
    row = ""
    if links:
        row = (f"<div style='margin:4px 0 2px 4px;'>{' &nbsp;·&nbsp; '.join(links)} "
               f"<span style='color:{_css(muted)};font-size:small;'>keeps going from the plan</span></div>")
    return card + row


def unique_call_id(payload: Any, turn_seq: int) -> str:
    """A call id that is unique across turns for a ``step_started`` / ``step_finished`` payload.

    Providers do not guarantee unique tool-call ids (some local servers answer ``call_0`` in every
    response), and rows, thumbnails and the Project all key on them: the panel keys on
    ``t<turn_seq>.<ordinal>.<call_id>`` (the ordinal is identical in the start and the finish payload).
    ``""`` for a payload that has no call id at all."""
    data = _load(payload) or {}
    call_id = _text(data.get("call_id"))
    ordinal = _int(data.get("ordinal"))
    if not call_id and ordinal <= 0:
        return ""
    return f"t{_int(turn_seq)}.{ordinal}.{call_id}"


# ---------------------------------------------------------------------------
# artifacts strip

_ARTIFACT_CHIPS = (
    ("fcstd", "open-doc", "📄"),
    ("notes", "open-notes", "📝"),
    ("rebuild", "open-rebuild", "⚙"),
    ("export", "open-exports", "📦"),
    ("provenance", "open-provenance", "🔖"),
)


def artifacts_strip_html(artifacts: dict | None, *, muted: str = "#888888") -> str:
    """The artifacts strip from an ``artifacts_changed`` payload ``{"project_dir", "files": [{"kind",
    "name", "path"}]}``: one chip per FCStd document (the primary ``model.FCStd`` first, others carry
    their file name as the anchor argument; ``*.FCBak`` backups are ignored), NOTES.md, rebuild.py,
    exports (ONE chip with the count), provenance, and the project folder. Anchors are ``oscar://artifact/<action>[/<arg>]``. ``""`` when
    there is nothing to show; snapshots are not chips (they are thumbnails)."""
    data = _load(artifacts)
    if not data:
        return ""
    files = [f for f in (data.get("files") or [])
             if isinstance(f, dict) and not _text(f.get("name")).lower().endswith(".fcbak")]
    chips: list[str] = []

    def chip(action: str, label: str, icon: str, arg: str = "") -> None:
        target = f"oscar://artifact/{action}" + (f"/{quote(arg, safe='')}" if arg else "")
        chips.append(f"<a href='{target}' style='text-decoration:none;'>{icon} {_esc(label)}</a>")

    for kind, action, icon in _ARTIFACT_CHIPS:
        group = [f for f in files if _text(f.get("kind")) == kind]
        if not group:
            continue
        if kind == "fcstd":
            group.sort(key=lambda f: (_text(f.get("name")) != "model.FCStd", _text(f.get("name"))))
            for index, item in enumerate(group):
                name = _text(item.get("name")) or "model.FCStd"
                chip(action, name, icon, "" if index == 0 and name == "model.FCStd" else name)
        elif kind == "export":
            chip(action, f"exports ({len(group)})" if len(group) > 1 else
                 (_text(group[0].get("name")) or "exports"), icon)
        else:
            chip(action, _text(group[0].get("name")) or kind, icon)
    if _text(data.get("project_dir")):
        chip("open-dir", "Oscar project", "📁")
    if not chips:
        return ""
    return (f"<div style='margin:3px 0;'><span style='color:{_css(muted)};'>Artifacts:</span> "
            + " &nbsp;·&nbsp; ".join(chips) + "</div>")


# ---------------------------------------------------------------------------
# heartbeat


def heartbeat_text(progress: dict | None, *, now: float) -> str:
    """The one-line heartbeat under the live card from a ``progress`` payload, e.g.
    ``round 7/30 · BUILD · thinking 12 s · 1.8k chars``.

    ``now`` is the panel clock (``time.time()``, same clock as ``started_at``). For ``running_tool`` and
    ``validating`` the worker thread is blocked and cannot tick, so the elapsed time is ``now -
    started_at``; every other activity uses the payload's ``elapsed_round``. Activities: thinking,
    writing, tool_args, running_tool, retrying (``retry`` = ``[attempt, max]``), waiting_approval,
    paused, validating. A provider stream silent for 30 s or more adds ``no data for N s``. ``""`` for
    a missing or empty payload; never raises."""
    if not isinstance(progress, dict) or not progress:
        return ""
    parts: list[str] = []
    rnd, cap = _int(progress.get("round")), _int(progress.get("round_cap"))
    if rnd > 0:
        parts.append(f"round {rnd}/{cap}" if cap > 0 else f"round {rnd}")
    phase = _text(progress.get("phase"))
    if phase:
        parts.append(phase)
    activity = _text(progress.get("activity")) or "thinking"
    tool = _text(progress.get("tool"))
    started = _num(progress.get("started_at"))
    elapsed = _num(progress.get("elapsed_round"))
    if activity in ("running_tool", "validating") and started > 0:
        elapsed = max(0.0, _num(now) - started)
    secs = _fmt_secs(elapsed)
    chars = 0
    if activity == "thinking":
        parts.append(f"thinking {secs}")
        chars = _int(progress.get("reasoning_chars"))
    elif activity == "writing":
        parts.append(f"writing {secs}")
        chars = _int(progress.get("text_chars"))
    elif activity == "tool_args":
        parts.append(f"writing {tool or 'tool'} arguments {secs}")
        chars = _int(progress.get("tool_chars"))
    elif activity == "running_tool":
        parts.append(f"running {tool or 'tool'} {secs}")
    elif activity == "validating":
        parts.append(f"validating {secs}")
    elif activity == "retrying":
        retry = progress.get("retry")
        if isinstance(retry, (list, tuple)) and len(retry) == 2:
            parts.append(f"retrying {_int(retry[0])}/{_int(retry[1])}")
        else:
            parts.append("retrying")
    elif activity == "waiting_approval":
        parts.append("waiting for your approval")
    elif activity == "paused":
        parts.append("paused")
    else:
        parts.append(activity.replace("_", " "))
    if chars > 0:
        parts.append(_fmt_chars(chars))
    idle = _num(progress.get("idle_s"))
    if activity in _LIVE_PHASE_ACTIVITIES and idle >= 30:
        parts.append(f"no data for {_fmt_secs(idle)}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# steps


_STEP_FIELDS = {
    "call_id": _text, "name": _text, "title": _text, "args_summary": _text, "round": _int, "ordinal": _int,
    "mutating": _flag, "geometry": _flag, "phase": _text, "started_at": _num, "ok": _flag,
    "skipped": _flag, "text": _text, "dur_s": _num, "code": _text, "args_json": _text,
    "step_index": lambda v: _int(v, -1), "snapshot_path": _text, "snapshot_note": _text,
    "checkpoint": _text, "script_path": _text,
}


@dataclass
class StepView:
    """One tool call of a turn: the ``step_started`` payload fields, the ``step_finished`` payload
    fields and ``running`` (True from ``step_started`` until ``step_finished``)."""

    call_id: str = ""
    name: str = ""
    title: str = ""
    args_summary: str = ""
    round: int = 0
    ordinal: int = 0
    mutating: bool = False
    geometry: bool = False
    phase: str = ""
    started_at: float = 0.0
    ok: bool = True
    skipped: bool = False
    text: str = ""
    dur_s: float = 0.0
    code: str = ""
    args_json: str = ""
    delta: dict = field(default_factory=dict)
    health: dict | None = None
    step_index: int = -1
    snapshot_path: str = ""
    snapshot_note: str = ""
    checkpoint: str = ""
    script_path: str = ""
    running: bool = False
    changed_props: dict = field(default_factory=dict)   # {object name: [changed property names]} from delta["changed"] dicts
    rev: int = 0   # bumped by every ``apply`` / ``touch``: a cheap change signature for the panel's fragment cache
    superseded: bool = False   # a revert / rewind discarded this step's geometry (set by the panel, never by a payload)

    def touch(self) -> None:
        """Mark the view as changed (``running`` was flipped from outside)."""
        self.rev += 1

    @property
    def can_rewind(self) -> bool:
        """True when the row offers ``rewind:`` (the model can go back to just after this step): a finished,
        successful step of the project journal that has a checkpoint and was not discarded since."""
        return (not self.running and self.ok and not self.skipped and not self.superseded
                and self.step_index >= 1 and bool(self.checkpoint))

    @property
    def label(self) -> str:
        """The row title: the step title, else the tool name."""
        return self.title or self.name or "tool"

    @property
    def icon(self) -> str:
        """``◌`` running, ``↷`` skipped, ``✓`` ok, ``✗`` failed."""
        if self.running:
            return "◌"
        if self.skipped:
            return "↷"
        return "✓" if self.ok else "✗"

    @property
    def failed(self) -> bool:
        """True for a finished call that failed (a skipped call did not fail)."""
        return not self.running and not self.skipped and not self.ok

    @property
    def default_expanded(self) -> bool:
        """A failed step opens with its full output (the user must see why)."""
        return self.failed

    def changed_objects(self) -> list[str]:
        """Names of the objects this step added or changed (the targets of the ``show:`` anchor)."""
        names: list[str] = []
        for key in ("added", "changed"):
            values = self.delta.get(key) if isinstance(self.delta, dict) else None
            names += [_text(v) for v in values or [] if _text(v)]
        return names

    def apply(self, payload: dict) -> None:
        """Merge a ``step_started`` / ``step_finished`` payload: known keys are coerced to their field
        type, unknown keys are ignored, absent keys keep their value."""
        for key, convert in _STEP_FIELDS.items():
            if key in payload and payload[key] is not None:
                setattr(self, key, convert(payload[key]))
        delta = payload.get("delta")
        if isinstance(delta, dict):
            # ``changed`` entries are dicts ({"name", "props", ...}) in the Project step record; a plain
            # name is accepted too (the hand-written 5.10 fixtures). The view keeps names only.
            self.delta = {k: [n for n in (_delta_name(e) for e in v) if n]
                          for k, v in delta.items() if isinstance(v, (list, tuple))}
            props: dict[str, list[str]] = {}
            for entry in delta.get("changed") or []:
                if isinstance(entry, dict) and _delta_name(entry) and isinstance(entry.get("props"), (list, tuple)):
                    props[_delta_name(entry)] = [_text(p) for p in entry["props"] if _text(p)][:8]
            self.changed_props = props
        health = payload.get("health")
        if isinstance(health, dict):
            self.health = health
        self.rev += 1


def _delta_name(entry: Any) -> str:
    """The object name of a delta entry: a string, or the ``name`` of a ``{"name", "props", ...}`` dict."""
    if isinstance(entry, dict):
        return _text(entry.get("name"))
    return _text(entry)


def _delta_chips(delta: dict, muted: str, limit: int = 6) -> str:
    chips: list[tuple[str, str]] = []
    for key, mark, color in (("added", "+", _OK), ("changed", "~", _WARN), ("removed", "-", _BAD)):
        for name in (delta or {}).get(key) or []:
            if _text(name):
                chips.append((f"{mark}{_text(name)}", color))
    shown = "".join(f" <span style='color:{c};'>{_esc(t)}</span>" for t, c in chips[:limit])
    if len(chips) > limit:
        shown += f" <span style='color:{muted};'>+{len(chips) - limit} more</span>"
    return shown


def _health_chip(health: dict | None) -> str:
    if not isinstance(health, dict):
        return ""
    errors, warnings = _int(health.get("errors")), _int(health.get("warnings"))
    if errors > 0:
        return f" <span style='color:{_BAD};'>⚠ {errors} error{'s' if errors != 1 else ''}</span>"
    if warnings > 0:
        return f" <span style='color:{_WARN};'>⚠ {warnings} warning{'s' if warnings != 1 else ''}</span>"
    return ""


def step_row_html(view: StepView, bid: int, index: int, *, muted: str = "#888888",
                  expanded: bool = False) -> str:
    """One titled tool-call row of the turn card.

    Collapsed: icon, title, duration, delta chips (``+Shell ~Lid``), a health chip when the call left
    errors / warnings, a ``rewind:<bid>:<index>`` link when the step has a checkpoint (``StepView.can_rewind``;
    a discarded step is struck through and offers none), and the thumbnail (``<img src='es-img://step/<call_id>' width='160'>`` inside a
    ``show:<bid>:<index>`` anchor) or the ``snapshot_note`` in muted text (not the expected ``not a geometry
    tool`` of a non-geometry step). Expanded adds the script in a
    ``<pre>`` with a ``copy:<bid>:<index>`` anchor (or the compact arguments), the full output (clipped
    at ``DETAIL_LIMIT``), the delta lists and the health line. The head anchor toggles the row
    (``step:<bid>:<index>``). Everything from the payload is HTML-escaped."""
    bid, index, muted = _int(bid), _int(index), _css(muted)
    head = f"{view.icon} {view.label}"
    if view.running:
        head += " …"
    elif view.dur_s > 0:
        head += f" · {_fmt_dur(view.dur_s)}"
    if view.skipped and not view.running:
        head += " · skipped"
    if view.superseded:
        head = f"{head} · discarded"
    row = (
        "border:1px solid rgba(128,128,128,0.28);border-radius:8px;"
        "padding:4px 8px;margin:2px 0 2px 14px;"
    )
    parts = [
        f"<a href='step:{bid}:{index}' style='color:{muted};text-decoration:none;'>"
        f"{'<s>' + _esc(head) + '</s>' if view.superseded else _esc(head)}</a>",
        _delta_chips(view.delta, muted),
        _health_chip(view.health),
    ]
    if view.can_rewind:
        parts.append(f" <a href='rewind:{bid}:{index}' style='color:{muted};font-size:small;text-decoration:none;'"
                     f" title='Restore the model to just after this step'>↶ rewind</a>")
    if view.snapshot_path:
        parts.append(
            f"<div style='padding:3px 0 0 2px;'><a href='show:{bid}:{index}'>"
            f"<img src='{thumb_url(view.call_id)}' width='160'></a></div>")
    elif view.snapshot_note and view.snapshot_note != _NO_NOTE:
        parts.append(f"<div style='color:{muted};font-size:small;padding:2px 0 0 2px;'>"
                     f"{_esc(view.snapshot_note)}</div>")
    if expanded and not view.running:
        parts.append(_step_detail_html(view, bid, index, muted))
    return f"<div style='{row}'>{''.join(parts)}</div>"


def _step_detail_html(view: StepView, bid: int, index: int, muted: str) -> str:
    out: list[str] = []
    label = f"color:{muted};font-size:small;padding:4px 0 0 4px;"
    if view.code:
        out.append(f"<div style='{label}'>code · <a href='copy:{bid}:{index}'>copy</a></div>"
                   f"<pre>{_esc(clip_text(view.code))}</pre>")
    elif view.args_json:
        out.append(f"<div style='{label}'>arguments: {_esc(clip_text(view.args_json, 4000))}</div>")
    if view.text:
        out.append(f"<div style='white-space:pre-wrap;padding:3px 0 0 4px;'>"
                   f"{_esc(clip_text(view.text))}</div>")
    elif not view.code:
        out.append(f"<div style='{label}'>(no output)</div>")
    for key, mark in (("added", "added"), ("changed", "changed"), ("removed", "removed")):
        names = [_text(n) for n in (view.delta or {}).get(key) or [] if _text(n)]
        if names:
            more = f" … +{len(names) - 30} more" if len(names) > 30 else ""
            if key == "changed" and view.changed_props:
                shown = [f"{n} ({', '.join(view.changed_props[n])})" if view.changed_props.get(n) else n
                         for n in names[:30]]
            else:
                shown = names[:30]
            out.append(f"<div style='{label}'>{mark}: {_esc(', '.join(shown))}{more}</div>")
    if isinstance(view.health, dict) and _text(view.health.get("line")):
        out.append(f"<div style='{label}'>{_esc(view.health.get('line'))}</div>")
    return "".join(out)


# ---------------------------------------------------------------------------
# plan strip

_PLAN_ICONS = {"pending": "○", "running": "◌", "done": "✓", "blocked": "✗", "skipped": "↷", "changed": "↻"}
_PLAN_TERMINAL = ("done", "blocked", "skipped", "changed")


def _plan_steps(ledger: dict | None) -> list[dict]:
    steps = ledger.get("steps") if isinstance(ledger, dict) else None
    return [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []


def _current_index(steps: list[dict]) -> int:
    """Index of the running step, else of the first non-terminal required step, else ``-1``."""
    for i, step in enumerate(steps):
        if _text(step.get("status")) == "running":
            return i
    for i, step in enumerate(steps):
        if _flag(step.get("required"), True) and _text(step.get("status")) not in _PLAN_TERMINAL:
            return i
    return -1


def plan_strip_html(ledger: dict | None, *, accent: str = "#1e6ee8", muted: str = "#888888",
                    collapsed: bool = False) -> str:
    """The pinned plan strip from a Ledger JSON dict; ``""`` when there are no steps.

    Header ``▾ Plan · step 4/10 · round 7/30 · BUILD`` is the ``plan:toggle`` anchor (``step`` is the
    1-based position of the current step among all steps; at the end it is the total). One row per step
    with the status icon (``○`` pending, ``◌`` running, ``✓`` done, ``✗`` blocked, ``↷`` skipped,
    ``↻`` changed), id, title, stage progress ``(1/2)`` for staged steps, ``(claimed)`` for a done step
    without app evidence, ``(optional)``, and the note of a blocked / skipped / changed step on a
    second line; the current step is bold in the accent colour. A muted ``intent: <name>`` line follows the
    header unless the intent is ``other`` / unset (the intent chip). ``collapsed`` keeps only the header
    and the current step. The function never collapses by itself."""
    steps = _plan_steps(ledger)
    if not steps:
        return ""
    accent, muted = _css(accent), _css(muted)
    current = _current_index(steps)
    ordinal = (current + 1) if current >= 0 else len(steps)
    head = [f"step {ordinal}/{len(steps)}"]
    rnd, cap = _int(ledger.get("round")), _int(ledger.get("round_cap"))
    if rnd > 0:
        head.append(f"round {rnd}/{cap}" if cap > 0 else f"round {rnd}")
    if _text(ledger.get("phase")):
        head.append(_text(ledger.get("phase")))
    caret = "▸" if collapsed else "▾"
    out = [f"<div><a href='plan:toggle' style='color:{muted};text-decoration:none;'>"
           f"{caret} Plan · {_esc(' · '.join(head))}</a></div>"]
    intent = _text(ledger.get("intent")).strip()
    if not collapsed and intent and intent != "other":
        out.append(f"<div style='color:{muted};font-size:small;margin:0 0 2px 8px;'>intent: {_esc(intent)}</div>")
    for i, step in enumerate(steps):
        if collapsed and i != current:
            continue
        status = _text(step.get("status")) or "pending"
        icon = _PLAN_ICONS.get(status, "?")
        color = {"done": _OK, "blocked": _BAD, "running": accent}.get(status, muted)
        title = _esc(step.get("title") or step.get("id") or "step")
        if i == current:
            title = f"<b style='color:{accent};'>{title}</b>"
        tags: list[str] = []
        stages = step.get("stages")
        if isinstance(stages, list) and stages:
            done = step.get("stages_done")
            tags.append(f"({len(done) if isinstance(done, list) else 0}/{len(stages)})")
        if status == "done" and _flag(step.get("claimed")):
            tags.append("(claimed)")
        if not _flag(step.get("required"), True):
            tags.append("(optional)")
        tail = f" <span style='color:{muted};'>{_esc(' '.join(tags))}</span>" if tags else ""
        row = (f"<div style='margin:1px 0 1px 8px;'><span style='color:{color};'>{icon}</span> "
               f"<span style='color:{muted};'>{_esc(step.get('id'))}</span> {title}{tail}")
        note = _text(step.get("note")).strip()
        if note and status in ("blocked", "skipped", "changed"):
            row += (f"<div style='color:{muted};font-size:small;margin-left:22px;'>"
                    f"{_esc(clip_text(note, 160))}</div>")
        out.append(row + "</div>")
    reference = _text(ledger.get("reference_status"))
    if not collapsed and ledger.get("needs_reference") and reference and reference != "none":
        out.append(f"<div style='color:{muted};font-size:small;margin:2px 0 0 8px;'>"
                   f"reference: {_esc(reference)}</div>")
    return "".join(out)


# ---------------------------------------------------------------------------
# the turn model


class TurnView:
    """The model of one turn, fed with the worker's JSON signal payloads.

    GUI-thread only (the signals arrive queued on the GUI thread). Every ``on_*`` method accepts the
    JSON string or a dict, ignores malformed input and never raises. ``clock`` (default
    ``time.time``) is injectable; ``now`` arguments use the same clock.

    Attributes: ``steps`` (``list[StepView]`` in call order, one per ``call_id``: a duplicate
    ``step_started`` / ``step_finished`` updates the same row), ``plan`` (last Ledger dict),
    ``progress`` (last progress dict), ``artifacts``, ``outcome`` / ``summary`` (set by
    ``on_outcome``), ``live_text`` (streamed text of the current round, cleared by ``on_round_reset``),
    ``thought_s`` (seconds the panel measured for reasoning rounds).
    """

    def __init__(self, *, clock: Callable[[], float] = time.time, started_at: float | None = None):
        self._clock = clock
        self.t0 = float(clock() if started_at is None else started_at)
        self.finished_at: float | None = None
        self.steps: list[StepView] = []
        self.plan: dict | None = None
        self.progress: dict | None = None
        self.progress_at: float | None = None   # panel clock time the latest progress payload arrived
        self.artifacts: dict | None = None
        self.outcome = ""
        self.summary = ""
        self.live_text = ""
        self.resets: list[str] = []
        self.thought_s = 0.0
        self.has_thoughts = False
        self._by_id: dict[str, StepView] = {}
        self._toggled: set[str] = set()

    # -- payload consumers --------------------------------------------------------------------

    def on_plan(self, payload: Any) -> bool:
        """``plan_changed``: store the ledger dict; False (state unchanged) for a malformed payload."""
        data = _load(payload)
        if data is None:
            return False
        self.plan = data
        return True

    def on_step_started(self, payload: Any) -> StepView | None:
        """``step_started``: add a running row (one row per ``call_id``: a repeated start for a known
        call updates it, and a call that already finished stays finished)."""
        data = _load(payload)
        if data is None:
            return None
        call_id = _text(data.get("call_id"))
        view = self._by_id.get(call_id) if call_id else None
        if view is None:
            view = StepView(running=True)
            view.apply(data)
            self.steps.append(view)
            if call_id:
                self._by_id[call_id] = view
        elif view.running:
            view.apply(data)
        return view

    def on_step_finished(self, payload: Any) -> StepView | None:
        """``step_finished``: complete the row of that ``call_id`` (else the last running row of the same
        tool name, else a new row), set ok / skipped / text / delta / health / snapshot."""
        data = _load(payload)
        if data is None:
            return None
        call_id = _text(data.get("call_id"))
        view = self._by_id.get(call_id) if call_id else None
        if view is None:
            name = _text(data.get("name"))
            view = next((v for v in reversed(self.steps) if v.running and (not name or v.name == name)), None)
        if view is None:
            view = StepView()
            self.steps.append(view)
        view.apply(data)
        view.running = False
        view.touch()
        if "ok" not in data:
            view.ok = True
        if view.call_id:
            self._by_id[view.call_id] = view
        return view

    def on_progress(self, payload: Any) -> bool:
        """``progress``: keep the latest payload for the heartbeat; False for a malformed payload."""
        data = _load(payload)
        if data is None:
            return False
        self.progress = data
        self.progress_at = float(self._clock())
        return True

    def on_round_reset(self, reason: str = "") -> None:
        """``round_reset``: the partial reply of the current round is void (clears ``live_text``)."""
        self.live_text = ""
        self.resets.append(_text(reason))
        del self.resets[:-50]

    def on_text(self, chunk: str) -> None:
        """Accumulate streamed answer text of the current round (kept to the last 200000 characters)."""
        self.live_text = (self.live_text + _text(chunk))[-200000:]

    def on_artifacts(self, payload: Any) -> bool:
        """``artifacts_changed``: store the payload for the strip; False for a malformed payload."""
        data = _load(payload)
        if data is None:
            return False
        self.artifacts = data
        return True

    def on_outcome(self, outcome: str, summary: str = "") -> None:
        """``turn_outcome``: the turn is over. Rows still marked running (a stopped turn) are closed as
        failed so the header never says ``Working`` afterwards."""
        self.outcome = _text(outcome)
        self.summary = _text(summary)
        self.finish()

    def finish(self) -> None:
        """Mark the turn finished without an outcome (legacy ``turn_finished`` only); idempotent."""
        if self.finished_at is None:
            self.finished_at = float(self._clock())
        for view in self.steps:
            if view.running:
                view.running = False
                view.ok = False
                view.text = view.text or "Interrupted before the step finished."
                view.touch()

    def add_thought(self, seconds: float) -> None:
        """Account one reasoning round (the panel measures it; reasoning text is not a JSON payload)."""
        self.thought_s += max(0.0, _num(seconds))
        self.has_thoughts = True

    # -- queries ------------------------------------------------------------------------------

    @property
    def finished(self) -> bool:
        """True once ``on_outcome`` / ``finish`` ran."""
        return self.finished_at is not None

    @property
    def running_steps(self) -> list[StepView]:
        """The rows that have started and not finished."""
        return [v for v in self.steps if v.running]

    def elapsed(self, now: float | None = None) -> float:
        """Seconds since the turn began (frozen at the end once finished)."""
        end = self.finished_at if self.finished_at is not None else float(self._clock() if now is None else now)
        return max(0.0, end - self.t0)

    def header(self, now: float | None = None) -> str:
        """The turn card header. While the turn is live: ``✻ Working… 12s`` (NEVER ``Done``, also between
        rounds and while a step runs). After the outcome the legacy format ``✻ Thought 2.2s · Ran 2
        steps · 14.0s`` (``✻ Done`` when nothing ran and the outcome is ``answered`` / ``validated`` / unset,
        ``✻ Ended`` otherwise), `` · ⚠ failed steps`` when a step failed, then the outcome badge
        `` · ✓ validated`` appended AFTER the legacy text (``header_text``)."""
        if not self.finished:
            return f"✻ Working… {self.elapsed(now):.0f}s"
        return header_text(thought_s=self.thought_s if self.has_thoughts else None, steps=len(self.steps),
                           total_s=self.elapsed(), failed=any(v.failed for v in self.steps), outcome=self.outcome)

    @property
    def header_kind(self) -> str:
        """``live`` while running, then the outcome badge kind (``ok`` / ``warn`` / ``error`` /
        ``neutral``); a finished turn without an outcome is ``ok`` unless a step failed (``warn``)."""
        if not self.finished:
            return "live"
        if self.outcome:
            return outcome_badge(self.outcome)[1]
        return "warn" if any(v.failed for v in self.steps) else "ok"

    def toggle(self, key: str) -> None:
        """Flip the user's expand / collapse choice of the row ``key`` (see ``row_key``)."""
        self._toggled ^= {key}

    def row_key(self, view: StepView) -> str:
        """The stable key of a row: its ``call_id``, else its position."""
        return view.call_id or f"#{self.steps.index(view)}"

    def is_expanded(self, view: StepView) -> bool:
        """Whether the row shows its detail: the default (failed rows open) flipped by ``toggle``."""
        return view.default_expanded != (self.row_key(view) in self._toggled)

    def rows_html(self, bid: int, *, muted: str = "#888888") -> str:
        """All step rows rendered with their current expansion state."""
        return "".join(
            step_row_html(v, bid, i, muted=muted, expanded=self.is_expanded(v))
            for i, v in enumerate(self.steps))

    def heartbeat(self, now: float | None = None) -> str:
        """``heartbeat_text`` of the latest progress (``""`` once the turn is finished).

        Without ``now`` (what the panel's 1 s timer calls) the view's own clock is used and, for the
        provider activities (``thinking`` / ``writing`` / ``tool_args``), the seconds since the payload
        arrived are added to ``elapsed_round`` and ``idle_s``: the counter keeps moving between two payloads
        and a stream that went silent shows ``no data for N s`` even when no further payload arrives. An
        explicit ``now`` is taken as "the payload is current" and nothing is added (``now`` is then also the
        clock ``started_at`` of a running tool is compared with)."""
        if self.finished:
            return ""
        progress = self.progress
        if now is None:
            now = float(self._clock())
            if isinstance(progress, dict) and self.progress_at is not None \
                    and _text(progress.get("activity")) in _LIVE_PHASE_ACTIVITIES:
                gap = max(0.0, now - self.progress_at)
                if gap > 0:
                    progress = dict(progress)
                    progress["elapsed_round"] = _num(progress.get("elapsed_round")) + gap
                    progress["idle_s"] = _num(progress.get("idle_s")) + gap
        return heartbeat_text(progress, now=float(now))

    def plan_view(self) -> dict | None:
        """The latest ledger with the newest round / cap / phase of the latest ``progress`` payload merged in
        (the ledger JSON only arrives when the plan changes; the round counter moves every round)."""
        plan = self.plan
        progress = self.progress
        if plan is None or not isinstance(progress, dict):
            return plan
        merged = dict(plan)
        if _int(progress.get("round")) > 0:
            merged["round"] = _int(progress.get("round"))
        if _int(progress.get("round_cap")) > 0:
            merged["round_cap"] = _int(progress.get("round_cap"))
        if _text(progress.get("phase")):
            merged["phase"] = _text(progress.get("phase"))
        return merged

    def plan_html(self, *, accent: str = "#1e6ee8", muted: str = "#888888", collapsed: bool = False) -> str:
        """``plan_strip_html`` of the latest plan (``plan_view``)."""
        return plan_strip_html(self.plan_view(), accent=accent, muted=muted, collapsed=collapsed)

    def summary_html(self) -> str:
        """``summary_card_html`` of the outcome's Build Summary (``""`` when there is none)."""
        return summary_card_html(self.summary)

    def artifacts_html(self, *, muted: str = "#888888") -> str:
        """``artifacts_strip_html`` of the latest artifacts payload."""
        return artifacts_strip_html(self.artifacts, muted=muted)


# ---------------------------------------------------------------------------
# persisted turn records (``blocks_meta``) and the restore card

META_VERSION = 1
_META_STEPS = 60
_META_TURNS = 20


def _delta_text(delta: dict) -> str:
    chips = []
    for key, mark in (("added", "+"), ("changed", "~"), ("removed", "-")):
        chips += [f"{mark}{_text(n)}" for n in (delta or {}).get(key) or [] if _text(n)]
    return " ".join(chips[:6]) + (f" +{len(chips) - 6} more" if len(chips) > 6 else "")


def make_turn_meta(request: str, outcome: str, header: str, steps: "list[StepView]", *,
                   root: str = "") -> dict:
    """The compact JSON-safe record of ONE finished turn that a chat persists as ``blocks_meta``.

    ``{"v", "request" (<= 160 chars), "outcome", "header", "steps": [{"title", "ok", "skipped", "dur",
    "delta", "snapshot"}]}`` with at most 60 steps (the first ones; ``"more"`` counts the rest). ``snapshot``
    is the PNG path relative to ``root`` (the project folder) when it lies inside it, else the absolute path
    (restored chats resolve it against the project root, 9.3). Never raises."""
    items = []
    for view in list(steps)[:_META_STEPS]:
        snap = _text(view.snapshot_path)
        if snap and root and os.path.isabs(snap):
            try:
                rel = os.path.relpath(snap, root)
            except ValueError:   # another drive (Windows)
                rel = ".."
            if not rel.startswith(".."):
                snap = rel.replace(os.sep, "/")
        items.append({"title": clip_text(view.label, 80), "ok": bool(view.ok), "skipped": bool(view.skipped),
                      "dur": round(_num(view.dur_s), 1), "delta": _delta_text(view.delta), "snapshot": snap})
    meta = {"v": META_VERSION, "request": clip_text(" ".join(_text(request).split()), 160),
            "outcome": _text(outcome), "header": _text(header), "steps": items}
    if len(steps) > _META_STEPS:
        meta["more"] = len(steps) - _META_STEPS
    return meta


def trim_turn_metas(metas: Any) -> list[dict]:
    """``metas`` filtered to well-formed records and cut to the newest 20 (a damaged chat file never breaks the
    restore card)."""
    good = [m for m in metas if isinstance(m, dict) and isinstance(m.get("steps"), list)] \
        if isinstance(metas, list) else []
    return good[-_META_TURNS:]


def restore_thumbs(meta: Any, key: str, limit: int = 4) -> list[tuple[str, str]]:
    """``(thumbnail id, stored snapshot path)`` of the newest ``limit`` steps of the newest turn that have a
    snapshot (oldest first); ``key`` namespaces the ids (the restore card's block id)."""
    metas = trim_turn_metas(meta)
    found: list[tuple[str, str]] = []
    for turn in reversed(metas):
        steps = [s for s in turn.get("steps") or [] if isinstance(s, dict) and _text(s.get("snapshot"))]
        if steps:
            found = [(f"restore.{_text(key)}.{i}", _text(s.get("snapshot"))) for i, s in enumerate(steps[-limit:])]
            break
    return found


def restore_card_html(title: str, messages: int, meta: Any, *, notice: str = "", muted: str = "#888888",
                      key: str = "r", max_turns: int = 3, thumbs: "set[str] | None" = None) -> str:
    """The card shown when a saved chat is opened again: the legacy ``↩ Restored “title” — N messages`` line,
    the interruption ``notice`` (``chats.recovery_notice``) in warning colour, then the newest ``max_turns``
    turn records (request, header with its badge, up to 8 step titles with their object delta) and the
    thumbnails of the last turn that the panel could load (``thumbs``: the ids it registered; ``None`` = all).
    Every text is escaped."""
    muted = _css(muted)
    out = [f"<div style='color:{muted};font-size:small;margin:3px 0 3px 4px;'>↩ Restored “{_esc(title)}” — "
           f"{_int(messages)} messages of task context. Continue where you left off.</div>"]
    if _text(notice).strip():
        out.append(f"<div style='color:{_WARN};margin:3px 0 3px 4px;'>⚠ {_esc(_text(notice).strip())}</div>")
    metas = trim_turn_metas(meta)[-max(1, int(max_turns)):]
    for turn in metas:
        request = _text(turn.get("request")) or "(no request text)"
        head = _text(turn.get("header"))
        out.append(f"<div style='margin:4px 0 0 4px;'><b>{_esc(request)}</b> "
                   f"<span style='color:{muted};'>{_esc(head)}</span>")
        steps = [s for s in turn.get("steps") or [] if isinstance(s, dict)]
        for step in steps[:8]:
            icon = "↷" if step.get("skipped") else ("✓" if step.get("ok", True) else "✗")
            delta = f" <span style='color:{muted};'>{_esc(step.get('delta'))}</span>" if _text(step.get("delta")) else ""
            out.append(f"<div style='margin-left:14px;color:{muted};'>{icon} {_esc(step.get('title'))}{delta}</div>")
        hidden = len(steps) - 8 + _int(turn.get("more"))
        if hidden > 0:
            out.append(f"<div style='margin-left:14px;color:{muted};'>… +{hidden} more steps</div>")
        out.append("</div>")
    ids = [tid for tid, _path in restore_thumbs(meta, key) if thumbs is None or tid in thumbs]
    if ids:
        out.append("<div style='margin:4px 0 0 4px;'>"
                   + " ".join(f"<img src='{thumb_url(tid)}' width='120'>" for tid in ids) + "</div>")
    return "".join(out)


# ---------------------------------------------------------------------------
# the OriginShape account: the header badge, its menu and the ACCOUNT cards
#
# ``state`` is the account view the panel keeps (``settings_ui.AccountHub.state``): ``{"known": bool, "signed_in":
# bool, "email": str, "snapshot": AccountSnapshot.to_dict() | None, "update": "" | "available" | "required",
# "latest_version": str, "min_version": str, "oscar_version": str, "error": str, "error_code": str}``. Any field may be
# missing or of the wrong type: these functions never raise.

ACCOUNT_CARD_KINDS = ("onboarding", "signed_out", "waitlisted", "no_credits", "outdated")

# The only targets an account card may link to. They name an action of the panel; which page an action opens is
# decided by the panel from its own configuration, never taken from the link.
ACCOUNT_TARGETS = frozenset({
    "oscar://account/signin", "oscar://account/device", "oscar://account/buy", "oscar://account/subscribe",
    "oscar://account/refresh", "oscar://account/waitlist", "oscar://account/manage", "oscar://account/download",
    "oscar://settings",
})

# ``account_problem`` codes (``ProviderError.code`` of kind "auth" / "account") that have a card of their own
_PROBLEM_CARDS = {"signed_out": "signed_out", "insufficient_credits": "no_credits", "waitlisted": "waitlisted"}
_WARN_CARDS = ("signed_out", "waitlisted", "no_credits", "outdated")
_URL_PROBLEMS = ("invalid_url", "insecure_url")


def format_credits(value: Any) -> str:
    """``42.5 credits``, ``1 credit``, ``1,250 credits``; a negative or unreadable value reads as ``0 credits`` (the
    available credits dip below zero while requests hold reservations)."""
    number = max(0.0, _num(value))
    text = f"{number:,.2f}".rstrip("0").rstrip(".")
    return f"{text} credit" if text == "1" else f"{text} credits"


def _account_snapshot(state: Any) -> dict:
    snapshot = state.get("snapshot") if isinstance(state, dict) else None
    return snapshot if isinstance(snapshot, dict) else {}


def _account_email(state: Any) -> str:
    email = _account_snapshot(state).get("email") or (state.get("email") if isinstance(state, dict) else "")
    return _text(email).strip()


def account_badge(state: Any) -> tuple[str, str, str]:
    """``(text, tone, tooltip)`` of the header badge: ``Sign in``, ``42 credits``, ``0 credits``, ``Waitlist``,
    ``Update Oscar`` (``state["update"]`` is ``"required"``: this Oscar, ``state["oscar_version"]``, is older than
    the server's minimum; the tooltip says which versions) ...
    ``tone`` is ``"warn"`` when the account cannot chat as it is (no credits left, suspended, a server URL Oscar
    refuses), else ``""``. The text is plain (the panel puts it on a button, never into HTML)."""
    if not isinstance(state, dict) or not state.get("known"):
        return "…", "", "Checking your OriginShape sign-in…"
    error = _text(state.get("error")).strip()
    if _text(state.get("error_code")) in _URL_PROBLEMS:
        return "Sign in", "warn", error or "The OriginShape server set in Oscar's settings cannot be used."
    if not state.get("signed_in"):
        return "Sign in", "", "Sign in to OriginShape to chat with your account (no API key needed)."
    snapshot = _account_snapshot(state)
    email = _account_email(state)
    parts = [f"Signed in as {email}" if email else "Signed in to OriginShape"]
    status = _text(snapshot.get("status"))
    update, mine = _text(state.get("update")), _text(state.get("oscar_version")).strip()
    if update == "required":
        minimum = _text(state.get("min_version")).strip()
        parts.append(f"this Oscar ({mine}) is older than the server supports: update to {minimum} or newer")
        return "Update Oscar", "warn", " · ".join(parts)
    if update == "available":
        parts.append(f"Oscar {_text(state.get('latest_version')).strip()} is available (this is {mine})")
    if status == "waitlisted":
        text, tone = "Waitlist", ""
        parts.append("your account is on the waitlist")
    elif status == "suspended":
        text, tone = "Suspended", "warn"
        parts.append("your account is suspended")
    elif status == "approved":
        available = _num(snapshot.get("credits_available"))
        text, tone = format_credits(available), ("warn" if available <= 0 else "")
        low = " (running low)" if snapshot.get("credits_low") is True and available > 0 else ""
        parts.append(f"{text} left{low}")
    else:   # signed in, the account not loaded yet
        text, tone = "Signed in", ""
    plan = _text(snapshot.get("plan_name") or snapshot.get("plan")).strip()
    if plan:
        parts.append(f"{plan} plan")
    if error:
        parts.append(f"last check failed: {error}")
    return text, tone, " · ".join(parts)


def account_menu(state: Any) -> list[tuple[str, str]]:
    """The badge menu as ``(action, label)`` rows: ``("", label)`` is a disabled line, ``("-", "")`` a separator.
    The actions are ``signin``, ``device``, ``download`` (an update of Oscar is available or required), ``buy``,
    ``manage``, ``refresh`` and ``signout``."""
    if not isinstance(state, dict) or not state.get("known"):
        return [("", "Checking your OriginShape sign-in…"), ("-", ""), ("refresh", "Refresh")]
    update = [("download", f"Update Oscar to {latest}…" if latest else "Update Oscar…")
              for latest in [_text(state.get("latest_version")).strip()]
              if _text(state.get("update")) in ("available", "required")]
    if not state.get("signed_in"):
        return [("signin", "Sign in with browser…"), ("device", "Sign in with a code…"), *update]
    email = _account_email(state)
    rows = [("", f"Signed in as {email}" if email else "Signed in to OriginShape")]
    if _text(_account_snapshot(state).get("status")) == "waitlisted":
        rows.append(("", "Your account is on the waitlist"))
    return rows + update + [("-", ""), ("buy", "Buy credits…"), ("manage", "Manage account…"), ("refresh", "Refresh"),
                            ("-", ""), ("signout", "Sign out")]


def account_card(kind: str, *, reason: str = "", returned: bool = False) -> tuple[str, list[tuple[str, str]]]:
    """``(markdown, actions)`` of an ``ACCOUNT`` card. ``kind`` is one of ``ACCOUNT_CARD_KINDS`` (anything else:
    ``("", [])``). ``reason="no_key"`` words the onboarding card for a profile that lacks its API key. ``returned``
    says the message the user sent is back in the input (its very first request failed)."""
    back = " Your message is back in the input: send it again when you are ready." if returned else ""
    signin = [("oscar://account/signin", "Sign in with browser"), ("oscar://account/device", "Sign in with a code")]
    own_key = ("oscar://settings", "Use my own API key")
    if kind == "onboarding":
        if reason == "no_key":
            text = ("**This profile has no API key.** Add your provider's key in ⚙ Settings, or sign in to OriginShape "
                    "to chat with your account instead: no key needed.")
        else:
            text = ("**Sign in to OriginShape** to chat with Oscar: your account pays for the AI models with credits, "
                    "so you need no API key. You can also use your own provider's key in ⚙ Settings.")
        return text, signin + [own_key]
    if kind == "signed_out":
        return ("**Your OriginShape sign-in has ended.** Sign in again to continue." + back,
                [("oscar://account/signin", "Sign in again"), ("oscar://account/device", "Sign in with a code")])
    if kind == "waitlisted":
        return ("**Your OriginShape account is on the waitlist.** You can chat as soon as it is approved; you will "
                "hear from us by email." + back, [("oscar://account/refresh", "Check again"), own_key])
    if kind == "no_credits":
        return ("**You have no OriginShape credits left.** Buy credits or choose a plan to continue." + back,
                [("oscar://account/buy", "Buy credits"), ("oscar://account/subscribe", "See plans"), own_key])
    if kind == "outdated":
        return ("**This Oscar is too old for your OriginShape account.** Update Oscar from FreeCAD's Addon Manager or "
                "the download page, then restart FreeCAD." + back,
                [("oscar://account/download", "Get the update"), ("oscar://account/refresh", "Check again"), own_key])
    return "", []


def account_card_html(markdown: str, actions: "list[tuple[str, str]] | None" = None, *, kind: str = "",
                      muted: str = "#888888") -> str:
    """An ``ACCOUNT`` card: a caption, the markdown in a bordered box (warning-coloured for ``signed_out``,
    ``waitlisted`` and ``no_credits``) and a row of action links. Only the panel's own account targets
    (``ACCOUNT_TARGETS``) become links; every label is escaped. ``""`` when there is neither text nor a link."""
    markdown = _text(markdown).strip()
    body = md_to_html(clip_text(markdown, 4000)) if markdown else ""
    links = [f"<a href='{_text(target)}' style='text-decoration:none;'>▶ {_esc(label)}</a>"
             for target, label in actions or [] if _text(target) in ACCOUNT_TARGETS]
    if not body and not links:
        return ""
    row = f"<div style='margin:4px 0 2px 0;'>{' &nbsp;·&nbsp; '.join(links)}</div>" if links else ""
    border = _WARN if _text(kind) in _WARN_CARDS else "rgba(128,128,128,0.28)"
    return (f"<div style='border:1px solid {border};border-radius:8px;padding:4px 10px;margin:6px 0 2px 4px;'>"
            f"<div style='color:{_css(muted)};font-size:small;'>OriginShape account</div>{body}{row}</div>")


def account_problem_card(problem: Any) -> str:
    """The card kind for an ``account_problem`` payload (the JSON string or its dict): ``signed_out``,
    ``no_credits`` or ``waitlisted``; ``""`` when the problem has no card (the turn's error line is shown)."""
    data = _load(problem)
    if data is None or _text(data.get("kind")) not in ("auth", "account"):
        return ""
    return _PROBLEM_CARDS.get(_text(data.get("code")), "")
