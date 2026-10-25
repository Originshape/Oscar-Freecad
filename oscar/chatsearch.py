"""Searching the panel's chats (the menu's **Search chats…**): their titles and every prompt and reply.

The rule: the query's words are matched as word prefixes in any order, case- and accent-insensitive, and every word
must be found ("brack m3" finds "M3 bracket"). A chat whose title matches comes first; each result carries a snippet
around the first match in the messages, with the matched words marked for the list.

Pure logic, no Qt, no FreeCAD: unit-testable in any interpreter.
"""

import re
import unicodedata
from dataclasses import dataclass

MAX_RESULTS = 50
SNIPPET_CHARS = 90        # around the first match
MAX_TERMS = 8

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class Found:
    """One chat that matches: its index in the list searched, its title, the role and snippet of the first message
    that matches (empty when only the title does), and the matched terms (for highlighting)."""
    index: int
    title: str
    role: str
    snippet: str
    terms: tuple


def _fold(text: str) -> str:
    """Lower case without accents ("Résumé" -> "resume")."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def terms(query: str) -> tuple:
    """The query's words, folded, at most ``MAX_TERMS``, duplicates removed."""
    out = []
    for word in _WORD.findall(_fold(query or "")):
        if word not in out:
            out.append(word)
    return tuple(out[:MAX_TERMS])


def _has_all(text: str, wanted: tuple) -> bool:
    words = _WORD.findall(_fold(text))
    return all(any(word.startswith(term) for word in words) for term in wanted)


def _first_at(text: str, wanted: tuple) -> int:
    """Where the first word starting with one of ``wanted`` begins in ``text`` (-1 when none). Folding keeps
    character positions for the common case (one folded character per character); a mismatch only shifts the
    snippet a little."""
    folded = _fold(text)
    best = -1
    for match in _WORD.finditer(folded):
        if any(match.group(0).startswith(term) for term in wanted):
            best = match.start()
            break
    return min(best, len(text) - 1) if best >= 0 else -1


def snippet(text: str, wanted: tuple, width: int = SNIPPET_CHARS) -> str:
    """One line of ``text`` around its first match: whitespace collapsed, ``…`` where it was cut."""
    flat = " ".join(str(text or "").split())
    at = _first_at(flat, wanted)
    if at < 0:
        return flat[:width] + ("…" if len(flat) > width else "")
    start = max(0, at - width // 3)
    end = min(len(flat), start + width)
    start = max(0, end - width)
    return ("…" if start > 0 else "") + flat[start:end] + ("…" if end < len(flat) else "")


def find(chats: list, query: str, limit: int = MAX_RESULTS) -> list:
    """The chats matching ``query``, best first. ``chats`` holds ``(title, [(role, text), ...])`` per chat, in the
    order the results' ``index`` refers to."""
    wanted = terms(query)
    if not wanted:
        return []
    titled, other = [], []
    for index, (title, messages) in enumerate(chats):
        # the title, or one message, holds every word
        title_hit = _has_all(title, wanted)
        message = next(((role, text) for role, text in messages if text and _has_all(text, wanted)), None)
        if not title_hit and message is None:
            continue
        role, text = message if message else ("", "")
        found = Found(index=index, title=title, role=role, snippet=snippet(text, wanted) if text else "", terms=wanted)
        (titled if title_hit else other).append(found)
    return (titled + other)[:limit]


def mark(text: str, wanted: tuple) -> list:
    """``text`` as ``[(piece, matched)]``: the words that start with a term are ``matched`` (the list bolds them)."""
    out, last = [], 0
    folded = _fold(text)
    for match in _WORD.finditer(folded):
        if any(match.group(0).startswith(term) for term in wanted) and len(folded) == len(text):
            if match.start() > last:
                out.append((text[last:match.start()], False))
            out.append((text[match.start():match.end()], True))
            last = match.end()
    if last < len(text):
        out.append((text[last:], False))
    return out
