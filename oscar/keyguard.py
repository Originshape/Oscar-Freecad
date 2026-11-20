"""Keys, and pieces of keys, in text that leaves this computer for OriginShape: the synced AI model settings (a
profile's name and model) and the website's Test in FreeCAD (its message). ONE policy for both, a superset of the
server's own (``ai_settings/keys.py`` in the app repository), so Oscar never sends what the server would redact or
refuse:

* **a whole credential by its shape** (``context_guard``: provider and OriginShape keys, tokens, private keys), and
  every secret this process has handled (``context_guard.remember``);
* **a key this computer holds, by value** (``secrets``): the whole key, and any word of 4 or more letters, digits,
  ``-`` or ``_`` that is a piece of it, such as its last characters ("ending in 4f2a");
* **a key a provider echoed masked**: a word with a run of mask characters (``*``, ``•``, ``●``, ``∙``), an ellipsis
  or ``..`` against letters or digits, or a run of four ``x`` (``sk-proj-****…Ab9Z``, ``****abcd``, ``sk-..abcd``,
  ``sk-xxxxxxxx1234``); a word of mask characters alone takes the next word with it ("**** abcd");
* **a word that starts like a key** (``KEY_PREFIXES``: ``sk-``, ``gsk_``, ``xai-`` ...).

``scrub(text, secrets)`` replaces each of them with ``[redacted]``; ``looks_like_a_key(text, secrets)`` says whether
there is one. Pure Python; linear in the text (word scans, no pattern that backtracks across words).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from . import context_guard as _context_guard

REDACTED = "[redacted]"
MIN_VALUE = 8          # a key shorter than this (a local server's placeholder) is not looked for by value
MIN_PIECE_SOURCE = 16  # pieces are looked for only of a key at least this long (a real key, not "lm-studio")
MIN_PIECE = 4          # the shortest piece of a key that counts
KEY_PREFIXES = ("sk-", "gsk_", "xai-", "pplx-", "csk-", "fw_", "tgp_v1_", "AIza", "os_live_", "os_test_", "hf_", "r8_",
                "nvapi-", "ghp_", "gho_", "ghs_", "github_pat_", "glpat-", "xoxb-", "xoxp-", "AKIA", "ASIA")

# The server's rules (keys.py), word by word: a word is a run of non-space characters.
_SPACED_WORD = re.compile(r"\S+")
_MASKED = re.compile(r"[*•●∙]{3,}|[xX]{4,}|(?:…|\.{2,})(?=[A-Za-z0-9])")
_MASKED_IN_KEYISH_WORD = re.compile(r"[*•●]{2}|(?<=[A-Za-z0-9])(?:…|\.{3,})")
_KEYISH = re.compile(r"[0-9_-]")
_ONLY_MASK = re.compile(r"[*•●∙.…]{3,}")
_KEY_TAIL = re.compile(r"[A-Za-z0-9_-]{2,12}[.,;:)\]'\"]?")
# Pieces of a key, and the start of a word that may be a key: runs of the characters keys are made of.
_KEY_CHARS = re.compile(r"[A-Za-z0-9_\-]+")


def _usable(secrets: Iterable[str] | None, minimum: int) -> list[str]:
    return sorted({s.strip() for s in secrets or () if isinstance(s, str) and len(s.strip()) >= minimum},
                  key=len, reverse=True)


def _masked_word(word: str) -> bool:
    """A word that is a masked key, or holds one (the server's ``_masked``)."""
    if _MASKED.search(word):
        return True
    return bool(_KEYISH.search(word) and _MASKED_IN_KEYISH_WORD.search(word))


def _starts_like_a_key(run: str) -> bool:
    """A run of key characters that starts like a key (``sk-proj-short``, ``gsk_abc``) and is longer than the prefix."""
    return any(run.startswith(prefix) and len(run) > len(prefix) for prefix in KEY_PREFIXES)


def _piece_of(word: str, keys: list[str]) -> bool:
    lowered = word.lower()
    return len(word) >= MIN_PIECE and any(lowered in key.lower() for key in keys)


def looks_like_a_key(text: str, secrets: Iterable[str] | None = ()) -> bool:
    """True when ``text`` holds a key or a piece of one (see the module docstring)."""
    if not text:
        return False
    if any(secret in text for secret in _usable(secrets, MIN_VALUE)):
        return True
    if _context_guard.redact(text) != text:
        return True
    keys = _usable(secrets, MIN_PIECE_SOURCE)
    for match in _KEY_CHARS.finditer(text):
        if _starts_like_a_key(match.group(0)) or (keys and _piece_of(match.group(0), keys)):
            return True
    return any(_masked_word(m.group(0)) for m in _SPACED_WORD.finditer(text))


def scrub(text: str, secrets: Iterable[str] | None = ()) -> str:
    """``text`` with every key, piece of a key and masked key replaced by ``REDACTED`` (see the module docstring). In
    this order: the keys held here by value, whole keys by their shape, masked words (the server's rule, with the word
    after a word of mask characters alone), then runs of key characters that start like a key or are a piece of one
    held here."""
    if not text:
        return text
    for secret in _usable(secrets, MIN_VALUE):
        text = text.replace(secret, REDACTED)
    text = _context_guard.redact(text, replacement=REDACTED)
    out: list[str] = []
    last = 0
    redact_next = False
    for match in _SPACED_WORD.finditer(text):
        word = match.group(0)
        out.append(text[last:match.start()])
        last = match.end()
        if redact_next and _KEY_TAIL.fullmatch(word):
            out.append(REDACTED)
            redact_next = False
            continue
        redact_next = False
        if word == REDACTED or not _masked_word(word):
            out.append(word)
            continue
        out.append(REDACTED)
        redact_next = bool(_ONLY_MASK.fullmatch(word))
    out.append(text[last:])
    text = "".join(out)
    keys = _usable(secrets, MIN_PIECE_SOURCE)

    def run(match: re.Match) -> str:
        chars = match.group(0)
        if chars == "redacted":
            return chars
        return REDACTED if _starts_like_a_key(chars) or (keys and _piece_of(chars, keys)) else chars

    return _KEY_CHARS.sub(run, text)
