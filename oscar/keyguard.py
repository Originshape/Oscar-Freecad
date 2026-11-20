"""Keys, and pieces of keys, in text that leaves this computer for OriginShape: the synced AI model settings (a
profile's name and model) and the website's Test in FreeCAD (its message). ONE policy for both, a superset of the
server's own (``ai_settings/keys.py`` in the app repository), so Oscar never sends what the server would redact or
refuse:

* **a whole credential by its shape** (``context_guard``: provider and OriginShape keys, tokens, private keys), and
  every secret this process has handled (``context_guard.remember``);
* **a key this computer holds, by value** (``secrets``): the whole key; in its body (the key without its provider's
  prefix, ``KEY_PREFIX``: ``sk-proj-`` is in every OpenAI key, so "My project" is no piece of one), any piece of 8 or
  more characters anywhere, and its last 4 to 8 characters as a word of their own ("ending in 4f2a", "…4f2a"); both
  case-sensitive, as keys are;
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
PIECE_ANYWHERE = 8     # a piece of a key's body this long counts wherever it is
TAIL = (4, 8)          # the body's last 4 to 8 characters count as a word of their own
# The prefixes of the keys providers issue (the same keys context_guard knows by shape), longest first: never a piece.
KEY_PREFIX = re.compile(r"^(?:sk-ant-api\d{2}-|sk-ant-admin\d{2}-|sk-ant-|sk-proj-|sk-svcacct-|sk-or-v1-|sk-|gsk_|"
                        r"xai-|pplx-|csk-|fw_|tgp_v1_|AIza|os_live_|os_test_|hf_|r8_|nvapi-|gh[pousr]_|github_pat_|"
                        r"glpat-|xox[abprs]-|AKIA|ASIA|whsec_|GOCSPX-)")
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


def key_body(key: str) -> str:
    """``key`` without its provider's prefix (``KEY_PREFIX``)."""
    match = KEY_PREFIX.match(key)
    return key[match.end():] if match else key


class _Pieces:
    """The pieces of the keys held here that count (see the module docstring), for runs of key characters."""

    def __init__(self, secrets: Iterable[str] | None) -> None:
        self.bodies = [body for body in (key_body(k) for k in _usable(secrets, MIN_PIECE_SOURCE))
                       if len(body) >= PIECE_ANYWHERE]
        self.grams = {body[i:i + PIECE_ANYWHERE] for body in self.bodies
                      for i in range(len(body) - PIECE_ANYWHERE + 1)}

    def __bool__(self) -> bool:
        return bool(self.bodies)

    def hit(self, run: str) -> bool:
        """``run`` (a whole run of key characters: a word of its own) holds 8 characters of a body, or is its tail."""
        if len(run) >= PIECE_ANYWHERE and any(run[i:i + PIECE_ANYWHERE] in self.grams
                                              for i in range(len(run) - PIECE_ANYWHERE + 1)):
            return True
        return TAIL[0] <= len(run) <= TAIL[1] and any(len(body) > len(run) and body.endswith(run)
                                                       for body in self.bodies)


def looks_like_a_key(text: str, secrets: Iterable[str] | None = ()) -> bool:
    """True when ``text`` holds a key or a piece of one (see the module docstring)."""
    if not text:
        return False
    if any(secret in text for secret in _usable(secrets, MIN_VALUE)):
        return True
    if _context_guard.redact(text) != text:
        return True
    pieces = _Pieces(secrets)
    for match in _KEY_CHARS.finditer(text):
        if _starts_like_a_key(match.group(0)) or (pieces and pieces.hit(match.group(0))):
            return True
    return any(_masked_word(m.group(0)) for m in _SPACED_WORD.finditer(text))


def scrub(text: str, secrets: Iterable[str] | None = ()) -> str:
    """``text`` with every key, piece of a key and masked key replaced by ``REDACTED`` (see the module docstring). In
    this order: the keys held here by value, whole keys by their shape, masked words (the server's rule, with the word
    after a word of mask characters alone), then runs of key characters that start like a key or are a piece of one
    held here (8 characters of its body anywhere, its tail as a word of its own)."""
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
    pieces = _Pieces(secrets)

    def run(match: re.Match) -> str:
        chars = match.group(0)
        if chars == "redacted":
            return chars
        return REDACTED if _starts_like_a_key(chars) or (pieces and pieces.hit(chars)) else chars

    return _KEY_CHARS.sub(run, text)
