"""PKCE (RFC 7636) and the OAuth ``state``: random verifiers, the S256 challenge and a constant-time state check.

The verifier is the secret half of the browser sign-in: it stays in this process until the code is exchanged, and
only its SHA-256 (the challenge) travels through the browser. Neither the verifier nor the state may ever appear in
a log line or an exception message, so the errors below name the problem, never the value.
"""

import base64
import hashlib
import hmac
import re
import secrets

VERIFIER_LENGTH = 64           # RFC 7636: 43 to 128 characters
_VERIFIER_RE = re.compile(r"[A-Za-z0-9\-._~]{43,128}")   # the RFC's unreserved characters


def new_verifier(length: int = VERIFIER_LENGTH) -> str:
    """A fresh random verifier of ``length`` characters (base64url, a subset of the allowed alphabet)."""
    if not 43 <= length <= 128:
        raise ValueError("a PKCE verifier has 43 to 128 characters")
    verifier = ""
    while len(verifier) < length:
        verifier += secrets.token_urlsafe(length)
    return verifier[:length]


def is_valid_verifier(verifier: object) -> bool:
    return isinstance(verifier, str) and _VERIFIER_RE.fullmatch(verifier) is not None


def challenge(verifier: str) -> str:
    """The S256 challenge: ``BASE64URL(SHA256(ASCII(verifier)))`` without padding."""
    if not is_valid_verifier(verifier):
        raise ValueError("not a valid PKCE verifier (43 to 128 unreserved characters)")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def new_state() -> str:
    """A fresh random ``state`` (256 bits, base64url)."""
    return secrets.token_urlsafe(32)


def states_match(expected: str, received: object) -> bool:
    """Constant-time comparison of the state Oscar sent with the one the callback carried (False for a missing or
    non-text value). Bytes are compared, so a non-ASCII value from a forged request cannot raise."""
    if not expected or not isinstance(received, str):
        return False
    return hmac.compare_digest(expected.encode("utf-8"), received.encode("utf-8"))
