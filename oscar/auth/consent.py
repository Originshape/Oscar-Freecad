# SPDX-License-Identifier: Apache-2.0
"""OriginShape's Terms and Privacy Policy: has the account accepted them, and accepting them (pure: no Qt).

``GET /v1/me`` says which Terms the account accepted (``consent``) and which are current (``legal``: the version and
the two pages); ``AccountSnapshot`` carries both (``consent_known``, ``terms_current``, ``legal_terms_version``,
``link_terms``, ``link_privacy``). A server from before the Terms sends neither: then nothing is asked and nothing is
recorded (``needed`` and ``recorded`` are both False).

* ``needed(snapshot)``: the account has not accepted the current Terms. Until it does, it can't be used for chat
  (``blocks_chat``); the user's own providers (an API key, Ollama ...) keep working.
* ``recorded(snapshot)``: the account accepted the current Terms, so its FreeCAD chats are kept (``oscar.records``).
* ``Prompt``: what the consent prompt shows (``NOTICE``, word for word) and what its Continue does. The box is ticked
  by default; ticked, ``submit`` sends ``POST /v1/me/consent`` (``accept_body``: ``accept_terms`` true, the current
  ``legal.terms_version``, ``marketing_emails`` null, which leaves that choice as it was); unticked it sends nothing
  and the account stays blocked (``BLOCKED``).
* ``accept(session, version)``: the request. 200 puts the answer's ``consent`` into the session's snapshot
  (``AuthSession.note_consent``: no second request, and ``on_change`` tells the panel). 409 means the Terms changed
  since the prompt was shown: ``/v1/me`` is fetched again and ``TermsChanged`` carries the new snapshot, so the prompt
  can show the new version.
* ``mark_needed(session)``: a recording was refused with 403 ``terms_required``; ``/v1/me`` is fetched again, and when
  it still says the Terms are accepted the snapshot is marked otherwise, so the prompt shows.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, replace
from typing import Any

from .account import AccountSnapshot
from .transport import AuthError, error_details, is_unavailable, unavailable

CONSENT_PATH = "/v1/me/consent"
TIMEOUT: tuple[float, float] = (5.0, 15.0)
NOTICE = ("OriginShape keeps your chats, prompts, files and feedback to run and improve the service. They are never "
          "used to train AI models.")
AGREE = "I agree to the Terms and the Privacy Policy"
BLOCKED = ("Your OriginShape account can't be used for chat until you accept the Terms and the Privacy Policy. Your "
           "own API key, or a local model such as Ollama, still works.")
CHANGED = "OriginShape's Terms have changed since you last saw them. Please read the new version."


class TermsChanged(AuthError):
    """409: the version accepted is no longer the current one. ``snapshot`` is the account as ``/v1/me`` now says
    (None when it could not be fetched)."""

    code = "terms_changed"

    def __init__(self, message: str, *, snapshot: AccountSnapshot | None = None, status: int = 409):
        super().__init__(message, status=status)
        self.snapshot = snapshot


def _get(snapshot: Any, name: str) -> Any:
    """A field of an ``AccountSnapshot`` or of its ``to_dict()`` (the panel's account state), else None."""
    if isinstance(snapshot, AccountSnapshot):
        return getattr(snapshot, name, None)
    return snapshot.get(name) if isinstance(snapshot, dict) else None


def needed(snapshot: Any) -> bool:
    """The account must accept the current Terms before it can chat: the server says which it accepted, and that is
    not the current version. A suspended account is not asked (it can't chat either way)."""
    return (_get(snapshot, "consent_known") is True and _get(snapshot, "terms_current") is not True
            and _get(snapshot, "status") != "suspended")


def recorded(snapshot: Any) -> bool:
    """The account accepted the current Terms: its chats in FreeCAD are kept (``oscar.records``)."""
    return _get(snapshot, "consent_known") is True and _get(snapshot, "terms_current") is True


def blocks_chat(snapshot: Any, *, account_profile: bool) -> bool:
    """True when a chat on the OriginShape account must wait for the Terms. Never for another provider."""
    return bool(account_profile) and needed(snapshot)


def accept_body(version: str) -> dict[str, Any]:
    """The ``POST /v1/me/consent`` body for accepting ``version``; ``marketing_emails`` null leaves it as it was."""
    return {"accept_terms": True, "terms_version": str(version or ""), "marketing_emails": None}


@dataclass
class Prompt:
    """The consent prompt for one version of the Terms. ``checked`` is the box (ticked by default); ``updated`` says
    the prompt is shown again because the Terms changed meanwhile (a 409)."""

    version: str
    terms_url: str = ""
    privacy_url: str = ""
    checked: bool = True
    updated: bool = False

    @classmethod
    def of(cls, snapshot: Any, *, updated: bool = False) -> "Prompt | None":
        """The prompt for the current Terms of ``snapshot``; None when nothing needs accepting."""
        if not needed(snapshot):
            return None
        return cls(str(_get(snapshot, "legal_terms_version") or ""), str(_get(snapshot, "link_terms") or ""),
                   str(_get(snapshot, "link_privacy") or ""), updated=updated)

    def agree_html(self) -> str:
        """``AGREE`` as rich text: "Terms" and "Privacy Policy" link to their pages (plain words when a page is not
        known). The URLs were checked where they were parsed (https, or http on this computer)."""
        def link(url: str, words: str) -> str:
            return f"<a href='{html.escape(url, quote=True)}'>{words}</a>" if url else words

        return f"I agree to the {link(self.terms_url, 'Terms')} and the {link(self.privacy_url, 'Privacy Policy')}"

    def note(self) -> str:
        """What the prompt says under the box: nothing while it is ticked, ``BLOCKED`` while it is not."""
        return "" if self.checked else BLOCKED

    def body(self) -> dict[str, Any] | None:
        """What Continue sends: ``accept_body`` while the box is ticked, nothing (None) while it is not."""
        return accept_body(self.version) if self.checked else None


def accept(session: Any, version: str, *, timeout: Any = TIMEOUT) -> AccountSnapshot | None:
    """``POST /v1/me/consent`` accepting ``version`` with ``session`` (an ``AuthSession``). Returns the snapshot with
    the new consent (None when there was no snapshot to update). Raises ``TermsChanged`` (after fetching ``/v1/me``
    again), ``SignedOut``, ``AuthUnavailable`` or ``AuthError``. Blocking: call it on a worker thread."""
    before = session.cached_snapshot()
    email = before.email if before is not None else ""
    reply = session.authorized_request("POST", CONSENT_PATH, json_body=accept_body(version), timeout=timeout)
    if 200 <= reply.status < 300:
        data = reply.json()
        consent = data.get("consent") if isinstance(data, dict) else None
        snapshot = session.note_consent(consent, email=email) if isinstance(consent, dict) else None
        return snapshot if snapshot is not None else session.account(force=True)
    if reply.status == 409:
        try:
            fresh = session.account(force=True)
        except AuthError:
            fresh = None
        raise TermsChanged(CHANGED, snapshot=fresh)
    if is_unavailable(reply.status):
        raise unavailable(reply)
    code, description = error_details(reply.body)
    if reply.status == 403 and description:   # an API key sign-in: the server says where to accept instead
        raise AuthError(description, code=code or "forbidden", status=403)
    raise AuthError(f"OriginShape did not take your answer to the Terms (HTTP {reply.status}).",
                    code=code or "http_error", status=reply.status)


def submit(prompt: Prompt, session: Any) -> AccountSnapshot | None:
    """Continue pressed: ``accept`` when the box is ticked; None, and nothing sent, when it is not (the account then
    stays blocked for chat)."""
    if prompt.body() is None:
        return None
    return accept(session, prompt.version)


def mark_needed(session: Any) -> None:
    """A recording was refused with 403 ``terms_required``: fetch ``/v1/me`` again, so the panel shows the prompt; when
    it still says the current Terms are accepted, mark them as not accepted in the snapshot (the server just said
    otherwise). Never raises."""
    try:
        snapshot = session.account(force=True)
    except Exception:  # noqa: BLE001 - unreachable now: the next account refresh asks again
        return
    if recorded(snapshot):
        stale = {"terms_version": snapshot.terms_version, "terms_accepted_at": snapshot.terms_accepted_at,
                 "terms_current": False}
        try:
            session.note_consent(stale, email=snapshot.email)
        except Exception:  # noqa: BLE001
            pass


def with_box(prompt: Prompt, checked: bool) -> Prompt:
    """``prompt`` with the box as the user left it."""
    return replace(prompt, checked=bool(checked))
