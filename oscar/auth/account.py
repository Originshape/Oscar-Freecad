"""The account the plugin shows, and the models it may use.

* ``AccountSnapshot``: parsed from ``GET /v1/me``. ``fetch_me()`` returns the session's
  snapshot (``AuthSession.account``). ``plugin_update(snapshot)`` compares this Oscar with the server's
  ``latest_version`` / ``min_version``: an update is ``"available"``, or ``"required"`` (the server no
  longer works with this Oscar; ``require_supported`` raises ``AuthError`` with code ``PLUGIN_OUTDATED``).
* ``ModelCatalog``: parsed from ``GET /v1/models``. ``fetch_models()`` fetches and caches it per API URL;
  the pickers, the capability table (``oscar.models``) and workspace jobs read it, so a model the server adds works
  in FreeCAD without a new Oscar.

Parsing is defensive. A missing or mistyped field becomes a neutral default (an empty string, 0 credits, False)
rather than an exception, and a link is kept only when it is https, or http on a loopback host, because the UI opens
it in the browser.
"""

import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .. import __version__
from . import config
from .transport import AuthError, InvalidServerURL, api_error

LINEUP: tuple[str, ...] = ("origin1", "origin1-super")   # the account's models before the server's list is known
# The retired spellings of the two tiers: the server still accepts them but never lists them in /v1/models, so their
# entry there is their tier's (``params.LEGACY_ACCOUNT_MODELS`` is this table).
RETIRED_ALIASES: dict[str, str] = {"oscar-default": "origin1-super", "oscar-fast": "origin1"}
DEFAULT_MODEL = "origin1"   # the server's default when its list names none (oscar-default is a retired spelling)
MODELS_TIMEOUT: tuple[float, float] = (5.0, 8.0)


def _text(value: Any, limit: int = 200) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(ch for ch in value.strip() if ch.isprintable())[:limit]


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number if math.isfinite(number) else 0.0


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0
    return max(int(value), 0)


def _link(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        return config.browser_url(value, what="account link")
    except InvalidServerURL:
        return ""


def _section(data: dict, key: str) -> dict:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class AccountSnapshot:
    """``/v1/me`` at ``fetched_at`` (``time.time()``), credits as the server counts them for the user."""

    status: str = ""                 # "approved" | "waitlisted" | "suspended"
    email: str = ""
    name: str = ""
    is_admin: bool = False
    plan: str = ""                   # "trial" | "pro" | "studio"
    plan_name: str = ""
    credits_balance: float = 0.0
    credits_available: float = 0.0
    credits_low: bool = False
    link_app: str = ""
    link_billing: str = ""
    link_manage: str = ""
    link_waitlist: str = ""
    link_download: str = ""
    plugin_latest_version: str = ""
    plugin_min_version: str = ""
    fetched_at: float = 0.0

    @property
    def approved(self) -> bool:
        return self.status == "approved"

    @property
    def waitlisted(self) -> bool:
        return self.status == "waitlisted"

    @property
    def suspended(self) -> bool:
        return self.status == "suspended"

    @property
    def out_of_credits(self) -> bool:
        """An approved account whose balance reached zero (the website's meaning; credits a running request holds
        don't count). The server stays authoritative."""
        return self.approved and self.credits_balance <= 0

    @property
    def links(self) -> dict[str, str]:
        return {"app": self.link_app, "billing": self.link_billing, "manage": self.link_manage,
                "waitlist": self.link_waitlist, "download": self.link_download}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "AccountSnapshot":
        """The inverse of ``to_dict``; unknown keys are ignored and mistyped values fall back to the defaults."""
        if not isinstance(data, dict):
            return cls()
        values: dict[str, Any] = {}
        for item in fields(cls):
            value = data.get(item.name)
            if item.type in (bool, "bool"):
                values[item.name] = value is True
            elif item.type in (float, "float"):
                values[item.name] = _number(value)
            elif item.name.startswith("link_"):
                values[item.name] = _link(value)
            else:
                values[item.name] = _text(value, 320)
        return cls(**values)

    def same_state(self, other: "AccountSnapshot | None") -> bool:
        """True when ``other`` shows the same account (``fetched_at`` aside)."""
        if other is None:
            return False
        mine, theirs = self.to_dict(), other.to_dict()
        mine.pop("fetched_at")
        theirs.pop("fetched_at")
        return mine == theirs


def parse_me(data: Any, *, fetched_at: float | None = None) -> AccountSnapshot:
    """``AccountSnapshot`` from a decoded ``/v1/me`` body; ``AuthError`` when it has no ``user`` object."""
    if not isinstance(data, dict) or not isinstance(data.get("user"), dict):
        raise AuthError("OriginShape sent an account answer Oscar cannot read.", code="bad_response")
    user = data["user"]
    org, credits, links, plugin = (_section(data, k) for k in ("org", "credits", "links", "plugin"))
    return AccountSnapshot(
        status=_text(user.get("status"), 32).lower(),
        email=_text(user.get("email"), 320),
        name=_text(user.get("name")),
        is_admin=user.get("is_admin") is True,
        plan=_text(org.get("plan"), 32).lower(),
        plan_name=_text(org.get("plan_name"), 64),
        credits_balance=_number(credits.get("balance")),
        credits_available=_number(credits.get("available")),
        credits_low=credits.get("low") is True,
        link_app=_link(links.get("app")),
        link_billing=_link(links.get("billing")),
        link_manage=_link(links.get("manage")),
        link_waitlist=_link(links.get("waitlist")),
        link_download=_link(links.get("download")),
        plugin_latest_version=_text(plugin.get("latest_version"), 32),
        plugin_min_version=_text(plugin.get("min_version"), 32),
        fetched_at=time.time() if fetched_at is None else fetched_at,
    )


# -- this Oscar against the server's versions ---------------------------------------------------------------------

# ``AuthError.code`` for an Oscar older than the server's ``min_version``: the server no longer works with it.
PLUGIN_OUTDATED = "plugin_outdated"
_VERSION_RE = re.compile(r"v?(\d{1,6}(?:\.\d{1,6}){0,3})")


def version_key(text: Any) -> tuple[int, ...] | None:
    """``"0.2.0"`` -> ``(0, 2, 0)``; a leading ``v`` and anything after the numbers (``-rc1``) are ignored, missing
    parts count as 0. None when ``text`` does not start with a version."""
    match = _VERSION_RE.match(text.strip()) if isinstance(text, str) else None
    if match is None:
        return None
    parts = [int(part) for part in match.group(1).split(".")]
    return tuple(parts + [0] * (4 - len(parts)))


def plugin_update(snapshot: Any, current: str = __version__) -> str:
    """What the server's ``/v1/me`` ``plugin`` versions say about this Oscar (``snapshot``: an ``AccountSnapshot``
    or its ``to_dict()``): ``"required"`` when it is older than ``min_version``, ``"available"`` when a newer
    ``latest_version`` exists, else ``""`` (also when a version is missing or unreadable: the server decides)."""
    if isinstance(snapshot, AccountSnapshot):
        snapshot = snapshot.to_dict()
    if not isinstance(snapshot, dict):
        return ""
    mine = version_key(current)
    if mine is None:
        return ""
    minimum = version_key(snapshot.get("plugin_min_version"))
    latest = version_key(snapshot.get("plugin_latest_version"))
    if minimum is not None and mine < minimum:
        return "required"
    if latest is not None and mine < latest:
        return "available"
    return ""


def outdated_message(snapshot: Any, current: str = __version__) -> str:
    """Why this Oscar can't use the account (``plugin_update`` is ``"required"``), else ``""``."""
    if plugin_update(snapshot, current) != "required":
        return ""
    data = snapshot.to_dict() if isinstance(snapshot, AccountSnapshot) else snapshot
    minimum = _text(data.get("plugin_min_version"), 32)
    return (f"This Oscar ({current}) is older than this OriginShape server supports: it needs Oscar {minimum} or "
            "newer. Update Oscar (FreeCAD's Addon Manager, or the download page), then restart FreeCAD.")


def require_supported(snapshot: Any, current: str = __version__) -> None:
    """Raise ``AuthError`` (code ``PLUGIN_OUTDATED``) when the server no longer works with this Oscar."""
    message = outdated_message(snapshot, current)
    if message:
        raise AuthError(message, code=PLUGIN_OUTDATED)


def fetch_me(api_url: str | None = None, *, force: bool = False) -> AccountSnapshot:
    """The signed-in account of ``api_url`` (default ``config.api_url()``): ``get_session(api_url).account(force)``.
    Blocking; raises ``SignedOut``, ``AuthUnavailable`` or ``AuthError``."""
    from .session import get_session   # session imports this module

    return get_session(api_url).account(force=force)


# -- models -------------------------------------------------------------------------------------------------

_MODEL_ID = re.compile(r"[A-Za-z0-9._:\-]{1,100}")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    display_name: str = ""
    context_window: int = 0
    max_output_tokens: int = 0
    capabilities: dict = field(default_factory=dict, compare=False)
    description: str = ""            # one line for the picker, in the server's words
    credit_multiplier: float = 0.0   # credits per unit of usage against the cheapest model; 0 = not said


@dataclass(frozen=True)
class ModelCatalog:
    default: str
    models: tuple[ModelInfo, ...] = ()
    fetched_at: float = 0.0

    def get(self, model_id: str) -> ModelInfo | None:
        return next((m for m in self.models if m.id == model_id), None)


def _capabilities(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, (str, int, float, bool))}


def parse_models(data: Any) -> ModelCatalog:
    """``ModelCatalog`` from a decoded ``/v1/models`` body; entries with a malformed id are skipped."""
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise AuthError("OriginShape sent a model list Oscar cannot read.", code="bad_response")
    models: list[ModelInfo] = []
    for item in data["data"][:50]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not _MODEL_ID.fullmatch(item["id"]):
            continue
        models.append(ModelInfo(
            id=item["id"], display_name=_text(item.get("display_name"), 80),
            context_window=_int(item.get("context_window")), max_output_tokens=_int(item.get("max_output_tokens")),
            capabilities=_capabilities(item.get("capabilities")), description=_text(item.get("description"), 300),
            credit_multiplier=_number(item.get("credit_multiplier"))))
    ids = [m.id for m in models]
    default = data.get("default")
    if not isinstance(default, str) or default not in ids:
        default = ids[0] if ids else DEFAULT_MODEL
    return ModelCatalog(default, tuple(models), time.time())


_CATALOG_LOCK = threading.Lock()
_CATALOGS: dict[str, ModelCatalog] = {}


def fetch_models(api_url: str | None = None) -> ModelCatalog:
    """``GET /v1/models`` with the session's token; the catalog is cached (``cached_model``). Blocking;
    raises ``SignedOut``, ``AuthUnavailable`` or ``AuthError``."""
    from .session import get_session   # session imports this module

    session = get_session(api_url)
    reply = session.authorized_request("GET", "/v1/models", timeout=MODELS_TIMEOUT)
    if reply.status != 200:
        raise api_error(reply, "the model list")
    catalog = parse_models(reply.json())
    with _CATALOG_LOCK:
        _CATALOGS[session.api_url] = catalog
    return catalog


def cached_models(api_url: str | None = None) -> ModelCatalog | None:
    """The last catalog fetched for ``api_url`` (default ``config.api_url()``), or None. No network."""
    try:
        url = config.normalise_url(api_url) if api_url else config.api_url()
    except AuthError:
        return None
    with _CATALOG_LOCK:
        return _CATALOGS.get(url)


def clear_model_cache() -> None:
    with _CATALOG_LOCK:
        _CATALOGS.clear()


def is_model_id(value: Any) -> bool:
    """True for a well-formed model alias (what ``/v1/models`` and a workspace job may name)."""
    return isinstance(value, str) and bool(_MODEL_ID.fullmatch(value))


def cached_model(alias: str, api_url: str | None = None) -> ModelInfo | None:
    """The cached ``/v1/models`` entry of ``alias`` (a retired alias: its tier's entry), or None (nothing fetched, or
    not listed). No network. Audit 2026-10-28: a retired alias fell back to a 32k window, 4096 output tokens and no
    caching, because the server never lists it."""
    with _CATALOG_LOCK:
        if not _CATALOGS:   # resolving the API URL reads the settings file: skip it while nothing was fetched
            return None
    catalog = cached_models(api_url)
    if catalog is None:
        return None
    return catalog.get(alias) or catalog.get(RETIRED_ALIASES.get(alias, ""))
