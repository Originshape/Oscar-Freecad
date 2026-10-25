# SPDX-License-Identifier: Apache-2.0
"""Is there a newer Oscar? ``GET {server}/v1/plugin``, public: no sign-in, no token, nothing about the user.

The OriginShape server (``oscar.auth.config.api_url()``, the app at ``app.originshape.design`` by default) answers
``{"latest_version", "min_version", "download_url"}``, the same numbers a signed-in ``/v1/me`` carries. So Oscar can
say an update is out to everyone, also to people who use their own AI provider and never sign in:
``account.plugin_update`` reads them (``"available"``, or ``"required"`` when this Oscar is older than the server's
minimum: then only the OriginShape account stops working, never a provider of your own).

The check is made at most once a day per server (``CHECK_EVERY_S``), on a daemon thread of the account hub
(``settings_ui.AccountHub.check_updates``), and on demand from **Oscar → Check for Updates…**. It sends what any web
request does (this computer's address, the ``Oscar-FreeCAD/<version>`` user agent) and nothing else. It is on by
default; the ``CheckForUpdates`` preference (Settings) or ``OSCAR_UPDATE_CHECK=0`` turns it off.

Pure python (no FreeCAD, no Qt): ``oscar.params`` is read lazily, and its absence (the pure tests) means "on".
"""

import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from . import __version__
from .auth import account, config, transport
from .auth.transport import AuthError

PREF = "CheckForUpdates"            # params boolean: the daily check is on (default)
ENV = "OSCAR_UPDATE_CHECK"          # "0" turns it off whatever the preference says (tests, managed installs)
PATH = "/v1/plugin"
CHECK_EVERY_S = 24 * 3600.0
TIMEOUT: transport.TimeoutSpec = (5.0, 8.0)

_LOCK = threading.Lock()
_CACHE: dict[str, "Release"] = {}   # api_url -> the last answer
_ASKED: dict[str, float] = {}       # api_url -> time.time() of the last attempt, answered or not


@dataclass(frozen=True)
class Release:
    """What the server says the current Oscar is; ``checked_at`` is ``time.time()`` of the answer."""

    latest_version: str = ""
    min_version: str = ""
    download_url: str = ""
    checked_at: float = 0.0

    def versions(self) -> dict[str, str]:
        """The two versions under the names ``account.plugin_update`` reads (those of a ``/v1/me`` snapshot)."""
        return {"plugin_latest_version": self.latest_version, "plugin_min_version": self.min_version}

    def update(self, current: str = __version__) -> str:
        """``""``, ``"available"`` or ``"required"`` for this Oscar (``current``: another version, for tests)."""
        return account.plugin_update(self.versions(), current)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(value: Any) -> str:
    return "".join(ch for ch in value.strip() if ch.isprintable())[:32] if isinstance(value, str) else ""


def parse(data: Any, *, checked_at: float | None = None) -> Release:
    """``Release`` from a decoded ``/v1/plugin`` body; ``AuthError`` when it isn't one."""
    if not isinstance(data, dict) or account.version_key(data.get("latest_version")) is None:
        raise AuthError("OriginShape answered the update check with something Oscar cannot read.",
                        code="bad_response")
    link = data.get("download_url")
    try:
        download = config.browser_url(link, what="download link") if isinstance(link, str) and link.strip() else ""
    except AuthError:
        download = ""
    at = time.time() if checked_at is None else checked_at
    return Release(_text(data.get("latest_version")), _text(data.get("min_version")), download,
                   at if math.isfinite(at) else 0.0)


def enabled() -> bool:
    """The daily check is on: not switched off by ``OSCAR_UPDATE_CHECK=0`` or the ``CheckForUpdates`` preference."""
    if (os.environ.get(ENV) or "").strip() == "0":
        return False
    try:
        from . import params
    except Exception:  # noqa: BLE001 - no FreeCAD (the pure tests): the default
        return True
    try:
        return bool(params.get_bool(PREF, True))
    except Exception:  # noqa: BLE001 - an unreadable settings file: the default
        return True


def cached(api_url: str | None = None) -> Release | None:
    """The last answer for ``api_url`` (default ``config.api_url()``), however old; None before the first. No network."""
    try:
        url = config.normalise_url(api_url) if api_url else config.api_url()
    except AuthError:
        return None
    with _LOCK:
        return _CACHE.get(url)


def due(api_url: str, *, now: float | None = None) -> bool:
    """True when ``api_url`` was last asked ``CHECK_EVERY_S`` ago or more (or never)."""
    now = time.time() if now is None else now
    with _LOCK:
        return now - _ASKED.get(api_url, float("-inf")) >= CHECK_EVERY_S


def fetch(api_url: str | None = None) -> Release:
    """``GET {api_url}/v1/plugin`` now (blocking, ``TIMEOUT``), remembered for ``cached``. Raises ``AuthUnavailable``
    (network, 429, 5xx) or ``AuthError`` (a server that doesn't answer like OriginShape)."""
    url = config.normalise_url(api_url) if api_url else config.api_url()
    with _LOCK:
        _ASKED[url] = time.time()
    reply = transport.request("GET", url + PATH, timeout=TIMEOUT)
    if transport.is_unavailable(reply.status):
        raise transport.unavailable(reply)
    if reply.status != 200:
        raise AuthError(f"{transport.host_of(url)} did not answer the update check (HTTP {reply.status}).",
                        code="http_error", status=reply.status)
    release = parse(reply.json())
    with _LOCK:
        _CACHE[url] = release
    return release


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
        _ASKED.clear()
