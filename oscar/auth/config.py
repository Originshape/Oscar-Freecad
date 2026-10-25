"""Where the OriginShape API lives (``api_url()``), and the URL rules every auth request follows.

The order is: ``OSCAR_API_URL`` (tests, a local stack), then the ``OriginShapeServer`` preference (the Server field in
Oscar's settings, read lazily through ``oscar.params``, which imports FreeCAD) while ``OriginShapeCustomServer`` (the
*Use a custom server* switch next to it) is on, then ``https://app.originshape.design``. The switch is off by default,
so a server typed in once for local testing is kept for next time but never decides where a sign-in goes on its own.
It is read at call time and never cached at import, so a changed preference applies to the next request.

OriginShape has one public address: the app, ``app.originshape.design``. It serves the API's paths as well
(``/.well-known``, ``/oauth``, ``/auth``, ``/v1``: the web app forwards them to the API), so the browser sign-in, the
tokens, the account, the gateway and the workspace all live on the origin the user signs in on. A retired address
(``RETIRED_API_URLS``, never served) reads as the default, so a Server field copied from early documentation works.

Tokens only travel encrypted. An ``http://`` URL is accepted for loopback hosts only (a local stack on
``http://localhost:8210``); any other plain-http URL, a URL carrying a user name or password, or one Oscar cannot
parse raises ``InvalidServerURL``.
"""

import ipaddress
import os
from urllib.parse import SplitResult, urlsplit, urlunsplit

from .transport import InvalidServerURL

DEFAULT_API_URL = "https://app.originshape.design"
RETIRED_API_URLS = frozenset({"https://api.originshape.design"})
ENV_API_URL = "OSCAR_API_URL"
PREF_SERVER = "OriginShapeServer"
PREF_CUSTOM_SERVER = "OriginShapeCustomServer"   # the Developer switch: PREF_SERVER applies only while it is on
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _params():
    """``oscar.params`` or None when FreeCAD (and so the settings store) is unavailable, as in the pure test run."""
    try:
        from .. import params
    except Exception:  # noqa: BLE001 - ImportError of FreeCAD, anything else at import
        return None
    return params


def configured_url() -> str:
    """The server setting as typed, before validation: the environment, the preference while the custom server is
    switched on, or the default."""
    env = (os.environ.get(ENV_API_URL) or "").strip()
    if env:
        return env
    params = _params()
    if params is not None:
        try:
            pref = (str(params.get_string(PREF_SERVER, "") or "").strip()
                    if params.get_bool(PREF_CUSTOM_SERVER, False) else "")
        except Exception:  # noqa: BLE001 - an unreadable settings file means "not set", never a crash
            pref = ""
        if pref:
            return pref
    return DEFAULT_API_URL


def api_url() -> str:
    """The OriginShape API root to use now, normalised (``https://host[:port][/path]``, no trailing slash); a retired
    address is the default. Raises ``InvalidServerURL`` for a URL tokens must not be sent to."""
    return normalise_url(configured_url())


def settings_dir() -> str | None:
    """The folder of Oscar's settings file (where the credential file fallback lives), or None when unknown."""
    params = _params()
    if params is not None:
        try:
            return os.path.dirname(os.path.abspath(params.settings_path()))
        except Exception:  # noqa: BLE001 - FreeCAD without a user folder
            pass
    override = os.environ.get("OSCAR_AI_SETTINGS")   # params.settings_path() honours the same variable
    return os.path.dirname(os.path.abspath(override)) if override else None


# -- URL rules ----------------------------------------------------------------------------------------------

def is_loopback_host(host: str) -> bool:
    """True for ``localhost`` and loopback IP literals (``127.0.0.0/8``, ``::1``)."""
    name = (host or "").strip().lower().rstrip(".")
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _split(url: object, what: str) -> tuple[SplitResult, str, int | None]:
    """``(parts, scheme, port)`` of a URL Oscar may use, or ``InvalidServerURL``. Never echoes a user name,
    password, path or query in the message."""
    text = url.strip() if isinstance(url, str) else ""
    if not text:
        raise InvalidServerURL(f"The {what} is empty.")
    if any(ch.isspace() or not ch.isprintable() for ch in text):
        raise InvalidServerURL(f"The {what} contains spaces or control characters.")
    try:
        parts = urlsplit(text)
        port = parts.port   # raises ValueError for a port that is not a number in range
    except ValueError:      # also an unbalanced IPv6 bracket
        parts, port = None, -1
    scheme = parts.scheme.lower() if parts is not None else ""
    if parts is None or port == -1 or scheme not in _DEFAULT_PORTS or not parts.hostname:
        raise InvalidServerURL(f"The {what} is not a valid http(s) URL.")
    if "@" in parts.netloc:
        raise InvalidServerURL(f"The {what} must not contain a user name or password.")
    if scheme == "http" and not is_loopback_host(parts.hostname):
        raise InvalidServerURL(f"The {what} must use https:// (plain http is allowed for localhost only, so tokens "
                               f"never travel unencrypted): http://{parts.hostname}", code="insecure_url")
    return parts, scheme, port


def _netloc(host: str, port: int | None, scheme: str) -> str:
    host = host.lower()
    if ":" in host:   # an IPv6 literal goes back into brackets
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return host
    return f"{host}:{port}"


def normalise_url(url: str, *, what: str = "OriginShape server URL") -> str:
    """An API root in canonical form: scheme and host in lower case, the default port dropped, no trailing slash,
    no query or fragment (so two spellings of one server compare equal, and ``issuer`` checks are exact). A retired
    address (``RETIRED_API_URLS``) is the default, here so that every session, catalog and metadata cache keyed by a
    server URL keys it under the address that is served. Raises ``InvalidServerURL``; the Settings dialog can use it
    to validate the Server field."""
    parts, scheme, port = _split(url, what)
    if parts.query or parts.fragment:
        raise InvalidServerURL(f"The {what} must not contain '?' or '#'.")
    canonical = urlunsplit((scheme, _netloc(parts.hostname, port, scheme), parts.path.rstrip("/"), "", ""))
    return DEFAULT_API_URL if canonical in RETIRED_API_URLS else canonical


def browser_url(url: str, *, what: str) -> str:
    """A URL Oscar may open in the browser or post to (authorize page, device page, account links, endpoints):
    https, or http on a loopback host. The query is kept (an endpoint may carry one), the fragment is dropped."""
    parts, scheme, port = _split(url, what)
    return urlunsplit((scheme, _netloc(parts.hostname, port, scheme), parts.path or "/", parts.query, ""))


def origin(url: str) -> tuple[str, str, int]:
    """``(scheme, host, port)`` with the default port filled in."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    return scheme, (parts.hostname or "").lower(), parts.port or _DEFAULT_PORTS.get(scheme, 0)


def same_origin(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` share scheme, host and port."""
    try:
        return origin(a) == origin(b)
    except ValueError:
        return False
