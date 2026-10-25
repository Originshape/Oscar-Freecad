"""Web tools: ``fetch_url`` and ``web_search``.

Both tools are permission gated through ``netguard.request_permission`` PER (tool, host): the user chooses Allow once (this turn) /
Always allow THIS host for THIS tool / Deny, a redirect to another host is another question, and every use writes a ``net_use``
audit event (host only, never the path or query). Nothing here is ever approved silently.

``fetch_url(url, purpose, max_chars=20000)`` reads ONE http(s) page and returns text:

* SSRF guard: ``netguard.check_public_url`` runs on the URL and on EVERY redirect hop (redirects are followed by hand, at most
  ``netguard.MAX_REDIRECTS``, never from https to http); loopback, link-local, private, metadata and internal names are refused;
* only ``text/html``, ``text/plain`` and ``application/json`` (a body with NUL bytes is binary and refused), at most
  ``MAX_BYTES`` (2 MB) counted on the DECODED stream, so a compression bomb stops at the cap;
* HTML becomes plain text with ``html.parser`` (script, style, nav, noscript, template, svg and head furniture dropped);
* the reply starts with the final URL, content type, size and sha256, then the clipped text; a page is recorded in the bound
  Project's provenance (guarded: a project problem never fails the fetch) with the ``purpose`` as the claim;
* ``DEFAULT_ALLOWLIST`` (vendor sites) only changes the wording of the question: a vendor host is described as a known vendor site,
  any other host gets a clearer warning. The user approves every host either way. The user can extend the list (setting
  ``web_allowlist_extra``, comma or space separated host names).

``web_search(query, max_results=5)`` queries a SearXNG-compatible JSON backend that the USER configures (setting
``search_backend_url`` or env ``OSCAR_SEARCH_URL``; the env wins). No backend is a ``ToolError`` that says how to
configure one. Consumer search engines are never scraped. The configured backend host is the user's own choice, so it may be on a
private network (a self-hosted instance); redirects away from it are checked like any other fetch. Results are compact
``[{title, url, snippet}]``.
"""

import hashlib
import json
import os
import re
import time
from html.parser import HTMLParser
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit

from . import context_guard as _context_guard
from . import netguard
from . import params as _params
from . import turnlog as _turnlog
from .tools import ToolError, current_context, tool

MAX_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_CHARS = 20000
MIN_MAX_CHARS = 500
TEXT_TYPES = ("text/html", "text/plain", "application/json")
DEFAULT_ALLOWLIST = (
    "raspberrypi.com", "datasheets.raspberrypi.com", "pip.raspberrypi.com", "github.com", "raw.githubusercontent.com",
    "docs.arduino.cc", "docs.espressif.com", "grabcad.com",
)
SEARCH_URL_ENV = "OSCAR_SEARCH_URL"
SEARCH_URL_SETTING = "search_backend_url"
ALLOWLIST_SETTING = "web_allowlist_extra"
_PHASES = ("INTAKE", "SOURCE", "PLAN")
_HEADERS = {"User-Agent": "Mozilla/5.0 (Oscar for FreeCAD; datasheet reader)", "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.1"}
_SNIPPET_CHARS = 300
_TITLE_CHARS = 160


# --- settings ---------------------------------------------------------------------------------------------------

def parse_hosts(text: str) -> list[str]:
    """Host names out of a comma / space / newline separated string (URLs are reduced to their host); lower case, unique, in order."""
    out: list[str] = []
    for raw in re.split(r"[,\s;]+", str(text or "")):
        raw = raw.strip()
        if not raw:
            continue
        host = (urlsplit(raw).hostname if "://" in raw else raw.split("/", 1)[0]) or ""
        host = host.strip().lower().rstrip(".")
        if host and host not in out:
            out.append(host)
    return out


def user_allowlist() -> list[str]:
    """The hosts the user added in Settings (``web_allowlist_extra``)."""
    try:
        return parse_hosts(_params.get_string(ALLOWLIST_SETTING, ""))
    except Exception:  # noqa: BLE001 - an unreadable settings file means "no extras"
        return []


def allowlist() -> list[str]:
    """Default vendor hosts plus the user's own."""
    return list(DEFAULT_ALLOWLIST) + [h for h in user_allowlist() if h not in DEFAULT_ALLOWLIST]


def known_host(host: str) -> bool:
    """True for a host on the (extended) vendor list; exact match, so ``www.x.com`` is not ``x.com``."""
    return str(host or "").strip().lower().rstrip(".") in allowlist()


def search_backend() -> str:
    """The configured search backend URL ('' when none): env ``OSCAR_SEARCH_URL`` first, then the ``search_backend_url`` setting."""
    value = (os.environ.get(SEARCH_URL_ENV) or "").strip()
    if not value:
        try:
            value = _params.get_string(SEARCH_URL_SETTING, "").strip()
        except Exception:  # noqa: BLE001
            value = ""
    return value


# --- HTML to text -----------------------------------------------------------------------------------------------

_SKIP = frozenset({"script", "style", "nav", "noscript", "template", "svg", "head", "iframe", "object"})
_BLOCK = frozenset({"p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "main", "header", "footer", "aside",
                    "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "dl", "dt", "dd", "hr", "form", "figure", "figcaption"})


class _TextExtractor(HTMLParser):
    """Visible text of an HTML page: the title first, block tags become line breaks, ``script`` / ``style`` / ``nav`` and friends are dropped."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._parts: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag == "body":
            self._skip = 0          # an unclosed <head> must not hide the whole page
        elif tag in _SKIP:
            self._skip += 1
        elif tag in _BLOCK and not self._skip:
            self._parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in _BLOCK and not self._skip:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK and not self._skip:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [re.sub(r"[ \t\r\f\v\xa0]+", " ", line).strip() for line in raw.split("\n")]
        body = "\n".join(line for line in lines if line)   # compact: no blank lines (fewer tokens for the model)
        title = re.sub(r"\s+", " ", self.title).strip()
        return f"{title}\n\n{body}".strip() if title else body


def html_to_text(html: str) -> str:
    """Plain text of ``html`` (stdlib parser; malformed markup is tolerated)."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - html.parser can raise on pathological input; keep what was read
        pass
    return parser.text()


# --- the guarded GET ----------------------------------------------------------------------------------------------

class _Page:
    """What ``_get`` returns: the decoded body and where it came from."""

    def __init__(self, final_url: str, content_type: str, raw: bytes, text: str):
        self.final_url, self.content_type, self.raw, self.text = final_url, content_type, raw, text
        self.sha256 = hashlib.sha256(raw).hexdigest()


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url or "").hostname or "").strip().lower().rstrip(".")
    except ValueError:
        return ""


def _ask(name: str, host: str, url: str, purpose: str, *, redirected_from: str = "") -> bool:
    """The per-(tool, host) permission question; True when the user said yes (once or always)."""
    if name == "web_search":
        summary = f"Oscar wants to send a web search to your configured search server:\n• {host}"
    elif redirected_from:
        summary = (f"The page redirected from '{redirected_from}' to ANOTHER site. Oscar wants to read:\n• {url}\n"
                   f"Host: {host}")
    else:
        summary = f"Oscar wants to read a web page:\n• {url}"
    if name == "fetch_url" and not known_host(host):
        summary += (f"\n\nWarning: '{host}' is not on the list of known vendor sites. Only allow it if you trust this site; "
                    "Oscar will only read its text (no files are saved).")
    elif name == "fetch_url":
        summary += f"\n\n'{host}' is a known vendor site."
    detail = f"Reason: {purpose}" if purpose else ""
    verdict = netguard.request_permission(summary, detail, tool=name, host=host)
    return verdict in ("allow", "always")


def _http_hint(code) -> str:
    if code in (401, 403):
        return (f"HTTP {code}: the site blocks automated reading. Do not retry it or other pages of that site; ask the user to open "
                "the page in a browser and paste the relevant text here, or attach the datasheet.")
    return f"HTTP {code}"


def _get(name: str, url: str, purpose: str, *, accept_types=TEXT_TYPES, max_bytes: int = MAX_BYTES, should_stop=None,
         timeout: float = 20.0, private_first_hop: bool = False, headers=None) -> _Page:
    """Guarded GET used by both tools. Raises ``ToolError`` (model-facing text) on refusal, denial, bad type, size or HTTP failure.

    Every hop (the URL and each redirect) passes ``netguard.check_public_url`` and, when its host is new, the permission question
    again. ``private_first_hop`` skips ONLY the SSRF check of the first URL (the user's own configured search backend).
    """
    import requests

    approved: set[str] = set()
    hops = 0
    previous = ""
    base_headers = dict(_HEADERS)
    base_headers.update(headers or {})
    while True:
        if should_stop is not None and should_stop():
            raise ToolError(f"{name}: stopped.")
        parts = urlsplit(url)
        host = _host_of(url)
        if parts.scheme.lower() not in ("http", "https") or not host:
            raise ToolError(f"{name}: only http(s) URLs with a host are supported, got {url!r}.")
        addresses: list[str] = []
        if not (private_first_hop and hops == 0):
            why = netguard.check_public_url(url)
            addresses = [] if why else netguard.take_validated(host)   # resolved ONCE: the request connects to exactly these
            if why:
                prefix = f"{name}: " + ("a redirect led to a URL that was refused. " if hops else "")
                raise ToolError(prefix + why.replace("download refused: ", "refused: ").rstrip(".") + ".")
        if host not in approved:
            if not _ask(name, host, url, purpose, redirected_from=previous if hops else ""):
                raise ToolError(f"{name}: the user DECLINED network access to '{host}'. Do not retry; continue without it or ask "
                                "the user to paste the text or attach the file.")
            approved.add(host)
        try:
            with netguard.pinned_dns(host, addresses):      # no second DNS lookup at connect time (rebinding)
                resp = requests.get(url, stream=True, timeout=(10, timeout), headers=base_headers, allow_redirects=False)
        except requests.RequestException as exc:
            netguard.audit_use(name, url, outcome="error")
            raise ToolError(f"{name}: request to '{host}' failed: {type(exc).__name__}: {str(exc)[:160]}.") from None
        try:
            code = getattr(resp, "status_code", 0)
            location = resp.headers.get("Location") if getattr(resp, "headers", None) is not None else None
            if code in (301, 302, 303, 307, 308) and location:
                hops += 1
                if hops > netguard.MAX_REDIRECTS:
                    raise ToolError(f"{name}: more than {netguard.MAX_REDIRECTS} redirects.")
                target = urljoin(url, str(location))
                if parts.scheme.lower() == "https" and urlsplit(target).scheme.lower() == "http":
                    raise ToolError(f"{name}: refused a redirect from https to http ({host}).")
                previous, url = host, target
                continue
            if code >= 400:
                netguard.audit_use(name, url, outcome=f"http_{code}")
                raise ToolError(f"{name}: {_http_hint(code)} from '{host}'.")
            page = _read_body(name, resp, url, accept_types, max_bytes, should_stop)
        finally:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
        netguard.audit_use(name, url, outcome="ok", bytes=len(page.raw), content_type=page.content_type)
        return page


def _read_body(name: str, resp, url: str, accept_types, max_bytes: int, should_stop) -> _Page:
    headers = resp.headers
    content_type = str(headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if content_type not in accept_types:
        shown = content_type or "no content type"
        raise ToolError(f"{name}: '{_host_of(url)}' returned {shown}; only {', '.join(accept_types)} can be read (files are never "
                        "downloaded here; use source_reference_part or download_file for a CAD file).")
    declared = headers.get("Content-Length")
    try:
        if declared not in (None, "") and int(declared) > max_bytes and not headers.get("Content-Encoding"):
            raise ToolError(f"{name}: the page is {int(declared) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB.")
    except ValueError:
        pass
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_content(chunk_size=1 << 15):
        if should_stop is not None and should_stop():
            raise ToolError(f"{name}: stopped.")
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise ToolError(f"{name}: the page is larger than the {max_bytes / 1e6:.0f} MB limit; read stopped.")
        chunks.append(chunk)
    raw = b"".join(chunks)
    if b"\x00" in raw[:8192]:
        raise ToolError(f"{name}: the body is binary data, not text; only {', '.join(accept_types)} can be read.")
    charset = "utf-8"
    match = re.search(r"charset\s*=\s*([\w\-]+)", str(headers.get("Content-Type") or ""), re.IGNORECASE)
    if match:
        charset = match.group(1)
    try:
        text = raw.decode(charset, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")
    return _Page(getattr(resp, "url", None) if isinstance(getattr(resp, "url", None), str) and resp.url else url,
                 content_type, raw, text)


# --- fetch_url --------------------------------------------------------------------------------------------------

def _record_page(project, page: _Page, purpose: str) -> str:
    """Provenance entry of a fetched page; ``E<n>`` or '' (a project problem is logged, never raised)."""
    if project is None:
        return ""
    try:
        from . import project as _project

        kinds = getattr(_project, "PROVENANCE_KINDS", ())
        kind = "web_page" if "web_page" in kinds else "datasheet"
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        host = _host_of(page.final_url)
        return str(project.provenance_add(
            kind=kind, claim=(purpose or "web page read for the design")[:300], source=f"web:{host}", url=page.final_url,
            sha256=page.sha256, bytes=len(page.raw), detected_type="none", status="obtained",
            confidence="community", note=f"fetched_at {stamp}; {page.content_type}; "
                                          f"{'vendor site' if known_host(host) else 'host not on the vendor list'}"))
    except Exception as exc:  # noqa: BLE001 - recording must never fail the fetch
        _turnlog.log_event("project_error", hook="provenance_add", error=f"{type(exc).__name__}: {exc}"[:200])
        return ""


@tool(
    "fetch_url",
    "Read ONE web page and return its text (HTML converted to plain text, clipped). Use it for a datasheet or vendor documentation "
    "page. The user is asked first, per site; internal addresses and binary files are refused, and it does not save CAD files (use "
    "source_reference_part for that). Give a short 'purpose': it is shown to the user and recorded as the source's claim.",
    {"type": "object", "properties": {
        "url": {"type": "string", "description": "An http(s) URL of a page, ideally on an official vendor site."},
        "purpose": {"type": "string", "description": "Why this page is needed (shown in the permission dialog, recorded as provenance)."},
        "max_chars": {"type": "integer", "default": DEFAULT_MAX_CHARS, "minimum": MIN_MAX_CHARS, "maximum": DEFAULT_MAX_CHARS}},
     "required": ["url"]},
    category="net", mutating=False, phases=_PHASES,
)
def _fetch_url(url: str, purpose: str = "", max_chars: int = DEFAULT_MAX_CHARS):
    """Fetch, extract and report one page. Raises ``ToolError`` for bad arguments, a declined permission, a refused or failed fetch."""
    ctx = current_context()
    link = str(url or "").strip()
    why = str(purpose or "").strip()
    if not link.lower().startswith(("http://", "https://")):
        raise ToolError(f"fetch_url: 'url' must be an http(s) URL, got {link!r}; give the full address of the page.")
    try:
        limit = int(max_chars)
    except (TypeError, ValueError):
        raise ToolError(f"fetch_url: 'max_chars' must be an integer between {MIN_MAX_CHARS} and {DEFAULT_MAX_CHARS}, got {max_chars!r}.") from None
    limit = max(MIN_MAX_CHARS, min(DEFAULT_MAX_CHARS, limit))
    page = _get("fetch_url", link, why, should_stop=getattr(ctx, "should_stop", None))
    if page.content_type == "text/html":
        body = html_to_text(page.text)
    else:
        body = page.text.strip()
    if not body:
        raise ToolError(f"fetch_url: '{_host_of(page.final_url)}' returned no readable text (an empty page or one that needs JavaScript); "
                        "ask the user to paste the text.")
    shown = body[:limit]
    entry = _record_page(getattr(ctx, "project", None), page, why)
    lines = [f"FETCHED {link}", f"final_url: {page.final_url}", f"content_type: {page.content_type}", f"bytes: {len(page.raw)}",
             f"sha256: {page.sha256}",
             f"text: {len(shown)} of {len(body)} chars" + (" (clipped; raise max_chars or fetch a more specific page)" if len(body) > limit else ""),
             f"provenance: {entry or 'not recorded' + (' (no project bound)' if getattr(ctx, 'project', None) is None else '')}", "",
             _context_guard.untrusted("this web page", shown)]
    return "\n".join(lines)


# --- web_search ---------------------------------------------------------------------------------------------------

_NOT_CONFIGURED = (
    "web_search: no search backend is configured. Set 'Search backend URL' in Settings > Network permissions (the address of a "
    "SearXNG instance with the JSON format enabled, for example http://localhost:8080) or the environment variable "
    f"{SEARCH_URL_ENV}. Until then do not call web_search again: ask the user to paste the page text or attach the datasheet, or "
    "read a known official page with fetch_url.")


def _search_endpoint(base: str, query: str, count: int) -> str:
    parts = urlsplit(base)
    path = parts.path if parts.path not in ("", "/") else "/search"
    extra = [(k, v) for k, v in parse_qsl(parts.query) if k not in ("q", "format")]
    qs = urlencode(extra + [("q", query), ("format", "json")], quote_via=quote)
    return f"{parts.scheme}://{parts.netloc}{path}?{qs}"


def _clip(text, limit: int) -> str:
    out = re.sub(r"\s+", " ", str(text or "")).strip()
    return out if len(out) <= limit else out[:limit - 1].rstrip() + "…"


@tool(
    "web_search",
    "Search the web through the user's configured search server and return compact results (title, url, snippet). The user is asked "
    "first. Use it to find an official datasheet or CAD page, then read it with fetch_url. If no search server is configured the "
    "tool says so: then ask the user to paste the page text or attach the file.",
    {"type": "object", "properties": {
        "query": {"type": "string"},
        "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10}},
     "required": ["query"]},
    category="net", mutating=False, phases=_PHASES,
)
def _web_search(query: str, max_results: int = 5):
    """Query the configured SearXNG-compatible backend; returns ``SEARCH RESULTS`` plus a JSON list of ``{title, url, snippet}``."""
    ctx = current_context()
    text = str(query or "").strip()
    if not text:
        raise ToolError("web_search: 'query' is empty; give the words to search for.")
    try:
        count = int(max_results)
    except (TypeError, ValueError):
        raise ToolError(f"web_search: 'max_results' must be an integer from 1 to 10, got {max_results!r}.") from None
    count = max(1, min(10, count))
    base = search_backend()
    if not base:
        raise ToolError(_NOT_CONFIGURED)
    if urlsplit(base).scheme.lower() not in ("http", "https") or not _host_of(base):
        raise ToolError(f"web_search: the configured search backend {base!r} is not an http(s) URL; fix it in Settings > Network "
                        f"permissions or {SEARCH_URL_ENV}.")
    endpoint = _search_endpoint(base, text, count)
    page = _get("web_search", endpoint, f"search: {text[:120]}", accept_types=("application/json",), max_bytes=MAX_BYTES,
                should_stop=getattr(ctx, "should_stop", None), private_first_hop=True,
                headers={"Accept": "application/json"})
    try:
        data = json.loads(page.text)
    except ValueError:
        raise ToolError("web_search: the search backend did not return JSON; enable the json format on the SearXNG server "
                        "(search.formats in settings.yml) or fix the URL.") from None
    raw = data.get("results") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise ToolError("web_search: the search backend's reply has no 'results' list; it must be SearXNG compatible.")
    results = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        link = str(item.get("url") or "").strip()
        if not link.lower().startswith(("http://", "https://")):
            continue
        results.append({"title": _clip(item.get("title"), _TITLE_CHARS), "url": link,
                        "snippet": _clip(item.get("content") or item.get("snippet"), _SNIPPET_CHARS)})
        if len(results) >= count:
            break
    head = f"SEARCH RESULTS ({len(results)}) for {text!r}"
    if not results:
        return head + "\nNo results. Try other words, or ask the user for the page."
    return head + "\n" + _context_guard.untrusted("web search results", json.dumps(results, ensure_ascii=False))
