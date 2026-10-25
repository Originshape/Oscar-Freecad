"""Settings dialog for Oscar: three tabs — Account, AI model, Tools & permissions.

**Account** (``oscarSettingsAccountTab``): the OriginShape account card (``objectName`` ``oscarAccount``). Signed out it
explains what the account is for and offers *Sign in with OriginShape* (the browser, ``oscarAccountSignIn``, the
primary action), *Sign in with a code* (``oscarAccountSignInCode``, for a browser that cannot reach FreeCAD) and *Use
an OriginShape API key* (``oscarAccountUseKey``); signed in it shows the email, plan and credits
(``oscarAccountEmail`` / ``oscarAccountPlan`` / ``oscarAccountCredits``), what a waitlisted, suspended or empty
account means (``oscarAccountWaitlist`` / ``oscarAccountNotice``), *Manage account*, *Buy credits*
(``oscarAccountBilling``) and *Sign out*, and the *Connect to the OriginShape workspace* switch
(``oscarConnectWorkspace`` / ``oscarConnectStatus``, oscar.connect) with, under it, *Approve scripts and network
access from the workspace* (``oscarConnectApprovals``, ``connect.REMOTE_PREF``, on by default). ``oscarAccountStatus``
says what is going on. A
collapsed *Developer: server address* section (``oscarAccountDeveloper``) holds the *Use a custom server* switch
(``oscarAccountServerCustom``, the ``OriginShapeCustomServer`` preference, off by default) and the Server field
(``oscarAccountServer``, the ``OriginShapeServer`` preference, editable and used only while the switch is on;
``OSCAR_API_URL`` overrides both) that points Oscar at another API for local testing. Both apply at once. Signing
in or out takes effect at once (it is not part of Save) and sets ``account_changed`` so the panel refreshes even when
the dialog is cancelled. A sign-in makes the OriginShape profile active (``params.ensure_account_profile``).

The account state the panel and this dialog show comes from one process-wide ``AccountHub`` (``account_hub()``):
``oscar.auth.session.on_change`` is pointed at it, and it refreshes ``GET /v1/me`` on a daemon thread, one fetch at
a time.

**AI model** (``oscarSettingsModelTab``): which profile is active, as a choice between *OriginShape account (uses
credits)* (``oscarUseAccount``) and *Your own provider (your API key)* (``oscarUseOwnProvider``). The account choice
picks one of the account's models by its name (``oscarAccountModel``: "Origin1" = ``origin1``, "Origin1 Super" =
``origin1-super``; a retired alias is shown as its tier) and can be tested (``oscarAccountTest``); choosing it adds the account profile when there is none. The own-provider
choice picks which of the user's profiles is used (``oscarOwnProfileInUse``) and shows the profile editor
(``oscarByokSection``): the list of the user's own profiles (the account profile is never listed there) and, for the
selected one, provider preset, base URL, API key and model. Choosing a mode only changes which profile Save makes
active (``params.set_active_profile``); the panel's combo still switches between all profiles. Presets that are not
usable yet (``params.COMING_SOON_PROVIDERS``, empty today) close the provider list, greyed out: visible, never
selectable.

Model selection: once a base URL (and key, if the provider needs one) is
set, "Load models" fetches the provider's /models endpoint and offers the
names as a dropdown (still typeable for endpoints that don't list models).
Temperature and max tokens are PREDEFINED per provider/model and applied
automatically; they are not shown as settings — the "Advanced overrides" section can replace them.

The Test button runs ``provider_check.run_connection_test``: a plain reply, ONE
tool call (native, else text mode) and ONE tiny image, and remembers what the
model can do (on the account it sends three tiny requests, which cost a few credits). An "Advanced" section holds
the per-profile overrides (max output, temperature, reasoning, context window, tool mode, native Ollama); they are
stored in ``Profile.reasoning`` / ``Profile.caps_override`` and survive the
per-turn recompute (``params.effective_limits``).

**Tools & permissions** (``oscarSettingsToolsTab``): "Network permissions" (``oscarNetPermissions``) lists what Oscar's
tools may reach: one row per (tool, host) grant (``oscarNetGrantList``), plus the legacy global "Always allow" pref as
an inactive, revocable row. Remove (``oscarNetGrantRemove``) and Revoke all (``oscarNetGrantRevokeAll``) take effect at
once (they are not part of Save) and run on a worker thread like the other jobs. "Web access" holds the search backend
URL (``oscarSearchBackendUrl``, setting ``search_backend_url``) and extra trusted web hosts (``oscarWebAllowlistExtra``,
setting ``web_allowlist_extra``), saved with Save. "Scripts" (``oscarScripts``) holds *Let Oscar run Python scripts it
writes (you approve each one)* (``oscarAllowCode``, the ``AllowCodeExecution`` preference read by ``codeguard``; off by
default), saved with Save.

Look: helper text is the palette's text colour blended towards the window colour (``tone_color``), never the palette's
Mid role or a disabled grey, so it stays readable in FreeCAD's light and dark themes; colours are recomputed when the
palette changes.

Nothing blocking runs on the GUI thread: the model list, the connection test and every account action (sign-in,
the code flow, the API key check, sign-out) run on worker threads
(``settings_model.run_in_background``). A worker never touches a Qt object (a
QObject whose last reference is dropped on a worker thread is destroyed there,
which crashes): it posts its result into a plain ``_Mailbox`` that a GUI-thread
timer drains. Closing the dialog cancels the jobs (a browser sign-in closes its 127.0.0.1 listener). One user-started
job (Test, model list) runs at a time:
its button turns into its Cancel and the other waits; a change of the selected profile or provider cancels it, and
its answer is applied to the profile that asked for it, never to whichever one is shown when it arrives.

A key belongs to the provider it was entered for (choosing another provider clears it). A Save that cannot write the
settings file says so and keeps the dialog open; a dialog whose profiles did not come from the settings file (it could
not be read when the dialog opened) refuses to save, so it can never replace the user's profiles.
"""

import copy
import json
import os
import threading
import time

from PySide import QtCore, QtGui, QtWidgets

from . import __version__
from . import codeguard as _codeguard
from . import connect as _connect
from . import params as _params
from . import settings_model as _model
from . import updates as _updates
from .auth import account as _auth_account
from .auth import config as _auth_config
from .auth import discovery as _auth_discovery
from .auth import keystore as _keystore
from .auth import session as _auth_session
from .auth.device import DeviceSignIn
from .auth.loopback import LoopbackSignIn
from .auth.transport import AuthError, SignedOut
from .panel_render import format_credits
from .params import Profile

# Personal OriginShape API keys are turned off on the server for now (API_KEYS_ENABLED): the "Use an OriginShape API
# key" sign-in is hidden; the code path stays for when keys come back.
OFFER_API_KEY_SIGNIN = False
ACCOUNT_REFRESH_AGE_S = 60.0       # a snapshot older than this is fetched again by the next refresh
SIGN_IN_TIMEOUT_S = 300.0          # how long a browser sign-in waits for the browser
_ACCOUNT_JOBS = ("account_signin", "account_code", "account_poll", "account_key", "account_signout")

class _Mailbox:
    """Thread-safe hand-over from worker threads to the GUI thread: workers ``post``, a GUI timer ``drain``s."""

    def __init__(self):
        self._lock = threading.Lock()
        self._items: list[tuple] = []

    def post(self, *item) -> None:
        with self._lock:
            self._items.append(item)

    def drain(self) -> list[tuple]:
        with self._lock:
            items, self._items = self._items, []
        return items


# -- the OriginShape account, shared by this dialog and the panel ------------------------------------------------------

def _blank_account_state(api_url: str = "") -> dict:
    return {"api_url": api_url, "known": False, "signed_in": False, "kind": "", "email": "", "snapshot": None,
            "models": [], **_update_fields(api_url, None), "error": "", "error_code": ""}


def _update_fields(api_url: str, snapshot: dict | None) -> dict:
    """What the panel says about a newer Oscar: from the signed-in account's ``/v1/me`` when there is one, else from
    the public update check (``oscar.updates``, cached; no network). ``update`` is ``""``, ``"available"`` or
    ``"required"``."""
    release = _updates.cached(api_url) if api_url else None
    if snapshot:
        latest, minimum = snapshot.get("plugin_latest_version") or "", snapshot.get("plugin_min_version") or ""
        download = snapshot.get("link_download") or (release.download_url if release is not None else "")
    elif release is not None:
        latest, minimum, download = release.latest_version, release.min_version, release.download_url
    else:
        latest = minimum = download = ""
    versions = {"plugin_latest_version": latest, "plugin_min_version": minimum}
    return {"update": _auth_account.plugin_update(versions), "latest_version": latest, "min_version": minimum,
            "download_url": download, "oscar_version": __version__}


class AccountHub(QtCore.QObject):
    """The process-wide view of the OriginShape sign-in that the panel and the settings dialog share.

    ``state`` is ``{"api_url", "known", "signed_in", "kind", "email", "snapshot": AccountSnapshot.to_dict() | None,
    "models": the aliases of the cached ``/v1/models``, "update" ("" | "available" | "required"), "latest_version",
    "min_version", "download_url", "oscar_version", "error", "error_code"}`` for the configured server (``known`` turns
    True once its stored sign-in was read). The update fields come from the account's ``/v1/me``, else from the public
    update check (``check_updates``: ``GET /v1/plugin`` at most once a day, also for people who never sign in).
    ``changed`` (that state as JSON) is emitted on the GUI thread whenever it changes. Updates arrive through
    ``_incoming`` from any thread: ``oscar.auth.session.on_change`` (sign-in, sign-out, a new account snapshot) is
    pointed at it, and so is the refresh thread. The state itself is always re-read from the session on the GUI thread
    (no I/O once the session is loaded), so a late event can never undo a newer one.

    ``refresh(force)`` returns at once. Local facts are read on the GUI thread only when that is cheap: a keystore in
    memory or in the credential file, or a session this process has already loaded. The OS keychain (a subprocess that
    may wait for an unlock) and ``GET /v1/me`` run on ONE daemon thread at a time; a refresh asked for meanwhile runs
    once more when it ends (coalesced). The hub lives as long as the process, so a late result never reaches a dead
    object. GUI thread only, apart from emitting ``_incoming``.
    """

    changed = QtCore.Signal(str)
    _incoming = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.state: dict = _blank_account_state()
        self._thread: threading.Thread | None = None
        self._seq = 0                         # numbers the refresh threads: only the latest one's end counts
        self._again: bool | None = None       # a refresh asked for while one ran (its ``force``)
        self._loaded: set[str] = set()        # API URLs whose stored sign-in this process has read
        self._update_thread: threading.Thread | None = None
        self._incoming.connect(self._on_incoming)
        _auth_session.on_change = self._incoming.emit

    # -- reading ------------------------------------------------------------------------------------------------------

    @staticmethod
    def configured_api_url() -> tuple[str, "AuthError | None"]:
        """``(api_url, None)``, or ``(the URL as configured, InvalidServerURL)`` for a server Oscar refuses."""
        try:
            return _auth_config.api_url(), None
        except AuthError as e:
            return _auth_config.configured_url().strip(), e

    def links(self) -> dict[str, str]:
        """The account pages the browser may open (``billing``, ``manage``, ``waitlist``, ``download``, ``app``): from
        the account snapshot, else from the server's cached sign-in metadata; ``""`` when unknown. Both are validated
        where they are parsed (https, or http on this computer). No network."""
        snapshot = self.state.get("snapshot") or {}
        links = {name: str(snapshot.get(f"link_{name}") or "")
                 for name in ("billing", "manage", "waitlist", "download", "app")}
        url = self.state.get("api_url") or ""
        if url and not all(links.values()):
            try:
                meta = _auth_discovery.cached(url)
            except AuthError:
                meta = None
            if meta is not None:
                links["billing"] = links["billing"] or meta.billing_url
                links["manage"] = links["manage"] or meta.manage_url
                links["app"] = links["app"] or meta.app_origin
                links["download"] = links["download"] or meta.download_url
        links["download"] = links["download"] or str(self.state.get("download_url") or "")
        return links

    # -- refreshing ---------------------------------------------------------------------------------------------------

    def refresh(self, force: bool = False) -> None:
        """Bring ``state`` up to date for the configured server (see the class docstring). ``force`` fetches the
        account even when the snapshot is young."""
        url, problem = self.configured_api_url()
        if problem is not None:
            self._set({**_blank_account_state(url), "known": True, "error": str(problem), "error_code": problem.code})
            return
        if url != self.state["api_url"]:
            self._set(_blank_account_state(url))      # another server: nothing is known about it yet
        if self._thread is not None and self._thread.is_alive():
            self._again = bool(self._again) or force
            return
        if url not in self._loaded and _keystore.mode() in ("memory", "file"):
            self._loaded.add(url)                     # at most a small local file: cheap enough for the GUI thread
        if url in self._loaded:
            self._sync(url)
            state = self.state
            snapshot = state["snapshot"] or {}
            age = time.time() - float(snapshot.get("fetched_at") or 0.0)
            if not state["signed_in"] or not (force or not snapshot or age > ACCOUNT_REFRESH_AGE_S):
                return
        self._start(url, force)

    def check_updates(self, force: bool = False) -> None:
        """Ask the server whether a newer Oscar is out (``oscar.updates``), on a daemon thread: at most once a day per
        server, and only while the check is on (Settings), unless ``force`` (Oscar → Check for Updates…). Returns at
        once; the answer lands in ``state`` through ``_incoming``."""
        url, problem = self.configured_api_url()
        if problem is not None or (self._update_thread is not None and self._update_thread.is_alive()):
            return
        if not force and (not _updates.enabled() or not _updates.due(url)):
            return
        emit = self._incoming.emit

        def work() -> None:   # daemon thread: one GET, then the signal
            try:
                _updates.fetch(url)
            except Exception:  # noqa: BLE001 - unreachable or unreadable: nothing to say, asked again tomorrow
                pass
            try:
                emit(json.dumps({"event": "release", "api_url": url}))
            except RuntimeError:                      # the application is quitting
                pass

        self._update_thread = threading.Thread(target=work, name="oscar-update-check", daemon=True)
        self._update_thread.start()

    def _start(self, url: str, force: bool) -> None:
        emit = self._incoming.emit
        self._seq += 1
        seq = self._seq

        def work() -> None:   # daemon thread: the keystore read and /v1/me; never a Qt object but the signal
            payload = {"event": "refresh", "api_url": url, "seq": seq, "error": "", "error_code": ""}
            try:
                session = _auth_session.get_session(url)
                if session.has_session():
                    session.account(force=force)      # a changed snapshot also arrives through on_change
                    try:                              # the server's models, for the pickers and the capabilities
                        _auth_account.fetch_models(url)
                    except AuthError:                 # the built-in lineup until the next refresh
                        pass
            except SignedOut:
                pass                                  # a state, not an error: the session says so now
            except AuthError as e:                    # unreachable, refused, unreadable: the message is safe to show
                payload.update(error=str(e), error_code=e.code)
            except Exception as e:  # noqa: BLE001 - a bug: its text could quote anything, so only its type
                payload.update(error=f"unexpected {type(e).__name__}", error_code="error")
            try:
                emit(json.dumps(payload))
            except RuntimeError:                      # the application is quitting
                pass

        self._thread = threading.Thread(target=work, name="oscar-account", daemon=True)
        self._thread.start()

    def _on_incoming(self, payload: str) -> None:
        """GUI thread: an ``on_change`` event of a session, or the end of a refresh."""
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            return
        if not isinstance(data, dict):
            return
        event, url = str(data.get("event") or ""), str(data.get("api_url") or "")
        if event == "release":                        # the update check answered: not an event of the session
            self.note_release(url)
            return
        if url:
            self._loaded.add(url)                     # any event of a session means it is loaded
        ended = event == "refresh" and data.get("seq") == self._seq   # the thread in flight is done
        if ended:
            self._thread = None
        if url and url == self.state["api_url"]:
            if event == "refresh":
                self._sync(url, error=(str(data.get("error") or ""), str(data.get("error_code") or "")))
            else:
                self._sync(url, error=("", ""))       # a sign-in, a sign-out or a new snapshot: the server answered
        if ended:
            again, self._again = self._again, None
            if again is not None:
                self.refresh(force=again)
        elif event == "signed_in" and url == self.state["api_url"]:
            self.refresh()                            # the account of the new sign-in, for the badge

    def note_release(self, url: str = "") -> None:
        """GUI thread: the update check of ``url`` (default: the configured server) answered; only the update fields
        change (the sign-in isn't read, so no keychain is asked)."""
        url = url or self.state.get("api_url") or ""
        if url and url == self.state["api_url"]:
            self._set({**self.state, **_update_fields(url, self.state.get("snapshot"))})

    def _sync(self, url: str, error: "tuple[str, str] | None" = None) -> None:
        """``state`` from the loaded session of ``url`` (no I/O); ``error`` replaces the last refresh error."""
        session = _auth_session.get_session(url)
        signed_in = session.has_session()
        snapshot = session.cached_snapshot() if signed_in else None
        if error is None:   # keep what the last refresh of this server said
            error = (self.state["error"], self.state["error_code"]) if self.state["api_url"] == url else ("", "")
        message, code = error if signed_in else ("", "")   # signed out, there is nothing to have failed
        catalog = _auth_account.cached_models(url) if signed_in else None
        self._set({"api_url": url, "known": True, "signed_in": signed_in, "kind": session.kind if signed_in else "",
                   "email": session.email if signed_in else "",
                   "snapshot": snapshot.to_dict() if snapshot is not None else None,
                   "models": [m.id for m in catalog.models] if catalog is not None else [],
                   **_update_fields(url, snapshot.to_dict() if snapshot is not None else None),
                   "error": message, "error_code": code})

    def _set(self, state: dict) -> None:
        if state == self.state:
            return
        self.state = state
        self.changed.emit(json.dumps(state))

    # -- actions the panel starts -------------------------------------------------------------------------------------

    def sign_out(self) -> None:
        """Sign out of the configured server on a daemon thread (the revocation is a network call with a short
        timeout); the new state arrives through ``on_change``."""
        url, problem = self.configured_api_url()
        if problem is not None:
            return

        def work() -> None:
            try:
                _auth_session.get_session(url).sign_out()
            except Exception:  # noqa: BLE001 - sign_out forgets the sign-in even when the server cannot be told
                pass

        threading.Thread(target=work, name="oscar-account-signout", daemon=True).start()


_HUB: "AccountHub | None" = None


def account_hub() -> AccountHub:
    """The process-wide ``AccountHub``, created on first use. Call it on the GUI thread."""
    global _HUB
    if _HUB is None:
        _HUB = AccountHub()
    return _HUB


def reset_account_hub() -> None:
    """Forget the hub (tests): the next ``account_hub()`` builds a fresh one and takes ``on_change`` over again."""
    global _HUB
    _HUB = None


def _no_browser(_url: str) -> bool:
    """The opener a sign-in flow keeps: the dialog opens the browser itself, on the GUI thread, so the flow (which
    lives on a worker thread) never holds a reference to a Qt object."""
    return False


def _after_sign_in(session) -> dict:
    """Worker thread, right after a sign-in was kept: make the OriginShape profile the active one and load the
    account. Every message in the result is safe to show."""
    result = {"profile": "", "profile_error": "", "store_error": session.last_store_error, "account_error": "",
              "file_notice": ""}
    try:
        result["profile"] = _params.ensure_account_profile(activate=True)
    except OSError as e:   # SettingsUnavailable: the settings file is locked right now; nothing was written
        result["profile_error"] = str(e)
    try:
        session.account(force=True)
    except AuthError as e:
        result["account_error"] = str(e)
    try:
        if _keystore.file_notice_pending():
            result["file_notice"] = _keystore.file_notice_text()
            _keystore.acknowledge_file_notice()
    except Exception:  # noqa: BLE001 - the notice is a nicety
        pass
    return result


def _account_job(work):
    """Wrap an account job: an ``AuthError`` comes back as ``{"error", "code"}`` (its message is safe to show), any
    other failure as its type only (its text could quote anything)."""
    def run(cancelled):
        try:
            return work(cancelled)
        except AuthError as e:
            return {"error": str(e), "code": e.code}
        except Exception as e:  # noqa: BLE001
            return {"error": f"Oscar hit an unexpected {type(e).__name__}.", "code": "error"}
    return run


# -- look: colours derived from the palette, readable in light and dark themes -----------------------------------------

def _is_dark(palette) -> bool:
    return palette.color(QtGui.QPalette.Window).lightness() < 128


def _mix(a, b, share_b: float):
    """``a`` moved ``share_b`` of the way towards ``b``."""
    return QtGui.QColor(round(a.red() + (b.red() - a.red()) * share_b),
                        round(a.green() + (b.green() - a.green()) * share_b),
                        round(a.blue() + (b.blue() - a.blue()) * share_b))


def _css(color) -> str:
    return f"rgb({color.red()}, {color.green()}, {color.blue()})"


def tone_color(palette, tone: str) -> str:
    """A CSS colour for ``tone`` on ``palette``'s window, or ``""`` for the normal text colour.

    ``muted`` (helper text) is the text colour 30 % of the way towards the window colour: about 5:1 contrast on a dark
    theme and 8:1 on a light one (the palette's Mid role, used before, all but vanished on dark themes). ``error`` and
    ``ok`` are a red / green picked for the theme's brightness."""
    dark = _is_dark(palette)
    if tone == "error":
        return "#ff8f85" if dark else "#b3261e"
    if tone == "ok":
        return "#7fd99a" if dark else "#1b7a3d"
    if tone == "muted":
        return _css(_mix(palette.color(QtGui.QPalette.WindowText), palette.color(QtGui.QPalette.Window), 0.3))
    return ""


class _StatusLabel(QtWidgets.QLabel):
    """A plain-text status line that takes no room while it is empty (an empty form row still adds its spacing)."""

    def __init__(self, name: str = ""):
        super().__init__("")
        if name:
            self.setObjectName(name)
        self.setTextFormat(QtCore.Qt.PlainText)      # what a server or provider says is data, never markup
        self.setWordWrap(True)
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.setVisible(False)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt's name
        super().setText(text)
        self.setVisible(bool(text))


# The account's models by the names the user sees, with one line on each: the server's own (``/v1/models``, cached by
# the account hub), else this built-in lineup (``params.ACCOUNT_MODEL_CHOICES``) until the server has answered. A
# retired alias (``params.LEGACY_ACCOUNT_MODELS``) is shown as its tier.
ACCOUNT_MODEL_NAMES = {
    "origin1": (_params.ACCOUNT_MODEL_LABELS["origin1"], "Recommended for most work."),
    "origin1-super": (_params.ACCOUNT_MODEL_LABELS["origin1-super"],
                      "The most capable model for CAD work. Uses more credits than Origin1."),
}


def account_model_names() -> list[tuple[str, str, str]]:
    """``(alias, name, line)`` of each model the account offers: the cached ``/v1/models`` in the server's order,
    else the built-in lineup. No network."""
    try:
        catalog = _auth_account.cached_models()
    except Exception:  # noqa: BLE001 - an unusable server URL: the built-in lineup needs nothing
        catalog = None
    if catalog is not None and catalog.models:
        return [(m.id, m.display_name or ACCOUNT_MODEL_NAMES.get(m.id, (m.id, ""))[0],
                 m.description or ACCOUNT_MODEL_NAMES.get(m.id, ("", ""))[1]) for m in catalog.models]
    aliases = [str(m) for m in _params.ORIGINSHAPE_PRESET.get("models") or _params.ACCOUNT_MODEL_CHOICES]
    return [(alias, *ACCOUNT_MODEL_NAMES.get(alias, (alias, ""))) for alias in aliases]

_TAB_ACCOUNT, _TAB_MODEL, _TAB_TOOLS = 0, 1, 2


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Oscar — Settings")
        self.setMinimumSize(640, 500)
        self.resize(800, 700)
        # ``_from_file`` is False when the settings file could not be read just now: the list is then an in-memory
        # starter profile, and saving it would replace the user's profiles (``_save`` refuses).
        self.profiles, self._from_file = _params.load_profiles_checked()
        # The ACTIVE profile object: a rename in the dialog must not change which profile is active
        self._active = next((p for p in self.profiles if p.name == _params.active_profile_name()), None)
        # The own profile "Your own provider" goes back to, and a blank one that choice added (dropped again when the
        # account is chosen before it got a key).
        self._own_choice = self._active if self._active is not None and not _params.is_account_profile(self._active) \
            else None
        self._starter: Profile | None = None
        # Set by a sign-in, a sign-out or a new server: they take effect at once, so whoever opened the dialog
        # refreshes its view even when the dialog is then cancelled.
        self.account_changed = False
        self._hub = account_hub()
        self._account_failure = ""               # the "could not load your account" line, while it is shown
        self._jobs: dict[str, tuple] = {}        # kind -> (job id, BackgroundJob, on_done, button, busy text)
        self._mailbox = _Mailbox()
        self._job_seq = 0
        self._job_t0 = 0.0
        self._test_account = False               # the running Test is the account's (its own label and button)
        self._ticker = QtCore.QTimer(self)       # "Testing… 12 s" once a second while a job runs
        self._ticker.setInterval(1000)
        self._ticker.timeout.connect(self._tick)
        self._poller = QtCore.QTimer(self)       # drains the mailbox on the GUI thread
        self._poller.setInterval(40)
        self._poller.timeout.connect(self._drain_mailbox)
        self._loading = False
        # The profile whose data the editor is showing. _stash_editor MUST resolve the profile through THIS, never
        # through list.currentRow(): the currentRowChanged signal fires AFTER the row has already moved, so stashing
        # by current row overwrote the newly selected profile with the previous editor state — the "name shows but
        # model and API key are empty" bug. Held by identity (``_editor_row`` is its index), so rows that move under
        # it (an added account profile, a removal) never redirect a stash.
        self._editor_profile: Profile | None = None
        self._rows: list[Profile] = []           # the own-provider list: row -> profile

        self._build_ui()
        self._reload_list(select_profile=self._own_choice)
        self._hub.changed.connect(self._on_account_state)
        self._hub.refresh(force=True)          # plan, credits and status, on a daemon thread when signed in
        self._render_account()
        # Own-provider users usually come to change a key or a model; everybody else starts at the account.
        self.tabs.setCurrentIndex(_TAB_MODEL if self._own_choice is not None and self._active is self._own_choice
                                  else _TAB_ACCOUNT)

    # -- the editor row (an index into ``profiles``, resolved through the profile it shows) -------------------------

    @property
    def _editor_row(self) -> int:
        return self._index_of(self._editor_profile)

    @_editor_row.setter
    def _editor_row(self, row: int) -> None:
        self._editor_profile = self.profiles[row] if 0 <= row < len(self.profiles) else None

    def _index_of(self, profile) -> int:
        return next((i for i, p in enumerate(self.profiles) if p is profile), -1) if profile is not None else -1

    # -- look -------------------------------------------------------------------------------------------------------

    def _tone(self, widget, tone: str) -> None:
        """Colour ``widget``'s text for ``tone`` (see ``tone_color``); remembered so a palette change recolours it."""
        widget.setProperty("oscarTone", tone)
        color = tone_color(self.palette(), tone)
        widget.setStyleSheet(f"color: {color};" if color else "")

    def _hint(self, text: str = "", name: str = "", plain: bool = False) -> QtWidgets.QLabel:
        """A short helper line under a control: wrapped, slightly smaller, in the muted colour."""
        label = QtWidgets.QLabel(text)
        if name:
            label.setObjectName(name)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)   # a wrapped line given spare height stays put
        if plain:
            label.setTextFormat(QtCore.Qt.PlainText)
        font = label.font()
        if font.pointSizeF() > 0:
            font.setPointSizeF(max(8.0, font.pointSizeF() * 0.92))
            label.setFont(font)
        self._tone(label, "muted")
        return label

    @staticmethod
    def _heading(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        return label

    def _apply_tones(self) -> None:
        for widget in self.findChildren(QtWidgets.QWidget):
            tone = widget.property("oscarTone")
            if isinstance(tone, str):
                self._tone(widget, tone)
        self._style_primary(self.signin_btn)
        if self.account_billing_btn.property("oscarPrimary"):
            self._style_primary(self.account_billing_btn)

    def _style_primary(self, button) -> None:
        """The one main action of a page: filled with the theme's highlight colour."""
        pal = self.palette()
        fill = pal.color(QtGui.QPalette.Highlight)
        text = pal.color(QtGui.QPalette.HighlightedText)
        off_fill = _mix(pal.color(QtGui.QPalette.Button), pal.color(QtGui.QPalette.Window), 0.3)
        off_text = tone_color(pal, "muted")
        button.setStyleSheet(
            f"QPushButton {{ background-color: {_css(fill)}; color: {_css(text)}; border: 1px solid "
            f"{_css(fill.darker(120))}; border-radius: 4px; padding: 6px 18px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: {_css(fill.lighter(112))}; }}"
            f"QPushButton:pressed {{ background-color: {_css(fill.darker(112))}; }}"
            f"QPushButton:focus {{ border: 1px solid {_css(text)}; }}"
            f"QPushButton:disabled {{ background-color: {_css(off_fill)}; color: {off_text}; border-color: "
            f"{_css(off_fill)}; }}")

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.PaletteChange and hasattr(self, "signin_btn"):
            self._apply_tones()

    @staticmethod
    def _page(name: str):
        """A tab page that scrolls vertically when the window is small: ``(scroll area, its column layout)``."""
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName(name)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        scroll.viewport().setAutoFillBackground(False)
        body.setAutoFillBackground(False)
        column = QtWidgets.QVBoxLayout(body)
        column.setContentsMargins(14, 14, 14, 14)
        column.setSpacing(14)
        return scroll, column

    @staticmethod
    def _collapser(text: str, name: str = "") -> QtWidgets.QToolButton:
        toggle = QtWidgets.QToolButton()
        if name:
            toggle.setObjectName(name)
        toggle.setText(text)
        toggle.setCheckable(True)
        toggle.setAutoRaise(True)
        toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        toggle.setArrowType(QtCore.Qt.RightArrow)
        return toggle

    # -- building -----------------------------------------------------------------------------------------------------

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("oscarSettingsTabs")
        self.tabs.setDocumentMode(False)
        outer.addWidget(self.tabs, 1)

        page, column = self._page("oscarSettingsAccountTab")
        column.addWidget(self._build_account())
        column.addWidget(self._build_developer())
        column.addStretch(1)
        self.tabs.addTab(page, "Account")

        page, column = self._page("oscarSettingsModelTab")
        column.addWidget(self._build_choice())
        column.addWidget(self._build_profiles())
        column.addStretch(1)
        self.tabs.addTab(page, "AI model")

        page, column = self._page("oscarSettingsToolsTab")
        column.addWidget(self._build_net_permissions())
        column.addWidget(self._build_web())
        column.addWidget(self._build_scripts())
        column.addWidget(self._build_updates())
        column.addStretch(1)
        self.tabs.addTab(page, "Tools && permissions")

        # Where a failed Save explains itself; the dialog stays open.
        self.save_label = QtWidgets.QLabel("")
        self.save_label.setObjectName("oscarSaveStatus")
        self.save_label.setTextFormat(QtCore.Qt.PlainText)      # an OS error text is data, never markup
        self.save_label.setWordWrap(True)
        self._tone(self.save_label, "error")
        self.save_label.setVisible(False)
        outer.addWidget(self.save_label)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._style_primary(self.signin_btn)

    # -- Tools & permissions ------------------------------------------------------------------------------------------

    def _build_scripts(self) -> QtWidgets.QGroupBox:
        """The Scripts group: the ``AllowCodeExecution`` switch of ``codeguard`` (saved with Save)."""
        box = QtWidgets.QGroupBox("Scripts")
        box.setObjectName("oscarScripts")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)
        self.allow_code_check = QtWidgets.QCheckBox("Let Oscar run Python scripts it writes (you approve each one)")
        self.allow_code_check.setObjectName("oscarAllowCode")
        self.allow_code_check.setChecked(_params.get_bool(_codeguard.PREF, False))
        layout.addWidget(self.allow_code_check)
        info = ("Off: Oscar builds with its CAD tools only. On: Oscar may also write Python scripts; each one is shown "
                "to you before it runs, and you decide.")
        if _codeguard.env_override():
            info += " (OSCAR_ALLOW_CODE=1 is set, so scripts run without asking in this session.)"
        layout.addWidget(self._hint(info))
        return box

    def _build_updates(self) -> QtWidgets.QGroupBox:
        """The Updates group: the daily check for a newer Oscar (``updates.PREF``, saved with Save)."""
        box = QtWidgets.QGroupBox("Updates")
        box.setObjectName("oscarUpdates")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)
        self.update_check = QtWidgets.QCheckBox("Check for a newer Oscar once a day")
        self.update_check.setObjectName("oscarUpdateCheck")
        self.update_check.setChecked(_params.get_bool(_updates.PREF, True))
        layout.addWidget(self.update_check)
        info = (f"This is Oscar {__version__}. The check asks the OriginShape server for the latest version number; it "
                "sends nothing about you or your work. Oscar → Check for Updates… asks at any time.")
        if (os.environ.get(_updates.ENV) or "").strip() == "0":
            info += f" ({_updates.ENV}=0 is set, so Oscar doesn't check in this session.)"
        layout.addWidget(self._hint(info))
        return box

    def _build_net_permissions(self):
        """The Network permissions group: the grants with Remove / Revoke all."""
        box = QtWidgets.QGroupBox("Network permissions")
        box.setObjectName("oscarNetPermissions")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)
        layout.addWidget(self._hint("Sites Oscar's tools may reach without asking you again. Removing an entry takes "
                                    "effect at once."))
        self.net_empty = self._hint("Nothing is allowed yet: Oscar asks you before a tool goes online.",
                                    "oscarNetGrantEmpty")
        layout.addWidget(self.net_empty)
        self.net_list = QtWidgets.QListWidget()
        self.net_list.setObjectName("oscarNetGrantList")
        self.net_list.setMaximumHeight(120)
        self.net_list.currentRowChanged.connect(lambda _row: self._update_net_buttons())
        layout.addWidget(self.net_list)
        row = QtWidgets.QHBoxLayout()
        self.net_remove_btn = QtWidgets.QPushButton("Remove")
        self.net_remove_btn.setObjectName("oscarNetGrantRemove")
        self.net_remove_btn.setAutoDefault(False)
        self.net_remove_btn.clicked.connect(self._remove_net_grant)
        self.net_revoke_all_btn = QtWidgets.QPushButton("Revoke all")
        self.net_revoke_all_btn.setObjectName("oscarNetGrantRevokeAll")
        self.net_revoke_all_btn.setAutoDefault(False)
        self.net_revoke_all_btn.clicked.connect(self._revoke_all_net_grants)
        row.addWidget(self.net_remove_btn)
        row.addWidget(self.net_revoke_all_btn)
        row.addStretch(1)
        layout.addLayout(row)
        self._net_rows: list[dict] = []
        self._set_net_rows(_model.net_grant_rows())
        return box

    def _build_web(self):
        """The web settings: the search backend and extra trusted hosts (saved with Save)."""
        box = QtWidgets.QGroupBox("Web access")
        box.setObjectName("oscarWebAccess")
        form = QtWidgets.QFormLayout(box)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.search_url_edit = QtWidgets.QLineEdit(_params.get_string("search_backend_url", ""))
        self.search_url_edit.setObjectName("oscarSearchBackendUrl")
        self.search_url_edit.setPlaceholderText("http://localhost:8080")
        self.allow_extra_edit = QtWidgets.QLineEdit(_params.get_string("web_allowlist_extra", ""))
        self.allow_extra_edit.setObjectName("oscarWebAllowlistExtra")
        self.allow_extra_edit.setPlaceholderText("docs.vendor.io, parts.example.com")
        form.addRow("Search backend", self.search_url_edit)
        form.addRow("", self._hint("A SearXNG server with JSON output turned on. Oscar's web search needs one; leave "
                                   "it empty if you don't use web search."))
        form.addRow("Extra hosts", self.allow_extra_edit)
        form.addRow("", self._hint("More web hosts to trust, separated by commas. Oscar still asks you before it "
                                   "reaches each one."))
        return box

    def _set_net_rows(self, rows: list) -> None:
        self._net_rows = list(rows)
        self.net_list.clear()
        for item in self._net_rows:
            self.net_list.addItem(item["label"])
        self.net_list.setVisible(bool(self._net_rows))
        self.net_empty.setVisible(not self._net_rows)
        self._update_net_buttons()

    def _update_net_buttons(self) -> None:
        busy = "netgrants" in self._jobs
        self.net_remove_btn.setEnabled(not busy and 0 <= self.net_list.currentRow() < len(self._net_rows))
        self.net_revoke_all_btn.setEnabled(not busy and bool(self._net_rows))

    def _net_job(self, fn) -> None:
        """Run a grant write on a worker thread; the list is refreshed from its result (never from the worker)."""
        def work(_cancelled):
            fn()
            return _model.net_grant_rows()

        def done(result):
            if result.ok:
                self._set_net_rows(result.value)
                self.save_label.setVisible(False)
            else:
                self.save_label.setText(f"✗ Could not change the network permissions: {result.error}")
                self.save_label.setVisible(True)
                self._update_net_buttons()

        self._start_job("netgrants", work, done)
        self._update_net_buttons()

    def _remove_net_grant(self) -> None:
        row = self.net_list.currentRow()
        if 0 <= row < len(self._net_rows):
            entry = dict(self._net_rows[row])
            self._net_job(lambda: _model.revoke_net_grant(entry))

    def _revoke_all_net_grants(self) -> None:
        self._net_job(_model.revoke_all_net_grants)

    # -- Account ------------------------------------------------------------------------------------------------------

    def _build_account(self) -> QtWidgets.QGroupBox:
        """The OriginShape account card: signed out, what it is for and the three ways in; signed in, who, plan,
        credits, what the account's state means, its actions and the workspace switch; a status line."""
        box = QtWidgets.QGroupBox("OriginShape account")
        box.setObjectName("oscarAccount")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(10)
        self.account_stack = QtWidgets.QStackedWidget()
        self.account_stack.setObjectName("oscarAccountStack")
        layout.addWidget(self.account_stack)

        # -- signed out
        out = QtWidgets.QWidget()
        out.setObjectName("oscarAccountSignedOut")
        out_layout = QtWidgets.QVBoxLayout(out)
        out_layout.setContentsMargins(0, 0, 0, 0)
        out_layout.setSpacing(8)
        headline = self._heading("You're not signed in")
        headline.setObjectName("oscarAccountHeadline")
        out_layout.addWidget(headline)
        intro = QtWidgets.QLabel("Sign in to use Oscar with your OriginShape credits — no API key needed.")
        intro.setObjectName("oscarAccountIntro")
        intro.setWordWrap(True)
        out_layout.addWidget(intro)
        self.signin_btn = QtWidgets.QPushButton("Sign in with OriginShape")
        self.signin_btn.setObjectName("oscarAccountSignIn")
        self.signin_btn.setToolTip("Opens the OriginShape sign-in page in your browser; it hands the result back to "
                                   "Oscar on this computer")
        self.signin_btn.clicked.connect(self._account_sign_in)
        primary = QtWidgets.QHBoxLayout()
        primary.addWidget(self.signin_btn)
        primary.addStretch(1)
        out_layout.addSpacing(2)
        out_layout.addLayout(primary)
        out_layout.addWidget(self._hint("Opens your browser. When you're done there, Oscar picks up the sign-in by "
                                        "itself."))
        out_layout.addSpacing(6)
        out_layout.addWidget(self._heading("Other ways to sign in"))
        self.signin_code_btn = QtWidgets.QPushButton("Sign in with a code")
        self.signin_code_btn.setObjectName("oscarAccountSignInCode")
        self.signin_code_btn.setToolTip("Shows a short code to enter on the OriginShape website (works on any device)")
        self.signin_code_btn.clicked.connect(self._account_sign_in_code)
        self.use_key_btn = QtWidgets.QPushButton("Use an OriginShape API key")
        self.use_key_btn.setObjectName("oscarAccountUseKey")
        self.use_key_btn.setToolTip("Sign in with a personal OriginShape API key (os_live_…), created on the website")
        self.use_key_btn.clicked.connect(self._account_show_key_row)
        others = QtWidgets.QGridLayout()
        others.setHorizontalSpacing(12)
        others.setVerticalSpacing(6)
        others.addWidget(self.signin_code_btn, 0, 0)
        others.addWidget(self._hint("Use this if the browser can't reach FreeCAD — on another device, a remote "
                                    "desktop or behind a strict firewall. You type a short code on the website."),
                         0, 1)
        others.addWidget(self.use_key_btn, 1, 0)
        key_hint = self._hint("A personal key (os_live_…) you created on the OriginShape website.")
        others.addWidget(key_hint, 1, 1)
        if not OFFER_API_KEY_SIGNIN:   # the server has personal keys turned off (API_KEYS_ENABLED)
            self.use_key_btn.hide()
            key_hint.hide()
        others.setColumnStretch(1, 1)
        out_layout.addLayout(others)
        for button in (self.signin_btn, self.signin_code_btn, self.use_key_btn):
            button.setAutoDefault(False)          # Enter keeps meaning Save
        self.account_code_label = QtWidgets.QLabel("")
        self.account_code_label.setObjectName("oscarAccountCode")
        self.account_code_label.setTextFormat(QtCore.Qt.PlainText)
        self.account_code_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        font = self.account_code_label.font()
        if font.pointSize() > 0:
            font.setPointSize(font.pointSize() + 6)
        font.setBold(True)
        self.account_code_label.setFont(font)
        self.account_code_label.setVisible(False)
        out_layout.addWidget(self.account_code_label)
        self.account_key_row = QtWidgets.QWidget()
        key_row = QtWidgets.QHBoxLayout(self.account_key_row)
        key_row.setContentsMargins(0, 0, 0, 0)
        self.account_key_edit = QtWidgets.QLineEdit()
        self.account_key_edit.setObjectName("oscarAccountKey")
        self.account_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.account_key_edit.setPlaceholderText("os_live_…")
        self.account_key_btn = QtWidgets.QPushButton("Use this key")
        self.account_key_btn.setObjectName("oscarAccountKeyApply")
        self.account_key_btn.setAutoDefault(False)
        self.account_key_btn.clicked.connect(self._account_use_key)
        key_row.addWidget(self.account_key_edit, 1)
        key_row.addWidget(self.account_key_btn)
        self.account_key_row.setVisible(False)
        out_layout.addWidget(self.account_key_row)
        self.account_stack.addWidget(out)

        # -- signed in
        signed_in = QtWidgets.QWidget()
        signed_in.setObjectName("oscarAccountSignedIn")
        in_layout = QtWidgets.QVBoxLayout(signed_in)
        in_layout.setContentsMargins(0, 0, 0, 0)
        in_layout.setSpacing(8)
        self._account_form = QtWidgets.QFormLayout()
        self._account_form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.account_email = QtWidgets.QLabel("")
        self.account_email.setObjectName("oscarAccountEmail")
        font = self.account_email.font()
        font.setBold(True)
        self.account_email.setFont(font)
        self.account_plan = QtWidgets.QLabel("")
        self.account_plan.setObjectName("oscarAccountPlan")
        self.account_credits = QtWidgets.QLabel("")
        self.account_credits.setObjectName("oscarAccountCredits")
        font = self.account_credits.font()
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * 1.6)
        font.setBold(True)
        self.account_credits.setFont(font)
        for label in (self.account_email, self.account_plan, self.account_credits):
            label.setTextFormat(QtCore.Qt.PlainText)      # what the server says is data, never markup
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._account_form.addRow("Signed in as", self.account_email)
        self._account_form.addRow("Plan", self.account_plan)
        self._account_form.addRow("Credits", self.account_credits)
        in_layout.addLayout(self._account_form)
        self.account_waitlist = QtWidgets.QLabel(
            "Your account is on the waitlist. You can use Oscar with OriginShape as soon as it is approved. Until "
            "then you can use your own provider (AI model tab).")
        self.account_waitlist.setObjectName("oscarAccountWaitlist")
        self.account_waitlist.setWordWrap(True)
        self.account_waitlist.setVisible(False)
        in_layout.addWidget(self.account_waitlist)
        self.account_notice = QtWidgets.QLabel("")
        self.account_notice.setObjectName("oscarAccountNotice")
        self.account_notice.setWordWrap(True)
        self.account_notice.setVisible(False)
        in_layout.addWidget(self.account_notice)
        actions = QtWidgets.QHBoxLayout()
        self.account_manage_btn = QtWidgets.QPushButton("Manage account")
        self.account_manage_btn.setObjectName("oscarAccountManage")
        self.account_manage_btn.setToolTip("Opens your OriginShape account page (plan, credits, devices, API keys)")
        self.account_manage_btn.clicked.connect(self._account_manage)
        self.account_billing_btn = QtWidgets.QPushButton("Buy credits")
        self.account_billing_btn.setObjectName("oscarAccountBilling")
        self.account_billing_btn.setToolTip("Opens the OriginShape billing page in your browser")
        self.account_billing_btn.clicked.connect(self._account_billing)
        self.signout_btn = QtWidgets.QPushButton("Sign out")
        self.signout_btn.setObjectName("oscarAccountSignOut")
        self.signout_btn.setToolTip("Signs this computer out of OriginShape")
        self.signout_btn.clicked.connect(self._account_sign_out)
        for button in (self.account_manage_btn, self.account_billing_btn):
            button.setAutoDefault(False)
            actions.addWidget(button)
        actions.addStretch(1)
        self.signout_btn.setAutoDefault(False)
        actions.addWidget(self.signout_btn)
        in_layout.addLayout(actions)
        # Connect mode (oscar.connect): tasks started in the OriginShape workspace run in this FreeCAD. Applied at
        # once, like the rest of this section.
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        in_layout.addSpacing(4)
        in_layout.addWidget(line)
        self.connect_check = QtWidgets.QCheckBox("Connect to the OriginShape workspace")
        self.connect_check.setObjectName("oscarConnectWorkspace")
        self.connect_check.setChecked(_params.get_bool(_connect.PREF, False))
        self.connect_check.toggled.connect(self._apply_connect)
        in_layout.addWidget(self.connect_check)
        self.connect_status = self._hint("", "oscarConnectStatus", plain=True)
        in_layout.addWidget(self.connect_status)
        # The browser may answer this computer's script and network questions (connect.REMOTE_PREF, on by default).
        self.remote_check = QtWidgets.QCheckBox("Approve scripts and network access from the workspace")
        self.remote_check.setObjectName("oscarConnectApprovals")
        self.remote_check.setToolTip(
            "Questions Oscar asks here can also be answered in the browser, where you see the script first. "
            "Anyone signed in to your OriginShape account can then run code on this computer through Oscar. "
            "'Always allow' is only ever offered here.")
        self.remote_check.setChecked(_params.get_bool(_connect.REMOTE_PREF, _connect.REMOTE_DEFAULT))
        self.remote_check.setEnabled(self.connect_check.isChecked())
        self.remote_check.toggled.connect(lambda on: _params.set_bool(_connect.REMOTE_PREF, bool(on)))
        self.connect_check.toggled.connect(self.remote_check.setEnabled)
        in_layout.addWidget(self.remote_check)
        self._watch_connect_state()
        self.account_stack.addWidget(signed_in)

        self.account_status = QtWidgets.QLabel("")
        self.account_status.setObjectName("oscarAccountStatus")
        self.account_status.setTextFormat(QtCore.Qt.PlainText)    # progress and errors (server text) are data
        self.account_status.setWordWrap(True)
        self.account_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.account_status.setVisible(False)
        layout.addWidget(self.account_status)
        return box

    def _build_developer(self) -> QtWidgets.QWidget:
        """Collapsed: the Server field, for testing against another OriginShape API."""
        section = QtWidgets.QWidget()
        section.setObjectName("oscarAccountDeveloper")
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.server_toggle = self._collapser("Developer: server address", "oscarAccountDeveloperToggle")
        self.server_toggle.toggled.connect(self._toggle_server)
        layout.addWidget(self.server_toggle)
        self.server_box = QtWidgets.QWidget()
        box = QtWidgets.QFormLayout(self.server_box)
        box.setContentsMargins(22, 0, 0, 0)
        box.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.server_custom = QtWidgets.QCheckBox("Use a custom server")
        self.server_custom.setObjectName("oscarAccountServerCustom")
        self.server_custom.setToolTip(f"Off: Oscar signs in at {_auth_config.DEFAULT_API_URL}. On: Oscar uses the "
                                      "Server below. Applies at once.")
        self.server_edit = QtWidgets.QLineEdit()
        self.server_edit.setObjectName("oscarAccountServer")
        self.server_edit.setPlaceholderText(f"{_auth_config.DEFAULT_API_URL} (default)")
        self.server_edit.setToolTip("Only for testing against another OriginShape API, e.g. http://localhost:8210 (the "
                                    "API, not the website). Plain http is accepted for this computer only. "
                                    "Applies at once.")
        env = (os.environ.get(_auth_config.ENV_API_URL) or "").strip()
        hint = (f"Off, Oscar signs in at {_auth_config.DEFAULT_API_URL}. On, it uses the Server instead: only for "
                "testing against another OriginShape API (the API, not the website), e.g. http://localhost:8210. "
                "Plain http works for this computer only. Applies at once.")
        if env:
            self.server_custom.setChecked(True)
            self.server_custom.setEnabled(False)
            self.server_edit.setText(env)
            self.server_edit.setEnabled(False)
            self.server_edit.setToolTip(f"Set by the {_auth_config.ENV_API_URL} environment variable.")
            hint = f"Set by the {_auth_config.ENV_API_URL} environment variable for this session."
        else:
            self.server_custom.setChecked(self._custom_server_on())
            self.server_edit.setText(self._stored_server())
            self.server_edit.setEnabled(self.server_custom.isChecked())
        self.server_custom.toggled.connect(self._apply_server_switch)
        self.server_edit.editingFinished.connect(self._apply_server_field)
        box.addRow(self.server_custom)
        box.addRow("Server", self.server_edit)
        self.server_problem = _StatusLabel("oscarAccountServerProblem")
        self._tone(self.server_problem, "error")
        under = QtWidgets.QVBoxLayout()
        under.setSpacing(4)
        under.addWidget(self.server_problem)
        under.addWidget(self._hint(hint))
        box.addRow("", under)
        layout.addWidget(self.server_box)
        # Open when Oscar is not talking to the default server, so that is never a surprise.
        self.server_toggle.setChecked(self.server_custom.isChecked())
        self._toggle_server(self.server_toggle.isChecked())
        return section

    def _toggle_server(self, shown: bool) -> None:
        self.server_box.setVisible(shown)
        self.server_toggle.setArrowType(QtCore.Qt.DownArrow if shown else QtCore.Qt.RightArrow)

    def _server_says(self, text: str) -> None:
        """A problem with the Server field, next to the field (opened) as well as in the account's status line."""
        self.server_problem.setText(text)
        if text:
            self.server_toggle.setChecked(True)

    # -- AI model -----------------------------------------------------------------------------------------------------

    def _build_choice(self) -> QtWidgets.QGroupBox:
        """Which AI Oscar uses: the OriginShape account or one of the user's own provider profiles."""
        box = QtWidgets.QGroupBox("Which AI should Oscar use?")
        box.setObjectName("oscarModelChoice")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)

        self.use_account_radio = QtWidgets.QRadioButton("OriginShape account (uses credits)")
        self.use_account_radio.setObjectName("oscarUseAccount")
        self.use_account_radio.clicked.connect(lambda *_: self._choose(account=True))
        self._mode_group.addButton(self.use_account_radio)
        layout.addWidget(self.use_account_radio)
        indent = QtWidgets.QVBoxLayout()
        indent.setContentsMargins(26, 0, 0, 0)
        indent.setSpacing(6)
        indent.addWidget(self._hint("No API key needed: each reply is paid for with your OriginShape credits."))
        self.account_details = QtWidgets.QWidget()
        self.account_details.setObjectName("oscarAccountChoiceDetails")
        details = QtWidgets.QVBoxLayout(self.account_details)
        details.setContentsMargins(0, 2, 0, 4)
        details.setSpacing(6)
        state_row = QtWidgets.QHBoxLayout()
        self.ai_account_state = QtWidgets.QLabel("")
        self.ai_account_state.setObjectName("oscarAccountChoiceState")
        self.ai_account_state.setTextFormat(QtCore.Qt.PlainText)
        self.ai_account_state.setWordWrap(True)
        self.goto_account_btn = QtWidgets.QPushButton("Go to Account")
        self.goto_account_btn.setObjectName("oscarGoToAccount")
        self.goto_account_btn.setAutoDefault(False)
        self.goto_account_btn.clicked.connect(lambda *_: self.tabs.setCurrentIndex(_TAB_ACCOUNT))
        state_row.addWidget(self.ai_account_state, 1)
        state_row.addWidget(self.goto_account_btn)
        details.addLayout(state_row)
        model_form = QtWidgets.QFormLayout()
        model_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        model_row = QtWidgets.QHBoxLayout()
        self.account_model_combo = QtWidgets.QComboBox()
        self.account_model_combo.setObjectName("oscarAccountModel")
        self.account_model_combo.setMinimumWidth(200)
        self.account_model_combo.currentIndexChanged.connect(self._on_account_model_picked)
        self.account_test_btn = QtWidgets.QPushButton("Test")
        self.account_test_btn.setObjectName("oscarAccountTest")
        self.account_test_btn.setAutoDefault(False)
        self.account_test_btn.setToolTip("Sends three tiny requests through your OriginShape account (they cost a few "
                                         "credits) and remembers what the model can do")
        self.account_test_btn.clicked.connect(lambda *_: self._test(account=True))
        model_row.addWidget(self.account_model_combo, 1)
        model_row.addWidget(self.account_test_btn)
        model_form.addRow("Model", model_row)
        self.account_model_hint = self._hint("", "oscarAccountModelHint")
        model_form.addRow("", self.account_model_hint)
        details.addLayout(model_form)
        self.account_test_label = _StatusLabel("oscarAccountTestResult")
        details.addWidget(self.account_test_label)
        indent.addWidget(self.account_details)
        layout.addLayout(indent)

        layout.addSpacing(4)
        self.use_own_radio = QtWidgets.QRadioButton("Your own provider (your API key)")
        self.use_own_radio.setObjectName("oscarUseOwnProvider")
        self.use_own_radio.clicked.connect(lambda *_: self._choose(account=False))
        self._mode_group.addButton(self.use_own_radio)
        layout.addWidget(self.use_own_radio)
        indent = QtWidgets.QVBoxLayout()
        indent.setContentsMargins(26, 0, 0, 0)
        indent.setSpacing(6)
        indent.addWidget(self._hint("Anthropic, OpenAI, Google Gemini, a local Ollama model and more: you bring the "
                                    "API key and the provider bills you."))
        self.own_details = QtWidgets.QWidget()
        self.own_details.setObjectName("oscarOwnChoiceDetails")
        own = QtWidgets.QFormLayout(self.own_details)
        own.setContentsMargins(0, 2, 0, 0)
        own.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.own_active_combo = QtWidgets.QComboBox()
        self.own_active_combo.setObjectName("oscarOwnProfileInUse")
        self.own_active_combo.setMinimumWidth(200)
        self.own_active_combo.activated.connect(self._on_own_choice)
        own.addRow("Profile in use", self.own_active_combo)
        indent.addWidget(self.own_details)
        layout.addLayout(indent)
        return box

    def _build_profiles(self) -> QtWidgets.QGroupBox:
        """The user's own provider profiles: list (never the account profile) and the editor of the selected one."""
        self.byok_box = QtWidgets.QGroupBox("Your provider profiles")
        self.byok_box.setObjectName("oscarByokSection")
        outer = QtWidgets.QVBoxLayout(self.byok_box)
        outer.setSpacing(8)
        outer.addWidget(self._hint("Set up as many as you like (work and personal keys, cloud and local models); the "
                                   "panel's model menu switches between them."))
        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(12)
        outer.addLayout(columns)

        # Left: profile list + add/remove
        left = QtWidgets.QVBoxLayout()
        self.list = QtWidgets.QListWidget()
        self.list.setObjectName("oscarProfileList")
        self.list.setMinimumWidth(150)
        self.list.setMaximumWidth(210)
        self.list.currentRowChanged.connect(self._on_row_changed)
        left.addWidget(self.list, 1)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        add_btn = QtWidgets.QPushButton("Add")
        add_btn.setObjectName("oscarProfileAdd")
        add_btn.setToolTip("Add a profile for another provider or key")
        add_btn.clicked.connect(self._add)
        dup_btn = QtWidgets.QPushButton("Duplicate")
        dup_btn.setObjectName("oscarProfileDuplicate")
        dup_btn.clicked.connect(self._duplicate)
        del_btn = QtWidgets.QPushButton("Remove")
        del_btn.setObjectName("oscarProfileRemove")
        del_btn.clicked.connect(self._remove)
        for button in (add_btn, dup_btn, del_btn):
            button.setAutoDefault(False)
            row.addWidget(button)
        left.addLayout(row)
        columns.addLayout(left)

        # Right: editor
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self._form = form
        form.setVerticalSpacing(8)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setObjectName("oscarProfileName")
        self.name_edit.editingFinished.connect(self._sync_current_name)
        form.addRow("Profile name", self.name_edit)

        # Item data: the index in ``params.PROVIDER_PRESETS`` (the account's preset is not offered: choosing the
        # account is the radio button above).
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.setObjectName("oscarProfileProvider")
        for index, preset in enumerate(_params.PROVIDER_PRESETS):
            if not preset.get("account"):
                self.preset_combo.addItem(preset["label"], index)
        self._add_coming_soon(self.preset_combo)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        form.addRow("Provider", self.preset_combo)

        self.base_edit = QtWidgets.QLineEdit()
        self.base_edit.setObjectName("oscarProfileBaseUrl")
        self.base_edit.setPlaceholderText("https://…")
        form.addRow("Base URL", self.base_edit)
        form.addRow("", self._hint("Keep the provider's default unless you run your own server or proxy."))

        self.key_edit = QtWidgets.QLineEdit()
        self.key_edit.setObjectName("oscarProfileKey")
        self.key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.key_edit.setPlaceholderText("Paste your provider's API key")
        form.addRow("API key", self.key_edit)

        # Model: dropdown populated from the provider's /models endpoint;
        # still editable for endpoints without a listing.
        model_row = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setObjectName("oscarProfileModel")
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.model_combo.currentIndexChanged.connect(self._on_model_picked)
        self.model_combo.setMinimumWidth(180)
        self.load_btn = QtWidgets.QToolButton()
        self.load_btn.setObjectName("oscarProfileLoadModels")
        self.load_btn.setText("↻")
        self.load_btn.setToolTip("Load the model list from the provider's /models endpoint")
        self.load_btn.clicked.connect(self._load_models)
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.load_btn)
        form.addRow("Model", model_row)

        # Temperature and max tokens are predefined per provider and model: computed (they are stored with the
        # profile) but not shown as settings; "Advanced overrides" replaces them and its last line says what is used.
        self.temp_spin = QtWidgets.QDoubleSpinBox(self)
        self.temp_spin.setRange(0.0, 1.0)
        self.temp_spin.setSingleStep(0.1)
        self._make_predefined(self.temp_spin)
        self.temp_spin.hide()
        self.max_tokens_spin = QtWidgets.QSpinBox(self)
        self.max_tokens_spin.setRange(256, 65536)
        self.max_tokens_spin.setSingleStep(256)
        self._make_predefined(self.max_tokens_spin)
        self.max_tokens_spin.hide()

        test_row = QtWidgets.QHBoxLayout()
        self.test_btn = QtWidgets.QPushButton("Test connection")
        self.test_btn.setObjectName("oscarProfileTest")
        self.test_btn.setAutoDefault(False)
        self.test_btn.setToolTip("Checks the connection, one tool call and one tiny image, and remembers what "
                                 "the model can do")
        self.test_btn.clicked.connect(lambda *_: self._test())
        test_row.addWidget(self.test_btn)
        test_row.addStretch(1)
        form.addRow("", test_row)
        self.test_label = _StatusLabel("oscarProfileTestResult")      # a provider's error body is data, never markup
        form.addRow("", self.test_label)

        self._build_advanced(form)
        right = QtWidgets.QVBoxLayout()
        right.addLayout(form)
        right.addStretch(1)                      # the rows stay together at the top, whatever the list's height
        columns.addLayout(right, 1)

        note = self._hint("Your provider's API key is saved in Oscar's settings file on this computer, readable only "
                          "by your user account; use a key you can revoke. Your OriginShape sign-in is kept "
                          "separately, in the system keychain.", "oscarKeyStorageNote")
        outer.addWidget(note)
        return self.byok_box

    def _account_profile(self) -> Profile | None:
        """The account profile the AI model tab edits: the active one when that is an account profile, else the
        first."""
        if self._active is not None and _params.is_account_profile(self._active) and self._index_of(self._active) >= 0:
            return self._active
        return next((p for p in self.profiles if _params.is_account_profile(p)), None)

    def _own_profiles(self) -> list:
        return [p for p in self.profiles if not _params.is_account_profile(p)]

    def _shown_active(self) -> Profile | None:
        """The profile Save will make active."""
        if self._active is not None and self._index_of(self._active) >= 0:
            return self._active
        return self.profiles[0] if self.profiles else None

    def _choose(self, account: bool) -> None:
        """The user picked "OriginShape account" or "Your own provider": that becomes the profile Save makes active.
        The account profile is added when there is none; an own provider without any profile starts a blank one from
        Anthropic (dropped again if the account is chosen before it got a key)."""
        self._stash_editor()
        shown = self._editor_profile
        if account:
            profile = self._account_profile()
            if profile is None:
                profile = _params.account_profile(self._unique_name(_params.ACCOUNT_PROFILE_NAME))
                self.profiles.insert(0, profile)
            starter = self._starter
            if starter is not None and not starter.api_key and self._index_of(starter) >= 0:
                del self.profiles[self._index_of(starter)]
                if self._own_choice is starter:
                    self._own_choice = None
                if shown is starter:
                    shown = None
            self._starter = None
            self._active = profile
        else:
            own = self._own_profiles()
            target = next((p for p in own if p is self._own_choice), None) \
                or next((p for p in own if p is shown), None) or (own[0] if own else None)
            if target is None:
                target = self._new_profile()
                self.profiles.append(target)
                self._starter = target
            self._active = target
            self._own_choice = target
            shown = target
        self._reload_list(select_profile=shown)

    def _on_own_choice(self, index: int) -> None:
        """The "Profile in use" menu: that profile becomes the active one and the editor shows it."""
        if 0 <= index < len(self._rows):
            profile = self._rows[index]
            self._active = profile
            self._own_choice = profile
            if self.list.currentRow() != index:
                self.list.setCurrentRow(index)
            self._refresh_choice()

    def _refresh_choice(self) -> None:
        """The AI model tab from the data: which choice is on, the profile in use, the account's model."""
        active = self._shown_active()
        account = _params.is_account_profile(active)
        (self.use_account_radio if account else self.use_own_radio).setChecked(True)
        self.account_details.setVisible(account)
        self.own_details.setVisible(not account)
        self.byok_box.setVisible(not account)
        self.own_active_combo.blockSignals(True)
        try:
            self.own_active_combo.clear()
            for profile in self._rows:
                self.own_active_combo.addItem(profile.name)
            row = next((i for i, p in enumerate(self._rows) if p is active), -1)
            self.own_active_combo.setCurrentIndex(row)
        finally:
            self.own_active_combo.blockSignals(False)
        for i, profile in enumerate(self._rows):      # the profile in use stands out in the list
            item = self.list.item(i)
            if item is None:
                continue
            font = item.font()
            font.setBold(profile is active and not account)
            item.setFont(font)
            item.setToolTip("In use" if profile is active and not account else "")
        self._show_account_model()

    def _show_account_model(self) -> None:
        profile = self._account_profile()
        model = profile.model if profile is not None and profile.model else _params.ORIGINSHAPE_PRESET["default_model"]
        model = _params.account_model_alias(model)   # a retired alias is shown (and kept on Save) as its tier
        choices = account_model_names()
        if model not in [alias for alias, _name, _line in choices]:
            choices.append((model, model, ""))   # one this server doesn't list: shown as it is, never replaced silently
        self._model_lines = {alias: line for alias, _name, line in choices}
        combo = self.account_model_combo
        combo.blockSignals(True)
        try:
            combo.clear()
            for alias, name, line in choices:
                combo.addItem(name, alias)
                combo.setItemData(combo.count() - 1, line or name, QtCore.Qt.ToolTipRole)
            combo.setCurrentIndex(max(0, combo.findData(model)))
        finally:
            combo.blockSignals(False)
        self._show_account_model_hint()

    def _account_test_says(self, text: str) -> None:
        self.account_test_label.setText(text)

    def _show_account_model_hint(self) -> None:
        alias = self.account_model_combo.currentData() or ""
        self.account_model_hint.setText(getattr(self, "_model_lines", {}).get(alias, ""))
        self.account_model_hint.setVisible(bool(self.account_model_hint.text()))

    def _on_account_model_picked(self, _index: int) -> None:
        alias = self.account_model_combo.currentData()
        profile = self._account_profile()
        if alias and profile is not None:
            profile.model = str(alias)
            profile.temperature, profile.max_tokens = _params.predefined_limits(profile.provider, "", profile.model)
        if self._job_running("test") and self._test_account:
            self._cancel_job("test")
        self._account_test_says("")
        self._show_account_model_hint()

    # -- connect mode ----------------------------------------------------------------------------------------------

    _CONNECT_TEXT = {
        "off": "Tasks you start in the OriginShape workspace (app.originshape.design) run here, in this FreeCAD. "
               "Scripts and network access still ask you first.",
        "connecting": "Connecting to the workspace…",
        "connected": "Connected: tasks from the workspace run here.",
        "working": "Working on a task from the workspace.",
        "stopped": "Not connected.",
    }

    def _apply_connect(self, on: bool) -> None:
        _params.set_bool(_connect.PREF, bool(on))
        from . import connect_gui

        connect_gui.setting_changed()
        self._show_connect_state()

    def _watch_connect_state(self) -> None:
        from . import connect_gui

        connector = connect_gui.current()
        if connector is not None:
            connector.state_changed.connect(self._show_connect_state)
        self._show_connect_state()

    def _show_connect_state(self, *_args) -> None:
        from . import connect_gui

        connector = connect_gui.current()
        state, message = connector.state if connector is not None else ("off", "")
        if state == "offline":
            text = f"Can’t reach the workspace right now; Oscar keeps trying. {message}".strip()
        else:
            text = message or self._CONNECT_TEXT.get(state, "")
        self.connect_status.setText(text)

    @staticmethod
    def _stored_server() -> str:
        return str(_params.get_string(_auth_config.PREF_SERVER, "") or "").strip()

    @staticmethod
    def _custom_server_on() -> bool:
        return bool(_params.get_bool(_auth_config.PREF_CUSTOM_SERVER, False))

    def _open_url(self, url: str) -> bool:
        """Open ``url`` in the desktop's browser; False when that did not work (the caller shows the address)."""
        try:
            return bool(QtGui.QDesktopServices.openUrl(QtCore.QUrl(url)))
        except Exception:  # noqa: BLE001 - no browser: the address is shown instead
            return False

    def _account_say(self, text: str, error: bool = False) -> None:
        self.account_status.setText(text)
        self._tone(self.account_status, "error" if error else "")
        self.account_status.setVisible(bool(text))

    def _account_busy(self) -> bool:
        return any(kind in self._jobs for kind in _ACCOUNT_JOBS)

    def _cancel_account_jobs(self) -> None:
        for kind in _ACCOUNT_JOBS:
            self._cancel_job(kind)
        self.account_code_label.setVisible(False)

    def _on_account_state(self, _payload: str = "") -> None:
        """The hub's state changed (any sign-in or sign-out, a new snapshot, the server's models)."""
        self._render_account()
        if hasattr(self, "account_model_combo"):
            self._show_account_model()

    def _show_account_page(self, index: int) -> None:
        """Switch the account card's page; only the page shown takes room (a QStackedWidget is otherwise as tall as
        its tallest page, which left a large empty band around the signed-out text)."""
        self.account_stack.setCurrentIndex(index)
        for i in range(self.account_stack.count()):
            page = self.account_stack.widget(i)
            policy = QtWidgets.QSizePolicy.Preferred if i == index else QtWidgets.QSizePolicy.Ignored
            page.setSizePolicy(policy, policy)
        self.account_stack.adjustSize()

    @staticmethod
    def _credits_number(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _render_account(self) -> None:
        """Show the hub's state: which page, the account lines, and which buttons can be used now."""
        if not hasattr(self, "account_stack"):
            return
        state = self._hub.state
        busy = self._account_busy()
        snapshot = state.get("snapshot") or {}
        status = snapshot.get("status")
        email = str(snapshot.get("email") or state.get("email") or "")
        if state.get("signed_in"):
            self._show_account_page(1)
            self.account_key_row.setVisible(False)
            self.account_code_label.setVisible(False)
            self.account_email.setText(email or "(loading…)")
            plan = str(snapshot.get("plan_name") or snapshot.get("plan") or "")
            if state.get("kind") == "api_key":
                plan = f"{plan} (personal API key)" if plan else "(personal API key)"
            self.account_plan.setText(plan or "(loading…)")
            waitlisted = status == "waitlisted"
            notice, tone = "", ""
            if status == "approved":
                available = snapshot.get("credits_available")
                low = " (running low)" if snapshot.get("credits_low") is True else ""
                self.account_credits.setText(format_credits(available) + low)
                if self._credits_number(available) <= 0:
                    notice, tone = ("You're out of credits, so Oscar can't use your account until you add more. Buy "
                                    "credits, or switch to your own provider on the AI model tab."), "error"
                elif low:
                    notice, tone = "You're running low on credits.", "muted"
            elif status == "suspended":
                self.account_credits.setText("Unavailable")
                notice, tone = ("Your account is suspended, so Oscar can't use it. Open Manage account to see "
                                "why."), "error"
            else:
                self.account_credits.setText("(loading…)" if not snapshot else "—")
            update = state.get("update")
            if update == "required":   # nothing else works until Oscar is updated: said first
                notice, tone = _auth_account.outdated_message(snapshot), "error"
            elif update == "available" and not notice:
                latest = str(state.get("latest_version") or "")
                notice, tone = (f"Oscar {latest} is available (this is {__version__}). Update it from "
                                "FreeCAD's Addon Manager."), "muted"
            label = self._account_form.labelForField(self.account_credits)
            for widget in (label, self.account_credits):
                if widget is not None:
                    widget.setVisible(not waitlisted)
            self.account_waitlist.setVisible(waitlisted)
            self.account_notice.setText(notice)
            self._tone(self.account_notice, tone)
            self.account_notice.setVisible(bool(notice))
            self.account_billing_btn.setVisible(status not in ("waitlisted", "suspended"))
            out_of_credits = status == "approved" and self._credits_number(snapshot.get("credits_available")) <= 0
            if out_of_credits:                    # then buying credits is the thing to do
                self._style_primary(self.account_billing_btn)
            else:
                self.account_billing_btn.setStyleSheet("")
            self.account_billing_btn.setProperty("oscarPrimary", out_of_credits)
            self.account_billing_btn.setEnabled(not busy)
            self.account_manage_btn.setEnabled(not busy)
            self.signout_btn.setEnabled(not busy)
        else:
            self._show_account_page(0)
            signing_in = self._job_running("account_signin")
            coding = self._job_running("account_code") or self._job_running("account_poll")
            known = bool(state.get("known"))
            self.signin_btn.setText("Cancel sign-in" if signing_in else "Sign in with OriginShape")
            self.signin_code_btn.setText("Cancel" if coding else "Sign in with a code")
            self.signin_btn.setEnabled(known and (signing_in or not busy))
            self.signin_code_btn.setEnabled(known and (coding or not busy))
            self.use_key_btn.setEnabled(known and not busy)
            self.account_key_btn.setEnabled(not busy)
            if not known and not self.account_status.text():
                self._account_say("Checking your OriginShape sign-in…")
        url_problem = state.get("error_code") in ("invalid_url", "insecure_url")
        problem = state.get("error") if url_problem else ""
        # a signed-in account that could not be loaded (the server is unreachable, say): said once, in the status
        # line, unless an action is talking there; cleared when a later refresh worked
        failed = f"Could not load your account: {state.get('error')}" \
            if state.get("signed_in") and state.get("error") and not url_problem else ""
        shown = self.account_status.text()
        if problem and not busy:
            self._account_say(problem, error=True)
            self._server_says(problem)
        elif failed and not busy and (not shown or shown == self._account_failure):
            self._account_say(failed, error=True)
            self._account_failure = failed
        elif shown and shown == self._account_failure and not failed:
            self._account_say("")
            self._account_failure = ""
        elif state.get("known") and shown == "Checking your OriginShape sign-in…":
            self._account_say("")
        self._render_account_choice(state, snapshot, status, email)

    def _render_account_choice(self, state: dict, snapshot: dict, status, email: str) -> None:
        """The account line on the AI model tab."""
        if not state.get("known"):
            text, go = "Checking your OriginShape sign-in…", False
        elif not state.get("signed_in"):
            text, go = "You're not signed in yet. Sign in on the Account tab to use your credits.", True
        elif status == "waitlisted":
            text, go = f"Signed in as {email}. Your account is on the waitlist, so it can't be used yet.", True
        elif status == "suspended":
            text, go = f"Signed in as {email}. Your account is suspended.", True
        elif status == "approved":
            available = snapshot.get("credits_available")
            text = f"Signed in as {email} · {format_credits(available)} available"
            go = self._credits_number(available) <= 0
        else:
            text, go = f"Signed in as {email or '…'}", False
        self.ai_account_state.setText(text)
        self.goto_account_btn.setVisible(go)

    def start_account_action(self, action: str) -> None:
        """Start ``"signin"`` (the browser) or ``"device"`` (a code) as if its button had been clicked: the panel's
        account links and the Sign in command open the dialog this way. Nothing happens when already signed in or when
        an account action runs."""
        self.tabs.setCurrentIndex(_TAB_ACCOUNT)
        if self._hub.state.get("signed_in") or self._account_busy():
            return
        if action == "signin":
            self._account_sign_in()
        elif action == "device":
            self._account_sign_in_code()

    def _apply_server_switch(self, on: bool) -> None:
        """The *Use a custom server* switch: on, the Server field decides (an empty one is still the default); off,
        Oscar uses the default server and keeps the field for next time. It applies at once, like the field."""
        try:
            _params.set_bool(_auth_config.PREF_CUSTOM_SERVER, on)
        except OSError as e:
            self.server_custom.blockSignals(True)     # nothing was written: show what still applies
            self.server_custom.setChecked(not on)
            self.server_custom.blockSignals(False)
            self._account_say(f"Could not save the server: {e}", error=True)
            self._server_says(f"Could not save the server: {e}")
            return
        self.server_edit.setEnabled(on)
        if on:
            self.server_edit.setFocus()
        self._server_changed()

    def _apply_server_field(self) -> bool:
        """Store the Server field when it changed (validated: https, or http on this computer). It applies at once:
        the account shown is the one of that server. False, with the reason shown, when it cannot be used."""
        if not self.server_edit.isEnabled():          # OSCAR_API_URL decides, or the custom server is off
            return True
        text = self.server_edit.text().strip()
        if text == self._stored_server():
            return True
        value = ""
        if text:
            try:
                value = _auth_config.normalise_url(text, what="server URL")
            except AuthError as e:
                self._account_say(str(e), error=True)
                self._server_says(str(e))
                return False
        try:
            _params.set_string(_auth_config.PREF_SERVER, value)
        except OSError as e:
            self._account_say(f"Could not save the server: {e}", error=True)
            self._server_says(f"Could not save the server: {e}")
            return False
        self.server_edit.setText(value)
        self._server_changed()
        return True

    def _server_changed(self) -> None:
        """The server Oscar uses changed (the switch or the field): the account shown is the one of the new server."""
        self._server_says("")
        self.account_changed = True
        self._cancel_account_jobs()                   # they were started for the previous server
        url, _ = self._hub.configured_api_url()
        self._account_say(f"Server: {url}")
        self._hub.refresh()
        self._render_account()

    def _server_note(self, code) -> str:
        """After a sign-in failed because the server could not be reached or did not answer like OriginShape: which
        setting chose that server when it is not the default one, so a forgotten local test server is never a mystery.
        ``""`` otherwise."""
        if code not in ("unavailable", "bad_metadata"):
            return ""
        url, _ = self._hub.configured_api_url()
        if url == _auth_config.DEFAULT_API_URL:
            return ""
        if (os.environ.get(_auth_config.ENV_API_URL) or "").strip():
            return f" That server comes from the {_auth_config.ENV_API_URL} environment variable."
        return (f" That server comes from Developer: server address; turn off Use a custom server to sign in at "
                f"{_auth_config.DEFAULT_API_URL}.")

    def _account_api_url(self) -> str:
        """The API root an account action uses (the Server field applied first); ``""`` with the reason shown."""
        if not self._apply_server_field():
            return ""
        url, problem = self._hub.configured_api_url()
        if problem is not None:
            self._account_say(str(problem), error=True)
            return ""
        return url

    def _account_sign_in(self) -> None:
        """Sign in with the browser: a 127.0.0.1 listener (bound here: fast), the authorize page in the browser, and a
        worker that waits for the answer. A second click cancels it."""
        if self._job_running("account_signin"):
            self._cancel_job("account_signin")    # the worker sees the flag within 0.1 s and closes the listener
            self._account_say("Sign-in cancelled.")
            self._render_account()
            return
        url = self._account_api_url()
        if not url:
            return
        self._cancel_account_jobs()
        try:
            flow = LoopbackSignIn(url, open_browser=_no_browser)
            authorize = flow.prepare()
        except (OSError, AuthError) as e:
            self._account_say(f"Could not start the sign-in: {e}", error=True)
            return
        if self._open_url(authorize):
            self._account_say("Finish signing in in your browser; Oscar is waiting for it.")
        else:
            self._account_say(f"Open this address in your browser to sign in: {authorize}")

        def work(cancelled):
            tokens = flow.wait(cancelled, timeout=SIGN_IN_TIMEOUT_S)
            session = _auth_session.get_session(url)
            session.complete_sign_in(tokens)
            return _after_sign_in(session)

        self._start_job("account_signin", _account_job(work), self._account_signed_in)
        self._render_account()

    def _account_sign_in_code(self) -> None:
        """Sign in with a code: ask the server for one (worker), show it and open the page, then a second worker
        polls until the user approved it there. A second click cancels it."""
        if self._job_running("account_code") or self._job_running("account_poll"):
            self._cancel_job("account_code")
            self._cancel_job("account_poll")
            self.account_code_label.setVisible(False)
            self._account_say("Sign-in cancelled.")
            self._render_account()
            return
        url = self._account_api_url()
        if not url:
            return
        self._cancel_account_jobs()
        flow = DeviceSignIn(url)

        def work(_cancelled):
            return {"flow": flow, "code": flow.start()}

        self._account_say("Asking OriginShape for a sign-in code…")
        self._start_job("account_code", _account_job(work), lambda result: self._account_code_ready(url, result))
        self._render_account()

    def _account_code_ready(self, url: str, result) -> None:
        info = result.value if result.ok and isinstance(result.value, dict) else {"error": result.error}
        if "flow" not in info:   # a failure is {"error", "code"}: its "code" is the reason, not a sign-in code
            self._account_say(f"Could not get a sign-in code: {info.get('error') or 'no answer'}"
                              f"{self._server_note(info.get('code'))}", error=True)
            self._render_account()
            return
        flow, code = info["flow"], info["code"]
        self.account_code_label.setText(code.user_code)
        self.account_code_label.setVisible(True)
        minutes = max(1, round(code.expires_in / 60))
        if self._open_url(code.browser_url):
            self._account_say(f"Check that the page that just opened shows the code {code.user_code}, then allow the "
                              f"sign-in there. The code expires in {minutes} min.")
        else:
            self._account_say(f"Open {code.verification_uri} and enter the code {code.user_code}. It expires in "
                              f"{minutes} min.")

        def work(cancelled):
            tokens = flow.poll(cancelled)
            session = _auth_session.get_session(url)
            session.complete_sign_in(tokens)
            return _after_sign_in(session)

        self._start_job("account_poll", _account_job(work), self._account_signed_in)
        self._render_account()

    def _account_show_key_row(self) -> None:
        self.account_key_row.setVisible(True)
        self.account_key_edit.setFocus()
        self._account_say("Paste a personal OriginShape API key: create one on the OriginShape website (it is shown "
                          "only once).")

    def _account_use_key(self) -> None:
        """Sign in with a personal API key: the server checks it first (worker); it is kept only when accepted."""
        key = self.account_key_edit.text().strip()
        if not key:
            self._account_say("Paste your OriginShape API key first.", error=True)
            return
        url = self._account_api_url()
        if not url:
            return
        self._cancel_account_jobs()

        def work(_cancelled):
            session = _auth_session.get_session(url)
            session.use_api_key(key)
            return _after_sign_in(session)

        self._account_say("Checking the key with OriginShape…")
        self._start_job("account_key", _account_job(work), self._account_signed_in)
        self._render_account()

    def _account_signed_in(self, result) -> None:
        """GUI thread: a sign-in (browser, code or API key) ended."""
        self.account_code_label.setVisible(False)
        info = result.value if result.ok and isinstance(result.value, dict) else {"error": result.error}
        if "error" in info:
            self._account_say(f"Not signed in: {info['error'] or 'no answer'}{self._server_note(info.get('code'))}",
                              error=True)
            self._render_account()
            return
        self.account_changed = True
        self.account_key_edit.clear()               # the key now lives in the keystore only
        self.account_key_row.setVisible(False)
        self._adopt_account_profile(info.get("profile") or "")
        notes = ["Signed in."]
        if info.get("profile_error"):
            notes.append(f"Oscar could not switch to the OriginShape profile ({info['profile_error']}); choose it in "
                         "the panel.")
        if info.get("store_error"):
            notes.append(f"The sign-in could not be saved, so it lasts until FreeCAD closes ({info['store_error']}).")
        if info.get("file_notice"):
            notes.append(info["file_notice"])
        if info.get("account_error"):
            notes.append(f"The account could not be loaded yet: {info['account_error']}")
        self._account_say(" ".join(notes))
        self._hub.refresh()
        self._render_account()

    def _adopt_account_profile(self, name: str) -> None:
        """After a sign-in: the account profile is in the list and is the one Save keeps active
        (``ensure_account_profile`` already made it active in the file)."""
        if not name:
            return
        self._stash_editor()
        profile = next((p for p in self.profiles if p.name == name and _params.is_account_profile(p)), None) \
            or next((p for p in self.profiles if _params.is_account_profile(p)), None)
        if profile is None:
            profile = _params.account_profile(self._unique_name(name))
            self.profiles.insert(0, profile)
        self._active = profile
        self._reload_list(select_profile=self._editor_profile)

    def _account_sign_out(self) -> None:
        url = self._hub.state.get("api_url") or self._account_api_url()
        if not url:
            return
        self._cancel_account_jobs()

        def work(_cancelled):
            _auth_session.get_session(url).sign_out()
            return {}

        self._account_say("Signing out…")
        self._start_job("account_signout", _account_job(work), self._account_signed_out)
        self._render_account()

    def _account_signed_out(self, result) -> None:
        info = result.value if result.ok and isinstance(result.value, dict) else {"error": result.error}
        self.account_changed = True
        if "error" in info:
            self._account_say(f"Sign-out failed: {info['error']}", error=True)
        else:
            self._account_say("Signed out.")
        self._hub.refresh()
        self._render_account()

    def _account_manage(self) -> None:
        self._open_account_page("manage")

    def _account_billing(self) -> None:
        self._open_account_page("billing")

    def _open_account_page(self, which: str) -> None:
        """Open the account's ``manage`` or ``billing`` page (the web app's start page when that one is unknown)."""
        links = self._hub.links()
        url = links.get(which) or links.get("manage") or links.get("app") or ""
        if not url:
            self._account_say("The account page is not known yet: Oscar needs to reach OriginShape first.", error=True)
            self._hub.refresh(force=True)
            return
        if not self._open_url(url):
            self._account_say(f"Open {url} in your browser.")

    def _build_advanced(self, form):
        """The collapsible Advanced section: overrides stored in ``Profile.reasoning`` / ``Profile.caps_override``."""
        self.adv_toggle = self._collapser("Advanced overrides", "oscarProfileAdvancedToggle")
        self.adv_toggle.setToolTip("Replace the predefined temperature, output length, reasoning, context window or "
                                   "tool mode of this profile. Most people never need this.")
        self.adv_toggle.toggled.connect(self._toggle_advanced)
        form.addRow("", self.adv_toggle)

        self.adv_box = QtWidgets.QWidget()
        adv = QtWidgets.QFormLayout(self.adv_box)
        adv.setContentsMargins(0, 0, 0, 0)

        self.adv_max_out = QtWidgets.QSpinBox()
        self.adv_max_out.setRange(0, _model.MAX_OUTPUT_RANGE[1])
        self.adv_max_out.setSingleStep(256)
        self.adv_max_out.setSpecialValueText("Automatic")
        self.adv_max_out.setToolTip("Largest reply the model may write (tokens). Automatic = the predefined value "
                                    "for the provider and model.")
        adv.addRow("Max output", self.adv_max_out)

        temp_row = QtWidgets.QHBoxLayout()
        self.adv_temp_check = QtWidgets.QCheckBox("Override")
        self.adv_temp = QtWidgets.QDoubleSpinBox()
        self.adv_temp.setRange(*_model.TEMPERATURE_RANGE)
        self.adv_temp.setSingleStep(0.1)
        self.adv_temp.setValue(0.2)
        self.adv_temp.setEnabled(False)
        self.adv_temp_check.toggled.connect(self.adv_temp.setEnabled)
        temp_row.addWidget(self.adv_temp_check)
        temp_row.addWidget(self.adv_temp, 1)
        adv.addRow("Temperature", temp_row)

        self.adv_reasoning = QtWidgets.QComboBox()
        for value, label in (("auto", "Automatic (medium for planning, low after)"), ("off", "Off"),
                             ("low", "Low"), ("medium", "Medium"), ("high", "High")):
            self.adv_reasoning.addItem(label, value)
        self.adv_reasoning.setToolTip("How much the model may think. Only sent to models that have a reasoning "
                                      "control.")
        adv.addRow("Reasoning", self.adv_reasoning)

        self.adv_ctx = QtWidgets.QSpinBox()
        self.adv_ctx.setRange(0, _model.CONTEXT_WINDOW_RANGE[1])
        self.adv_ctx.setSingleStep(1024)
        self.adv_ctx.setSpecialValueText("Automatic")
        self.adv_ctx.setToolTip("Context window in tokens. For Ollama this is the num_ctx sent with every request "
                                "(default 16384).")
        adv.addRow("Context window", self.adv_ctx)

        self.adv_tools = QtWidgets.QComboBox()
        for value, label in (("auto", "Automatic (detected and remembered)"), ("native", "Native tool calling"),
                             ("text", "Text protocol (for models without tool support)")):
            self.adv_tools.addItem(label, value)
        adv.addRow("Tool mode", self.adv_tools)

        self.adv_ollama = QtWidgets.QCheckBox("Use Ollama's native API (sends the context window with every request)")
        self.adv_ollama.setChecked(True)
        adv.addRow("", self.adv_ollama)

        self.adv_effective = self._hint("", "oscarProfileEffective", plain=True)
        adv.addRow("", self.adv_effective)

        self.adv_box.setVisible(False)
        form.addRow("", self.adv_box)
        for widget in (self.adv_max_out, self.adv_ctx):
            widget.valueChanged.connect(self._on_advanced_changed)
        self.adv_temp_check.toggled.connect(self._on_advanced_changed)
        self.adv_temp.valueChanged.connect(self._on_advanced_changed)
        for combo in (self.adv_reasoning, self.adv_tools):
            combo.currentIndexChanged.connect(self._on_advanced_changed)
        self.adv_ollama.toggled.connect(self._on_advanced_changed)

    def _toggle_advanced(self, shown: bool):
        self.adv_box.setVisible(shown)
        self.adv_toggle.setArrowType(QtCore.Qt.DownArrow if shown else QtCore.Qt.RightArrow)

    def _advanced_values(self) -> dict:
        """The Advanced widgets as ``settings_model`` values."""
        return {
            "max_output": int(self.adv_max_out.value()),
            "temperature": float(self.adv_temp.value()) if self.adv_temp_check.isChecked() else None,
            "reasoning": self.adv_reasoning.currentData() or "auto",
            "context_window": int(self.adv_ctx.value()),
            "tools": self.adv_tools.currentData() or "auto",
            "ollama_native": self.adv_ollama.isChecked(),
        }

    def _load_advanced(self, profile):
        """Show ``profile``'s overrides in the Advanced widgets (without triggering their change handlers)."""
        values = _model.read_advanced(profile) if profile is not None else _model.read_advanced(Profile())
        was = self._loading
        self._loading = True
        try:
            self.adv_max_out.setValue(values["max_output"])
            self.adv_temp_check.setChecked(values["temperature"] is not None)
            self.adv_temp.setValue(values["temperature"] if values["temperature"] is not None else 0.2)
            self.adv_reasoning.setCurrentIndex(max(0, self.adv_reasoning.findData(values["reasoning"])))
            self.adv_ctx.setValue(values["context_window"])
            self.adv_tools.setCurrentIndex(max(0, self.adv_tools.findData(values["tools"])))
            self.adv_ollama.setChecked(values["ollama_native"])
        finally:
            self._loading = was
        has_override = any((values["max_output"], values["temperature"] is not None, values["context_window"],
                            values["reasoning"] != "auto", values["tools"] != "auto", not values["ollama_native"]))
        if has_override and not self.adv_toggle.isChecked():
            self.adv_toggle.setChecked(True)
        self._refresh_effective()

    def _on_advanced_changed(self, *_args):
        if self._loading:
            return
        self._apply_predefined()
        self._refresh_effective()

    def _refresh_effective(self):
        """The 'what will be used' line, computed from a scratch copy of the profile with the widgets applied."""
        row = self._editor_row
        if not (0 <= row < len(self.profiles)):
            self.adv_effective.setText("")
            return
        scratch = copy.deepcopy(self.profiles[row])
        preset = self._preset()
        if preset:
            scratch.provider = preset["provider"]
        scratch.base_url = self.base_edit.text().strip()
        scratch.model = self.model_combo.currentText().strip()
        problems = _model.apply_advanced(scratch, self._advanced_values())
        text = _model.describe_effective(scratch)
        if problems:   # an invalid value is not stored: say so instead of letting the spin box pretend
            text += " Not applied: " + "; ".join(problems) + "."
        self.adv_effective.setText(text)

    # -- background jobs ----------------------------------------------------------------------------------

    def _start_job(self, kind: str, fn, on_done, *, button=None, busy_text: str = ""):
        """Run ``fn(cancelled_event)`` on a worker thread; ``on_done(JobResult)`` runs on the GUI thread.

        A second job of the same ``kind`` cancels the first one. The result of a cancelled or replaced job is dropped.
        ``fn`` must not capture ``self`` or any Qt object: it may reach the GUI only through ``self._mailbox``.
        """
        self._cancel_job(kind)
        self._job_seq += 1
        job_id = self._job_seq
        mailbox = self._mailbox

        def deliver(result):
            mailbox.post("job", job_id, result)

        job = _model.run_in_background(fn, deliver, name=f"oscar-settings-{kind}")
        self._jobs[kind] = (job_id, job, on_done, button, busy_text)
        if button is not None:
            button.setProperty("idle_text", button.property("idle_text") or button.text())
            button.setText(busy_text or "Cancel")
            # One user-started job at a time: the other action buttons wait until it ends (this one is its Cancel).
            for other in self._action_buttons():
                other.setEnabled(other is button)
        self._job_t0 = time.monotonic()
        if not self._ticker.isActive():
            self._ticker.start()
        if not self._poller.isActive():
            self._poller.start()

    def _action_buttons(self) -> tuple:
        """The buttons that start a user job (Test, the account's Test, model list)."""
        return self.test_btn, self.load_btn, self.account_test_btn

    def _release_buttons(self, button) -> None:
        """A job ended: its button is idle again and every action button is usable (while a user job still runs,
        only its own button: a background job that ends must not wake the buttons it put to sleep)."""
        if button is not None:
            button.setText(button.property("idle_text") or button.text())
        running = next((self._jobs[kind] for kind in ("test", "models") if kind in self._jobs), None)
        for other in self._action_buttons():
            other.setEnabled(running is None or other is running[3])
        if hasattr(self, "net_list") and "netgrants" not in self._jobs:
            self._update_net_buttons()
        if hasattr(self, "account_stack"):
            self._render_account()

    def _cancel_job(self, kind: str):
        entry = self._jobs.pop(kind, None)
        if entry is None:
            return
        entry[1].cancel()
        self._release_buttons(entry[3])

    def _cancel_row_jobs(self) -> None:
        """The row / provider the editor shows changed: the Test and model list that were started for the
        previous one must not answer into the new one."""
        running = [kind for kind in ("test", "models") if kind in self._jobs
                   and not (kind == "test" and self._test_account)]     # the account's Test is not about the row
        for kind in running:
            self._cancel_job(kind)
        if running:
            self.test_label.setText("")

    def _cancel_all_jobs(self):
        for kind in list(self._jobs):
            self._cancel_job(kind)
        self._ticker.stop()

    def _job_running(self, kind: str) -> bool:
        return kind in self._jobs

    def _drain_mailbox(self):
        """GUI thread: hand every posted result to its handler (results of cancelled jobs are dropped here)."""
        for item in self._mailbox.drain():
            if item[0] == "job":
                self._on_job_done(item[1], item[2])
        if not self._jobs:
            self._ticker.stop()
            if not self._mailbox._items:
                self._poller.stop()

    def _on_job_done(self, job_id: int, result):
        for kind, entry in list(self._jobs.items()):
            if entry[0] == job_id:
                del self._jobs[kind]
                self._release_buttons(entry[3])
                entry[2](result)
                break

    def _tick(self):
        """Once a second while a job runs: say so (elapsed seconds), so a slow local model never looks frozen."""
        elapsed = int(time.monotonic() - self._job_t0)
        if "test" in self._jobs:
            if self._test_account:
                self._account_test_says(f"Testing… {elapsed} s")
            else:
                self.test_label.setText(f"Testing… {elapsed} s (a local model may need a minute to load)")


    def done(self, result):
        self._cancel_all_jobs()
        self._poller.stop()
        super().done(result)


    @staticmethod
    def _add_coming_soon(combo: QtWidgets.QComboBox) -> None:
        """List the announced providers after the presets, below a separator and greyed out: they can be seen, not
        chosen. They carry no preset index (``_preset()`` is ``{}`` for them)."""
        if not _params.COMING_SOON_PROVIDERS:
            return
        combo.insertSeparator(combo.count())
        for soon in _params.COMING_SOON_PROVIDERS:
            combo.addItem(f"{soon['label']} — Coming soon")
            index = combo.count() - 1
            combo.model().item(index).setEnabled(False)
            combo.setItemData(index, soon["note"], QtCore.Qt.ToolTipRole)

    @staticmethod
    def _make_predefined(spin: QtWidgets.QAbstractSpinBox) -> None:
        tip = "Predefined by provider and model"
        spin.setReadOnly(True)
        spin.setToolTip(tip)
        spin.setStatusTip(tip)

    # -- list management (the user's own profiles; the account profile is never listed) ----------------------------

    def _reload_list(self, select_profile=None, select_name: str = ""):
        """Rebuild the own-provider list and select ``select_profile`` (or the one named ``select_name``, else the
        first). The editor follows through the row change; the AI model tab is refreshed."""
        self._loading = True
        self.list.clear()
        self._rows = self._own_profiles()
        for profile in self._rows:
            self.list.addItem(profile.name)
        self._loading = False
        row = next((i for i, p in enumerate(self._rows) if p is select_profile), -1)
        if row < 0 and select_name:
            row = next((i for i, p in enumerate(self._rows) if p.name == select_name), -1)
        if row < 0 and self._rows:
            row = 0
        if row >= 0:
            self.list.setCurrentRow(row)
        else:
            self._stash_editor()
            self._load_editor(None)
        self._refresh_choice()

    def _selected(self) -> Profile | None:
        row = self.list.currentRow()
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def _on_row_changed(self, row: int):
        if self._loading:
            return
        self._cancel_row_jobs()       # its profile is no longer the one shown
        self._stash_editor()          # store into the profile the editor showed
        profile = self._selected()
        self._load_editor(profile, self._index_of(profile))

    def _on_preset_changed(self, _index: int):
        preset = self._preset()
        if self._loading or not preset:    # not a usable preset (coming soon)
            return
        self._cancel_row_jobs()       # it was started for the previous provider
        self.base_edit.setText(preset["base_url"])
        self.model_combo.setCurrentText(preset["default_model"])
        # A key belongs to the provider it was entered for: sending it to the newly chosen provider would leak
        # it, and keeping it would fail with a confusing 401.
        self.key_edit.clear()
        self._apply_predefined()
        self._refresh_effective()

    def _select_preset(self, preset: dict) -> None:
        for i in range(self.preset_combo.count()):
            data = self.preset_combo.itemData(i)
            if isinstance(data, int) and 0 <= data < len(_params.PROVIDER_PRESETS) \
                    and _params.PROVIDER_PRESETS[data] is preset:
                self.preset_combo.setCurrentIndex(i)
                return

    def _on_model_picked(self, _index: int):
        if self._loading:
            return
        self._apply_predefined()
        self._refresh_effective()

    # -- predefined sampling values -------------------------------------------

    def _apply_predefined(self):
        """Set temperature/max tokens from the provider+model; never editable."""
        provider = self._preset().get("provider", "")
        temperature, max_tokens = _params.predefined_limits(
            provider, self.base_edit.text().strip(), self.model_combo.currentText().strip()
        )
        self.adv_max_out.setSpecialValueText(f"Automatic ({max_tokens})")
        # what the Advanced overrides replace is used instead
        if self.adv_max_out.value():
            max_tokens = self.adv_max_out.value()
        if self.adv_temp_check.isChecked():
            temperature = self.adv_temp.value()
        self.temp_spin.setValue(temperature)
        self.max_tokens_spin.setValue(max_tokens)

    # -- editor <-> data -------------------------------------------------------

    def _stash_editor(self):
        row = self._editor_row
        if not (0 <= row < len(self.profiles)):
            return
        profile = self.profiles[row]
        self._apply_predefined()  # normalize before storing
        profile.name = self._unique_name(self.name_edit.text().strip() or profile.name, exclude=profile)
        preset = self._preset()
        if preset:
            profile.provider = preset["provider"]
        profile.base_url = self.base_edit.text().strip()
        profile.model = self.model_combo.currentText().strip()
        profile.temperature = float(self.temp_spin.value())
        profile.max_tokens = int(self.max_tokens_spin.value())
        _model.apply_advanced(profile, self._advanced_values())
        profile.api_key = self.key_edit.text().strip()
        if preset.get("account"):     # never: the account's preset is not offered here; kept as a guard
            profile.base_url, profile.api_key = "", ""

    def _load_editor(self, profile: Profile | None, row: int = -1):
        self._editor_row = row
        for widget in (self.name_edit, self.preset_combo, self.base_edit, self.key_edit, self.model_combo,
                       self.load_btn, self.test_btn, self.adv_toggle):
            widget.setEnabled(profile is not None)
        if profile is None:
            for widget in (self.name_edit, self.base_edit, self.key_edit):
                widget.clear()
            self.model_combo.clear()
            self.model_combo.setEditText("")
            self._load_advanced(None)
            self._apply_predefined()
            return
        self._loading = True
        self.name_edit.setText(profile.name)
        self.base_edit.setText(profile.base_url)
        self.key_edit.setText(profile.api_key)
        self.model_combo.clear()
        if profile.model:
            self.model_combo.addItem(profile.model)
            self.model_combo.setCurrentIndex(0)
        # picked by base URL (several presets share the openai_compat provider)
        self._select_preset(_params.preset_for(profile.provider, profile.base_url))
        self._loading = False
        self._load_advanced(profile)
        self._apply_predefined()

    def _unique_name(self, wanted: str, exclude=None) -> str:
        """``wanted``, or ``wanted 2`` / ``wanted 3`` ... when another profile (not ``exclude``) already uses it: two
        profiles with one name make the panel's picker and ``active_profile`` ambiguous."""
        taken = {p.name.casefold() for p in self.profiles if p is not exclude}
        if wanted.casefold() not in taken:
            return wanted
        n = 2
        while f"{wanted} {n}".casefold() in taken:
            n += 1
        return f"{wanted} {n}"

    def _sync_current_name(self):
        profile = self._selected()
        if profile is None:
            return
        new_name = self._unique_name(self.name_edit.text().strip(), exclude=profile)
        if new_name.strip() and new_name != profile.name:
            profile.name = new_name
            self._loading = True
            self.list.currentItem().setText(new_name)
            if self.name_edit.text().strip() != new_name:
                self.name_edit.setText(new_name)    # the editor shows the name that was really taken
            self._loading = False
            self._refresh_choice()                  # the "Profile in use" menu shows the new name

    # -- add / duplicate / remove ----------------------------------------------

    def _new_profile(self) -> Profile:
        """A new own-provider profile from the preset shown (Anthropic when there is none)."""
        preset = self._preset()
        if not preset or preset.get("account"):
            preset = _params.preset_for("anthropic")
        return Profile(
            name=self._unique_name(f"Profile {len(self.profiles) + 1}"),
            provider=preset["provider"],
            base_url=preset["base_url"],
            model=preset["default_model"],
            temperature=preset.get("temperature", 0.2),
            max_tokens=preset.get("max_tokens", 8192),
        )

    def _add(self):
        self._stash_editor()
        profile = self._new_profile()
        self.profiles.append(profile)
        self._reload_list(select_profile=profile)

    def _duplicate(self):
        self._stash_editor()
        source = self._selected()
        if source is None:
            return
        clone = copy.deepcopy(source)
        clone.name = self._unique_name(f"{source.name} copy")
        self.profiles.append(clone)
        self._reload_list(select_profile=clone)

    def _remove(self):
        row = self.list.currentRow()
        if not (0 <= row < len(self._rows)):
            return
        if len(self.profiles) <= 1:
            QtWidgets.QMessageBox.information(
                self, "Oscar", "Keep at least one profile."
            )
            return
        profile = self._rows[row]
        # By identity: list.remove() compares by value and would take the first profile that merely EQUALS the
        # selected one.
        del self.profiles[self._index_of(profile)]
        # The editor still shows the removed profile: forget it, or the row change that follows the reload would
        # stash the removed profile's name / URL / model / key into the profile that took its place.
        self._editor_profile = None
        if self._own_choice is profile:
            self._own_choice = None
        if self._starter is profile:
            self._starter = None
        if self._active is profile:            # the first profile takes over, as Save always did
            self._active = self.profiles[0]
        own = self._own_profiles()
        self._reload_list(select_profile=own[min(row, len(own) - 1)] if own else None)

    # -- model listing -----------------------------------------------------------

    def _load_models(self):
        """Fetch /models for the currently edited provider settings (on a worker thread)."""
        if self._job_running("models"):
            self._cancel_job("models")
            self.test_label.setText("Model list cancelled.")
            return
        self._stash_editor()
        profile = self._selected()
        if profile is None:
            return
        if not profile.base_url:
            self.test_label.setText("✗ Set the Base URL first, then load models.")
            return
        self.test_label.setText("Loading models…")
        snapshot = copy.deepcopy(profile)

        def work(_cancelled):
            from .providers import list_models

            return list_models(snapshot)

        self._start_job("models", work, self._models_loaded, button=self.load_btn, busy_text="✕")

    def _models_loaded(self, result):
        """GUI thread: show the fetched names; the user's own model stays selectable even when it is not listed."""
        models, error = result.value if result.ok and result.value else ([], result.error)
        if not models:
            self.test_label.setText(f"✗ Could not load models: {error or 'no answer'}")
            return
        current = self.model_combo.currentText().strip()
        self._loading = True
        self.model_combo.clear()
        self.model_combo.addItems(models)
        missing = bool(current) and current not in models
        if missing:
            self.model_combo.insertItem(0, current)
        self._loading = False
        if current:
            self.model_combo.setCurrentText(current)
        else:
            self.model_combo.setCurrentIndex(0)
        self._apply_predefined()
        self._refresh_effective()
        note = f" — your model “{current}” is not in the list; it was kept." if missing else " — pick one from the list."
        self.test_label.setText(f"✓ {len(models)} models loaded{note}")

    # -- test / save -------------------------------------------------------------

    def _preset(self) -> dict:
        """The preset the editor's Provider menu shows (``{}`` for an entry that is not a usable preset)."""
        data = self.preset_combo.currentData()
        if isinstance(data, int) and 0 <= data < len(_params.PROVIDER_PRESETS):
            return _params.PROVIDER_PRESETS[data]
        return {}

    def _test(self, account: bool = False):
        """Connection test on a worker thread: reply, one tool call, one tiny image; remembers what it learns.
        ``account`` tests the account profile (the AI model tab's Test), else the profile in the editor."""
        if self._job_running("test"):
            account_test = self._test_account
            self._cancel_job("test")
            self._test_says(account_test, "Test cancelled.")
            return
        if account:
            profile = self._account_profile()
        else:
            self._stash_editor()
            profile = self._selected()
        if profile is None:
            return
        self._test_account = account
        self._test_says(account, "Testing…")
        snapshot = copy.deepcopy(profile)

        def work(cancelled):
            from .provider_check import run_connection_test

            return run_connection_test(snapshot, should_stop=cancelled.is_set)

        button = self.account_test_btn if account else self.test_btn
        self._start_job("test", work, self._test_finished, button=button, busy_text="Cancel test")

    def _test_says(self, account: bool, text: str) -> None:
        if account:
            self._account_test_says(text)
        else:
            self.test_label.setText(text)

    def _test_finished(self, result):
        if not result.ok or result.value is None:
            self._test_says(self._test_account, f"✗ {result.error or 'the test failed'}")
            return
        report = result.value
        text = report.summary() or "(no result)"
        if not report.ok:
            text += "\n✗ This model cannot be used as it is."
        elif report.learned:
            text += "\n(remembered: " + ", ".join(f"{k} = {v}" for k, v in sorted(report.learned.items())) + ")"
        self._test_says(self._test_account, text)
        if not self._test_account:
            self._refresh_effective()

    def _save_failed(self, text: str, tab: int | None = None) -> None:
        self.save_label.setText(text)
        self.save_label.setVisible(True)
        if tab is not None:
            self.tabs.setCurrentIndex(tab)

    def _save(self):
        self._stash_editor()
        self._cancel_all_jobs()
        if not self._from_file:
            self._save_failed("✗ Oscar could not read its settings file when this window opened (another program may "
                              "have had it open), so saving now could replace your profiles. Close and reopen "
                              "Settings.")
            return
        problem = _model.validate_search_url(self.search_url_edit.text())
        if problem:
            self._save_failed("✗ " + problem, _TAB_TOOLS)
            self.search_url_edit.setFocus()
            return
        if not self._apply_server_field():
            self._save_failed("✗ " + (self.account_status.text() or "The OriginShape server URL cannot be used."),
                              _TAB_ACCOUNT)
            return
        seen: set[str] = set()
        for profile in self.profiles:   # profiles stored by an older build may share a name
            if profile.name.casefold() in seen:
                profile.name = self._unique_name(profile.name, exclude=profile)
            seen.add(profile.name.casefold())
        # The active profile follows its object: a rename must not switch Oscar to another model. Only when it
        # was removed does the first profile become active.
        try:
            _params.save_profiles(self.profiles)
            _params.set_string("search_backend_url", self.search_url_edit.text().strip())
            _params.set_string("web_allowlist_extra", self.allow_extra_edit.text().strip())
            _params.set_bool(_codeguard.PREF, self.allow_code_check.isChecked())
            _params.set_bool(_updates.PREF, self.update_check.isChecked())
            if self._active is not None and any(p is self._active for p in self.profiles):
                _params.set_active_profile(self._active.name)
            elif _params.active_profile_name() not in {p.name for p in self.profiles}:
                _params.set_active_profile(self.profiles[0].name)
        except OSError as e:
            # The settings file is locked or unreadable right now (a scanner or sync client, permissions): say so
            # and keep what was typed; the dialog stays open.
            self._save_failed(f"✗ Could not save the settings: {e}")
            return
        self.accept()
