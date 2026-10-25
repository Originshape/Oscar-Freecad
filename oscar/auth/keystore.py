"""Where Oscar keeps the OriginShape sign-in between sessions: one entry per API URL.

An entry holds only ``{"kind": "oauth", "refresh_token", "email"}`` or ``{"kind": "api_key", "api_key", "email"}``;
access tokens live in memory only. ``OSCAR_CREDENTIAL_STORE`` picks the backend at call time:

* ``auto`` (default): the operating system's store; when it is missing or fails, the file below, with a one-time
  notice for the UI (``file_notice_pending()`` / ``acknowledge_file_notice()``). The file only ever holds the entry
  of a save that had to fall back, so ``load`` reads it first: an older OS entry (a used refresh token) never wins.
* ``keychain``: the operating system's store only (an error when it cannot be used).
* ``file``: ``oscar_credentials.json`` next to Oscar's settings file (``OSCAR_CREDENTIAL_FILE`` overrides the path),
  readable by the user only (0600) and replaced atomically.
* ``memory``: this process only. The test suite's default, so no test touches a real keychain.

Operating-system stores:

* macOS Keychain through ``/usr/bin/security -i`` with the commands on STDIN, so the secret never appears in a
  process argument list (``ps`` shows those to every user).
* Windows Credential Manager through ``ctypes`` (``CredWriteW`` / ``CredReadW`` / ``CredDeleteW`` / ``CredFree``).
* Linux Secret Service through ``secret-tool``, with the secret on stdin.

They store the entry as one blob, ``osk1.`` + base64url(JSON), which needs no quoting anywhere. The service is
"Oscar for FreeCAD" and the account ``originshape:`` + 12 hex digits of the API URL's SHA-256.

The module is deliberately NOT called ``credentials.py``: the repository's ``.gitignore`` ignores ``credentials*``,
so such a file would silently be left out of the release.
"""

import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .. import context_guard as _context_guard
from .. import fsretry as _fsretry
from .. import turnlog as _turnlog
from . import config
from .transport import AuthError, clean_token

SERVICE = "Oscar for FreeCAD"
FILE_NAME = "oscar_credentials.json"
ENV_STORE = "OSCAR_CREDENTIAL_STORE"
ENV_FILE = "OSCAR_CREDENTIAL_FILE"
MODES = ("auto", "keychain", "file", "memory")
KINDS = ("oauth", "api_key")
BLOB_PREFIX = "osk1."
SUBPROCESS_TIMEOUT_S = 30.0   # a locked keychain may ask for its password: give the user time, but not forever
_FILE_VERSION = 1


class KeystoreError(AuthError):
    """A credential store failed. The message names the store, never the secret."""

    code = "keystore"


class KeystoreUnavailable(KeystoreError):
    """This store cannot be used here (no tool, no D-Bus session, no settings folder); ``auto`` falls back."""

    code = "keystore_unavailable"


# -- the entry --------------------------------------------------------------------------------------------

def _clean_email(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return text if len(text) <= 320 and text.isprintable() else ""


@dataclass(frozen=True, repr=False)
class StoredCredential:
    """A stored sign-in: ``kind`` ``"oauth"`` (``secret`` is the refresh token) or ``"api_key"``."""

    kind: str
    secret: str
    email: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown credential kind {self.kind!r}")
        object.__setattr__(self, "secret", clean_token(self.secret, "credential",
                                                       error="A stored sign-in has an unexpected format."))
        object.__setattr__(self, "email", _clean_email(self.email))
        _context_guard.remember(self.secret)   # every sign-in this process handles stays out of the model's context

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, ("refresh_token" if self.kind == "oauth" else "api_key"): self.secret,
                "email": self.email}

    @classmethod
    def from_dict(cls, data: Any) -> "StoredCredential | None":
        """The entry from its dict form, or None when it is damaged."""
        if not isinstance(data, dict) or data.get("kind") not in KINDS:
            return None
        kind = data["kind"]
        secret = data.get("refresh_token" if kind == "oauth" else "api_key")
        try:
            return cls(kind, secret, data.get("email") or "")
        except AuthError:
            return None

    def __repr__(self) -> str:
        return f"StoredCredential(kind={self.kind!r}, email={self.email!r}, secret=<hidden>)"


def account_name(api_url: str) -> str:
    """The store's account for one API URL (``originshape:<12 hex>``), so the store's UI never shows the URL."""
    return "originshape:" + hashlib.sha256(api_url.encode("utf-8")).hexdigest()[:12]


def encode_blob(cred: StoredCredential) -> str:
    raw = json.dumps(cred.to_dict(), separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return BLOB_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_blob(text: Any) -> StoredCredential | None:
    """The entry of a blob, or None for anything else (never raises)."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text.startswith(BLOB_PREFIX):
        return None
    body = text[len(BLOB_PREFIX):]
    try:
        data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)).decode("utf-8"))
    except (binascii.Error, ValueError):
        return None
    return StoredCredential.from_dict(data)


# -- backends ---------------------------------------------------------------------------------------------

class Backend:
    """One place to keep entries. ``read`` returns None when there is no entry; ``delete`` of a missing entry is
    fine; failures raise ``KeystoreError``."""

    name = ""

    def available(self) -> bool:
        return True

    def read(self, account: str) -> StoredCredential | None:
        raise NotImplementedError

    def write(self, account: str, cred: StoredCredential) -> None:
        raise NotImplementedError

    def delete(self, account: str) -> None:
        raise NotImplementedError


_MEMORY: dict[str, StoredCredential] = {}
_MEMORY_LOCK = threading.Lock()


class MemoryBackend(Backend):
    name = "memory"

    def read(self, account: str) -> StoredCredential | None:
        with _MEMORY_LOCK:
            return _MEMORY.get(account)

    def write(self, account: str, cred: StoredCredential) -> None:
        with _MEMORY_LOCK:
            _MEMORY[account] = cred

    def delete(self, account: str) -> None:
        with _MEMORY_LOCK:
            _MEMORY.pop(account, None)


_SAFE_ARG = re.compile(r"[A-Za-z0-9 :._\-]+")


def _quoted(value: str) -> str:
    """``value`` in double quotes for the line parser of ``security -i``. Only characters that need no escaping are
    accepted (the blob is base64url, the account ``originshape:<hex>``), so no value can break out of its argument."""
    if not _SAFE_ARG.fullmatch(value):
        raise KeystoreError("Oscar refused to pass an unexpected value to the macOS keychain.")
    return f'"{value}"'


def _not_found(returncode: int, stderr: str) -> bool:
    return returncode == 44 or "could not be found" in stderr   # errSecItemNotFound


class MacKeychainBackend(Backend):
    """The login keychain through ``security -i``. The item is created by ``/usr/bin/security`` itself, which is
    therefore trusted to read it back without a prompt."""

    name = "keychain"
    TOOL = "/usr/bin/security"
    # ``security -i`` reads its commands into a 4096-byte line buffer and runs whatever is left over as a SECOND
    # command, so a longer line would store a truncated blob. Refused instead (``auto`` then uses the file).
    MAX_COMMAND_CHARS = 4000

    def __init__(self, run: Callable[..., Any] = subprocess.run):
        self._run = run

    def available(self) -> bool:
        return sys.platform == "darwin" and os.access(self.TOOL, os.X_OK)

    def _security(self, command: str) -> tuple[int, str, str]:
        if len(command) > self.MAX_COMMAND_CHARS:
            raise KeystoreError("The sign-in is too large for the macOS keychain.")
        failure = ""
        try:
            proc = self._run([self.TOOL, "-i"], input=command + "\n", capture_output=True, text=True,
                             timeout=SUBPROCESS_TIMEOUT_S, check=False)
        except (OSError, subprocess.SubprocessError) as e:
            failure = type(e).__name__
        if failure:
            raise KeystoreUnavailable(f"The macOS keychain could not be used ({failure}).")
        return proc.returncode, proc.stdout or "", proc.stderr or ""

    def read(self, account: str) -> StoredCredential | None:
        rc, out, err = self._security(f"find-generic-password -a {_quoted(account)} -s {_quoted(SERVICE)} -w")
        if rc == 0:
            for line in out.splitlines():
                cred = decode_blob(line)
                if cred is not None:
                    return cred
            return None   # an item Oscar cannot read (damaged): treated as absent, replaced by the next save
        if _not_found(rc, err):
            return None
        raise KeystoreError(f"The macOS keychain could not read the sign-in (security exit status {rc}).")

    def write(self, account: str, cred: StoredCredential) -> None:
        rc, _out, _err = self._security(f"add-generic-password -U -a {_quoted(account)} -s {_quoted(SERVICE)} "
                                        f"-l {_quoted(SERVICE)} -w {_quoted(encode_blob(cred))}")
        if rc != 0:
            raise KeystoreError(f"The macOS keychain did not save the sign-in (security exit status {rc}).")

    def delete(self, account: str) -> None:
        rc, _out, err = self._security(f"delete-generic-password -a {_quoted(account)} -s {_quoted(SERVICE)}")
        if rc != 0 and not _not_found(rc, err):
            raise KeystoreError(f"The macOS keychain did not remove the sign-in (security exit status {rc}).")


CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2
ERROR_NOT_FOUND = 1168
CRED_MAX_BLOB_BYTES = 5 * 512
_CRED_TYPES: tuple | None = None


def cred_types() -> tuple:
    """``(CREDENTIALW, PCREDENTIALW)`` for the Credential Manager calls, built on first use (only Windows and the
    tests need ``ctypes.wintypes``)."""
    global _CRED_TYPES
    if _CRED_TYPES is None:
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

        class CREDENTIALW(ctypes.Structure):
            _fields_ = [("Flags", wintypes.DWORD), ("Type", wintypes.DWORD), ("TargetName", wintypes.LPWSTR),
                        ("Comment", wintypes.LPWSTR), ("LastWritten", FILETIME),
                        ("CredentialBlobSize", wintypes.DWORD), ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
                        ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD),
                        ("Attributes", ctypes.c_void_p), ("TargetAlias", wintypes.LPWSTR),
                        ("UserName", wintypes.LPWSTR)]

        _CRED_TYPES = (CREDENTIALW, ctypes.POINTER(CREDENTIALW))
    return _CRED_TYPES


class WindowsCredentialBackend(Backend):
    """Windows Credential Manager (a generic credential, persisted for this user on this machine). ``advapi32`` and
    ``last_error`` can be injected (the tests use a fake DLL); by default they are loaded on first use."""

    name = "credman"

    def __init__(self, advapi32: Any = None, last_error: Callable[[], int] | None = None):
        self._dll = advapi32
        self._last_error = last_error

    def available(self) -> bool:
        return self._dll is not None or os.name == "nt"

    def _api(self) -> Any:
        if self._dll is not None:
            return self._dll
        failure = ""
        try:
            import ctypes
            from ctypes import wintypes

            credential, pcredential = cred_types()
            dll = ctypes.WinDLL("advapi32", use_last_error=True)
            dll.CredWriteW.argtypes = [pcredential, wintypes.DWORD]
            dll.CredWriteW.restype = wintypes.BOOL
            dll.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(pcredential)]
            dll.CredReadW.restype = wintypes.BOOL
            dll.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
            dll.CredDeleteW.restype = wintypes.BOOL
            dll.CredFree.argtypes = [ctypes.c_void_p]
            dll.CredFree.restype = None
            self._dll, self._last_error = dll, ctypes.get_last_error
        except (OSError, AttributeError, ValueError) as e:
            failure = type(e).__name__
        if failure:
            raise KeystoreUnavailable(f"Windows Credential Manager could not be used ({failure}).")
        return self._dll

    def _error(self) -> int:
        try:
            return int(self._last_error()) if self._last_error is not None else 0
        except Exception:  # noqa: BLE001 - the error number is only for the message
            return 0

    @staticmethod
    def _target(account: str) -> str:
        return f"{SERVICE}/{account}"

    def read(self, account: str) -> StoredCredential | None:
        import ctypes

        api = self._api()
        _credential, pcredential = cred_types()
        handle = pcredential()
        if not api.CredReadW(self._target(account), CRED_TYPE_GENERIC, 0, ctypes.pointer(handle)):
            code = self._error()
            if code == ERROR_NOT_FOUND:
                return None
            raise KeystoreError(f"Windows Credential Manager could not read the sign-in (error {code}).")
        try:
            record = handle.contents
            data = ctypes.string_at(record.CredentialBlob, record.CredentialBlobSize)
        finally:
            api.CredFree(handle)
        return decode_blob(data.decode("ascii", "replace"))

    def write(self, account: str, cred: StoredCredential) -> None:
        import ctypes

        api = self._api()
        credential, _pcredential = cred_types()
        data = encode_blob(cred).encode("ascii")
        if len(data) > CRED_MAX_BLOB_BYTES:
            raise KeystoreError("The sign-in is too large for Windows Credential Manager.")
        buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        record = credential()
        record.Type = CRED_TYPE_GENERIC
        record.TargetName = self._target(account)
        record.UserName = account
        record.CredentialBlobSize = len(data)
        record.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
        record.Persist = CRED_PERSIST_LOCAL_MACHINE
        try:
            ok = api.CredWriteW(ctypes.pointer(record), 0)
        finally:
            ctypes.memset(buffer, 0, len(data))   # do not leave the secret in a buffer that outlives the call
        if not ok:
            raise KeystoreError(f"Windows Credential Manager did not save the sign-in (error {self._error()}).")

    def delete(self, account: str) -> None:
        api = self._api()
        if not api.CredDeleteW(self._target(account), CRED_TYPE_GENERIC, 0):
            code = self._error()
            if code != ERROR_NOT_FOUND:
                raise KeystoreError(f"Windows Credential Manager did not remove the sign-in (error {code}).")


class SecretToolBackend(Backend):
    """The freedesktop Secret Service (GNOME Keyring, KWallet ...) through ``secret-tool``."""

    name = "secret-service"

    def __init__(self, run: Callable[..., Any] = subprocess.run, which: Callable[[str], str | None] = shutil.which):
        self._run = run
        self._which = which

    def available(self) -> bool:
        return sys.platform.startswith("linux") and self._which("secret-tool") is not None

    def _secret_tool(self, args: list[str], stdin: str | None = None) -> tuple[int, str, str]:
        tool = self._which("secret-tool")
        if not tool:
            raise KeystoreUnavailable("secret-tool is not installed.")
        failure = ""
        try:
            proc = self._run([tool, *args], input=stdin, capture_output=True, text=True,
                             timeout=SUBPROCESS_TIMEOUT_S, check=False)
        except (OSError, subprocess.SubprocessError) as e:
            failure = type(e).__name__
        if failure:
            raise KeystoreUnavailable(f"The Secret Service could not be used ({failure}).")
        return proc.returncode, proc.stdout or "", proc.stderr or ""

    @staticmethod
    def _attributes(account: str) -> list[str]:
        return ["service", SERVICE, "account", account]

    def read(self, account: str) -> StoredCredential | None:
        rc, out, err = self._secret_tool(["lookup", *self._attributes(account)])
        if rc == 0:
            return decode_blob(out)
        if not err.strip():
            return None   # exit status 1 and no message: there is no such entry
        raise KeystoreUnavailable("The Secret Service is not available (is a keyring running?).")

    def write(self, account: str, cred: StoredCredential) -> None:
        rc, _out, _err = self._secret_tool(["store", "--label", SERVICE, *self._attributes(account)],
                                           stdin=encode_blob(cred))
        if rc != 0:
            raise KeystoreUnavailable("The Secret Service did not save the sign-in (is a keyring running?).")

    def delete(self, account: str) -> None:
        rc, _out, err = self._secret_tool(["clear", *self._attributes(account)])
        if rc != 0 and err.strip():
            raise KeystoreError("The Secret Service did not remove the sign-in.")


_FILE_LOCK = threading.RLock()


class FileBackend(Backend):
    """``oscar_credentials.json``: ``{"version": 1, "entries": {account: entry}, "notice_shown": bool}``, written
    to a 0600 temporary file in the same folder and moved over the old one, so a crash never leaves half a file."""

    name = "file"

    def __init__(self, path: str | None = None):
        self._path = path

    def path(self) -> str:
        if self._path:
            return self._path
        override = (os.environ.get(ENV_FILE) or "").strip()
        if override:
            return override
        folder = config.settings_dir()
        if not folder:
            raise KeystoreUnavailable("There is no settings folder to keep the sign-in in.")
        return os.path.join(folder, FILE_NAME)

    def available(self) -> bool:
        try:
            self.path()
        except KeystoreError:
            return False
        return True

    def load_state(self) -> tuple[dict, bool]:
        """``(state, damaged)``: a missing file is an empty state; a damaged one is an empty state flagged so that the
        next write replaces it (it only ever caches a sign-in: signing in again is the remedy)."""
        path = self.path()
        failure = ""
        data: Any = None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return _empty_state(), False
        except ValueError:   # not JSON, or not UTF-8
            return _empty_state(), True
        except OSError as e:
            failure = type(e).__name__
        if failure:
            raise KeystoreError(f"The credential file could not be read ({failure}).")
        if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
            return _empty_state(), True
        return data, False

    def save_state(self, state: dict) -> None:
        path = self.path()
        folder = os.path.dirname(os.path.abspath(path))
        failure = ""
        tmp = ""
        try:
            os.makedirs(folder, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".oscar_credentials.", suffix=".tmp", dir=folder)   # mode 0600
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=1, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o600)
            _fsretry.replace(tmp, path)
            tmp = ""
        except OSError as e:
            failure = type(e).__name__
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        if failure:
            raise KeystoreError(f"The credential file could not be written ({failure}).")

    def read(self, account: str) -> StoredCredential | None:
        with _FILE_LOCK:
            try:
                state, _damaged = self.load_state()
            except KeystoreUnavailable:   # no settings folder: nothing can have been stored there
                return None
        return StoredCredential.from_dict(state["entries"].get(account))

    def write(self, account: str, cred: StoredCredential) -> None:
        with _FILE_LOCK:
            state, _damaged = self.load_state()
            state["entries"][account] = cred.to_dict()
            state["version"] = _FILE_VERSION
            self.save_state(state)

    def delete(self, account: str) -> None:
        with _FILE_LOCK:
            try:
                state, damaged = self.load_state()
            except KeystoreUnavailable:
                return
            if account not in state["entries"] and not damaged:
                return
            state["entries"].pop(account, None)
            self.save_state(state)   # a damaged file is rewritten too: no stray copy of a secret survives sign-out


def _empty_state() -> dict:
    return {"version": _FILE_VERSION, "entries": {}, "notice_shown": False}


# -- the store --------------------------------------------------------------------------------------------

def mode() -> str:
    """``OSCAR_CREDENTIAL_STORE`` (``auto`` when unset or unknown)."""
    value = (os.environ.get(ENV_STORE) or "auto").strip().lower()
    return value if value in MODES else "auto"


def _os_backend() -> Backend | None:
    """This platform's store, or None (tests replace this function)."""
    if sys.platform == "darwin":
        return MacKeychainBackend()
    if os.name == "nt":
        return WindowsCredentialBackend()
    if sys.platform.startswith("linux"):
        return SecretToolBackend()
    return None


def _usable_os_backend() -> Backend | None:
    backend = _os_backend()
    return backend if backend is not None and backend.available() else None


def _log(outcome: str, **fields: Any) -> None:
    _turnlog.log_event("auth", event="keystore", outcome=outcome, **fields)


def backend_name() -> str:
    """The store a ``save`` would use first: ``memory``, ``file``, or the OS store's name."""
    current = mode()
    if current in ("memory", "file"):
        return current
    backend = _usable_os_backend()
    if backend is not None:
        return backend.name
    return "none" if current == "keychain" else "file"


def load(api_url: str) -> StoredCredential | None:
    """The stored sign-in of ``api_url`` (a normalised API URL), or None."""
    account = account_name(api_url)
    current = mode()
    if current == "memory":
        return MemoryBackend().read(account)
    if current == "file":
        return FileBackend().read(account)
    backend = _usable_os_backend()
    if current == "keychain":
        if backend is None:
            raise KeystoreUnavailable("No system keychain is available on this computer.")
        return backend.read(account)
    # auto: the file first. ``save`` removes the file's copy whenever the OS store took a write, so an entry there
    # means the LATEST save had to fall back to the file, and an OS entry is older: after a token rotation it would
    # be the used refresh token, whose reuse makes the server revoke the whole sign-in.
    try:
        cred = FileBackend().read(account)
    except KeystoreError:
        cred = None
    if cred is not None or backend is None:
        return cred
    failure = ""
    try:
        cred = backend.read(account)
    except KeystoreError as e:
        failure = e.code
    if failure:
        _log("read_failed", backend=backend.name, reason=failure)
    return cred


def save(api_url: str, cred: StoredCredential) -> str:
    """Store ``cred`` for ``api_url``; returns the name of the backend that holds it. Raises ``KeystoreError`` when
    no backend allowed by the mode could store it."""
    account = account_name(api_url)
    current = mode()
    if current == "memory":
        MemoryBackend().write(account, cred)
        return "memory"
    if current == "file":
        FileBackend().write(account, cred)
        return "file"
    backend = _usable_os_backend()
    if current == "keychain":
        if backend is None:
            raise KeystoreUnavailable("No system keychain is available on this computer.")
        backend.write(account, cred)
        return backend.name
    if backend is not None:
        failure = ""
        try:
            backend.write(account, cred)
        except KeystoreError as e:
            failure = e.code
        if not failure:
            _forget_file_copy(account)
            return backend.name
        _log("fallback", backend=backend.name, reason=failure)
    FileBackend().write(account, cred)
    if backend is not None:   # the OS store may still hold an older entry: remove it (best effort, see ``load``)
        try:
            backend.delete(account)
        except KeystoreError:
            pass
    return "file"


def delete(api_url: str) -> None:
    """Remove the stored sign-in of ``api_url`` from every store the mode allows (missing entries are fine).
    In ``auto`` mode failures are logged, not raised: the in-memory sign-out has already happened."""
    account = account_name(api_url)
    current = mode()
    if current == "memory":
        MemoryBackend().delete(account)
        return
    if current == "file":
        FileBackend().delete(account)
        return
    backend = _usable_os_backend()
    if current == "keychain":
        if backend is None:
            raise KeystoreUnavailable("No system keychain is available on this computer.")
        backend.delete(account)
        return
    if backend is not None:
        failure = ""
        try:
            backend.delete(account)
        except KeystoreError as e:
            failure = e.code
        if failure:
            _log("delete_failed", backend=backend.name, reason=failure)
    _forget_file_copy(account)


def _forget_file_copy(account: str) -> None:
    failure = ""
    try:
        FileBackend().delete(account)
    except KeystoreError as e:
        failure = e.code
    if failure:
        _log("delete_failed", backend="file", reason=failure)


# -- the one-time "stored in a file" notice -------------------------------------------------------------------

def credential_file_path() -> str:
    """The credential file's path, or ``""`` when there is no settings folder."""
    try:
        return FileBackend().path()
    except KeystoreError:
        return ""


def file_notice_pending() -> bool:
    """True when ``auto`` mode had to keep a sign-in in the credential file (no usable OS store) and the user has not
    been told yet. The UI shows ``file_notice_text()`` once, then calls ``acknowledge_file_notice()``."""
    if mode() != "auto":
        return False
    try:
        with _FILE_LOCK:
            state, _damaged = FileBackend().load_state()
    except KeystoreError:
        return False
    return bool(state["entries"]) and not state.get("notice_shown")


def file_notice_text() -> str:
    return ("No system keychain could be used, so your OriginShape sign-in is saved in a file that only your user "
            f"account can read: {credential_file_path()}")


def acknowledge_file_notice() -> None:
    """Remember that the notice was shown (in the file itself, so it survives restarts)."""
    with _FILE_LOCK:
        backend = FileBackend()
        try:
            state, _damaged = backend.load_state()
            if state["entries"] and not state.get("notice_shown"):
                state["notice_shown"] = True
                backend.save_state(state)
        except KeystoreError:
            pass


def reset_memory() -> None:
    """Forget every in-memory entry (tests)."""
    with _MEMORY_LOCK:
        _MEMORY.clear()
