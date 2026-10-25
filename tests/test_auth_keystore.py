"""oscar.auth.keystore: where the sign-in is kept.

* memory and file backends for real (the file: 0600, atomic replace, a damaged file, the path next to the settings);
* the macOS (``security -i``), Linux (``secret-tool``) and Windows (``ctypes`` Credential Manager) backends against
  emulators: the subprocess and the DLL are replaced, Oscar's own command building and marshalling run for real, and
  the secret is proven to travel on stdin only, never in a process argument list;
* the ``auto`` mode's fallback to the file and the one-time notice.

A round trip through the REAL macOS login keychain runs only with ``OSCAR_TEST_REAL_KEYCHAIN=1`` (it writes, reads and
deletes one uniquely named item). Pure python: the platform backends are exercised on any OS.
"""

import ctypes
import json
import os
import secrets
import shlex
import stat
import subprocess
import sys
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.auth import config, keystore  # noqa: E402
from oscar.auth.keystore import KeystoreError, KeystoreUnavailable, StoredCredential  # noqa: E402
from oscar.auth.transport import AuthError  # noqa: E402

API = "https://app.originshape.design"
OTHER = "http://127.0.0.1:8210"


def _secret(prefix: str = "fake-rt-") -> str:
    return prefix + secrets.token_urlsafe(32)


# -- emulators ----------------------------------------------------------------------------------------------

class FakeSecurity:
    """``/usr/bin/security -i`` for the three commands Oscar sends, parsed with shlex (quotes as the real one)."""

    NOT_FOUND = "security: SecKeychainSearchCopyNext: The specified item could not be found in the keychain.\n"

    def __init__(self):
        self.items: dict[tuple[str, str], str] = {}
        self.calls: list[tuple[list, str]] = []
        self.write_status = 0
        self.raise_error: BaseException | None = None

    def __call__(self, argv, *, input=None, capture_output=None, text=None, timeout=None, check=None):  # noqa: A002
        self.calls.append((list(argv), input))
        if self.raise_error is not None:
            raise self.raise_error
        assert argv == ["/usr/bin/security", "-i"], argv
        assert capture_output and text and timeout, "output captured as text, with a timeout"
        rc, out, err = 0, "", ""
        for line in (input or "").splitlines():
            words = shlex.split(line)
            command, args = words[0], words[1:]
            opts, flags, i = {}, set(), 0
            while i < len(args):
                if args[i] in ("-a", "-s", "-l") or (args[i] == "-w" and command == "add-generic-password"):
                    opts[args[i]] = args[i + 1]
                    i += 2
                else:
                    flags.add(args[i])
                    i += 1
            key = (opts["-a"], opts["-s"])
            if command == "add-generic-password":
                if self.write_status:
                    rc, err = self.write_status, "security: SecKeychainItemCreateFromContent: failed\n"
                elif key in self.items and "-U" not in flags:
                    rc, err = 45, "security: The specified item already exists in the keychain.\n"
                else:
                    self.items[key] = opts["-w"]
            elif command == "find-generic-password":
                assert "-w" in flags
                if key in self.items:
                    out = self.items[key] + "\n"
                else:
                    rc, err = 44, self.NOT_FOUND
            elif command == "delete-generic-password":
                if self.items.pop(key, None) is None:
                    rc, err = 44, self.NOT_FOUND
            else:
                rc, err = 1, f"unknown command {command}\n"
        return subprocess.CompletedProcess(argv, rc, out, err)


class FakeSecretTool:
    """``secret-tool`` store / lookup / clear, keyed by the attribute pairs; the secret is read from stdin."""

    def __init__(self):
        self.items: dict[tuple, str] = {}
        self.calls: list[tuple[list, str]] = []
        self.dbus_down = False

    def __call__(self, argv, *, input=None, capture_output=None, text=None, timeout=None, check=None):  # noqa: A002
        self.calls.append((list(argv), input))
        if self.dbus_down:
            return subprocess.CompletedProcess(argv, 1, "", "secret-tool: Cannot autolaunch D-Bus without X11 $DISPLAY\n")
        command, rest = argv[1], argv[2:]
        if command == "store":
            assert rest[:2] == ["--label", keystore.SERVICE]
            self.items[tuple(rest[2:])] = input
            return subprocess.CompletedProcess(argv, 0, "", "")
        key = tuple(rest)
        if command == "lookup":
            if key in self.items:
                return subprocess.CompletedProcess(argv, 0, self.items[key], "")
            return subprocess.CompletedProcess(argv, 1, "", "")
        if command == "clear":
            self.items.pop(key, None)
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 2, "", "usage\n")


class FakeAdvapi32:
    """Credential Manager's four calls, working on the real ``CREDENTIALW`` structures Oscar builds."""

    def __init__(self):
        self.store: dict[str, dict] = {}
        self.error = 0
        self.freed = 0
        self.fail_with = 0
        self._alive: list = []

    def last_error(self) -> int:
        return self.error

    def CredWriteW(self, pcred, flags):  # noqa: N802 - the Win32 name
        if self.fail_with:
            self.error = self.fail_with
            return 0
        record = pcred.contents
        self.store[record.TargetName] = {
            "user": record.UserName, "type": record.Type, "persist": record.Persist,
            "blob": ctypes.string_at(record.CredentialBlob, record.CredentialBlobSize)}
        return 1

    def CredReadW(self, target, cred_type, flags, ppcred):  # noqa: N802
        entry = self.store.get(target)
        if entry is None or cred_type != keystore.CRED_TYPE_GENERIC:
            self.error = keystore.ERROR_NOT_FOUND
            return 0
        credential, _pcredential = keystore.cred_types()
        data = entry["blob"]
        buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        record = credential()
        record.TargetName = target
        record.UserName = entry["user"]
        record.CredentialBlobSize = len(data)
        record.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
        self._alive.append((buffer, record))   # "allocated by the system" until CredFree
        ppcred[0] = ctypes.pointer(record)
        return 1

    def CredDeleteW(self, target, cred_type, flags):  # noqa: N802
        if self.store.pop(target, None) is None:
            self.error = keystore.ERROR_NOT_FOUND
            return 0
        return 1

    def CredFree(self, pointer):  # noqa: N802
        self.freed += 1


class _Available:
    """Wrap a backend so that ``available()`` is True on any OS (the CI pure job runs on Linux)."""

    def __init__(self, backend, *, available: bool = True):
        self._backend = backend
        self._available = available
        self.name = backend.name

    def available(self):
        return self._available

    def __getattr__(self, name):
        return getattr(self._backend, name)


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.dir = isolation.fresh_dir("keystore_")
        self.cred_file = os.path.join(self.dir, "creds", keystore.FILE_NAME)
        patcher = mock.patch.dict(os.environ, {keystore.ENV_STORE: "memory", keystore.ENV_FILE: self.cred_file})
        patcher.start()
        self.addCleanup(patcher.stop)
        keystore.reset_memory()
        self.addCleanup(keystore.reset_memory)

    def use(self, mode: str) -> None:
        os.environ[keystore.ENV_STORE] = mode

    def use_os_backend(self, backend, *, available: bool = True) -> None:
        patcher = mock.patch.object(keystore, "_os_backend", lambda: _Available(backend, available=available))
        patcher.start()
        self.addCleanup(patcher.stop)

    def file_json(self) -> dict:
        with open(self.cred_file, encoding="utf-8") as f:
            return json.load(f)

    def file_text(self) -> str:
        with open(self.cred_file, encoding="utf-8") as f:
            return f.read()


# -- tests ------------------------------------------------------------------------------------------------

class EntryTest(unittest.TestCase):
    def test_account_name(self):
        name = keystore.account_name(API)
        self.assertRegex(name, r"^originshape:[0-9a-f]{12}$")
        self.assertNotEqual(name, keystore.account_name(OTHER))
        self.assertEqual(name, keystore.account_name(API))

    def test_blob_round_trip(self):
        for cred in (StoredCredential("oauth", _secret(), "ada@example.com"),
                     StoredCredential("api_key", _secret("os_live_"))):
            blob = keystore.encode_blob(cred)
            self.assertRegex(blob, r"^osk1\.[A-Za-z0-9_-]+$")   # nothing that needs quoting anywhere
            self.assertEqual(keystore.decode_blob(blob), cred)
            self.assertEqual(keystore.decode_blob("  " + blob + "\n"), cred)
        for junk in (None, "", "osk1.", "osk1.!!!!", "osk2.e30", "plain-password", "osk1." + "e30"):
            self.assertIsNone(keystore.decode_blob(junk), junk)

    def test_the_entry_holds_only_what_the_spec_allows(self):
        oauth = StoredCredential("oauth", "  fake-rt-abcdefgh  ", "ada@example.com").to_dict()
        self.assertEqual(oauth, {"kind": "oauth", "refresh_token": "fake-rt-abcdefgh", "email": "ada@example.com"})
        key = StoredCredential("api_key", "os_live_abcdefgh").to_dict()
        self.assertEqual(key, {"kind": "api_key", "api_key": "os_live_abcdefgh", "email": ""})

    def test_validation_and_repr(self):
        secret = _secret()
        cred = StoredCredential("oauth", secret, "ada@example.com")
        self.assertNotIn(secret, repr(cred) + str(cred))
        with self.assertRaises(ValueError):
            StoredCredential("password", secret)
        with self.assertRaises(AuthError):   # a malformed secret
            StoredCredential("oauth", "has spaces in it")
        self.assertEqual(StoredCredential("oauth", secret, "bad\nemail").email, "")
        for damaged in (None, [], {"kind": "oauth"}, {"kind": "x", "refresh_token": secret},
                        {"kind": "oauth", "refresh_token": "a b"}):
            self.assertIsNone(StoredCredential.from_dict(damaged))


class ModeTest(StoreCase):
    def test_modes(self):
        for value, expected in (("auto", "auto"), ("KEYCHAIN", "keychain"), (" file ", "file"), ("memory", "memory"),
                                ("bogus", "auto"), ("", "auto")):
            os.environ[keystore.ENV_STORE] = value
            self.assertEqual(keystore.mode(), expected, value)
        del os.environ[keystore.ENV_STORE]
        self.assertEqual(keystore.mode(), "auto")

    def test_backend_names(self):
        self.assertEqual(keystore.backend_name(), "memory")
        self.use("file")
        self.assertEqual(keystore.backend_name(), "file")
        self.use("auto")
        self.use_os_backend(keystore.MacKeychainBackend(run=FakeSecurity()))
        self.assertEqual(keystore.backend_name(), "keychain")

    def test_the_default_os_backend_matches_the_platform(self):
        backend = keystore._os_backend()
        if sys.platform == "darwin":
            self.assertIsInstance(backend, keystore.MacKeychainBackend)
        elif os.name == "nt":
            self.assertIsInstance(backend, keystore.WindowsCredentialBackend)
        elif sys.platform.startswith("linux"):
            self.assertIsInstance(backend, keystore.SecretToolBackend)


class MemoryTest(StoreCase):
    def test_round_trip_one_entry_per_api_url(self):
        first, second = StoredCredential("oauth", _secret()), StoredCredential("api_key", _secret("os_live_"))
        self.assertIsNone(keystore.load(API))
        self.assertEqual(keystore.save(API, first), "memory")
        keystore.save(OTHER, second)
        self.assertEqual((keystore.load(API), keystore.load(OTHER)), (first, second))
        keystore.delete(API)
        self.assertIsNone(keystore.load(API))
        self.assertEqual(keystore.load(OTHER), second)
        keystore.delete(API)   # deleting a missing entry is fine
        self.assertFalse(os.path.exists(self.cred_file), "the memory store never writes a file")


class FileTest(StoreCase):
    def setUp(self):
        super().setUp()
        self.use("file")

    def test_round_trip_and_format(self):
        cred = StoredCredential("oauth", _secret(), "ada@example.com")
        self.assertEqual(keystore.save(API, cred), "file")
        self.assertEqual(keystore.load(API), cred)
        data = self.file_json()
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["entries"], {keystore.account_name(API): cred.to_dict()})
        keystore.save(OTHER, StoredCredential("api_key", _secret("os_live_")))
        self.assertEqual(len(self.file_json()["entries"]), 2)
        keystore.delete(API)
        self.assertIsNone(keystore.load(API))
        self.assertIsNotNone(keystore.load(OTHER))
        self.assertNotIn(cred.secret, self.file_text())

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_the_file_is_owner_only(self):
        keystore.save(API, StoredCredential("oauth", _secret()))
        self.assertEqual(stat.S_IMODE(os.stat(self.cred_file).st_mode), 0o600)
        os.chmod(self.cred_file, 0o644)   # loosened by hand: the next write puts it back
        keystore.save(API, StoredCredential("oauth", _secret()))
        self.assertEqual(stat.S_IMODE(os.stat(self.cred_file).st_mode), 0o600)

    def test_writes_are_atomic(self):
        first = StoredCredential("oauth", _secret())
        keystore.save(API, first)
        before = self.file_text()
        with mock.patch.object(keystore._fsretry, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(KeystoreError) as ctx:
                keystore.save(API, StoredCredential("oauth", _secret()))
        self.assertIn("OSError", str(ctx.exception))
        self.assertEqual(self.file_text(), before, "a failed write leaves the old file as it was")
        self.assertEqual(os.listdir(os.path.dirname(self.cred_file)), [keystore.FILE_NAME], "no temp file left")
        self.assertEqual(keystore.load(API), first)

    def test_a_damaged_file_reads_as_empty_and_is_replaced(self):
        os.makedirs(os.path.dirname(self.cred_file), exist_ok=True)
        leaked = _secret()
        for damage in ("[1, 2]", "", "\xff\xfe not utf8",
                       '{"entries": {"x": {"kind": "oauth", "refresh_token": "' + leaked + '"'):   # truncated
            with open(self.cred_file, "w", encoding="latin-1") as f:
                f.write(damage)
            self.assertIsNone(keystore.load(API), repr(damage))
        keystore.delete(API)   # sign-out rewrites a damaged file: no stray secret survives
        self.assertNotIn(leaked, self.file_text())
        cred = StoredCredential("oauth", _secret())
        keystore.save(API, cred)
        self.assertEqual(keystore.load(API), cred)

    def test_the_default_path_is_next_to_the_settings_file(self):
        del os.environ[keystore.ENV_FILE]
        settings = os.path.join(self.dir, "fc", "oscar_settings.json")
        params = mock.Mock()
        params.settings_path.return_value = settings
        with mock.patch.object(config, "_params", lambda: params):
            self.assertEqual(keystore.credential_file_path(), os.path.join(self.dir, "fc", "oscar_credentials.json"))
            keystore.save(API, StoredCredential("oauth", _secret()))
            self.assertTrue(os.path.isfile(os.path.join(self.dir, "fc", "oscar_credentials.json")))
        with mock.patch.object(config, "settings_dir", lambda: None):
            self.assertEqual(keystore.credential_file_path(), "")
            with self.assertRaises(KeystoreUnavailable):
                keystore.save(API, StoredCredential("oauth", _secret()))
            self.assertIsNone(keystore.load(API))   # nothing to read is not an error

    def test_no_notice_when_the_file_was_chosen(self):
        keystore.save(API, StoredCredential("oauth", _secret()))
        self.assertFalse(keystore.file_notice_pending())


class AutoModeTest(StoreCase):
    def setUp(self):
        super().setUp()
        self.use("auto")
        self.security = FakeSecurity()
        self.use_os_backend(keystore.MacKeychainBackend(run=self.security))

    def test_the_os_store_is_used_first_and_the_file_is_not_touched(self):
        cred = StoredCredential("oauth", _secret(), "ada@example.com")
        self.assertEqual(keystore.save(API, cred), "keychain")
        self.assertEqual(keystore.load(API), cred)
        self.assertFalse(os.path.exists(self.cred_file))
        self.assertFalse(keystore.file_notice_pending())
        keystore.delete(API)
        self.assertIsNone(keystore.load(API))
        self.assertEqual(self.security.items, {})

    def test_a_failing_os_store_falls_back_to_the_file_with_a_one_time_notice(self):
        self.security.write_status = 51
        cred = StoredCredential("oauth", _secret())
        self.assertEqual(keystore.save(API, cred), "file")
        self.assertEqual(keystore.load(API), cred, "found in the file when the keychain has nothing")
        self.assertTrue(keystore.file_notice_pending())
        self.assertIn(self.cred_file, keystore.file_notice_text())
        keystore.acknowledge_file_notice()
        self.assertFalse(keystore.file_notice_pending())
        self.assertTrue(self.file_json()["notice_shown"], "remembered in the file: once, across restarts")
        keystore.save(OTHER, StoredCredential("oauth", _secret()))
        self.assertFalse(keystore.file_notice_pending())

    def test_an_unavailable_os_store_falls_back_to_the_file(self):
        self.security.raise_error = FileNotFoundError("no security tool")
        cred = StoredCredential("api_key", _secret("os_live_"))
        self.assertEqual(keystore.save(API, cred), "file")
        self.assertEqual(keystore.load(API), cred)
        keystore.delete(API)
        self.assertIsNone(keystore.load(API))

    def test_a_fallback_save_is_never_shadowed_by_an_older_os_entry(self):
        # A token rotation while the keychain refuses writes: the new refresh token lands in the file. The keychain's
        # old one is already USED at the server; reading it back would trigger reuse detection and end the sign-in.
        old = StoredCredential("oauth", _secret())
        self.assertEqual(keystore.save(API, old), "keychain")
        self.security.write_status = 51
        new = StoredCredential("oauth", _secret())
        self.assertEqual(keystore.save(API, new), "file")
        self.assertEqual(self.security.items, {}, "the older keychain entry was removed")
        self.assertEqual(keystore.load(API), new)
        self.security.items[(keystore.account_name(API), keystore.SERVICE)] = keystore.encode_blob(old)
        self.assertEqual(keystore.load(API), new, "even when that removal failed, the file's newer entry wins")
        self.security.write_status = 0
        newest = StoredCredential("oauth", _secret())
        self.assertEqual(keystore.save(API, newest), "keychain")
        self.assertEqual(self.file_json()["entries"], {}, "the keychain took over again")
        self.assertEqual(keystore.load(API), newest)

    def test_once_the_os_store_works_the_file_copy_is_removed(self):
        self.security.write_status = 51
        old = StoredCredential("oauth", _secret())
        keystore.save(API, old)
        self.security.write_status = 0
        new = StoredCredential("oauth", _secret())
        self.assertEqual(keystore.save(API, new), "keychain")
        self.assertEqual(keystore.load(API), new)
        self.assertNotIn(old.secret, self.file_text())
        self.assertEqual(self.file_json()["entries"], {})

    def test_no_os_store_at_all(self):
        patcher = mock.patch.object(keystore, "_os_backend", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        cred = StoredCredential("oauth", _secret())
        self.assertEqual(keystore.backend_name(), "file")
        self.assertEqual(keystore.save(API, cred), "file")
        self.assertEqual(keystore.load(API), cred)


class KeychainModeTest(StoreCase):
    def setUp(self):
        super().setUp()
        self.use("keychain")

    def test_without_an_os_store_it_is_an_error(self):
        self.use_os_backend(keystore.MacKeychainBackend(run=FakeSecurity()), available=False)
        self.assertEqual(keystore.backend_name(), "none")
        for call in (lambda: keystore.load(API), lambda: keystore.save(API, StoredCredential("oauth", _secret())),
                     lambda: keystore.delete(API)):
            with self.assertRaises(KeystoreUnavailable):
                call()
        self.assertFalse(os.path.exists(self.cred_file), "never the file in keychain mode")

    def test_failures_are_raised_not_hidden(self):
        security = FakeSecurity()
        security.write_status = 51
        self.use_os_backend(keystore.MacKeychainBackend(run=security))
        with self.assertRaises(KeystoreError):
            keystore.save(API, StoredCredential("oauth", _secret()))
        self.assertFalse(os.path.exists(self.cred_file))


class MacKeychainBackendTest(unittest.TestCase):
    def setUp(self):
        self.security = FakeSecurity()
        self.backend = keystore.MacKeychainBackend(run=self.security)
        self.account = keystore.account_name(API)

    def test_round_trip_with_the_secret_on_stdin_only(self):
        cred = StoredCredential("oauth", _secret(), "ada@example.com")
        self.backend.write(self.account, cred)
        self.assertEqual(self.backend.read(self.account), cred)
        replacement = StoredCredential("api_key", _secret("os_live_"))
        self.backend.write(self.account, replacement)   # -U: an update, not a duplicate
        self.assertEqual(self.backend.read(self.account), replacement)
        self.backend.delete(self.account)
        self.assertIsNone(self.backend.read(self.account))
        self.backend.delete(self.account)   # already gone: fine
        for argv, stdin in self.security.calls:
            self.assertEqual(argv, ["/usr/bin/security", "-i"])
        joined_args = " ".join(" ".join(argv) for argv, _ in self.security.calls)
        for secret in (cred.secret, replacement.secret, keystore.encode_blob(cred)):
            self.assertNotIn(secret, joined_args)
        add = next(stdin for _, stdin in self.security.calls if stdin.startswith("add-generic-password"))
        self.assertIn(f'-s "{keystore.SERVICE}"', add)
        self.assertIn(f'-a "{self.account}"', add)
        self.assertIn(" -U ", add)
        self.assertEqual(self.security.items, {})

    def test_failures(self):
        self.security.write_status = 51
        cred = StoredCredential("oauth", _secret())
        with self.assertRaises(KeystoreError) as ctx:
            self.backend.write(self.account, cred)
        self.assertIn("51", str(ctx.exception))
        self.assertNotIn(cred.secret, str(ctx.exception))
        for error in (FileNotFoundError("security"), subprocess.TimeoutExpired(["security"], 30)):
            self.security.raise_error = error
            with self.assertRaises(KeystoreUnavailable):
                self.backend.read(self.account)

    def test_an_item_oscar_cannot_read_is_treated_as_absent(self):
        self.security.items[(self.account, keystore.SERVICE)] = "not-an-oscar-blob"
        self.assertIsNone(self.backend.read(self.account))

    def test_an_unexpected_read_failure_is_an_error(self):
        broken = keystore.MacKeychainBackend(run=lambda argv, **kw: subprocess.CompletedProcess(argv, 36, "", "locked\n"))
        with self.assertRaises(KeystoreError):
            broken.read(self.account)

    def test_unsafe_values_never_reach_the_command_line(self):
        with self.assertRaises(KeystoreError):
            self.backend.read('x" -w "y')
        with self.assertRaises(KeystoreError):
            keystore._quoted("line\nbreak")
        self.assertEqual(self.security.calls, [])

    def test_an_entry_longer_than_a_security_command_line_is_refused_not_split(self):
        # ``security -i`` reads 4096-byte lines and runs the rest of a longer one as another command (checked
        # by hand on macOS 26): the item would hold a truncated blob. Refused before anything runs.
        huge = StoredCredential("oauth", _secret() + "x" * 3500, "ada@example.com")
        with self.assertRaises(KeystoreError) as ctx:
            self.backend.write(self.account, huge)
        self.assertNotIsInstance(ctx.exception, KeystoreUnavailable)
        self.assertNotIn(huge.secret, str(ctx.exception))
        self.assertEqual(self.security.calls, [])
        normal = StoredCredential("oauth", _secret(), "a" * 60 + "@example.com")
        self.backend.write(self.account, normal)   # a real-sized entry is far below the limit
        self.assertLess(len(self.security.calls[0][1]), keystore.MacKeychainBackend.MAX_COMMAND_CHARS // 4)
        directory = isolation.fresh_dir("keystore_huge_")
        with mock.patch.dict(os.environ, {keystore.ENV_STORE: "auto",
                                          keystore.ENV_FILE: os.path.join(directory, keystore.FILE_NAME)}), \
                mock.patch.object(keystore, "_os_backend", lambda: _Available(self.backend)):
            self.assertEqual(keystore.save(API, huge), "file", "auto mode keeps it in the file instead")
            self.assertEqual(keystore.load(API), huge)

    def test_available_only_on_macos(self):
        self.assertEqual(self.backend.available(), sys.platform == "darwin" and os.access("/usr/bin/security", os.X_OK))


class SecretToolBackendTest(unittest.TestCase):
    def setUp(self):
        self.tool = FakeSecretTool()
        self.backend = keystore.SecretToolBackend(run=self.tool, which=lambda name: "/usr/bin/secret-tool")
        self.account = keystore.account_name(API)

    def test_round_trip_with_the_secret_on_stdin_only(self):
        cred = StoredCredential("oauth", _secret(), "ada@example.com")
        self.backend.write(self.account, cred)
        self.assertEqual(self.backend.read(self.account), cred)
        self.backend.delete(self.account)
        self.assertIsNone(self.backend.read(self.account))
        store_argv, store_stdin = self.tool.calls[0]
        self.assertEqual(store_argv, ["/usr/bin/secret-tool", "store", "--label", keystore.SERVICE, "service",
                                      keystore.SERVICE, "account", self.account])
        self.assertEqual(store_stdin, keystore.encode_blob(cred))
        for argv, _ in self.tool.calls:
            self.assertNotIn(cred.secret, " ".join(argv))

    def test_no_keyring_running(self):
        self.tool.dbus_down = True
        with self.assertRaises(KeystoreUnavailable):
            self.backend.read(self.account)
        with self.assertRaises(KeystoreUnavailable):
            self.backend.write(self.account, StoredCredential("oauth", _secret()))

    def test_not_installed(self):
        backend = keystore.SecretToolBackend(run=self.tool, which=lambda name: None)
        self.assertFalse(backend.available())
        with self.assertRaises(KeystoreUnavailable):
            backend.read(self.account)

    def test_auto_mode_falls_back_to_the_file_without_a_keyring(self):
        directory = isolation.fresh_dir("keystore_linux_")
        path = os.path.join(directory, keystore.FILE_NAME)
        self.tool.dbus_down = True
        with mock.patch.dict(os.environ, {keystore.ENV_STORE: "auto", keystore.ENV_FILE: path}), \
                mock.patch.object(keystore, "_os_backend", lambda: _Available(self.backend)):
            cred = StoredCredential("oauth", _secret())
            self.assertEqual(keystore.save(API, cred), "file")
            self.assertEqual(keystore.load(API), cred)
            self.assertTrue(keystore.file_notice_pending())


class WindowsCredentialBackendTest(unittest.TestCase):
    def setUp(self):
        self.dll = FakeAdvapi32()
        self.backend = keystore.WindowsCredentialBackend(advapi32=self.dll, last_error=self.dll.last_error)
        self.account = keystore.account_name(API)

    def test_round_trip_through_real_credential_structures(self):
        cred = StoredCredential("oauth", _secret(), "ada@example.com")
        self.assertTrue(self.backend.available())
        self.backend.write(self.account, cred)
        target = f"{keystore.SERVICE}/{self.account}"
        entry = self.dll.store[target]
        self.assertEqual(entry["user"], self.account)
        self.assertEqual(entry["type"], keystore.CRED_TYPE_GENERIC)
        self.assertEqual(entry["persist"], keystore.CRED_PERSIST_LOCAL_MACHINE)
        self.assertEqual(entry["blob"], keystore.encode_blob(cred).encode("ascii"))
        self.assertEqual(self.backend.read(self.account), cred)
        self.assertEqual(self.dll.freed, 1, "CredFree releases what CredReadW returned")
        self.backend.delete(self.account)
        self.assertIsNone(self.backend.read(self.account))
        self.backend.delete(self.account)   # already gone: fine

    def test_failures_carry_the_error_number_only(self):
        cred = StoredCredential("oauth", _secret())
        self.dll.fail_with = 1312
        with self.assertRaises(KeystoreError) as ctx:
            self.backend.write(self.account, cred)
        self.assertIn("1312", str(ctx.exception))
        self.assertNotIn(cred.secret, str(ctx.exception))

    def test_an_oversized_entry_is_refused(self):
        with self.assertRaises(KeystoreError):
            self.backend.write(self.account, StoredCredential("api_key", "os_live_" + "k" * 3000))
        self.assertEqual(self.dll.store, {})

    def test_without_windows_it_is_unavailable(self):
        backend = keystore.WindowsCredentialBackend()
        self.assertEqual(backend.available(), os.name == "nt")
        if os.name != "nt":
            with self.assertRaises(KeystoreUnavailable):
                backend.read(self.account)


@unittest.skipUnless(sys.platform == "darwin" and os.environ.get("OSCAR_TEST_REAL_KEYCHAIN") == "1",
                     "writes to the real login keychain: set OSCAR_TEST_REAL_KEYCHAIN=1 on macOS to run it")
class RealKeychainTest(unittest.TestCase):
    def test_round_trip_in_the_login_keychain(self):
        backend = keystore.MacKeychainBackend()
        self.assertTrue(backend.available())
        account = keystore.account_name(f"https://oscar-keychain-test-{uuid.uuid4().hex}.invalid")
        self.addCleanup(backend.delete, account)
        cred = StoredCredential("oauth", _secret("test-refresh-"), "test@example.com")
        backend.write(account, cred)
        self.assertEqual(backend.read(account), cred)
        replacement = StoredCredential("api_key", _secret("os_live_"))
        backend.write(account, replacement)
        self.assertEqual(backend.read(account), replacement)
        backend.delete(account)
        self.assertIsNone(backend.read(account))
        backend.delete(account)


if __name__ == "__main__":
    unittest.main()
