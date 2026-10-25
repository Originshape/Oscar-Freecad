"""tests/support/isolation.py - keep every test away from the user's real data.

Call ``isolate_process()`` at the top of each test module BEFORE importing ``oscar``.
It is idempotent and process-wide: one temp sandbox per test process, removed at exit.
It sets OSCAR_* variables only (settings, projects, logs, cache, downloads, and the account and script
switches: no real keychain, no reachable OriginShape server, scripts allowed without a dialog).
"""

import atexit
import os
import shutil
import tempfile

_ROOT: str | None = None
PINNED: set[str] = set()   # OSCAR_* variables the caller (scripts/test.sh, CI) had already set: honoured, not overridden


def sandbox_root() -> str:
    """The sandbox directory (created on first use)."""
    return isolate_process()


def isolate_process() -> str:
    global _ROOT
    if _ROOT:
        return _ROOT
    _ROOT = tempfile.mkdtemp(prefix="oscar_test_")
    atexit.register(shutil.rmtree, _ROOT, ignore_errors=True)
    env = os.environ
    PINNED.update(k for k in ("OSCAR_AI_SETTINGS", "OSCAR_PROJECTS_DIR", "OSCAR_LOG_DIR",
                              "OSCAR_CACHE_DIR", "OSCAR_DOWNLOAD_DIR", "OSCAR_ALLOW_CODE", "OSCAR_UPDATE_CHECK",
                              "OSCAR_CREDENTIAL_STORE", "OSCAR_API_URL") if k in env)
    env.setdefault("OSCAR_AI_SETTINGS", os.path.join(_ROOT, "settings.json"))
    env.setdefault("OSCAR_PROJECTS_DIR", os.path.join(_ROOT, "projects"))
    env.setdefault("OSCAR_LOG_DIR", os.path.join(_ROOT, "logs"))
    env.setdefault("OSCAR_CACHE_DIR", os.path.join(_ROOT, "cache"))
    env.setdefault("OSCAR_DOWNLOAD_DIR", os.path.join(_ROOT, "downloads"))
    env.setdefault("OSCAR_ALLOW_CODE", "1")              # scripts run without the approval dialog
    env.setdefault("OSCAR_UPDATE_CHECK", "0")            # no daily update check (the tests of it turn it on)
    env.setdefault("OSCAR_CREDENTIAL_STORE", "memory")   # never the real keychain or credential file
    env.setdefault("OSCAR_API_URL", "http://127.0.0.1:9")  # unroutable: no test reaches a real OriginShape server
    # HOME is deliberately NOT overridden: tests that spawn the bundled python need the real user-site
    # packages found through HOME.
    return _ROOT


def fresh_dir(prefix: str = "t_") -> str:
    """A new empty directory inside the sandbox."""
    return tempfile.mkdtemp(prefix=prefix, dir=isolate_process())
