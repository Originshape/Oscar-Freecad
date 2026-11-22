"""Locate the workbench directory reliably.

The application's mod loader exec()'s InitGui.py with locals separate from
globals, so path resolution lives here instead — an imported module has
normal semantics and ``__file__`` works.
"""

import os


def mod_dir() -> str:
    """Directory of the Oscar workbench (…/Mod/Oscar)."""
    try:
        # This module lives at …/Oscar/oscar/paths.py — go up one.
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    import FreeCAD

    for base in (
        os.path.join(FreeCAD.getUserAppDataDir(), "Mod"),
        os.path.join(FreeCAD.getResourceDir(), "Mod"),
    ):
        candidate = os.path.join(base, "Oscar")
        if os.path.isfile(os.path.join(candidate, "InitGui.py")):
            return candidate
    return os.getcwd()


def icon(name: str) -> str:
    """Absolute path of the toolbar / panel icon ``name`` in the workbench's ``Resources/icons`` folder."""
    return os.path.join(mod_dir(), "Resources", "icons", name)



def cache_dir(*parts: str) -> str:
    """OSCAR_CACHE_DIR (read at CALL time) or ~/.cache/oscar, joined with ``parts``; the directory is created.
    Every module that caches (cad/headless, cad/engine, cad/scripting, refparts) uses this one function."""
    base = os.environ.get("OSCAR_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "oscar")
    path = os.path.join(base, *parts)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Projects (the per-chat dossier, oscar.project)

PROJECTS_FALLBACK = False
"""True when the last ``projects_dir()`` probe found the default root unwritable (macOS TCC denial of
~/Documents, a read-only or offline OneDrive/iCloud folder) and used the app-data fallback instead.
The first Project write turns it into a visible notice and ``manifest["storage_note"]``. The
OSCAR_PROJECTS_DIR override never sets it (it is exempt from the fallback)."""


def _default_projects_dir() -> str:
    return os.path.join(os.path.expanduser("~"), "Documents", "Oscar", "Projects")


def fallback_projects_dir() -> str:
    """``<FreeCAD user app data>/Oscar/Projects`` without creating it."""
    try:
        import FreeCAD

        base = os.path.join(FreeCAD.getUserAppDataDir(), "Oscar", "Projects")
    except Exception:
        base = os.path.join(os.path.expanduser("~"), ".local", "share", "Oscar", "Projects")
    return base


def _writable_dir(path: str) -> bool:
    """Create ``path`` if needed, then create and delete a temp file in it. Never raises."""
    import tempfile

    try:
        os.makedirs(path, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".oscar-probe-", dir=path)
        os.close(fd)
        os.remove(probe)
        return True
    except OSError:
        return False


def projects_dir(*, probe: bool = True) -> str:
    """The projects root: OSCAR_PROJECTS_DIR if set (read at CALL time, not import time; ``~`` expanded), else
    ``~/Documents/Oscar/Projects``. Never inside the repo.

    ``probe=True`` (the default) creates the directory and probes writability (create and delete a temp
    file). When the DEFAULT root is not writable it falls back to ``fallback_projects_dir()`` and sets
    ``PROJECTS_FALLBACK = True``; a successful probe of the default resets the flag. The environment
    override is exempt from the fallback: it is created and returned as given, and an unwritable override
    surfaces as the OSError of the first write (a test pointing at an unwritable directory must fail).

    ``probe=False`` does no disk work at all (the panel resolves the folder name on the GUI thread at send
    time; Project.ensure_dir() later falls back through ``fallback_projects_dir()`` on an OSError).
    """
    global PROJECTS_FALLBACK
    override = os.environ.get("OSCAR_PROJECTS_DIR")
    if override:
        override = os.path.expanduser(override)   # a quoted ``~/Projects`` in a shell profile is not expanded by the shell
        if probe:
            os.makedirs(override, exist_ok=True)
        return override
    base = _default_projects_dir()
    if not probe:
        return base
    if _writable_dir(base):
        PROJECTS_FALLBACK = False
        return base
    fallback = fallback_projects_dir()
    os.makedirs(fallback, exist_ok=True)
    PROJECTS_FALLBACK = True
    return fallback


def slugify(text: str, max_len: int = 32) -> str:
    """Lowercase ``[a-z0-9-]`` only, runs collapsed to one ``-``, stripped, at most ``max_len`` characters;
    accented Latin letters are folded to ASCII (``Gehäuse`` -> ``gehause``); an empty result becomes ``project``."""
    import re
    import unicodedata

    folded = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")
    slug = slug[: max(1, int(max_len))].strip("-")
    return slug or "project"


def project_dirname(chat_id: str, title: str) -> str:
    """``<slug(title)>-<tail>``: ``tail`` is the last ``-`` separated part of ``chat_id`` when it is 6 lowercase
    hex characters (the ``chats.new_session`` ids), else the first 6 hex characters of ``sha1(chat_id)``.
    Example: ``raspberry-pi-4-housing-3f9a1c``. Pure; the folder name is fixed when the Project is created."""
    import hashlib
    import re

    chat_id = str(chat_id or "")
    tail = chat_id.rsplit("-", 1)[-1]
    if not re.fullmatch(r"[0-9a-f]{6}", tail):
        tail = hashlib.sha1(chat_id.encode("utf-8")).hexdigest()[:6]
    return f"{slugify(title)}-{tail}"
