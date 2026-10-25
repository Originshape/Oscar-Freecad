"""Where a path in chat text points, said without the full folder: "Oscar project" instead of
``/Users/<name>/Documents/Oscar/Projects/<chat>-<id>``.

A reply names the files it made by their absolute paths. That is noise in the chat and it shows the user's account
name and folder layout, so:

* the panel shows a path inside a project as the link ``Oscar project`` (plus the part below the project folder,
  ``Oscar project/exports``) that opens it (``panel_render.md_to_html``);
* text that leaves this computer (the OriginShape workspace) says the same without a link (``redact``), and any
  other path under the home folder starts with ``~``.

Pure: no Qt and no FreeCAD (the roots come from ``oscar.paths``, which only asks FreeCAD for its data folder).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

LABEL = "Oscar project"
_STOP = "\\s`'\"<>|*()\\[\\]"   # characters that end a path in prose


def project_roots() -> list[str]:
    """The folders that hold Oscar's projects: the configured root and the app-data fallback (no disk work)."""
    from . import paths

    roots: list[str] = []
    for root in (paths.projects_dir(probe=False), paths.fallback_projects_dir()):
        root = os.path.normpath(os.path.expanduser(root)) if root else ""
        if root and root not in roots:
            roots.append(root)
    return roots


def _root_pattern(root: str) -> str:
    """``root`` as a regex that also matches it written with the other slash (a Windows path in model text)."""
    parts = [part for part in re.split(r"[\\/]+", root) if part]
    lead = r"[\\/]" if root[:1] in "\\/" else ""
    return lead + r"[\\/]".join(re.escape(part) for part in parts)


def project_path_re(roots: Iterable[str] | None = None) -> re.Pattern[str] | None:
    """A regex for paths inside a project: group 1 is the project folder, group 2 the rest (``/exports/a.stl``)."""
    roots = [root for root in (project_roots() if roots is None else roots) if root]
    if not roots:
        return None
    alternatives = "|".join(_root_pattern(root) for root in sorted(roots, key=len, reverse=True))
    return re.compile(rf"((?:{alternatives})[\\/][^\\/{_STOP}]+)((?:[\\/][^\\/{_STOP}]+)*[\\/]?)")


def label(rest: str) -> str:
    """``Oscar project`` plus the part below the project folder (``Oscar project/exports``)."""
    rest = rest.replace("\\", "/").strip("/")
    return f"{LABEL}/{rest}" if rest else LABEL


def install() -> None:
    """Have the panel's markdown show project paths as ``Oscar project`` links (``panel_render`` stays pure, so
    the pattern is handed to it)."""
    from . import panel_render

    panel_render.set_project_paths(project_path_re())


def _redacted(match: re.Match[str]) -> str:
    """One project path as ``Oscar project[/rest]``; sentence punctuation after it stays outside."""
    whole = match.group(1) + match.group(2)
    kept = whole.rstrip(".,;:!?")
    return label(kept[len(match.group(1)):]) + whole[len(kept):]


def redact(text: str, *, roots: Iterable[str] | None = None, home: str | None = None) -> str:
    """``text`` for a reader on another computer: project paths become ``Oscar project[/...]`` and other paths
    under the home folder start with ``~``."""
    if not text:
        return text
    pattern = project_path_re(roots)
    if pattern is not None:
        text = pattern.sub(_redacted, text)
    home = os.path.normpath(os.path.expanduser("~") if home is None else home)
    if len(home) > 3:   # never "/" or "C:\"
        text = re.sub(_root_pattern(home) + rf"(?=[\\/{_STOP}]|$)", "~", text)
    return text
