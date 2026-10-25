"""``os.replace`` that rides out Windows sharing violations (pure python: os + time only).

POSIX replaces a file atomically even while another thread or process has it open for reading. Windows does not: replacing
a file that is open elsewhere (the panel reading ``steps.jsonl``, a second reader of ``settings.json``) raises
``PermissionError`` ([WinError 5] / 32). The reader closes within milliseconds, so a short bounded retry makes the atomic
write behave like it does everywhere else; after the last attempt the error is raised unchanged.
"""

import os
import time

ATTEMPTS = 40          # worst case about 2 s of waiting (delays grow 10 ms, 20 ms ... capped at 100 ms)
FIRST_DELAY_S = 0.01
MAX_DELAY_S = 0.1


def _windows() -> bool:
    return os.name == "nt"


def replace(src, dst) -> None:
    """``os.replace(src, dst)``; on Windows a ``PermissionError`` is retried ``ATTEMPTS`` times with a growing delay."""
    if not _windows():
        os.replace(src, dst)
        return
    for attempt in range(ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(min(FIRST_DELAY_S * (attempt + 1), MAX_DELAY_S))
