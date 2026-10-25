"""Checkpoint bookkeeping of :mod:`oscar.project` (private, PURE: no FreeCAD, no Qt).

``checkpoints/`` holds ``NNN.FCStd`` (+ sidecar ``NNN.json``) for every successful geometry step and
``T<last 4 of the turn id>.FCStd`` for the document as it was just before a turn's first geometry step
(``T<suffix>.<DocName>.FCStd`` for a second document of the same turn). This module lists them and applies
the keep rules; the FreeCAD work (``saveCopy`` / restore) stays in ``Project`` and ``project_gui``.
"""

import os
import re

from .project_util import _read_json


def list_entries(directory: str) -> list[dict]:
    """Every checkpoint of ``directory``: ``{"id", "stem", "kind" ("geometry" | "turn" | "other"), "index", "path",
    "bytes", "side", "ts"}``. Lock-free; ``_rescue`` copies and ``.tmp-`` files are excluded; an unreadable
    sidecar gives ``{}``."""
    out: list[dict] = []
    try:
        names = os.listdir(directory)
    except OSError:
        return out
    for name in names:
        if not name.endswith(".FCStd") or ".tmp-" in name or name.startswith("_rescue"):
            continue
        stem = name[: -len(".FCStd")]
        ckpt_id = stem.split(".", 1)[0]
        path = os.path.join(directory, name)
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        side = _read_json(os.path.join(directory, stem + ".json"), {}) or {}
        if re.fullmatch(r"\d+", ckpt_id):
            kind, index = "geometry", int(ckpt_id)
        elif re.fullmatch(r"T[0-9a-zA-Z]{4}", ckpt_id) and "turn" in side:
            kind, index = "turn", int(side.get("after_step", 0) or 0)
        else:
            kind, index = "other", int(side.get("step", 0) or 0)
        out.append({"id": ckpt_id, "stem": stem, "kind": kind, "index": index, "path": path, "bytes": size,
                    "side": side, "ts": str(side.get("ts", ""))})
    return out


def prune(entries: list[dict], *, keep_turns: int, keep_newest: int, keep_every: int, max_geometry: int,
          budget_bytes: int) -> None:
    """Keep ``checkpoints/`` bounded (file deletes only; the caller holds the Project lock).

    Turn checkpoints keep the newest ``keep_turns``. Geometry checkpoints are pruned oldest first while the
    total exceeds ``budget_bytes`` or more than ``max_geometry`` exist, but never every ``keep_every``-th,
    the newest ``keep_newest``, or the one a surviving turn checkpoint was taken after; when the byte budget
    still cannot be met the protected ones go too (oldest first, the newest always stays): the folder
    size, not a silent skip, is what is bounded."""
    if not entries:
        return

    def drop(entry: dict) -> None:
        for path in (entry["path"], os.path.join(os.path.dirname(entry["path"]), entry["stem"] + ".json")):
            try:
                os.remove(path)
            except OSError:
                pass
        entry["gone"] = True

    turns = sorted((e for e in entries if e["kind"] == "turn"), key=lambda e: (e["ts"], e["stem"]))
    for entry in turns[: max(0, len(turns) - keep_turns)]:
        drop(entry)
    geometry = sorted((e for e in entries if e["kind"] == "geometry"), key=lambda e: e["index"])
    keep_after = {e["index"] for e in turns if not e.get("gone")}
    protected = {e["index"] for e in geometry[-keep_newest:]}
    protected |= {e["index"] for e in geometry if e["index"] % keep_every == 0}
    protected |= keep_after

    def total() -> int:
        return sum(e["bytes"] for e in entries if not e.get("gone"))

    def alive() -> list[dict]:
        return [e for e in geometry if not e.get("gone")]

    while total() > budget_bytes or len(alive()) > max_geometry:
        victim = next((e for e in alive() if e["index"] not in protected), None)
        if victim is None:
            break
        drop(victim)
    if total() > budget_bytes:
        for victim in [e for e in alive()][:-1]:
            if total() <= budget_bytes:
                break
            drop(victim)
        for victim in [e for e in turns if not e.get("gone")][:-1]:
            if total() <= budget_bytes:
                break
            drop(victim)
