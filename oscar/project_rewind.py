"""Rewind / revert lookups of :mod:`oscar.project` (private mixin, read-only, no FreeCAD, no Qt).

``Project`` inherits :class:`RewindMixin`. The panel asks it WHAT a rewind to a step or a checkpoint revert of a
turn would change (``rewind_info``, ``turn_checkpoints``) before it confirms with the user, then performs the
change through ``Project.rewind_to`` / ``Project.restore`` and journals it. Nothing here touches the live
document or the journal.
"""


class RewindMixin:
    """Which checkpoint a rewind / turn revert would restore, and what it would discard."""

    def rewind_info(self, step_index: int, doc_name: str | None = None) -> dict:
        """What ``rewind_to(step_index)`` would do: ``{"ok", "error", "step", "checkpoint", "doc", "objects",
        "lost_steps"}``.

        ``ok`` is False (with the reason in ``error``) when the step does not exist or no geometry checkpoint
        exists at or before it (pruned, too big, never taken). ``checkpoint`` is the id the rewind restores (the
        newest geometry checkpoint at or before the step), ``doc`` its document, ``objects`` the object names the
        document holds after the restore (the sidecar's ``all_objects``; empty when the sidecar has none) and
        ``lost_steps`` the not yet superseded steps after ``step_index`` as ``{"index", "title", "tool"}``."""
        try:
            wanted = int(step_index)
        except (TypeError, ValueError):
            return {"ok": False, "error": f"step index must be an integer, got {step_index!r}."}
        last = self.last_step_index()
        if wanted < 0 or wanted > last:
            return {"ok": False, "error": f"there is no step {wanted}; steps run 1..{last}."}
        candidates = [e for e in self._checkpoint_entries() if e["kind"] == "geometry" and e["index"] <= wanted
                      and (not doc_name or str(e["side"].get("doc", "")).lower() == doc_name.lower())]
        if not candidates:
            return {"ok": False, "error": f"no checkpoint at or before step {wanted} (it may have been pruned)."}
        entry = max(candidates, key=lambda e: e["index"])
        steps, _ = self._journal()
        lost = [{"index": s.index, "title": s.title, "tool": s.tool}
                for s in steps if s.index > wanted and not s.superseded]
        side = entry["side"]
        return {"ok": True, "error": "", "step": wanted, "checkpoint": entry["id"], "doc": str(side.get("doc", "")),
                "objects": list(side.get("all_objects") or []), "lost_steps": lost}

    def turn_checkpoints(self, first_step: int) -> list[dict]:
        """The turn-start checkpoints (``T<suffix>``) of the turn that recorded step ``first_step``:
        ``[{"id", "doc", "objects"}]``, one per document the turn changed (empty when the step is unknown or the
        turn took none, e.g. it ran no geometry tool).

        Matched by the turn id stored in the step record and in the checkpoint sidecar, so a LATER turn's
        checkpoint is never taken for an earlier turn. ``objects`` is the sidecar's ``all_objects`` (``None`` when
        it has none: the restore then cannot be verified by name set)."""
        try:
            first = int(first_step)
        except (TypeError, ValueError):
            return []
        steps, _ = self._journal()
        record = next((s for s in steps if s.index == first), None)
        if record is None or not record.turn_id:
            return []
        out = []
        for entry in self._checkpoint_entries():
            if entry["kind"] != "turn" or str(entry["side"].get("turn", "")) != record.turn_id:
                continue
            objects = entry["side"].get("all_objects")
            out.append({"id": entry["id"], "doc": str(entry["side"].get("doc", "")),
                        "objects": list(objects) if isinstance(objects, list) else None})
        return out
