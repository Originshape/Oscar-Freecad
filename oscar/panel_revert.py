"""Turn revert of the Oscar panel (undo-stack based, checkpoint fallback), as a mixin of OscarPanel.

Split out of panel.py: which card may be reverted, the guards (never mid-flight, only the latest
non-reverted turn, refuse when the turn-start snapshot is missing or failed, mark the card reverted only
after a verified result) and the revert itself; a verified revert is also journaled in the chat's Project
(its steps become superseded) and in the carried plan (``Ledger.on_revert``, so the plan strip stops showing steps
done for geometry that is gone). When the undo stack cannot hold the turn, or must not undo it (``revert_turn``
reports ``incomplete``: a document the turn touched has a full stack - the GUI keeps 20 undo steps - or the turn
built PartDesign features FreeCAD's undo can crash on, ``undo_unsafe``) the turn-start checkpoint of the Project is
restored instead. That fallback refuses when a document the turn touched has no checkpoint, asks before it throws away
changes made after the turn, closes the documents the turn created only through the engine's ownership rules
(``close_turn_documents``: the user's documents stay open) and verifies the result by object-name-set equality before
the card is marked reverted. A PARTIAL revert (``revert_turn`` undid some steps but refused to touch later work or
could not undo them all: ``partial``) is reported as such: the card is not marked reverted and stays revertible, and
the model is told that part of the turn may still be there. ``_revert_turn`` returns True only when the card ended up
reverted. The mixin relies on the host panel for worker, blocks, _append, _render, _pending_notes, _project_of and
_log_panel_error.
"""

from typing import NamedTuple

from PySide import QtCore, QtWidgets

from .panel_blocks import Block


class _CheckpointRevert(NamedTuple):
    """Outcome of the checkpoint fallback."""

    ok: bool                 # every checkpoint was restored and verified
    message: str             # what happened (the card note), or why nothing was restored
    ids: str = ""            # the restored checkpoint ids
    cancelled: bool = False  # the user chose to keep later changes: nothing was restored, a retry may still work
    kept: tuple = ()         # "Name (reason)" of the documents the turn created that were left open


def _engine():
    from .cad import get_engine

    return get_engine()


class RevertMixin:
    """Revert methods of the panel (see the module docstring)."""

    def _latest_revertible(self) -> Block | None:
        """The only card that may be reverted: the latest finished, changed, not yet reverted turn
        (reverting an older one would also pop every later turn off the undo stack)."""
        return next((b for b in reversed(self.blocks)
                     if b.kind == Block.TOOL and b.revertible and not b.reverted), None)

    def _update_revert_flags(self) -> None:
        """Offer "revert changes" on the latest revertible card only."""
        latest = self._latest_revertible()
        for block in self.blocks:
            if block.kind == Block.TOOL and block.revertible:
                block.revert_enabled = block is latest

    @staticmethod
    def _snapshot_usable(snapshot) -> bool:
        """A turn-start snapshot that can drive a revert: a dict with a ``docs`` map that did not
        report failure (``begin_turn_snapshot`` returns ``success: False`` when it could not read the
        documents; an empty ``docs`` of a successful snapshot is legitimate: no document was open)."""
        return (isinstance(snapshot, dict) and snapshot.get("success", True) is not False
                and isinstance(snapshot.get("docs"), dict))

    @staticmethod
    def _kept_doc_names(result: dict) -> list[str]:
        """Names of the documents ``revert_turn`` left open on purpose (``kept_docs`` entries are
        ``{"doc": name, "reason": ...}`` dicts; plain names are tolerated)."""
        names = []
        for entry in (result or {}).get("kept_docs") or []:
            name = entry.get("doc") if isinstance(entry, dict) else entry
            if name and str(name) not in names:
                names.append(str(name))
        return names

    @staticmethod
    def _revert_residue(snapshot: dict, kept=()) -> list[str]:
        """After a revert: the documents that still hold more undo steps than at turn start, or that
        the turn created and are still open. Empty list = the revert is verified; ``None``-safe: any
        failure to look returns ``[]`` (the engine's own success is then trusted). ``kept`` names the
        documents the engine deliberately left open (``result["kept_docs"]``: the user changed them
        after the turn, so closing them would lose their work); they are not residue."""
        kept = {str(k) for k in (kept or ())}
        try:
            from .cad import get_engine

            now = get_engine().begin_turn_snapshot()
            if not isinstance(now, dict) or not isinstance(now.get("docs"), dict):
                return []
            start = snapshot["docs"]
            residue = [name for name, depth in start.items()
                       if name in now["docs"] and now["docs"][name] > depth and name not in kept]
            residue += [name for name in now["docs"] if name not in start and name not in kept]
            return residue
        except Exception:
            return []

    @staticmethod
    def _card_step_indexes(card: Block) -> list[int]:
        """The Project step indexes (>= 1) of the tool rows of ``card``."""
        return [s.view.step_index for s in card.steps if s.view is not None and s.view.step_index >= 1]

    def _journal_revert(self, card: Block, **how) -> None:
        """Tell the chat's Project that the steps of the reverted turn are gone (``Project.note_reverted``): its
        rebuild.py and NOTES.md must not keep replaying geometry the user just undid. ``how`` is passed on
        (``via`` / ``checkpoint`` of a checkpoint revert). Best effort: a failing journal never turns a verified
        revert into an error."""
        indexes = self._card_step_indexes(card)
        if not indexes:
            return
        mark = getattr(self._project_of(self._current_session()), "note_reverted", None)
        if not callable(mark):
            return
        try:
            mark(min(indexes), **how)
        except Exception as e:  # noqa: BLE001
            self._log_panel_error("revert_journal", e)

    def _ledger_on_revert(self, restored_step: int | None) -> None:
        """Let the carried plan forget the work the revert / rewind discarded (``Ledger.on_revert``; a no-op
        until the ledger offers it), then redraw the plan strip. Best effort."""
        session = self._current_session()
        stored = self._session_ledger(session)
        if session is None or not stored:
            return
        try:
            from .flow import Ledger

            ledger = Ledger.from_dict(stored)
            hook = getattr(ledger, "on_revert", None)
            if callable(hook):
                hook(restored_step)
                session["ledger"] = ledger.to_dict()
        except Exception as e:  # noqa: BLE001
            self._log_panel_error("revert_ledger", e)
        self._refresh_plan_strip()
        self._persist_soon()

    def _mark_steps_superseded(self, after_step: int) -> None:
        """Every tool row whose Project step index is above ``after_step`` was discarded: strike it through and
        take its rewind link away (a card of such steps can no longer be reverted either)."""
        for block in self.blocks:
            if block.kind != Block.TOOL:
                continue
            hit = False
            for step in block.steps:
                view = step.view
                if view is not None and view.step_index > after_step and not view.superseded:
                    view.superseded = True
                    view.touch()
                    hit = True
            if hit:
                block.revertible = False

    def _after_geometry_reverted(self, card: Block, **how) -> None:
        """Everything a verified revert of ``card`` changes besides the model: Project journal, struck-through
        rows, the carried plan (``on_revert``) and the artifacts strip."""
        self._journal_revert(card, **how)
        indexes = self._card_step_indexes(card)
        if indexes:
            self._mark_steps_superseded(min(indexes) - 1)
        self._ledger_on_revert(min(indexes) - 1 if indexes else None)
        self._artifacts = self._artifacts_of(self._current_session())
        self._refresh_artifacts_strip()

    def _revert_via_checkpoints(self, card: Block, result: dict | None = None, *,
                                remote: bool = False) -> _CheckpointRevert:
        """Revert fallback for a turn the undo stack cannot hold or must not undo (``result`` is what ``revert_turn``
        reported): restore the Project's turn-start checkpoint of every document the turn changed, verify each by
        object-name-set equality, then close the documents the turn created through the engine's ownership rules.

        Refused, with nothing restored, when the project has no checkpoint of the turn or misses one for a document
        the turn touched (``result["touched_docs"]``). When a document was changed after the turn (the engine's
        ``changed_since_turn``) the user is asked first (``_confirm_checkpoint_restore``, default No); a "No" restores
        nothing and leaves the card revertible. Documents the turn created are closed only by
        ``close_turn_documents`` (an engine without it closes nothing); the ones it keeps open are listed. Never
        raises. Nothing is reported as reverted unless every document verified. A ``remote`` revert (the workspace
        asked for it) asks nothing here: it restores nothing when a document changed after the
        turn, and says so (the person at FreeCAD decides what to keep)."""
        indexes = self._card_step_indexes(card)
        project = self._project_of(self._current_session())
        if not indexes or project is None or not callable(getattr(project, "turn_checkpoints", None)):
            return _CheckpointRevert(False, "this chat's project holds no checkpoint of the turn")
        try:
            wanted = project.turn_checkpoints(min(indexes))
        except Exception as e:  # noqa: BLE001
            return _CheckpointRevert(False, f"the project's checkpoints could not be read ({e})")
        start = card.snapshot.get("docs") if isinstance(card.snapshot, dict) else None
        wanted = [w for w in wanted if not isinstance(start, dict) or w["doc"] in start]
        if not wanted:
            return _CheckpointRevert(False, "no turn-start checkpoint was recorded for this turn")
        result = result if isinstance(result, dict) else {}
        touched = result.get("touched_docs")
        if touched is None:
            touched = result.get("incomplete_docs") or []
        covered = {w["doc"] for w in wanted}
        missing = [str(name) for name in touched if name not in covered]
        if missing:
            # restoring only some of the documents would leave the turn half reverted
            return _CheckpointRevert(False, f"no turn checkpoint for {', '.join(missing)}; nothing was restored")
        engine = _engine()
        check = getattr(engine, "changed_since_turn", None)
        changed: dict = {}
        if callable(check):
            try:
                changed = dict(check(card.snapshot, [w["doc"] for w in wanted]) or {})
            except Exception as e:  # noqa: BLE001 - when in doubt, ask
                changed = {w["doc"]: f"it could not be checked ({e})" for w in wanted}
        if changed and remote:
            names = ", ".join(str(name) for name in changed)
            return _CheckpointRevert(False, f"{names} changed in FreeCAD after this turn, so nothing was restored; "
                                     "revert it in FreeCAD to choose what to keep", cancelled=True)
        if changed and not self._confirm_checkpoint_restore(changed):
            return _CheckpointRevert(False, "you chose to keep the changes made after this turn, so nothing was restored",
                                     cancelled=True)
        done, failed = [], []
        for item in wanted:
            try:
                restored = project.restore(item["doc"], item["id"])
            except Exception as e:  # noqa: BLE001
                failed.append(f"{item['doc']}: {e}")
                continue
            names = set(getattr(restored, "objects_after", None) or [])
            if not restored.ok:
                failed.append(f"{item['doc']}: {restored.error}")
            elif item["objects"] is not None and names != set(item["objects"]):
                extra, missing_names = sorted(names - set(item["objects"])), sorted(set(item["objects"]) - names)
                failed.append(f"{item['doc']}: object names differ from the checkpoint "
                              f"(extra {extra[:5]}, missing {missing_names[:5]})")
            else:
                done.append(item)
        if failed:
            return _CheckpointRevert(False, "; ".join(failed))
        closing = self._close_turn_documents(engine, card.snapshot)
        closed = [str(name) for name in closing.get("closed_docs") or []]
        reasons = {str(e.get("doc")): str(e.get("reason") or "") for e in closing.get("kept_docs") or []
                   if isinstance(e, dict)}
        kept = tuple(f"{name} ({reasons[name]})" if reasons.get(name) else name
                     for name in self._kept_doc_names(closing))
        ids = ", ".join(sorted({d["id"] for d in done}))
        message = f"restored checkpoint {ids} ({', '.join(d['doc'] for d in done)})"
        if closed:
            message += f"; closed {', '.join(closed)}"
        if kept:
            message += f"; left open: {', '.join(kept)}"
        return _CheckpointRevert(True, message, ids, kept=kept)

    def _close_turn_documents(self, engine, snapshot) -> dict:
        """``engine.close_turn_documents(snapshot)``: the documents the turn created, closed under the engine's
        ownership rules. An engine without the method (a test double) closes nothing: closing a document the user
        owns would lose unsaved work, so the fallback never closes on its own."""
        close = getattr(engine, "close_turn_documents", None)
        if not callable(close):
            return {}
        try:
            report = close(snapshot)
        except Exception as e:  # noqa: BLE001
            self._log_panel_error("revert_close", e)
            return {}
        return report if isinstance(report, dict) else {}

    def _confirm_checkpoint_restore(self, changed: dict) -> bool:
        """Ask before a checkpoint restore throws away changes made after the turn (``{doc: reason}``); default No.
        Tests replace this method."""
        lines = "\n".join(f"• {name}: {reason}" for name, reason in changed.items())
        box = QtWidgets.QMessageBox(self)
        box.setObjectName("oscarCheckpointRestoreDialog")
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Revert from the checkpoint")
        box.setTextFormat(QtCore.Qt.PlainText)   # document names and undo labels are data, never markup
        box.setText("This turn can only be reverted by restoring whole documents from the checkpoint taken before it. "
                    "These documents changed after the turn, and those changes would be lost too:\n\n"
                    f"{lines}\n\nRestore them anyway?")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        box.setDefaultButton(QtWidgets.QMessageBox.No)
        return box.exec() == QtWidgets.QMessageBox.Yes

    def _confirm_revert(self, card: Block) -> bool:
        """An older turn can only be reverted while later turns left the same documents alone (the engine never undoes
        someone else's work to get at it), so say what to expect before trying. The latest revertible turn - the only
        one the panel offers - has no later turn and needs no question."""
        later = [b for b in self.blocks[self.blocks.index(card) + 1:]
                 if b.kind == Block.TOOL and b.revertible and not b.reverted] if card in self.blocks else []
        if not later:
            return True
        ret = QtWidgets.QMessageBox.question(
            self, "Oscar",
            f"{len(later)} later turn{'s have' if len(later) != 1 else ' has'} "
            f"worked on the model since. Oscar only reverts this turn in "
            f"documents the later work did not touch, and leaves the rest "
            f"as it is. Try anyway?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        return ret == QtWidgets.QMessageBox.Yes

    def _revert_turn(self, card: Block, *, remote: bool = False) -> bool:
        """Undo every CAD change made during that turn (undo-stack pops to
        the depth recorded at turn start; documents created by the turn are
        closed). True when the revert happened (the card is marked reverted, so a caller may build on it); False when it
        was refused, failed, found nothing to undo or was only partly done.

        Guards: never mid-flight; only the latest non-reverted turn; refused when the turn-start
        snapshot is missing or failed; and the card is marked reverted only AFTER the engine reported
        success and a fresh snapshot confirms the documents are back at their recorded depth. A PARTIAL revert
        (the engine undid some steps but refused to touch later work or could not undo every step) changed the
        documents, so it is reported as partial, the card stays revertible and the model is told what state it is in."""
        if self.worker is not None:  # never mid-flight
            return False
        if not card.revertible or card.reverted:
            return False
        if card is not self._latest_revertible():
            self._append(Block.TOOL, "Only the latest turn can be reverted: revert the newer turn first.")
            self._render()
            return False
        if not self._snapshot_usable(card.snapshot):
            card.revertible = False
            self._append(Block.TOOL, "Cannot revert this turn: its starting state was not recorded.")
            self._update_revert_flags()
            self._render()
            return False
        if not remote and not self._confirm_revert(card):  # the workspace asked its user already (§14)
            return False
        try:
            from .cad import get_engine

            result = get_engine().revert_turn(card.snapshot)
        except Exception as e:
            result = {"success": False, "error": str(e)}
        undone = result.get("undone") or {}
        closed = result.get("closed_docs") or []
        if result.get("success") and (undone or closed):
            kept = self._kept_doc_names(result)
            residue = self._revert_residue(card.snapshot, kept)
            if residue:
                note = ("Revert incomplete: " + ", ".join(residue)
                        + " still hold changes from this turn. Use Edit > Undo or reopen the file.")
            else:
                card.reverted = True
                self._after_geometry_reverted(card)
                note = "↩ Turn reverted"
                if undone:
                    note += f" — {sum(undone.values())} undo step(s) in {', '.join(undone)}"
                if closed:
                    note += f"; closed {', '.join(closed)}"
                if kept:
                    note += f"; left open because you changed them since: {', '.join(kept)}"
                self._pending_notes.append(
                    "[I reverted every CAD change from one of your earlier turns"
                    + (f"; documents closed: {', '.join(closed)}" if closed else "")
                    + (f"; documents kept open (the user changed them afterwards): {', '.join(kept)}" if kept else "")
                    + ". Objects that turn created or edited are back to their "
                    "previous state — check the workspace state below.]"
                )
        elif result.get("success"):
            card.revertible = False  # nothing on the undo stack: there is nothing to offer again
            note = "Nothing to undo for this turn (is the undo stack enabled?)."
        elif result.get("incomplete"):
            fallback = self._revert_via_checkpoints(card, result, remote=remote)
            unsafe = bool(result.get("undo_unsafe"))
            if fallback.ok:
                card.reverted = True
                self._after_geometry_reverted(card, via="checkpoint", checkpoint=fallback.ids)
                note = f"↩ Turn reverted from the project checkpoint — {fallback.message}"
                if unsafe:
                    note += " (the undo is unsafe for PartDesign builds, so the project checkpoint was used)"
                self._pending_notes.append(
                    "[I reverted every CAD change from one of your earlier turns by restoring the project checkpoint "
                    "taken before it. Objects that turn created or edited are back to their previous state"
                    + (f"; documents it created that were left open: {', '.join(fallback.kept)}" if fallback.kept else "")
                    + " — check the workspace state below.]")
            elif fallback.cancelled:
                note = f"Revert cancelled: {fallback.message}."   # the card stays revertible: the user may change their mind
            else:
                card.revertible = False  # neither the undo stack nor a checkpoint can do it: a retry cannot succeed
                note = f"Revert refused: {result.get('error')} Checkpoint fallback failed: {fallback.message}."
        elif undone or closed:
            # Some steps WERE undone but not all (the engine refused to touch later work, or an undo failed): the
            # documents changed, so this is neither "failed" nor "reverted". The card stays revertible (a retry
            # may finish it) and the model learns that part of the turn is still there.
            reason = result.get("error") or "some changes could not be undone"
            note = "↩ Turn only partly reverted"
            if undone:
                note += f" — {sum(undone.values())} undo step(s) in {', '.join(undone)}"
            if closed:
                note += f"; closed {', '.join(closed)}"
            note += f" ({reason})"
            self._pending_notes.append(
                "[I tried to revert one of your earlier turns but it was only partly undone"
                + (f"; documents closed: {', '.join(closed)}" if closed else "")
                + f" ({reason}). Part of what that turn created or edited may still be there — "
                "check the workspace state below.]")
        else:
            note = f"Revert failed: {result.get('error')}"
        self._append(Block.TOOL, note)
        self._update_revert_flags()
        self._render()
        return card.reverted
