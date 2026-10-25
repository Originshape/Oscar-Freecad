"""Per-step rewind of the Oscar panel (the ``rewind:<bid>:<idx>`` anchor), as a mixin of ``OscarPanel``.

A titled step row of a turn card links to ``rewind:<bid>:<idx>``: restore the model to just after that step from
the Project checkpoint recorded for it (``Project.rewind_to``, which re-activates the live document and journals
the rewind so rebuild.py / NOTES.md never replay the discarded steps).

Guards, each with a clear message and no change to the model: no rewind while a turn runs, the step must belong to
the chat's Project and have a checkpoint that still exists (checkpoints are pruned), the user confirms a dialog
that says exactly which objects disappear and which later steps are lost, and a restore whose validation fails
leaves the model as it was (the Project puts the rescue copy back). After a successful rewind the later rows are
struck through, the carried plan forgets them (``Ledger.on_revert``), the plan and artifacts strips are redrawn and
a note tells the model on its next message.

The mixin relies on the host panel for ``worker``, ``blocks``, ``_step_at``, ``_project_of``,
``_current_session``, ``_pending_notes``, ``_on_notice``, ``_render``, ``_update_revert_flags`` and the revert
mixin's ``_mark_steps_superseded`` / ``_ledger_on_revert``.
"""

import FreeCAD as App
from PySide import QtWidgets

_LIST_LIMIT = 8


def _short(names: list[str], limit: int = _LIST_LIMIT) -> str:
    """``A, B, C`` clipped to ``limit`` names with ``… +N more``."""
    shown = ", ".join(names[:limit])
    return shown + (f" … +{len(names) - limit} more" if len(names) > limit else "")


class RewindMixin:
    """Rewind methods of the panel (see the module docstring)."""

    def _confirm_rewind(self, text: str) -> bool:
        """The confirmation dialog (default answer: No). Tests replace this method."""
        box = QtWidgets.QMessageBox(self)
        box.setObjectName("oscarRewindDialog")
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Rewind the model")
        box.setText(text)
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        box.setDefaultButton(QtWidgets.QMessageBox.No)
        return box.exec() == QtWidgets.QMessageBox.Yes

    @staticmethod
    def _live_object_names(doc_name: str) -> list[str] | None:
        """Object names of the open document ``doc_name`` (``None`` when it is not open)."""
        try:
            doc = App.listDocuments().get(doc_name)
            return None if doc is None else [o.Name for o in doc.Objects]
        except Exception:  # noqa: BLE001
            return None

    def _rewind_confirmation_text(self, view, info: dict) -> str:
        """What the dialog says: the document restored, the objects that disappear, the steps that are lost."""
        doc = info["doc"] or "the document"
        keep = set(info["objects"])
        live = self._live_object_names(info["doc"])
        lines = [f"Restore '{doc}' to just after step {view.step_index} ({view.label}) from checkpoint "
                 f"{info['checkpoint']}?", ""]
        if live is None:
            lines.append(f"The document '{doc}' is not open; the rewind will be refused.")
        elif keep:
            gone = [n for n in live if n not in keep]
            back = [n for n in info["objects"] if n not in set(live)]
            lines.append("Objects that disappear: " + (_short(gone) if gone else "none") + ".")
            if back:
                lines.append("Objects that come back: " + _short(back) + ".")
            lines.append("Everything else in the document returns to its state at that step "
                         "(parameters and edits included).")
        else:
            lines.append("The whole document returns to its state at that step (the checkpoint lists no objects "
                         "to compare).")
        lost = info["lost_steps"]
        if lost:
            lines.append(f"Later steps that are lost ({len(lost)}): "
                         + _short([f"{s['index']} {s['title']}" for s in lost]) + ".")
        else:
            lines.append("No later steps are lost.")
        lines.append("")
        lines.append("One Ctrl+Z reverts the rewind.")
        return "\n".join(lines)

    def _on_rewind_anchor(self, spec: str) -> None:
        """``rewind:<bid>:<idx>`` (``spec`` is ``"<bid>:<idx>"``)."""
        if self.worker is not None:
            self._on_notice("A turn is running: stop it or wait for it to finish before rewinding the model.")
            return
        step = self._step_at(spec)
        view = getattr(step, "view", None)
        if view is None or view.step_index < 1:
            self._on_notice("This step is not recorded in the project, so the model cannot be rewound to it.")
            return
        if view.superseded:
            self._on_notice("This step was already discarded by an earlier revert or rewind.")
            return
        if not view.checkpoint:
            self._on_notice(f"No checkpoint exists for step {view.step_index}, so the model cannot be rewound to it.")
            return
        project = self._project_of(self._current_session())
        if project is None or not callable(getattr(project, "rewind_info", None)):
            self._on_notice("This chat has no project to rewind.")
            return
        try:
            info = project.rewind_info(view.step_index)
        except Exception as e:  # noqa: BLE001
            self._log_panel_error("rewind_info", e)
            self._on_notice(f"Cannot rewind: the project could not be read ({e}).")
            return
        if not info.get("ok"):
            self._on_notice(f"Cannot rewind to step {view.step_index}: {info.get('error')}")
            return
        if info.get("checkpoint") != view.checkpoint:
            self._on_notice(f"The checkpoint of step {view.step_index} no longer exists (the nearest is "
                            f"{info.get('checkpoint')}), so the model cannot be rewound exactly to it.")
            return
        if not self._confirm_rewind(self._rewind_confirmation_text(view, info)):
            return
        try:
            result = project.rewind_to(view.step_index)
        except Exception as e:  # noqa: BLE001
            self._log_panel_error("rewind", e)
            self._on_notice(f"Rewind failed: {e}")
            return
        if not result.ok:
            self._on_notice(f"Rewind to step {view.step_index} failed and the model was left as it was: "
                            f"{result.error}")
            return
        self._after_rewind(view, info, result)

    def _after_rewind(self, view, info: dict, result) -> None:
        """A verified rewind: rows, plan, strips, the note for the model and the notice."""
        n = view.step_index
        self._mark_steps_superseded(n)
        self._ledger_on_revert(n)
        self._artifacts = self._artifacts_of(self._current_session())
        self._refresh_artifacts_strip()
        lost = info.get("lost_steps") or []
        names = _short(list(result.objects_after or info.get("objects") or []), 12)
        self._pending_notes.append(
            f"[The user rewound the model to step {n} ({view.label}): '{info['doc']}' was restored from its "
            f"checkpoint {info['checkpoint']}"
            + (f"; objects now present: {names}" if names else "")
            + (f". Steps {lost[0]['index']}..{lost[-1]['index']} ({len(lost)} step(s)) were discarded: their geometry "
               "is gone and the project journal no longer counts them" if lost else "")
            + ". Do not assume anything built after that step exists; check the workspace state below.]")
        self._update_revert_flags()
        self._on_notice(f"Rewound to step {n} ({view.label}); {len(lost)} later step(s) discarded.")
        self._render()
        self._fit_view_after_turn(force=True)
