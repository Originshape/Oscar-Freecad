"""Task chats of the Oscar panel (sessions) and their per-chat build state, as a mixin of ``OscarPanel``.

A session is the dict ``chats.new_session()`` returns plus panel-only keys: ``blocks`` (the rendered cards,
``None`` until first shown), ``pending_notes`` and ``_project`` (the bound ``oscar.project.Project``, never
persisted). The persisted optional keys are the chat's build state:

- ``project_dir``: the absolute folder of the chat's Project (set when a turn binds it);
- ``ledger``: the plan the next turn carries over, kept from ``plan_changed`` and from the worker's
  ledger at turn end;
- ``blocks_meta``: compact records of the finished turns (``panel_render.make_turn_meta``) that the restore
  card shows when the chat is opened again.

The Project is created per session AFTER the chat got its title (``_bind_project``: the folder name derives
from it) and is lazy: nothing is written until the first mutating call or work turn on the worker thread.

An empty chat shows the welcome text (``WELCOME``) and, when the active profile cannot chat yet (the OriginShape
account without a sign-in, an Anthropic profile without its key), the onboarding card under it
(``_append_empty_state``; the panel's ``_sync_onboarding`` keeps that card in step with the sign-in).

The mixin relies on the host panel for ``chat_combo``, ``del_chat_btn``, ``history`` / ``blocks`` / ``worker``,
``_append``, ``_render``, ``_set_status``, ``_refresh_session_widgets`` (the plan / artifacts strips), ``_thumbs`` and
``_sync_onboarding``.
"""

import os

from PySide import QtWidgets

from .panel_blocks import Block
from .panel_render import restore_thumbs, trim_turn_metas

WELCOME = """**Oscar** — your CAD copilot is ready.

Ask me to build, inspect or modify models and I'll drive the application
directly: primitives, booleans, drafts, FEM runs, screenshots I can actually
see.

Try:
- `create a document and build a flange, 80 mm diameter, 8 mm thick, 6 bolt holes`
- `what's in this document?`
- `mesh the bracket and run a static analysis`

I run tools that change the open document — check the model after each turn.
"""

_RESTORE_THUMBS = 4   # snapshots of the newest turn the restore card shows


class SessionMixin:
    """Session methods of the panel (see the module docstring)."""

    # ------------------------------------------------------------------
    # task chats: new / switch / clear / delete (persisted via oscar.chats)
    # ------------------------------------------------------------------

    def _busy(self) -> bool:
        return self.worker is not None or getattr(self, "_building", False)

    def _append_empty_state(self) -> None:
        """What an empty chat shows: the welcome text, then the onboarding card when the account asks for one."""
        self._welcome = self._append(Block.ASSISTANT, WELCOME)
        sync = getattr(self, "_sync_onboarding", None)
        if callable(sync):
            sync()

    def _load_sessions(self) -> None:
        """Restore persisted chats; the newest becomes current."""
        from . import chats as _chats

        self.sessions = []
        for meta in _chats.list_sessions():
            if meta.get("messages") == 0 and _chats.is_unused(meta["id"]):
                _chats.delete(meta["id"])   # an empty leftover (a "New chat" nobody wrote in): clean it up
                continue
            try:
                session = _chats.load(meta["id"])
            except Exception as e:  # noqa: BLE001 - one damaged chat file must never keep the panel from opening
                self._log_panel_error("load_chat", e)
                continue
            if session is not None and session["history"]:
                session["blocks"] = None  # lazily rendered on first switch
                session["_saved"] = True  # it has a file: an edit (clear) must reach it even when the chat is empty now
                self.sessions.append(session)
        if not self.sessions:
            self.sessions = [_chats.new_session()]
        # restore=True: the chat the user returns to after a restart gets its restore card (turn records, thumbnails
        # and, after a crash, the "interrupted in the middle of a step" notice); an empty chat gets none
        self._activate_session(self.sessions[0])

    def _activate_session(self, session: dict, restore: bool = True) -> None:
        """Point history/blocks at a session and refresh the picker, the plan strip and the artifacts strip."""
        self.history = session["history"]
        blocks = session.get("blocks")
        if blocks is None:
            # First activation this run: fresh view; a compact restore card stands in for the original
            # rendering (the model-side context — what actually matters — is fully intact).
            self.blocks = []
            self._append_empty_state()
            if restore and len(session["history"]) > 1:
                self._append_restore_card(session)
            session["blocks"] = self.blocks
        else:
            self.blocks = blocks
            # the welcome banner of THIS chat (still there until its first message), not the previous chat's
            self._welcome = next((b for b in blocks if b.kind == Block.ASSISTANT and b.text == WELCOME), None)
            sync = getattr(self, "_sync_onboarding", None)
            if callable(sync):
                sync()              # the sign-in may have changed while another chat was shown
        self._turn_view = None
        # Recency order: the active chat is always sessions[0], so the picker reads "most recent first" and a
        # delete / switch falls back to the chat that was used last instead of whatever was loaded first.
        try:
            self.sessions.remove(session)
        except ValueError:
            pass
        self.sessions.insert(0, session)
        self._rebuild_chat_combo()
        self._refresh_session_widgets()
        self._render()

    def _append_restore_card(self, session: dict) -> Block:
        """The card of a reopened chat: the legacy "Restored" line, the interruption notice (a crash in the
        middle of a step), the newest turn records of ``blocks_meta`` and the thumbnails of the last turn
        (their stored relative paths are resolved against the project folder)."""
        block = self._append(Block.RESTORE, session["title"])
        block.messages = len(session["history"])
        try:
            from . import chats as _chats

            block.detail = _chats.recovery_notice(session)
        except Exception:  # noqa: BLE001 - a notice is a nicety
            block.detail = ""
        block.meta = trim_turn_metas(session.get("blocks_meta"))
        root = session.get("project_dir") or ""
        for thumb_id, stored in restore_thumbs(block.meta, str(block.bid), _RESTORE_THUMBS):
            path = stored if os.path.isabs(stored) else (os.path.join(root, stored) if root else "")
            if path and self._thumbs.load(thumb_id, path) is not None:
                block.thumbs.add(thumb_id)
        return block

    def _rebuild_chat_combo(self) -> None:
        self._session_loading = True
        self.chat_combo.clear()
        for session in self.sessions:
            title = session["title"] or "New chat"
            self.chat_combo.addItem(title)
        current = next((i for i, s in enumerate(self.sessions)
                        if s["history"] is self.history), 0)
        self.chat_combo.setCurrentIndex(current)
        self.del_chat_btn.setEnabled(len(self.sessions) > 1)
        self._session_loading = False

    def _current_session(self) -> dict | None:
        return next((s for s in self.sessions
                     if s["history"] is self.history), None)

    @property
    def _pending_notes(self) -> list[str]:
        """Facts the model must learn with the next message, kept per chat session so a revert in one
        chat never leaks a note into another (not persisted: it dies with the app, as before)."""
        session = self._current_session()
        if session is None:
            return self._orphan_notes
        return session.setdefault("pending_notes", [])

    @_pending_notes.setter
    def _pending_notes(self, notes: list[str]) -> None:
        session = self._current_session()
        if session is None:
            self._orphan_notes = list(notes)
        else:
            session["pending_notes"] = list(notes)

    def _persist_current(self) -> None:
        session = self._current_session()
        if session is None:
            return
        if not session["history"] and not session.get("_saved"):
            return   # a chat nobody wrote in is not a chat: clicking "new chat" five times leaves no empty files behind
        try:
            from . import chats as _chats

            _chats.save(session)
            session["_saved"] = True
        except Exception as e:  # noqa: BLE001 - a full disk must not break the chat, but it must leave a trace
            self._log_panel_error("persist", e)

    def _persist_soon(self) -> None:
        """Persist the current chat within about 1.5 s (one write for a burst of plan changes)."""
        timer = getattr(self, "_persist_timer", None)
        if timer is not None and not timer.isActive():
            timer.start()

    def _on_chat_picked(self, index: int) -> None:
        if self._session_loading:
            return
        if self._busy():
            self._set_status("Wait for the current turn to finish before "
                             "switching chats.")
            self._rebuild_chat_combo()
            return
        if not (0 <= index < len(self.sessions)):
            return
        self._persist_current()
        self._activate_session(self.sessions[index])

    def _new_chat(self) -> None:
        if self._busy():
            self._set_status("Wait for the current turn to finish first.")
            return
        current = self._current_session()
        if current is not None and not current["history"] and current["title"] == "New chat":
            # already sitting in an unused chat: a second one would only be a duplicate empty entry in the picker
            self._set_status("Already in a new chat.")
            self.input.setFocus()
            return
        from . import chats as _chats

        self._persist_current()
        session = _chats.new_session()
        session["blocks"] = None
        self.sessions.insert(0, session)
        self._activate_session(session, restore=False)

    def _clear_chat(self) -> None:
        if self._busy():
            self._set_status("Wait for the current turn to finish first.")
            return
        session = self._current_session()
        if session is None:
            return
        ret = QtWidgets.QMessageBox.question(
            self, "Oscar",
            f"Clear all messages of “{session['title']}”?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if ret != QtWidgets.QMessageBox.Yes:
            return
        session["history"].clear()
        session.pop("pending_notes", None)
        session["ledger"] = None          # the plan belonged to the messages that are gone
        session["blocks_meta"] = []
        session["title"] = "New chat"
        session["blocks"] = None
        self._activate_session(session, restore=False)
        self._persist_current()
        self._set_status("Chat cleared.")

    def _delete_chat(self) -> None:
        if self._busy():
            self._set_status("Wait for the current turn to finish first.")
            return
        session = self._current_session()
        if session is None or len(self.sessions) <= 1:
            return
        ret = QtWidgets.QMessageBox.question(
            self, "Oscar",
            f"Delete the chat “{session['title']}” and its messages? "
            f"This cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if ret != QtWidgets.QMessageBox.Yes:
            return
        from . import chats as _chats

        _chats.delete(session["id"])   # the chat file only: the project folder holds the user's files and stays
        self.sessions.remove(session)
        # the chat used last (sessions are in recency order) that has something in it - never an empty "New chat"
        # while a real chat exists; a chat not shown yet in this run gets its restore card
        self._activate_session(next((s for s in self.sessions if s["history"]), self.sessions[0]))

    def _title_current_from(self, text: str) -> None:
        """Name the chat after its first user prompt (combo shows it)."""
        session = self._current_session()
        if session is None or session["title"] != "New chat" or session.get("named"):
            return
        from . import chats as _chats

        session["title"] = _chats.title_from_prompt(text)
        self._rebuild_chat_combo()

    # ------------------------------------------------------------------
    # the chat's build state: Project, plan, artifacts
    # ------------------------------------------------------------------

    def _bind_project(self, session: dict | None):
        """The ``oscar.project.Project`` of ``session`` (created once, cached on the session), or ``None`` when
        the module is unavailable or the binding failed (the turn then runs without a Project).

        Call it AFTER ``_title_current_from`` (the folder name derives from the title and is fixed at creation).
        Nothing is written to disk here: the folder is created lazily on the worker thread."""
        if session is None:
            return None
        project = session.get("_project")
        if project is None:
            try:
                from .project import Project

                project = Project.for_chat(session["id"], session.get("title") or "")
            except ModuleNotFoundError as e:
                if e.name not in ("oscar.project", f"{__package__}.project"):
                    self._log_panel_error("project", e)
                return None
            except Exception as e:  # noqa: BLE001 - recording is best effort, the turn must still run
                self._log_panel_error("project", e)
                return None
            session["_project"] = project
        root = getattr(project, "root", "")
        if root:
            session["project_dir"] = root
        return project

    def _project_of(self, session: dict | None):
        """The bound Project of ``session`` without creating one; a saved chat with a ``project_dir`` re-opens it
        (``None`` when the folder is gone or holds no ``project.json``)."""
        if session is None:
            return None
        project = session.get("_project")
        if project is not None:
            return project
        root = session.get("project_dir") or ""
        if root and os.path.isfile(os.path.join(root, "project.json")):
            try:
                from .project import Project

                project = Project.open(root)
            except Exception:  # noqa: BLE001
                return None
            session["_project"] = project
        return project

    def _session_ledger(self, session: dict | None = None) -> dict | None:
        """The plan (ledger dict) of ``session`` (default: the current chat), ``None`` when it has none."""
        session = session if session is not None else self._current_session()
        ledger = session.get("ledger") if session is not None else None
        return ledger if isinstance(ledger, dict) else None

    def _resume_text(self) -> str:
        """The message the ``continue:`` link sends: the ledger's own resume prompt (the worker recognises
        exactly that text as a continuation), ``Continue the task.`` when the plan is unavailable."""
        ledger = self._session_ledger()
        if ledger:
            try:
                from .flow import Ledger

                return Ledger.from_dict(ledger).resume_prompt()
            except Exception:  # noqa: BLE001
                pass
        return "Continue the task."

    def _log_panel_error(self, where: str, error: Exception) -> None:
        """One telemetry line for a panel-side failure that must not interrupt the user."""
        try:
            from .turnlog import log_event

            log_event("panel_error", where=where, error=f"{type(error).__name__}: {error}"[:200])
        except Exception:  # noqa: BLE001
            pass

    def _round_persister(self, session: dict):
        """The worker's ``on_round`` callback: persist ``session`` (history, plan, turn records) after every
        provider round. Runs on the worker thread; ``chats.save`` is thread safe."""
        from . import chats as _chats

        def persist(_history) -> None:
            _chats.persist_round(session)

        return persist
