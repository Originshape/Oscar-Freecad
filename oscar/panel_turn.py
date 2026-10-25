"""The live turn of the Oscar panel: what the worker's signals do to the screen, as a mixin of ``OscarPanel``.

Signals consumed. The legacy ones keep their exact behaviour (``text_delta``,
``reasoning_delta``, ``tool_started``, ``tool_finished``, ``turn_finished``, ``notice``, ``flow_paused``); the
additive ones feed a ``panel_render.TurnView`` and the widgets built here:

- ``plan_changed`` -> the pinned plan strip (never auto-collapsed; ``plan:toggle`` is the user's choice) and
  ``session["ledger"]`` (what the next turn carries over);
- ``step_started`` / ``step_finished`` -> the SAME ``Step`` row the legacy ``tool_started`` created (rich view:
  title, delta, health, thumbnail, code / output on demand; the legacy ``tool_finished`` that follows never
  replaces what ``step_finished`` set; never two rows for one call);
- ``progress`` -> the heartbeat row (``round 7/30 · BUILD · thinking 12 s``), re-drawn once a second by the panel's
  own timer, and the Act-now button;
- ``round_reset`` -> the partial reply of the round is void and is removed;
- ``artifacts_changed`` -> the artifacts strip; ``turn_outcome`` -> the outcome badge on the card header and the
  Build Summary card, which is rendered ONCE (``turn_finished``'s text is cut at the first ``## Build Summary``);
- ``account_problem`` (just before ``turn_outcome``: the turn failed for an OriginShape account reason) -> the
  matching ``ACCOUNT`` card replaces the raw "Provider error: ..." line (``panel_render.account_problem_card``), and
  when the turn's very first request failed the message goes back into the input (``_give_back_turn``).

Thumbnails: each snapshot PNG is decoded once into ``self._thumbs`` and served to the log document by URL
(``panel_blocks.LogBrowser``); the registration is repeated after every ``setHtml``.

Provider call ids are not unique across turns (some local servers answer ``call_0`` every time), so rows and
thumbnails are keyed by ``panel_render.unique_call_id`` (turn sequence + ordinal + id).

The mixin relies on the host panel for ``blocks``, ``worker``, ``log``, ``_append``, ``_render``, ``_set_status``,
``_set_busy``, ``_on_notice``, ``_current_session`` / ``_session_ledger`` / ``_project_of`` / ``_persist_*``
(``panel_session``), ``_MUTATING_TOOLS`` and the status / render timers.
"""

import json
import os
import time

from PySide import QtCore, QtWidgets

from .panel_blocks import Block, Step, ThumbCache, _accent, _font_size, _muted
from .panel_render import (TurnView, account_problem_card, artifacts_strip_html, header_text, make_turn_meta,
                           outcome_badge, plan_strip_html, split_build_summary, trim_turn_metas, unique_call_id)

# Status texts that only mean "busy": the first streamed text supersedes them. Anything else (waiting
# for an approval, pausing, stopping, a tool name) stays until the next status change.
_GENERIC_STATUS = ("Thinking", "Working…")
_PROVIDER_ACTIVITIES = ("thinking", "writing", "tool_args")   # Act now only makes sense while a provider call is open
_MAX_THUMB_LOADS_PER_PASS = 2                                 # decode at most this many PNGs in one render pass


class TurnMixin:
    """Live-turn methods of the panel (see the module docstring)."""

    # ------------------------------------------------------------------
    # construction (called by OscarPanel.__init__ / _build_ui)
    # ------------------------------------------------------------------

    def _init_turn_state(self) -> None:
        self._turn_view: TurnView | None = None
        self._turn_seq = 0                              # keys rows and thumbnails of one turn apart from every other turn
        self._steps_by_call: dict[str, Step] = {}
        self._thumbs = ThumbCache()
        self._thumb_retry = False
        self._plan_collapsed = False
        self._plan_markup = ""
        self._artifacts: dict | None = None
        self._artifacts_markup = ""
        self._turn_outcome = ("", "")
        self._turn_request = ""
        self._act_now_armed = True
        self._carry_images: list[str] = []              # images of a steered message that could not be delivered
        self._turn_account: dict | None = None           # this turn's account_problem payload, decoded
        self._turn_sent: dict | None = None              # what _send put in (for _give_back_turn)

    def _build_flow_widgets(self) -> None:
        """The plan strip, the artifacts strip, the heartbeat row (the host places them in its layout)."""
        self.plan_strip = QtWidgets.QTextBrowser()
        self.plan_strip.setObjectName("oscarPlanStrip")
        self.plan_strip.setOpenLinks(False)
        self.plan_strip.setOpenExternalLinks(False)
        self.plan_strip.anchorClicked.connect(self._on_anchor)
        self.plan_strip.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.plan_strip.setStyleSheet(
            "QTextBrowser { background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.25);"
            " border-radius: 8px; }")
        self.plan_strip.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.plan_strip.hide()

        self.artifacts_strip = QtWidgets.QLabel()
        self.artifacts_strip.setObjectName("oscarArtifactsStrip")
        self.artifacts_strip.setTextFormat(QtCore.Qt.RichText)
        self.artifacts_strip.setWordWrap(True)
        self.artifacts_strip.setOpenExternalLinks(False)
        self.artifacts_strip.linkActivated.connect(self._on_anchor_text)
        self.artifacts_strip.hide()

        self.heartbeat_label = QtWidgets.QLabel("")
        self.heartbeat_label.setObjectName("oscarHeartbeat")
        self.heartbeat_label.setTextFormat(QtCore.Qt.PlainText)   # the tool name comes from the model: never markup
        self.heartbeat_label.setStyleSheet(f"color:{_muted()};font-size:small;")
        self.act_now_btn = QtWidgets.QPushButton("Act now")
        self.act_now_btn.setObjectName("oscarActNowButton")
        self.act_now_btn.setToolTip("Stop the model's thinking and make it act now")
        self.act_now_btn.setFlat(True)
        self.act_now_btn.setStyleSheet(
            "QPushButton { padding:1px 8px; border:1px solid rgba(128,128,128,0.4); border-radius:8px;"
            " font-size:small; }")
        self.act_now_btn.clicked.connect(self._act_now)
        self.heartbeat_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.heartbeat_row)
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(6)
        row.addWidget(self.heartbeat_label, 1)
        row.addWidget(self.act_now_btn)
        self.heartbeat_row.hide()

        self._heartbeat_timer = QtCore.QTimer(self)
        self._heartbeat_timer.setInterval(1000)
        self._heartbeat_timer.timeout.connect(self._update_heartbeat)
        self._persist_timer = QtCore.QTimer(self)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(1500)
        self._persist_timer.timeout.connect(self._persist_current)

    # ------------------------------------------------------------------
    # plan strip / artifacts strip / heartbeat
    # ------------------------------------------------------------------

    def _plan_source(self) -> dict | None:
        view = self._turn_view
        if self.worker is not None and view is not None and view.plan is not None:
            return view.plan_view()
        return self._session_ledger()

    def _refresh_plan_strip(self) -> None:
        """Redraw the pinned plan strip from the live ledger (or the chat's stored one); hidden without steps.
        The markup is compared first: a progress tick that changes nothing visible costs one string compare."""
        markup = plan_strip_html(self._plan_source(), accent=_accent(), muted=_muted(),
                                 collapsed=self._plan_collapsed)
        if markup == self._plan_markup:
            return
        self._plan_markup = markup
        if not markup:
            self.plan_strip.clear()
            self.plan_strip.hide()
            return
        bar = self.plan_strip.verticalScrollBar()
        scrolled = bar.value()
        self.plan_strip.setHtml(f"<html><body style='font-size:{_font_size()}pt;'>{markup}</body></html>")
        self.plan_strip.show()
        self._fit_plan_strip()
        bar.setValue(min(scrolled, bar.maximum()))   # a plan the user scrolled stays where it was

    def _fit_plan_strip(self) -> None:
        """Height of the strip: its content, at most about 30% of the panel (it scrolls beyond that)."""
        document = self.plan_strip.document()
        width = self.plan_strip.viewport().width()
        document.setTextWidth(max(160, width))
        wanted = int(document.size().height()) + 12
        limit = max(90, int(self.height() * 0.30))
        self.plan_strip.setFixedHeight(min(wanted, limit))

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if not self.plan_strip.isHidden():
            self._fit_plan_strip()

    def _artifacts_of(self, session: dict | None) -> dict | None:
        """The artifacts payload of ``session`` read from its Project folder (``None`` without a folder)."""
        project = self._project_of(session)
        root = getattr(project, "root", "") if project is not None else ""
        if not root or not os.path.isdir(root):
            return None
        try:
            files = list(project.artifact_list())
        except Exception as e:  # noqa: BLE001
            self._log_panel_error("artifact_list", e)
            files = []
        return {"project_dir": root, "files": files}

    def _refresh_artifacts_strip(self) -> None:
        markup = artifacts_strip_html(self._artifacts, muted=_muted())
        if markup == self._artifacts_markup:
            return
        self._artifacts_markup = markup
        self.artifacts_strip.setText(markup)
        self.artifacts_strip.setVisible(bool(markup))

    def _refresh_session_widgets(self) -> None:
        """The chat changed (switched, cleared, restored): plan strip, artifacts strip and heartbeat follow."""
        session = self._current_session()
        self._artifacts = self._artifacts_of(session)
        self._plan_markup = "\u0000"   # force one redraw
        self._refresh_plan_strip()
        self._refresh_artifacts_strip()
        self._update_heartbeat()

    def _update_heartbeat(self) -> None:
        """Heartbeat row: shown while a turn runs. The text comes from the latest ``progress`` payload and is
        re-asked every second (the elapsed counters keep moving between payloads)."""
        busy = self.worker is not None
        self.heartbeat_row.setVisible(busy)
        if not busy:
            return
        view = self._turn_view
        text = view.heartbeat() if view is not None else ""
        self.heartbeat_label.setText(text or "starting…")
        activity = ""
        if view is not None and isinstance(view.progress, dict):
            activity = str(view.progress.get("activity") or "")
        self.act_now_btn.setEnabled(self._act_now_armed and activity in _PROVIDER_ACTIVITIES
                                    and callable(getattr(self.worker, "act_now", None)))
        self.act_now_btn.setVisible(callable(getattr(self.worker, "act_now", None)))

    def _act_now(self) -> None:
        """Act-now button: ask the worker to cut the current provider generation and act (watchdog path)."""
        act = getattr(self.worker, "act_now", None)
        if not callable(act):
            return
        self._act_now_armed = False   # one press per round: each press counts toward the watchdog's trip budget
        self.act_now_btn.setEnabled(False)
        try:
            act()
        except Exception as e:  # noqa: BLE001
            self._set_status(f"Could not ask Oscar to act now: {e}")
            return
        self._set_status("Asking Oscar to stop thinking and act…")

    # ------------------------------------------------------------------
    # thumbnails
    # ------------------------------------------------------------------

    def _abs_snapshot(self, path: str) -> str:
        """``path`` made absolute: a stored relative path is resolved against the chat's project folder."""
        if not path or os.path.isabs(path):
            return path
        project = self._project_of(self._current_session())
        root = getattr(project, "root", "") if project is not None else ""
        return os.path.join(root, path) if root else path

    def _mark_snapshot_missing(self, view) -> None:
        view.snapshot_path = ""
        view.snapshot_note = view.snapshot_note or "snapshot file not found"
        view.touch()

    def _register_thumbnails(self) -> None:
        """After every ``setHtml``: decode the thumbnails the rows need (at most two PNGs per pass, the rest on
        the next one) and hand every cached image to the new document. A file that cannot be read turns its row
        into the muted ``snapshot_note`` instead of a broken image.

        Only the rows that are DRAWN need an image: the step rows of an expanded card (a collapsed card shows its
        header alone) and the thumbnails of a restore card. Those are protected from eviction, so the cache can
        always hold what is on screen: with more steps than the cache has room for, evicting a thumbnail that the
        next pass needs again made the retry loop re-render the whole log forever."""
        loads, missing, pending = 0, False, False
        wanted: list = []   # (step, view) of the rows on screen that carry a snapshot
        keep: set[str] = set()
        for block in self.blocks:
            if block.kind == Block.RESTORE and block.thumbs:
                keep |= block.thumbs
            elif block.kind == Block.TOOL and block.expanded:
                for step in block.steps:
                    view = step.view
                    if view is not None and view.snapshot_path and step.call_id:
                        wanted.append((step, view))
                        keep.add(step.call_id)
        for block in self.blocks:
            if block.kind == Block.RESTORE and block.thumbs:
                alive = {t for t in block.thumbs if self._thumbs.has(t)}
                if alive != block.thumbs:   # the cache evicted some: the card must not point at vanished images
                    block.thumbs = alive
                    missing = True
        for step, view in wanted:
            if self._thumbs.has(step.call_id):
                continue
            if self._thumbs.failed(step.call_id):
                self._mark_snapshot_missing(view)
                missing = True
                continue
            if loads >= _MAX_THUMB_LOADS_PER_PASS:
                pending = True
                continue
            loads += 1
            if self._thumbs.load(step.call_id, self._abs_snapshot(view.snapshot_path), keep=keep) is None:
                self._mark_snapshot_missing(view)
                missing = True
        self.log.register_thumbnails(self._thumbs)
        if (missing or pending) and not self._thumb_retry:
            self._thumb_retry = True
            QtCore.QTimer.singleShot(0, self._retry_render)

    def _retry_render(self) -> None:
        self._thumb_retry = False
        self._render()

    # ------------------------------------------------------------------
    # the turn card
    # ------------------------------------------------------------------

    def _ensure_turn_view(self) -> TurnView:
        if self._turn_view is None:
            self._turn_view = TurnView()
        return self._turn_view

    def _ensure_card(self) -> Block:
        """This turn's single process card: thoughts AND tool steps."""
        if self._turn_card is None:
            card = self._append(Block.TOOL, "")
            card.expanded = True
            card.t0 = time.monotonic()
            card.snapshot = self._turn_snapshot
            card.workspace_message = str((self._turn_sent or {}).get("workspace_message") or "")
            self._turn_card = card
            if not self._status_timer.isActive():
                self._status_timer.start()  # drives the Working… timer
        return self._turn_card

    def _close_live_thought(self):
        card = self._turn_card
        if card is None or not card.steps:
            return
        last = card.steps[-1]
        if last.is_thought and last.dur == 0:
            last.dur = max(0.0, time.monotonic() - (last.t0 or time.monotonic()))
            last.expanded = False
            if self._turn_view is not None:
                self._turn_view.add_thought(last.dur)

    def _close_remark(self):
        card = self._turn_card
        if card is None or not card.steps:
            return
        last = card.steps[-1]
        if last.plain and last.live:
            last.live = False

    def _fold_open_text(self):
        """A pre-tool remark ("Let me set that up…") is not an answer —
        fold it into the process card as a muted note instead of leaving a
        frozen message above the live thinking (Claude/GPT chat layout:
        only final answers render as messages)."""
        block = self._current
        self._current = None
        if block is not None and block in self.blocks:
            # the remark moves INTO the card; left in place it would also
            # render as a chat message above the card (shown twice)
            self.blocks.remove(block)
        if block is None or not block.text.strip():
            return
        card = self._ensure_card()
        note = Step(summary=block.text.strip())
        note.plain = True
        card.steps.append(note)

    # -- legacy streaming callbacks (GUI thread, via queued signal connections) -----------------------

    def _on_reasoning_delta(self, chunk: str):
        # Reasoning models think before EACH round; every round becomes a
        # step inside the turn's process card, streaming its tail in gray.
        self._fold_open_text()
        self._close_remark()
        card = self._ensure_card()
        last = card.steps[-1] if card.steps else None
        if not (last and last.is_thought and last.dur == 0):
            last = Step(is_thought=True)
            last.t0 = time.monotonic()
            card.steps.append(last)
            card.expanded = True  # a new thinking round re-opens the card
        last.detail += chunk
        self._dirty = True  # the throttle only renders when dirty
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _on_text_delta(self, chunk: str):
        self._close_live_thought()
        if self._status in _GENERIC_STATUS:  # the model is visibly typing now
            self._status = ""
        if self._turn_card is not None:
            # Mid-flow remarks stream INSIDE the process card as a live
            # gray note — they never render as chat messages (so a
            # minimized card stays minimized); the round's final text
            # becomes the answer message when the turn ends.
            card = self._turn_card
            last = card.steps[-1] if card.steps else None
            if not (last and last.plain and last.live):
                last = Step("note")
                last.summary = ""   # Step() falls back to the name for an empty summary: the note starts blank
                last.plain = True
                last.live = True
                card.steps.append(last)
            last.summary += chunk
        else:
            if self._current is None:
                self._start_assistant_block()
            self._stream_buffer += chunk
            self._current.text = self._stream_buffer
        self._dirty = True  # the throttle only renders when dirty
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _on_tool_started(self, name: str):
        # Close the streaming block (model text preceding tool calls is
        # folded into the card) and add the call to this turn's card.
        self._render_timer.stop()
        self._flush_render()
        self._close_live_thought()
        self._close_remark()
        self._fold_open_text()
        card = self._ensure_card()
        step = Step(name, summary=f"{name} …")
        card.steps.append(step)
        self._active_step = step
        self._tool_t0 = time.monotonic()
        self._set_status(f"Running {name}")
        self._render()

    def _refresh_view_live(self, tool_label: str) -> None:
        """Throttled fit during a turn (follow-the-build, ``panel_follow``: never when the user moved the camera
        since the last fit) — only from the GUI thread (this is a slot on the worker's signals, already queued
        here)."""
        now = time.monotonic()
        if now - getattr(self, "_last_live_fit", 0.0) < 0.8:
            return
        self._last_live_fit = now
        self._fit_camera()

    def _refresh_credits_live(self) -> None:
        """During a turn on the OriginShape account, refresh the credits badge at most every 5 s: each tool call
        follows a model request the server has already settled, so the badge follows the spend as Oscar works
        (the end of the turn still forces a final refresh)."""
        sent = getattr(self, "_turn_sent", None)
        if not (sent and sent.get("account")):
            return
        now = time.monotonic()
        if now - getattr(self, "_last_live_credits", 0.0) < 5.0:
            return
        self._last_live_credits = now
        self._refresh_account(force=True, render=False)

    def _on_tool_finished(self, summary: str, ok: bool, result_text: str):
        self._close_live_thought()
        self._refresh_credits_live()
        step = self._active_step
        self._active_step = None
        dur = time.monotonic() - self._tool_t0 if self._tool_t0 else 0.0
        self._tool_t0 = 0.0
        if step is None:
            step = Step(summary or "tool")
            if self._turn_card is not None:
                self._turn_card.steps.append(step)
        step.summary = summary or "done"
        rich = step.view is not None and not step.view.running
        if rich:
            # step_finished already filled the row (full output, code, delta, health, thumbnail): the legacy
            # signal only completes what is still missing and never replaces the detail
            if step.ok is None:
                step.ok = ok
            if not step.dur:
                step.dur = dur
        else:
            step.ok = ok
            step.dur = dur
            step.detail = result_text.rstrip()
            step.expanded = not ok  # failures open with full context
        if ok:
            self._turn_used_cad = True
        self._set_status("Thinking")
        name = summary.split("(", 1)[0].strip()
        if ok and name in self._MUTATING_TOOLS:
            self._turn_mutated = True
            self._refresh_view_live(name)
        self._render()

    # -- new signals -----------------------------------------------------------------------------------

    def _unique_payload(self, payload) -> dict | None:
        """The decoded step payload with its ``call_id`` replaced by the turn-unique key (``None`` when the
        payload is not a JSON object)."""
        try:
            data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        data = dict(data)
        data["call_id"] = unique_call_id(data, self._turn_seq)
        return data

    def _on_plan_changed(self, payload) -> None:
        view = self._ensure_turn_view()
        if not view.on_plan(payload):
            return
        session = self._current_session()
        if session is not None:
            session["ledger"] = dict(view.plan)   # the plan the next turn carries over
        if self._artifacts is None:               # the first plan of a work turn: the project folder exists now
            self._artifacts = self._artifacts_of(session)
            self._refresh_artifacts_strip()
        self._refresh_plan_strip()
        self._persist_soon()

    def _on_step_started(self, payload) -> None:
        data = self._unique_payload(payload)
        if data is None:
            return
        view = self._ensure_turn_view().on_step_started(data)
        if view is None:
            return
        step = self._active_step
        if step is None or step.view is not None:
            # a worker that did not emit tool_started first, or a stale row of the previous call
            self._on_tool_started(view.name or "tool")
            step = self._active_step
        step.view = view
        step.call_id = view.call_id
        self._steps_by_call[view.call_id] = step
        self._act_now_armed = True
        self._set_status(f"Running {view.label}")
        self._render()

    def _on_step_finished(self, payload) -> None:
        data = self._unique_payload(payload)
        if data is None:
            return
        view = self._ensure_turn_view().on_step_finished(data)
        if view is None:
            return
        step = self._steps_by_call.get(view.call_id)
        if step is None:
            step = self._active_step if (self._active_step is not None and self._active_step.view is None) else None
            if step is None:
                step = Step(view.name or "tool", summary=view.label)
                self._ensure_card().steps.append(step)
            self._steps_by_call[view.call_id] = step
        step.view = view
        step.call_id = view.call_id
        step.summary = view.label
        step.ok = bool(view.ok)
        step.dur = view.dur_s
        step.detail = ""
        step.expanded = view.default_expanded   # a failed step opens with its full output
        if view.snapshot_path and self._thumbs.load(view.call_id, self._abs_snapshot(view.snapshot_path)) is None:
            self._mark_snapshot_missing(view)
        if view.ok and view.mutating and not view.skipped:
            self._turn_mutated = True
            self._turn_used_cad = True
            self._follow_step(view)
        self._render()

    def _on_progress(self, payload) -> None:
        view = self._ensure_turn_view()
        if not view.on_progress(payload):
            return
        self._update_heartbeat()
        self._refresh_plan_strip()   # round / phase in the strip header move with every round

    def _on_round_reset(self, reason: str = "") -> None:
        """The partial reply of the current round is void (retry, dropped stream, watchdog, gate): remove the
        live streamed text so the retry does not append to it."""
        if self._turn_view is not None:
            self._turn_view.on_round_reset(reason)
        self._act_now_armed = True
        self._close_live_thought()
        card = self._turn_card
        if card is not None and card.steps and card.steps[-1].plain and card.steps[-1].live:
            card.steps.pop()
        if self._current is not None:
            if self._current in self.blocks:
                self.blocks.remove(self._current)
            self._current = None
        self._stream_buffer = ""
        self._dirty = True
        self._flush_render()

    def _on_artifacts_changed(self, payload) -> None:
        view = self._ensure_turn_view()
        if not view.on_artifacts(payload):
            return
        self._artifacts = view.artifacts
        session = self._current_session()
        root = str((view.artifacts or {}).get("project_dir") or "")
        if session is not None and root:
            session["project_dir"] = root
        self._refresh_artifacts_strip()

    def _on_turn_outcome(self, outcome: str, summary: str) -> None:
        """Just BEFORE ``turn_finished``: keep the outcome and the Build Summary for the card."""
        self._turn_outcome = (str(outcome or ""), str(summary or ""))
        self._ensure_turn_view().on_outcome(outcome, summary)

    def _on_account_problem(self, payload) -> None:
        """Just before ``turn_outcome``: the turn failed for an account reason (JSON ``{"code", "kind", "status",
        "message", "retry_after", "data", "first_request"}``); ``_on_turn_finished`` shows it."""
        try:
            data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
        except (TypeError, ValueError):
            data = None
        self._turn_account = data if isinstance(data, dict) else None

    def _give_back_turn(self, sent: dict | None) -> bool:
        """The turn's very first request failed for an account reason: nothing was generated, so the message leaves
        the chat and goes back into the input (with its attachments, images and the notes it carried) and the plan is
        what it was before. ``sent`` is what ``_send`` recorded. Not when the turn already carried a steered message
        (those bubbles stay), and not for a workspace turn: its message came from the browser, which shows the
        problem, and FreeCAD's input box was never involved. True when done."""
        message = sent.get("message") if sent else None
        if sent and sent.get("workspace"):
            return False
        if message is None or not self.history or self.history[-1] is not message or self._queued_blocks:
            return False
        self.history.pop()
        bubble = sent.get("bubble")
        if bubble in self.blocks:
            self.blocks.remove(bubble)
        typed = self.input.toPlainText().strip()
        request = str(sent.get("typed") or "")
        if request:
            self.input.setPlainText(f"{request}\n\n{typed}" if typed else request)
        self.pending = list(dict.fromkeys(list(sent.get("pending") or []) + list(self.pending)))
        self._refresh_attach_bar()
        self._carry_images = list(sent.get("carried") or []) + list(self._carry_images)
        notes = list(sent.get("notes") or [])
        if notes:
            self._pending_notes = notes + [n for n in self._pending_notes if n not in notes]
        session = self._current_session()
        if session is not None:
            if sent.get("had_ledger"):
                session["ledger"] = sent.get("ledger")
            else:
                session.pop("ledger", None)
        return True

    # ------------------------------------------------------------------
    # end of the turn
    # ------------------------------------------------------------------

    def _fit_view_after_turn(self, force: bool = False):
        """Frame the model once when a turn that ran CAD tools ends.

        Mid-turn captures no longer move the user's camera (views.capture
        restores it), so the zoom the user kept through the turn may
        predate the new geometry. One fit — in the user's own
        orientation — shows the finished result, unless the user moved the camera since the last fit or turned
        "Follow the build" off (``force`` = an explicit user action such as opening a file or a rewind).
        Pure-chat turns never touch the camera at all.
        """
        self._fit_camera(force)

    def _adopt_worker_ledger(self, worker) -> None:
        """The plan the next turn carries: the worker's live ledger after a WORK turn; after a question /
        inspect / other turn the previous plan stays as it was (the worker's ledger is a throwaway then)."""
        ledger = getattr(worker, "ledger", None)
        session = self._current_session()
        if ledger is None or session is None:
            return
        try:
            if ledger.is_work_turn():
                session["ledger"] = ledger.to_dict()
        except Exception as e:  # noqa: BLE001
            self._log_panel_error("adopt_ledger", e)

    def _on_turn_finished(self, ok: bool, text: str):
        self._render_timer.stop()
        self._heartbeat_timer.stop()
        self._close_live_thought()
        outcome, summary = self._turn_outcome
        view = self._turn_view
        if view is not None and not view.finished:
            view.finish()
        # turn_finished is emitted from INSIDE run(); nulling the panel's
        # reference before the thread exits lets Python destroy the QThread
        # wrapper mid-run — Qt then aborts the whole app ("QThread:
        # Destroyed while thread is still running"). Hold the reference
        # until the thread has fully stopped, then queue C++ cleanup.
        worker = self.worker
        self.worker = None
        unsent: list[dict] = []
        if worker is not None:
            worker.wait()
            self._adopt_worker_ledger(worker)
            drain = getattr(worker, "drain_unsent", None)
            if callable(drain):  # steered messages that arrived too late to be delivered
                try:
                    unsent = list(drain() or [])
                except Exception:
                    unsent = []
            worker.deleteLater()
        if self._turn_used_cad:
            self._fit_view_after_turn()
        # The Build Summary is rendered once, as its own card: cut it from the text (invariant I2 puts it
        # after the model's commentary) and prefer the block turn_outcome carried.
        answer, text_summary = split_build_summary(text)
        summary = summary.strip() or text_summary
        # Collapse the turn's process card into its summary.
        card = self._turn_card
        self._turn_card = None
        if ok and answer and card is not None and card.steps:
            last = card.steps[-1]
            if last.plain and last.live:
                # it was the streaming preview of the final answer — the
                # answer message below supersedes it
                card.steps.pop()
        if card is not None:
            total = max(0.0, time.monotonic() - card.t0) if card.t0 else 0.0
            thoughts = [s for s in card.steps if s.is_thought]
            tools = [s for s in card.steps if not s.is_thought and not s.plain]
            failed = sum(1 for s in tools if s.ok is False)
            card.text = header_text(thought_s=sum(s.dur for s in thoughts) if thoughts else None, steps=len(tools),
                                    total_s=total, failed=bool(failed), outcome=outcome)
            card.header_kind = outcome_badge(outcome)[1] if outcome else ""
            card.t0 = 0.0
            # Offer a revert only when a mutating tool really succeeded AND the turn-start snapshot can
            # drive it (a card whose link can only refuse is worse than no link).
            card.revertible = self._turn_mutated and self._snapshot_usable(card.snapshot)
            if not failed:
                card.expanded = False
        problem, self._turn_account = self._turn_account, None
        sent, self._turn_sent = self._turn_sent, None
        given_back = bool(not ok and problem and problem.get("first_request")) and self._give_back_turn(sent)
        card_kind = account_problem_card(problem) if (problem and not ok) else ""
        if not ok and answer:
            if card_kind:   # the account card says what to do; the raw "Provider error: HTTP 402 ..." line does not
                self._show_account_card(card_kind, returned=given_back)
            else:
                self._append(Block.ERROR, answer)
                if given_back:
                    self._on_notice("Your message is back in the input: send it again when you are ready.")
        elif ok and answer:
            if self._current is not None and not self._current.text.strip():
                self._current.text = answer
            elif self._current is None:
                # turn ended without any streamed text (e.g. tool-only turn)
                self._append(Block.ASSISTANT, answer)
        self._current = None
        actions = [("continue:", "Continue")] if outcome == "capped" else []
        if summary or actions:
            self._append(Block.SUMMARY, summary).actions = actions
        if given_back:
            self._persist_current()   # the chat without the message; no turn record: nothing happened
        else:
            self._remember_turn(outcome, card)
        self._update_revert_flags()
        self._set_status("")
        self._set_busy(False)
        self._settle_queued(unsent, ok)
        self._finish_turn_widgets()
        if ok and answer and not given_back:
            self._maybe_offer_rating(answer, outcome, card)
        self._render()
        if problem or (sent and sent.get("account")):
            self._refresh_account(force=True)   # credits were spent, or the account is why the turn failed

    def _maybe_offer_rating(self, answer: str, outcome: str, card: Block | None) -> None:
        """Now and then, after a substantial turn, the "How did Oscar do?" card (``rating_gui``)."""
        try:
            from . import rating_gui

            rating_gui.maybe_offer(self, answer, outcome, card)
        except Exception as e:  # noqa: BLE001 - a rating card is never a reason for a turn to fail
            self._log_panel_error("rating_offer", e)

    def _remember_turn(self, outcome: str, card: Block | None) -> None:
        """Append the finished turn's compact record to the chat's ``blocks_meta`` and persist the chat."""
        session = self._current_session()
        view = self._turn_view
        if session is None:
            return
        if view is not None and (view.steps or outcome not in ("", "answered")):
            project = session.get("_project")
            root = getattr(project, "root", "") or session.get("project_dir") or ""
            meta = make_turn_meta(self._turn_request, outcome, card.text if card is not None else "",
                                  view.steps, root=root)
            session["blocks_meta"] = trim_turn_metas(list(session.get("blocks_meta") or []) + [meta])
        manifest = getattr(session.get("_project"), "manifest", None)
        note = str(manifest.get("storage_note") or "") if isinstance(manifest, dict) else ""
        if note and session.get("_storage_note") != note:
            session["_storage_note"] = note
            self._on_notice(note)
        self._persist_current()

    def _finish_turn_widgets(self) -> None:
        """Strips and heartbeat after a turn: the plan stays pinned with its final state, the artifacts are
        re-read from the project folder (exports, NOTES.md and rebuild.py were written at the end)."""
        self._act_now_armed = True
        fresh = self._artifacts_of(self._current_session())
        if fresh is not None:
            self._artifacts = fresh
        self._refresh_artifacts_strip()
        self._refresh_plan_strip()
        self._update_heartbeat()

    def _settle_queued(self, unsent: list[dict], ok: bool) -> None:
        """End of turn for the steered ("queued") bubbles: delivered ones become plain bubbles; the ones
        the worker could not deliver (``unsent``) leave the chat and are re-sent as the next turn after a
        normal finish, or put back into the input when the turn was stopped / failed (a stop must never
        start new work by itself). Images of an undelivered message travel with it.

        An undelivered item is matched to its bubble by what was handed to the worker (``Block.steer_key``: the
        typed text PLUS the attachment block when an image rode along), not by what the bubble shows, and it is
        re-sent as the typed text (``Block.steer_text``): the image travels as an image, never as the attachment
        note, and no second bubble appears."""
        items = [i for i in unsent if isinstance(i, dict) and str(i.get("text", "")).strip()]
        images = [img for i in items for img in (i.get("images") or []) if img]
        waiting = list(self._queued_blocks)
        texts: list[str] = []
        for item in items:
            key = str(item.get("text", "")).strip()
            block = next((b for b in waiting if (b.steer_key or b.text.strip()) == key), None)
            if block is not None:
                waiting.remove(block)
                if block in self.blocks:
                    self.blocks.remove(block)  # the re-send (or the input box) shows it again
            texts.append((block.steer_text.strip() if block is not None else "") or key)
        for block in self._queued_blocks:
            block.queued = False
        self._queued_blocks = []
        if not texts:
            return
        joined = "\n\n".join(texts)
        if ok:
            if images:
                QtCore.QTimer.singleShot(0, lambda: self._send_text(joined, images=images))
            else:
                QtCore.QTimer.singleShot(0, lambda: self._send_text(joined))
        else:
            self._carry_images = list(images)
            if not self.input.toPlainText().strip():
                self.input.setPlainText(joined)
            self._on_notice("Your queued message was not delivered; it is back in the input.")

    def _send_text(self, text: str, images=()) -> None:
        """Send ``text`` as a new turn without losing what the user is typing meanwhile."""
        if self.worker is not None:
            return
        typed = self.input.toPlainText()
        self.input.setPlainText(text)
        if images:
            self._carry_images = list(images)
        self._send()
        if typed.strip() and not self.input.toPlainText().strip():
            self.input.setPlainText(typed)

    def _continue_turn(self) -> None:
        """``continue:`` link (after the round cap): send the plan's resume prompt as the next message."""
        if self._busy():
            self._set_status("Wait for the current turn to finish first.")
            return
        self._send_text(self._resume_text())
