"""The Oscar chat panel — a dock widget beside the 3D view.

Pure QtWidgets (the application bundles no QtWebEngine). The conversation is
kept as render blocks and re-rendered as rich text with a light markdown
subset; streaming text updates are throttled through a timer so a fast token
stream does not re-render per chunk.

Layout: ``panel_render`` holds every pure (Qt-free) piece of presentation
logic (markdown, plan strip, step rows, heartbeat, badges, ``TurnView``),
``panel_blocks`` the ``Block`` / ``Step`` objects and the log browser,
``panel_session`` the task chats and their per-chat build state (Project,
plan, restore card), ``panel_turn`` what the worker's signals do to the screen
(plan strip, step rows, heartbeat, artifacts strip, outcome, Build Summary),
``panel_anchors`` every link, ``panel_revert`` the turn revert (undo stack, checkpoint
fallback), ``panel_rewind`` the per-step rewind and ``panel_follow`` follow-the-build;
this module is the widget itself and re-exports their public names.

Widgets located by object name (GUI tests): ``oscarLog``, ``oscarPlanStrip``,
``oscarArtifactsStrip``, ``oscarHeartbeat``, ``oscarStopButton``,
``oscarActNowButton``, ``oscarMenuButton`` (header menu with the ``oscarFollowAction`` toggle),
``oscarAccountBadge`` (the OriginShape account badge, menu ``oscarAccountMenu``).

The OriginShape account: the badge before ⚙ reads *Sign in* / *42 credits* / *0 credits* (warning colour) /
*Waitlist* and carries the account menu; it shows for the account profile or a sign-in. Its state comes from the
process-wide ``settings_ui.AccountHub`` (``on_change`` of the sign-in plus a coalesced background ``/v1/me``), refreshed
when the panel opens, on a profile change, after Settings closes, at the end of a turn and on sign-in; nothing here
touches the network. The send gate uses local facts only: the account profile without a sign-in gets the onboarding
card instead of a turn, a waitlisted account (fresh snapshot) the waitlist card; everything else goes to the server,
which decides about credits. A failed turn's ``account_problem`` (``AgentWorker``) shows the matching card instead of
the raw provider error, and when the turn's very first request failed the typed message goes back to the input.

Dock geometry follows the Help workbench pattern: persisted into user
parameters on dockLocationChanged, restored on creation, deduplicated by
objectName via findChild.
"""

import base64
import copy
import html
import os
import re
import threading
import time

import FreeCAD as App
import FreeCADGui as Gui
from PySide import QtCore, QtGui, QtWidgets

from . import __version__
from . import attachments as _attach
from . import panel_chats, panel_markup, paths
from . import params as _params
from . import settings_ui
from .agent import AgentWorker
from .panel_anchors import AnchorMixin
from .panel_blocks import (  # noqa: F401 - re-exported: tests and gui_smoke import these from oscar.panel
    _BLOCK_SEQ, _DETAIL_LIMIT, Block, LogBrowser, Step, ThumbCache, _accent, _font_size, _fg, _muted, _text_color)
from .panel_follow import FollowMixin
from .panel_render import (  # noqa: F401 - md_to_html stays importable from oscar.panel
    classify_file_link, md_to_html, TurnView, account_badge, account_card, account_menu)
from .panel_revert import RevertMixin
from .panel_rewind import RewindMixin
from .panel_session import WELCOME, SessionMixin  # noqa: F401 - WELCOME stays importable from oscar.panel
from .panel_turn import _GENERIC_STATUS, TurnMixin  # noqa: F401
from .providers import Message

DOCK_OBJECT_NAME = "OscarPanel"
_QUIT_WAIT_MS = 4000   # how long ``aboutToQuit`` waits for a running turn to end
_WAITLIST_FRESH_S = 300.0   # the send gate believes a "waitlisted" snapshot this young; the server decides after that


# ---------------------------------------------------------------------------
# The panel


def _scroll_target_for(value: int, maximum: int, tol: int = 8):
    """What the chat log should hold on to after the user moves the scrollbar: ``None`` = stay glued to the end
    (streaming follows), otherwise the pixel offset to keep."""
    return None if value >= maximum - tol else value


_STATIC_MUTATING = frozenset({
    "create_object", "edit_object", "delete_object", "import_file",
    "insert_part_from_library", "execute_code", "apply_feature_plan",
    "reapply_plan_feature", "run_fem_analysis",
})


def _workspace_message_id(workspace) -> str:
    """The OriginShape workspace message a connect-mode turn answers (its ``GuiTurn``'s job), or ""."""
    job = getattr(getattr(workspace, "turn", None), "job", None)
    return str(getattr(job, "message_id", "") or "")


def _mutating_tool_names() -> set[str]:
    """Tools whose success means the MODEL changed: the registry's ``mutating_names()`` when the tool
    registry offers it, always including the historical static list."""
    names = set(_STATIC_MUTATING)
    try:
        from . import tools as _tools

        names |= set(getattr(_tools, "mutating_names", lambda: ())())
    except Exception:
        pass
    return names


class OscarPanel(SessionMixin, TurnMixin, AnchorMixin, RevertMixin, RewindMixin, FollowMixin, QtWidgets.QWidget):
    """Chat content of the dock; the dock wrapper lives below."""

    # netguard's permission-wait announcements (any thread → GUI thread)
    _netguard_wait = QtCore.Signal(bool)

    # Tools whose success means the MODEL changed — after these the viewport refreshes so the user watches
    # the part take shape piece by piece (a human draws layer by layer; a single end-of-turn reveal reads as
    # "everything dumped at once"). Derived from the tool registry (``tools.mutating_names()``) plus the
    # historical static list.
    _MUTATING_TOOLS = _mutating_tool_names()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.blocks: list[Block] = []
        self.history: list[Message] = []
        self.worker: AgentWorker | None = None
        self.pending: list[str] = []  # attachment paths waiting to be sent
        self._stream_buffer = ""
        self._current: Block | None = None
        self._dirty = False
        self._welcome: Block | None = None
        self._status = ""  # "Thinking…" / "Running x…" while a turn is live
        self._status_dots = 0
        self._turn_card: Block | None = None  # this turn's process card
        self._turn_snapshot: dict = {}  # doc undo depths before the turn
        self._turn_used_cad = False  # any successful CAD tool ran this turn
        self._paused = False
        self._building = False  # the user message is being assembled (a PDF is read on a helper thread)
        self._active_step: Step | None = None
        self._tool_t0 = 0.0
        self._turn_mutated = False  # a mutating tool succeeded this turn (gates "revert changes")
        self._queued_blocks: list[Block] = []  # steered user bubbles waiting for delivery
        self._init_follow()  # follow-the-build preference and camera bookkeeping (panel_follow)
        self._init_turn_state()  # TurnView, plan / artifacts strips, thumbnails (panel_turn)
        self._frag_cache: dict[int, tuple[tuple, str]] = {}  # bid -> (signature, html) of settled blocks
        self._frag_style = ("", "")
        # Scroll bookkeeping across setHtml (Qt lays long documents out lazily, so the scroll range
        # keeps growing for a few ms after setHtml): see _render / _on_range_changed.
        self._scroll_mode = ""      # "" | "follow" (stay at the bottom) | "hold" (return to _scroll_target)
        self._scroll_target = 0
        self._scroll_set = 0        # the value this code last set; a different value means the user moved
        self._scroll_blocks = None  # the chat (blocks list) the scroll state belongs to
        # Task chats: multiple conversations, switchable, persisted. The
        # live session's history IS self.history (the worker holds that
        # list object for the whole turn — switching is blocked while busy).
        self.sessions: list[dict] = []
        self._session_loading = False
        # Facts the model must learn with the next message (e.g. the user
        # reverted a turn — otherwise it would keep building on objects
        # that no longer exist). Kept PER SESSION (see ``_pending_notes``).
        self._orphan_notes: list[str] = []

        self._build_ui()
        self._register_netguard_hook()
        self._register_quit_hook()
        self._init_account()
        self._append_empty_state()
        self._init_project_links()
        self._load_sessions()
        self._render()
        self._init_workspace()

    def _init_project_links(self) -> None:
        """Paths inside an Oscar project read as an "Oscar project" link in the chat, never the full folder."""
        try:
            from . import project_paths

            project_paths.install()
        except Exception as e:  # noqa: BLE001 - the chat still renders, with plain paths
            self._log_panel_error("project_links", e)

    def _init_workspace(self) -> None:
        """Connect mode (``oscar.connect_gui``): with its setting on, the OriginShape workspace's tasks run in this
        panel. Created after the netguard hook, which it wraps. A failure here never keeps the panel from opening."""
        self._workspace = None
        try:
            from .connect_gui import WorkspaceConnector

            self._workspace = WorkspaceConnector(self)
        except Exception as e:  # noqa: BLE001
            self._log_panel_error("workspace", e)

    def _register_netguard_hook(self):
        """While a network-permission dialog is pending, the status line
        must say WHY the turn is paused — a silent spinner for two minutes
        reads as a hang (that is exactly how the first permission request
        was lost).

        netguard announces from the agent's WORKER thread; widgets may only
        be touched on the GUI thread, so the hook just emits a signal (a
        queued, thread-safe hop) and the slot updates the status."""
        self._netguard_wait.connect(self._on_netguard_wait)
        try:
            from . import netguard

            netguard.on_permission_wait = self._netguard_wait.emit
        except Exception:
            pass

    def _register_quit_hook(self) -> None:
        """Quitting while a turn runs must not destroy a running ``QThread`` ("QThread: Destroyed while thread is
        still running" aborts the process): ``aboutToQuit`` stops the workers first."""
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown)

    def _shutdown(self) -> None:
        """The application is quitting (``aboutToQuit``): stop connect mode, ask the running turn to stop and give its
        thread a few seconds to end (it polls its stop flag between provider chunks and tool calls). Never raises."""
        workspace = getattr(self, "_workspace", None)
        if workspace is not None:
            try:
                workspace.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self._shutdown_workers(_QUIT_WAIT_MS)

    def _shutdown_workers(self, timeout_ms: int = _QUIT_WAIT_MS) -> None:
        """Stop and join the agent thread. A QThread still running when Qt tears down aborts the
        whole process ("QThread: Destroyed while thread is still running"), so a quit mid-turn would crash instead
        of exiting.

        A paused turn is released first (it only polls the flag it holds, and ``resume`` is idempotent), then told to
        stop. A worker with ``isRunning`` is joined in short slices while the GUI event loop keeps running: tool
        calls hop to the GUI thread through a queued signal (cad/dispatch.py), so a blocking ``wait`` here would
        leave a worker that sits in one stuck for the whole dispatch timeout. A worker that cannot be polled gets one
        blocking ``wait``. Never raises: a destroyed C++ thread object (``RuntimeError``) or any other failure of one worker is
        tolerated, the quit goes on."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        # snapshot: pumping events below can deliver _on_turn_finished, which clears the panel's own references
        for worker in (getattr(self, "worker", None),):
            if worker is None:
                continue
            try:
                resume = getattr(worker, "resume", None)
                if resume is not None:
                    resume()   # also cancels a pause that was requested but not yet reached
                worker.stop()
                running = getattr(worker, "isRunning", None)
                if running is None:
                    worker.wait(timeout_ms)
                    continue
                while running() and time.monotonic() < deadline:
                    QtWidgets.QApplication.processEvents()
                    worker.wait(50)
                if running():
                    App.Console.PrintWarning("Oscar: a worker thread did not stop before quit\n")
            except RuntimeError:
                pass  # the C++ thread object is already gone
            except Exception as e:  # noqa: BLE001 - the application is going away anyway
                log = getattr(self, "_log_panel_error", None)
                if callable(log):
                    log("shutdown", e)

    def _on_netguard_wait(self, waiting: bool):
        if waiting:
            self._set_status("Waiting for your approval — see the "
                             "Oscar permission dialog…")
        else:
            self._set_status("Working…")

    # -- UI construction ---------------------------------------------------

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Header: title + settings (the model picker lives at the input)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Oscar")
        title.setStyleSheet("font-weight:bold;")
        header.addWidget(title)
        subtitle = QtWidgets.QLabel("copilot")
        subtitle.setStyleSheet(f"color:{_muted()};font-size:small;padding-right:6px;")
        subtitle.setTextFormat(QtCore.Qt.RichText)    # "copilot", or a link when a newer Oscar is out
        subtitle.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        subtitle.linkActivated.connect(lambda _link: self._on_account_action("download"))
        header.addWidget(subtitle)
        self.subtitle = subtitle          # hidden while the account badge shows: the 280 px dock has no room for both
        header.addStretch(0)

        # Task chats: switcher + new + clear (each task keeps its own context)
        self.chat_combo = QtWidgets.QComboBox()
        self.chat_combo.setMinimumWidth(90)
        self.chat_combo.setToolTip("Task chats — switch between separate "
                                   "conversations")
        self.chat_combo.activated.connect(self._on_chat_picked)
        header.addWidget(self.chat_combo, 1)
        new_chat_btn = QtWidgets.QToolButton()
        new_chat_btn.setText("＋")
        new_chat_btn.setToolTip("New chat (fresh task context)")
        new_chat_btn.clicked.connect(self._new_chat)
        header.addWidget(new_chat_btn)
        clear_chat_btn = QtWidgets.QToolButton()
        clear_chat_btn.setText("🗑")
        clear_chat_btn.setToolTip("Clear this chat's messages (the chat "
                                  "itself stays)")
        clear_chat_btn.clicked.connect(self._clear_chat)
        header.addWidget(clear_chat_btn)
        del_chat_btn = QtWidgets.QToolButton()
        del_chat_btn.setText("✕")
        del_chat_btn.setToolTip("Delete this chat")
        del_chat_btn.clicked.connect(self._delete_chat)
        header.addWidget(del_chat_btn)
        self.del_chat_btn = del_chat_btn

        # The OriginShape account: "Sign in" / "42 credits" / "0 credits" / "Waitlist", with the account menu
        self.account_btn = QtWidgets.QToolButton()
        self.account_btn.setObjectName("oscarAccountBadge")
        self.account_btn.setAutoRaise(True)
        self.account_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.account_menu = QtWidgets.QMenu(self.account_btn)
        self.account_menu.setObjectName("oscarAccountMenu")
        self.account_btn.setMenu(self.account_menu)
        self.account_btn.hide()           # shown for the account profile or a sign-in (_apply_account_state)
        header.addWidget(self.account_btn)

        settings_btn = QtWidgets.QToolButton()
        settings_btn.setText("⚙")
        settings_btn.setToolTip("Settings (OriginShape account, providers, models, API keys)")
        settings_btn.clicked.connect(lambda _checked=False: self._open_settings())
        header.addWidget(settings_btn)
        menu_btn = QtWidgets.QToolButton()
        menu_btn.setObjectName("oscarMenuButton")
        menu_btn.setText("⋯")
        menu_btn.setToolTip("Panel options")
        menu_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.header_menu = QtWidgets.QMenu(menu_btn)
        self.header_menu.setObjectName("oscarHeaderMenu")
        self.follow_action = self.header_menu.addAction("Follow the build")
        self.follow_action.setObjectName("oscarFollowAction")
        self.follow_action.setCheckable(True)
        self.follow_action.setChecked(self._follow_build)
        self.follow_action.setToolTip("Highlight what each step creates and keep the model framed, "
                                      "unless you moved the camera")
        self.follow_action.toggled.connect(self._set_follow_build)
        self.header_menu.addSeparator()
        rename_action = self.header_menu.addAction("Rename chat…")
        rename_action.setObjectName("oscarRenameChatAction")
        rename_action.triggered.connect(lambda _checked=False: panel_chats.rename(self))
        search_action = self.header_menu.addAction("Search chats…")
        search_action.setObjectName("oscarSearchChatsAction")
        search_action.setToolTip("Find a chat by its title, what you asked or what Oscar answered")
        search_action.triggered.connect(lambda _checked=False: panel_chats.search(self))
        menu_btn.setMenu(self.header_menu)
        header.addWidget(menu_btn)
        root.addLayout(header)

        # Pinned plan strip (above the log, never auto-collapses), chat log, artifacts strip, heartbeat row
        self._build_flow_widgets()
        root.addWidget(self.plan_strip)

        # Chat log
        self.log = LogBrowser()
        self.log.setObjectName("oscarLog")
        self.log.setOpenLinks(False)  # every anchor goes through _on_anchor (web / file links via QDesktopServices)
        self.log.setOpenExternalLinks(False)
        self.log.anchorClicked.connect(self._on_anchor)
        self.log.verticalScrollBar().rangeChanged.connect(self._on_range_changed)
        root.addWidget(self.log, 1)
        root.addWidget(self.artifacts_strip)
        root.addWidget(self.heartbeat_row)

        # Pending attachments (chips; hidden while empty)
        self.attach_bar = QtWidgets.QWidget()
        attach_layout = QtWidgets.QHBoxLayout(self.attach_bar)
        attach_layout.setContentsMargins(0, 0, 0, 0)
        attach_layout.setSpacing(4)
        self.attach_bar.hide()
        root.addWidget(self.attach_bar)

        # Input area: one rounded container (z.ai-style) holding the text
        # edit and the controls row — attach, model picker, key hint.
        input_frame = QtWidgets.QFrame()
        input_frame.setStyleSheet(
            "QFrame { background: rgba(128,128,128,0.08);"
            " border: 1px solid rgba(128,128,128,0.25); border-radius: 14px; }"
        )
        input_box = QtWidgets.QVBoxLayout(input_frame)
        input_box.setContentsMargins(10, 6, 10, 6)
        input_box.setSpacing(4)

        self.input = QtWidgets.QPlainTextEdit()
        self.input.setPlaceholderText(
            "Ask Oscar… (Enter to send · Shift+Enter for a new line · Esc stops)"
        )
        self.input.setFixedHeight(64)
        self.input.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.input.setStyleSheet("QPlainTextEdit { background: transparent; border: none; }")
        self.input.installEventFilter(self)
        input_box.addWidget(self.input)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)
        self.attach_btn = QtWidgets.QToolButton()
        self.attach_btn.setText("📎")
        self.attach_btn.setToolTip(
            "Attach files for Oscar — images (the model sees them), text/CSV/JSON "
            "(inlined into the prompt), STEP/STL/DXF/FCStd… (imported as geometry)"
        )
        self.attach_btn.clicked.connect(self._pick_files)
        controls.addWidget(self.attach_btn)
        # The pen: draw on the 3D view where Oscar should change the model (panel_markup)
        self.markup_btn = QtWidgets.QToolButton()
        self.markup_btn.setObjectName("oscarMarkupButton")
        self.markup_btn.setText("✏️")
        self.markup_btn.setToolTip("Mark up the model — draw on the 3D view where Oscar should change it; "
                                   "the picture and the region go with your message")
        self.markup_btn.clicked.connect(self._mark_up)
        controls.addWidget(self.markup_btn)

        # Prompt templates (inserted into the input — copyable & editable
        # before sending). The parametric one is the canonical
        # feature-tree build request; "revert & re-run" redoes the last
        # build cleanly with it.
        self.templates_btn = QtWidgets.QToolButton()
        self.templates_btn.setText("📋")
        self.templates_btn.setToolTip("Prompt templates — click to insert "
                                      "into the input")
        self.templates_btn.clicked.connect(self._show_templates)
        controls.addWidget(self.templates_btn)

        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.setMinimumWidth(110)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_picked)
        controls.addWidget(self.profile_combo)
        controls.addStretch(1)
        # The model of the active profile sits where a hint used to be.
        self.model_label = QtWidgets.QLabel("")
        self.model_label.setStyleSheet(f"color:{_muted()};font-size:small;")
        controls.addWidget(self.model_label)

        # Stop: visible only while a turn runs; same
        # effect as Esc.
        self.stop_btn = QtWidgets.QPushButton("■")
        self.stop_btn.setObjectName("oscarStopButton")
        self.stop_btn.setFixedWidth(34)
        self.stop_btn.setToolTip("Stop the running turn (Esc)")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setStyleSheet(
            "QPushButton { background:#e06c5a; color:white; border:none;"
            " border-radius:8px; padding:4px 0; font-weight:600; }"
        )
        self.stop_btn.hide()
        controls.addWidget(self.stop_btn)

        # Enter sends; the trailing button mirrors the flow state
        # (send -> pause -> resume). Esc cancels a running turn.
        self.send_btn = QtWidgets.QPushButton("➤")
        self.send_btn.setFixedWidth(42)
        self.send_btn.setToolTip("Send (Enter)")
        self.send_btn.clicked.connect(self._on_flow_button)
        self.send_btn.setStyleSheet(
            f"QPushButton {{ background:{_accent()}; color:white; border:none;"
            f" border-radius:8px; padding:4px 0; font-weight:600; }}"
            f" QPushButton:disabled {{ background:palette(mid);"
            f" color:palette(window-text); }}"
        )
        controls.addWidget(self.send_btn)
        input_box.addLayout(controls)
        root.addWidget(input_frame)

        # Esc stops a running turn from ANYWHERE in the panel (the input's own
        # key filter only saw it while the input had focus). Enabled only
        # while busy, so an idle Esc still reaches the rest of the app.
        shortcut_cls = getattr(QtGui, "QShortcut", None) or getattr(QtWidgets, "QShortcut")
        self._esc_shortcut = shortcut_cls(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        self._esc_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self._esc_shortcut.activated.connect(self._stop)
        self._esc_shortcut.setEnabled(False)

        # Throttled re-render while streaming
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setInterval(100)
        self._render_timer.timeout.connect(self._flush_render)

        # Animated status dots while a turn is live (Thinking… / Running…)
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.setInterval(450)
        self._status_timer.timeout.connect(self._tick_status)

        self._reload_profiles()

    # -- profile / settings --------------------------------------------------

    def _reload_profiles(self):
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        names = [p.name for p in _params.load_profiles()]
        self.profile_combo.addItems(names)
        active = _params.active_profile_name()
        if active in names:
            self.profile_combo.setCurrentIndex(names.index(active))
        self.profile_combo.blockSignals(False)
        profile = _params.active_profile()
        self.model_label.setText(_params.model_display_name(profile))
        self._refresh_account()

    def _on_profile_picked(self, index: int):
        name = self.profile_combo.itemText(index)
        if name:
            try:
                _params.set_active_profile(name)
            except OSError as e:   # the settings file is locked right now: nothing was written, show what is active
                self._log_panel_error("profile", e)
                self._reload_profiles()
                self._flash_status(f"Could not switch the profile: {e}", 6000)
                return
        profile = _params.active_profile()
        self.model_label.setText(_params.model_display_name(profile))
        self._refresh_account()

    def _open_settings(self, account_action: str = ""):
        """The settings dialog (modal). ``account_action`` (``"signin"`` / ``"device"``) starts that sign-in at once.
        A sign-in or sign-out takes effect without Save, so a cancelled dialog refreshes the panel too."""
        dialog = settings_ui.SettingsDialog(self)
        if account_action:
            start = getattr(dialog, "start_account_action", None)
            if callable(start):
                start(account_action)
        if dialog.exec() or getattr(dialog, "account_changed", False):
            self._reload_profiles()
        else:
            self._refresh_account()
        dispose = getattr(dialog, "deleteLater", None)   # a closed dialog must not keep following the account hub
        if callable(dispose):
            dispose()

    # -- the OriginShape account ----------------------------------------------------------------------------

    def _init_account(self) -> None:
        """Follow the process-wide account hub and take its state in (no network here: the hub fetches on a daemon
        thread)."""
        self._account_ready = True
        settings_ui.account_hub().changed.connect(self._on_account_changed)
        self._refresh_account(render=False)

    def _refresh_account(self, force: bool = False, render: bool = True) -> None:
        """Bring the badge, the menu and the empty-state card up to date; the hub reads local facts at once and asks
        the server on its daemon thread (one fetch at a time; ``force`` asks even when the snapshot is young)."""
        if not getattr(self, "_account_ready", False):   # still in __init__ (_build_ui reloads the profiles)
            return
        try:
            hub = settings_ui.account_hub()
            hub.refresh(force=force)
            hub.check_updates()           # a newer Oscar? at most once a day, also for people who never sign in
        except Exception as e:  # noqa: BLE001 - the account is a nicety of the panel, never a reason for it to fail
            self._log_panel_error("account_refresh", e)
        self._apply_account_state(render=render)

    def _on_account_changed(self, _payload: str = "") -> None:
        """The hub's state changed (GUI thread): a sign-in or sign-out, a fresh account snapshot."""
        self._apply_account_state()

    def _account_state(self) -> dict:
        return dict(settings_ui.account_hub().state)

    def _apply_account_state(self, render: bool = True) -> None:
        state = self._account_state()
        try:
            profile = _params.active_profile()
        except Exception:  # noqa: BLE001 - an unreadable settings file reads as "no profile" here
            profile = None
        show = _params.is_account_profile(profile) or bool(state.get("signed_in"))
        text, tone, tooltip = account_badge(state)
        self.account_btn.setText(text)
        self.account_btn.setToolTip(tooltip)
        self.account_btn.setStyleSheet("QToolButton { color: #d9902b; font-weight: 600; }" if tone == "warn" else "")
        self._rebuild_account_menu(state)
        self.account_btn.setVisible(show)
        self.subtitle.setVisible(not show)
        # Without the badge (your own provider), a newer Oscar shows here instead: a link to the download page.
        newer = state.get("update") in ("available", "required") and state.get("latest_version")
        self.subtitle.setText(f"<a href='oscar://account/download'>Oscar {html.escape(str(newer))} is out</a>"
                              if newer else "copilot")
        self.subtitle.setToolTip(f"A newer Oscar is available (this is {__version__}). Click to get it."
                                 if newer else "")
        if self._sync_onboarding(profile, state) and render:
            self._render()

    def _rebuild_account_menu(self, state: dict) -> None:
        menu = self.account_menu
        menu.clear()
        for action, label in account_menu(state):
            if action == "-":
                menu.addSeparator()
                continue
            entry = menu.addAction(label)
            if not action:
                entry.setEnabled(False)
                continue
            entry.setObjectName(f"oscarAccount_{action}")
            entry.triggered.connect(lambda _checked=False, a=action: self._on_account_action(a))

    def _on_account_action(self, action: str) -> None:
        """A badge menu entry or an account link: ``signin``, ``device``, ``buy``, ``subscribe``, ``manage``,
        ``waitlist``, ``download``, ``refresh``, ``signout``. Pages come from the account's own links, never from a
        link's text."""
        if action in ("signin", "device"):
            self._open_settings(account_action=action)
        elif action in ("buy", "subscribe", "manage", "waitlist", "download"):
            links = settings_ui.account_hub().links()
            page = {"manage": "manage", "waitlist": "waitlist", "download": "download"}.get(action, "billing")
            url = links.get(page) or links.get("app") or ""
            if not url:
                self._refresh_account(force=True)
                self._flash_status("Oscar does not know the OriginShape website yet: it is checking your account; try "
                                   "again in a moment.", 6000)
                return
            self._open_external(url)
        elif action == "refresh":
            self._refresh_account(force=True)
            self._flash_status("Checking your OriginShape account…")
        elif action == "signout":
            settings_ui.account_hub().sign_out()
            self._flash_status("Signing out of OriginShape…")

    def _onboarding_reason(self, profile, state: dict) -> str:
        """Why the empty chat should offer the account: ``signed_out`` (the account profile, known to be signed out),
        ``no_key`` (an Anthropic profile without its key), else ``""``. Local facts only."""
        if profile is None:
            return ""
        if _params.is_account_profile(profile):
            known = state.get("known") and state.get("error_code") not in ("invalid_url", "insecure_url")
            return "signed_out" if known and not state.get("signed_in") else ""
        if not getattr(profile, "api_key", "") and getattr(profile, "provider", "") == "anthropic":
            return "no_key"
        return ""

    def _sync_onboarding(self, profile=None, state: dict | None = None) -> bool:
        """The onboarding card of the empty chat: added after the welcome text when ``_onboarding_reason`` asks for
        it (``profile``: the active one when None), removed when it no longer applies. Only the chat on screen carries
        it (``_activate_session`` syncs a chat again when it is shown). True when a block list changed."""
        state = self._account_state() if state is None else state
        if profile is None:
            try:
                profile = _params.active_profile()
            except Exception:  # noqa: BLE001 - an unreadable settings file reads as "no profile" here
                profile = None
        reason = self._onboarding_reason(profile, state)
        changed = False
        lists = [self.blocks] + [s.get("blocks") for s in getattr(self, "sessions", [])]
        seen: set[int] = set()
        for blocks in lists:
            if not isinstance(blocks, list) or id(blocks) in seen:
                continue
            seen.add(id(blocks))
            for block in [b for b in blocks if b.kind == Block.ACCOUNT and b.detail == "onboarding"]:
                if reason and blocks is self.blocks and getattr(block, "reason", "") == reason:
                    continue
                blocks.remove(block)
                changed = True
        welcome = self._welcome
        if reason and welcome is not None and welcome in self.blocks and not any(
                b.kind == Block.ACCOUNT and b.detail == "onboarding" for b in self.blocks):
            card = self._account_block("onboarding", reason=reason)
            self.blocks.insert(self.blocks.index(welcome) + 1, card)
            changed = True
        return changed

    def _account_block(self, kind: str, *, reason: str = "", returned: bool = False) -> Block:
        """A new (unattached) ``ACCOUNT`` block of ``kind``."""
        text, actions = account_card(kind, reason=reason, returned=returned)
        block = Block(Block.ACCOUNT, text, detail=kind)
        block.actions = actions
        block.reason = reason
        return block

    def _show_account_card(self, kind: str, *, reason: str = "", returned: bool = False) -> None:
        """Append an account card, unless the chat already ends with the same card (pressing Send again must not
        stack them)."""
        last = self.blocks[-1] if self.blocks else None
        if last is None or last.kind != Block.ACCOUNT or last.detail != kind or getattr(last, "reason", "") != reason:
            self.blocks.append(self._account_block(kind, reason=reason, returned=returned))
        self._render()

    def _send_blocker(self, profile) -> str:
        """Why a send must not start a turn, from LOCAL facts only (no network): ``no_profile``, ``no_key``, ``server``
        (the configured OriginShape server is refused), ``signed_out``, ``waitlisted``, ``outdated`` (this Oscar is
        older than the server's minimum version); ``""`` lets the turn go. The
        server stays authoritative: credits are never judged here (a stale "0" must not block someone who just bought
        some), and a sign-in that is not known yet (the OS keychain is still being read) is left to the turn."""
        if profile is None:
            return "no_profile"
        if not _params.is_account_profile(profile):
            if not profile.api_key and profile.provider == "anthropic":
                return "no_key"
            return ""
        state = self._account_state()
        if state.get("error_code") in ("invalid_url", "insecure_url"):
            return "server"
        if not state.get("known"):
            return ""
        if not state.get("signed_in"):
            return "signed_out"
        snapshot = state.get("snapshot") or {}
        fresh = time.time() - float(snapshot.get("fetched_at") or 0.0) < _WAITLIST_FRESH_S
        if snapshot.get("status") == "waitlisted" and fresh:
            return "waitlisted"
        # Unlike the waitlist, this is believed at any age: the server refuses a waitlisted account itself, but
        # nothing on the server enforces its minimum version, so this is the only gate. A stale "required" heals
        # at once: the card's send forces a refresh (_block_send), and so does the connect loop on each reconnect.
        if state.get("update") == "required":
            return "outdated"
        return ""

    def _block_send(self, blocker: str) -> None:
        """Show why the send did not start a turn; the typed text and the attachments stay where they are."""
        if blocker == "no_profile":
            self._append(Block.ERROR, "No chat profile is configured — open ⚙ settings and add one.")
            self._render()
        elif blocker == "server":
            problem = self._account_state().get("error") or "The OriginShape server URL cannot be used."
            self._append(Block.ERROR, f"{problem} Fix the Server field in ⚙ Settings (OriginShape account).")
            self._render()
        elif blocker == "no_key":
            self._show_account_card("onboarding", reason="no_key")
        elif blocker == "signed_out":
            self._show_account_card("onboarding", reason="signed_out")
        elif blocker == "waitlisted":
            self._show_account_card("waitlisted")
            self._refresh_account(force=True)     # approved meanwhile? the next send will know
        elif blocker == "outdated":
            self._show_account_card("outdated")
            self._refresh_account(force=True)     # the server may have lowered its minimum

    # -- conversation --------------------------------------------------------

    def _append(self, kind: str, text: str = "") -> Block:
        block = Block(kind, text)
        self.blocks.append(block)
        return block

    # -- attachments -----------------------------------------------------------

    _FILE_FILTERS = (
        "CAD, mesh & drawings "
        "(*.step *.stp *.iges *.igs *.brep *.stl *.obj *.3mf *.ply *.dxf *.svg *.FCStd);;"
        "Images (*.png *.jpg *.jpeg *.gif *.webp *.bmp *.tiff *.tif);;"
        "Text & data (*.txt *.md *.markdown *.csv *.tsv *.json *.xml *.yaml *.yml *.py *.scad *.log);;"
        "All files (*)"
    )

    PARAMPT = (
        "Build this component parametrically: one PartDesign Body, a "
        "\"Parameters\" spreadsheet where every dimension is a named cell, "
        "and real sketch-based features (Pad / Pocket / Revolution / Hole) "
        "bound to those parameters by expression. No scripting fused "
        "shapes, no boolean blobs — the feature tree must stay editable so "
        "changing one spreadsheet cell resizes the model. Build ONE "
        "feature per tool call so I can watch it take shape. When done, "
        "list the tree."
    )

    # Human labels of the starting messages in ``prompt.TEMPLATES`` (unknown keys get a title-cased label).
    _TEMPLATE_LABELS = {
        "enclosure_intake": "Enclosure for a board (insert)",
        "parametric_part": "Parametric part (insert)",
        "bracket": "Mounting bracket (insert)",
    }

    def _template_entries(self) -> list[tuple[str, str]]:
        """``(menu label, text)`` of the starting messages in ``prompt.TEMPLATES`` (guarded import: the menu
        keeps its historical entries when the prompt module is unavailable)."""
        try:
            from .prompt import TEMPLATES
        except Exception:  # noqa: BLE001 - a missing nicety must never break the menu
            return []
        return [(self._TEMPLATE_LABELS.get(key, key.replace("_", " ").capitalize() + " (insert)"), text)
                for key, text in TEMPLATES.items() if isinstance(text, str) and text.strip()]

    def _insert_template(self, text: str) -> None:
        """Put a template into the input (copyable and editable before sending) with its first
        ``<placeholder>`` selected, so typing overwrites it."""
        self.input.setPlainText(text)
        self.input.setFocus()
        match = re.search(r"<[^<>\n]{1,60}>", text)
        if match is not None:
            cursor = self.input.textCursor()
            cursor.setPosition(match.start())
            cursor.setPosition(match.end(), QtGui.QTextCursor.KeepAnchor)
            self.input.setTextCursor(cursor)

    def _show_templates(self):
        menu = QtWidgets.QMenu(self)
        insert_param = menu.addAction(
            "Parametric feature-tree build (insert)")
        rerun = menu.addAction(
            "Revert last turn & re-run parametric build")
        extra = {}
        entries = self._template_entries()
        if entries:
            menu.addSeparator()
            for label, text in entries:
                extra[menu.addAction(label)] = text
        chosen = menu.exec(QtGui.QCursor.pos())
        if chosen is insert_param:
            self.input.setPlainText(self.PARAMPT)
            self.input.setFocus()
        elif chosen is rerun:
            self._revert_and_rerun_parametric()
        elif chosen in extra:
            self._insert_template(extra[chosen])

    def _revert_and_rerun_parametric(self):
        """Clean slate, canonical prompt: revert the last revertible
        turn's CAD changes, then run the parametric build immediately."""
        if self._busy():
            self._set_status("Wait for the current turn to finish first.")
            return
        target = next((b for b in reversed(self.blocks)
                       if b.revertible and not b.reverted), None)
        if target is not None:
            if not self._revert_turn(target):
                # refused, failed or only partial (the card says why): rebuilding on top of the old geometry
                # would duplicate its parts
                self._on_notice("The revert did not happen, so the parametric build was not started.")
                self._set_status("Not rebuilding: the revert was not complete — see the note above.")
                return
        else:
            self._on_notice("No revertible turn found — building fresh.")
        self.input.setPlainText(self.PARAMPT)
        self._send()

    def _mark_up(self):
        """The pen's click (``panel_markup.start``): not while a turn runs, as the paperclip."""
        if self.worker is not None:
            return
        panel_markup.start(self, paths.cache_dir("markup"))

    def _pick_files(self):
        if self.worker is not None:
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Attach files for Oscar", os.path.expanduser("~"), self._FILE_FILTERS
        )
        self.add_attachments(paths)

    def add_attachments(self, paths) -> None:
        """Queue attachment paths (also the programmatic entry for tests)."""
        for path in paths:
            if path not in self.pending:
                self.pending.append(path)
        self._refresh_attach_bar()

    def _remove_attachment(self, path: str):
        if path in self.pending:
            self.pending.remove(path)
        self._refresh_attach_bar()

    def _refresh_attach_bar(self):
        layout = self.attach_bar.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not self.pending:
            self.attach_bar.hide()
            return
        for path in self.pending:
            chip = QtWidgets.QPushButton(f"{os.path.basename(path)}  ✕")
            chip.setToolTip(path)
            chip.setFlat(True)
            chip.setStyleSheet(
                "padding:1px 6px; border:1px solid palette(mid); border-radius:4px;"
            )
            chip.clicked.connect(lambda _checked=False, p=path: self._remove_attachment(p))
            layout.addWidget(chip)
        layout.addStretch(1)
        self.attach_bar.show()

    def _image_to_png_b64(self, path: str) -> str | None:
        """Decode an image and re-encode as downscaled PNG base64.

        PNG keeps the existing vision plumbing (Message.images) format-agnostic
        on the provider side; 1280 px is plenty of resolution for chat vision.
        """
        reader = QtGui.QImageReader(path)
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            return None
        longest = max(image.width(), image.height())
        if longest > 1280:
            scale = 1280.0 / longest
            image = image.scaled(
                max(1, round(image.width() * scale)),
                max(1, round(image.height() * scale)),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.WriteOnly)
        if not image.save(buffer, "PNG"):
            return None
        return base64.b64encode(bytes(buffer.data())).decode("ascii")

    def _late_image_attachments(self) -> tuple[list, list[str]]:
        """The pending attachments a steered message can carry: images that decode (``(attachments,
        base64 PNGs)``). Everything else (CAD files, text) stays pending for the next full turn."""
        found, encoded = [], []
        for path in list(self.pending):
            if _attach.classify(path) != "image":
                continue   # ``make`` hashes CAD files and PDFs (up to 256 MB): never on the GUI thread for a file that stays pending
            try:
                att = _attach.make(path)
            except Exception:  # noqa: BLE001 - an unreadable file simply stays pending
                continue
            if att.kind != "image":
                continue
            image = self._image_to_png_b64(att.path)
            if image:
                found.append(att)
                encoded.append(image)
        return found, encoded

    def _steer_while_busy(self) -> None:
        """Send pressed while a turn runs: hand the text (and any attached image) to the worker (``steer``)
        and show the bubble tagged "queued" instead of ignoring the click. Without ``AgentWorker.steer``
        (older worker) the text stays in the input and the status line says what to do."""
        text = self.input.toPlainText().strip()
        if not text:
            return
        steer = getattr(self.worker, "steer", None)
        if not callable(steer):
            self._set_status("Oscar is still working — press Esc to stop, then send again.")
            return
        attachments, images = self._late_image_attachments()
        content = _attach.build_user_message(text, attachments) if attachments else text
        try:
            if images:
                steer(content, images=images)
            else:
                steer(content)
        except Exception as e:
            self._set_status(f"Could not queue the message: {e}")
            return
        self.input.clear()
        display = text
        if attachments:
            display += "\n\n" + "\n".join(_attach.describe(a) for a in attachments)
            sent = {a.path for a in attachments}
            self.pending = [p for p in self.pending if p not in sent]
            self._refresh_attach_bar()
        block = self._append(Block.USER, display)
        block.queued = True
        block.steer_key = content.strip()
        block.steer_text = text
        self._queued_blocks.append(block)
        if self.pending:
            self._set_status("Attachments are sent with your next message once Oscar is done.")
        self._render()

    def _build_content(self, text: str, attachments: list) -> str:
        """The stored user message (``attachments.build_user_message``). Reading a PDF can take seconds (poppler up to
        20 s, the built-in reader much longer on a large file), so it runs on a helper thread while a local event
        loop keeps the window alive; the panel is locked meanwhile (``_busy`` is true: no second send, no chat
        switch). Everything else is assembled inline."""
        if not any(a.kind == "pdf" for a in attachments):
            return _attach.build_user_message(text, attachments)
        result: dict = {}

        def work() -> None:
            try:
                result["content"] = _attach.build_user_message(text, attachments)
            except BaseException as e:  # noqa: BLE001 - reported below, never lost on the thread
                result["error"] = e

        thread = threading.Thread(target=work, name="oscar-attachments", daemon=True)
        self._building = True
        self.input.setEnabled(False)
        self.send_btn.setEnabled(False)
        self._set_status("Reading the attached PDF…")
        loop = QtCore.QEventLoop(self)
        timer = QtCore.QTimer(self)
        timer.setInterval(50)
        timer.timeout.connect(lambda: None if thread.is_alive() else loop.quit())
        try:
            thread.start()
            timer.start()
            loop.exec()
        finally:
            timer.stop()
            self._building = False
            self.input.setEnabled(True)
            self.send_btn.setEnabled(True)
            self._set_status("")
        content = result.get("content")
        if content is None:   # the application is quitting, or the assembly failed: send the typed text with a plain note
            reason = result.get("error")
            self._log_panel_error("attachments", reason if isinstance(reason, Exception) else RuntimeError("interrupted"))
            note = "\n\n".join(f"[attached file: {a.name}] — could not be read here; path: {a.path}" for a in attachments)
            return f"{text}\n\n--- attached files ---\n{note}"
        return content

    def _send(self, workspace=None):
        """Start a turn with the input box's message, or, given ``workspace`` (a ``connect_gui.WorkspaceSend``),
        with a message from the OriginShape workspace: its text, on the account with its model, the input box
        left alone; ``workspace.turn`` is told when the turn can't start and gets the worker before it runs."""
        if self._building:
            return
        if self.worker is not None:
            if workspace is not None:
                workspace.turn.refuse("Oscar is busy with another turn in FreeCAD.")
                return
            self._steer_while_busy()
            return
        if workspace is not None:
            text, pending, profile = workspace.text.strip(), list(workspace.attachments), workspace.profile
        else:
            text, pending, profile = self.input.toPlainText().strip(), list(self.pending), _params.active_profile()
        if not text and not pending:
            return
        blocker = self._send_blocker(profile)
        if blocker:
            self._block_send(blocker)
            if workspace is not None:
                workspace.turn.refuse(blocker)
            return
        typed = text

        if workspace is None:
            self.input.clear()
            self.pending.clear()
            self._refresh_attach_bar()

        # The welcome banner (and the onboarding card under it) makes way for the real conversation.
        if self._welcome is not None and self._welcome in self.blocks:
            self.blocks.remove(self._welcome)
        self._welcome = None
        self.blocks[:] = [b for b in self.blocks if not (b.kind == Block.ACCOUNT and b.detail == "onboarding")]

        attachments = [_attach.make(path) for path in pending]
        images: list[str] = []
        for att in attachments:
            if att.kind == "image":
                encoded = self._image_to_png_b64(att.path)
                if encoded:
                    images.append(encoded)
                else:
                    att.kind = "other"  # unreadable image → metadata note only
        carried = list(self._carry_images)
        if self._carry_images:  # images of a steered message that arrived too late ride with its re-send
            images = list(self._carry_images) + images
            self._carry_images = []

        if not text:
            text = "Please work with the attached file(s)."
        content = self._build_content(text, attachments)
        notes = list(self._pending_notes)
        if self._pending_notes:
            content = "\n".join(self._pending_notes) + "\n\n" + content
            self._pending_notes.clear()
        display = text
        if attachments:
            display += "\n\n" + "\n".join(_attach.describe(a) for a in attachments)

        bubble = self._append(Block.USER, display)
        message = Message(role="user", content=content, images=images)
        self.history.append(message)
        self._title_current_from(text)
        self._persist_current()
        # The chat's Project is bound AFTER the title is set (the folder name derives from it); the folder
        # itself is created lazily on the worker thread, never here.
        session = self._current_session()
        project = self._bind_project(session)
        # What giving the message back takes, should the turn's very first request fail for an account reason
        # (panel_turn._give_back_turn): the stored message and its bubble, what the input held, the plan before it.
        self._turn_account = None
        self._turn_sent = {"message": message, "bubble": bubble, "typed": typed, "pending": pending, "carried": carried,
                           "notes": notes, "had_ledger": session is not None and "ledger" in session,
                           "workspace": workspace is not None,
                           # the workspace message this turn answers: its card is reverted by it
                           "workspace_message": _workspace_message_id(workspace),
                           "ledger": copy.deepcopy(session.get("ledger")) if session is not None else None,
                           "account": _params.is_account_profile(profile)}
        if workspace is not None and session is not None and self._turn_sent["workspace_message"]:
            # The workspace turns this FreeCAD run holds (runtime only, never saved: a restart forgets them, as it
            # forgets the cards and the undo stack). A revert from the browser can only reach these (§14).
            session.setdefault("_workspace_turns", set()).add(self._turn_sent["workspace_message"])
        # The assistant block is created lazily on the first text chunk, so
        # thinking and tool cards land ABOVE the response, in the order
        # they actually happened.
        self._set_busy(True)
        for block in self.blocks:  # no revert while a turn runs; _update_revert_flags re-enables the latest
            block.revert_enabled = False
        self._turn_seq += 1
        self._turn_view = TurnView()
        self._steps_by_call = {}
        self._turn_outcome = ("", "")
        self._turn_request = text
        self._act_now_armed = True
        self._render()

        try:
            self.worker = AgentWorker(
                self.history, project=project, ledger=self._session_ledger(session), request_text=text,
                attachments=[att.path for att in attachments],
                on_round=self._round_persister(session) if session is not None else None,
                session_id=str(session.get("id") or "") if session is not None else "",
                **({"profile": profile} if workspace is not None else {}))
        except Exception as e:
            self._append(Block.ERROR, str(e))
            self._set_busy(False)
            self._update_revert_flags()
            self._render()
            if workspace is not None:
                workspace.turn.refuse(str(e))
            return
        if workspace is not None:
            workspace.turn.attach(self.worker)
        # Snapshot document undo depths BEFORE anything runs, so this turn's
        # CAD changes can be reverted from its process card.
        try:
            from .cad import get_engine

            self._turn_snapshot = get_engine().begin_turn_snapshot()
        except Exception:
            self._turn_snapshot = {}
        self._turn_used_cad = False
        self._turn_mutated = False
        self._queued_blocks = []
        self.worker.text_delta.connect(self._on_text_delta)
        self.worker.reasoning_delta.connect(self._on_reasoning_delta)
        self.worker.tool_started.connect(self._on_tool_started)
        self.worker.tool_finished.connect(self._on_tool_finished)
        self.worker.turn_finished.connect(self._on_turn_finished)
        self.worker.notice.connect(self._on_notice)
        self.worker.flow_paused.connect(self._on_flow_paused)
        # the additive signals; a worker without one of them simply never feeds that widget
        for name, slot in (("plan_changed", self._on_plan_changed), ("step_started", self._on_step_started),
                           ("step_finished", self._on_step_finished), ("progress", self._on_progress),
                           ("round_reset", self._on_round_reset), ("artifacts_changed", self._on_artifacts_changed),
                           ("turn_outcome", self._on_turn_outcome), ("account_problem", self._on_account_problem)):
            signal = getattr(self.worker, name, None)
            if signal is not None and hasattr(signal, "connect"):
                signal.connect(slot)
        ended = getattr(self.worker, "finished", None)   # QThread.finished: the safety net of _worker_ended
        if ended is not None and hasattr(ended, "connect"):
            ended.connect(self._worker_ended)
        self.worker.start()
        self._turn_card = None
        self._heartbeat_timer.start()
        self._update_heartbeat()
        self._set_status("Thinking")

    def _worker_ended(self) -> None:
        """``QThread.finished`` of a turn worker. ``turn_finished`` (emitted by the same thread just before) has
        normally cleared ``self.worker`` already and nothing happens. When the thread ended WITHOUT it (an error the
        worker could not report), the turn is closed as failed so the panel is not left busy with a Stop button
        that does nothing. The check is repeated one event-loop turn later, so a ``turn_finished`` that is still
        queued is always handled first (never two endings of one turn)."""
        worker = self.worker
        if worker is not None:
            QtCore.QTimer.singleShot(0, lambda: self._close_dead_worker(worker))

    def _close_dead_worker(self, worker) -> None:
        if self.worker is not worker or getattr(worker, "isRunning", lambda: True)():
            return
        self._log_panel_error("worker_ended", RuntimeError("the worker thread ended without turn_finished"))
        self._on_turn_outcome("failed", "")
        self._on_turn_finished(False, "Oscar's worker stopped unexpectedly before it could report; see the log. "
                                      "Send your message again.")

    def _start_assistant_block(self):
        self._stream_buffer = ""
        self._current = self._append(Block.ASSISTANT, "")
        self._dirty = True

    def _stop(self):
        if self.worker is not None:
            if self._paused:
                self.worker.resume()
            self.worker.stop()
            self._set_status("stopping…")

    def _on_flow_button(self):
        """The trailing button mirrors the flow: send / pause / resume."""
        if self.worker is None:
            self._send()
            return
        if self._paused:
            self._paused = False
            self.worker.resume()
            self.send_btn.setText("⏸")
            self.send_btn.setToolTip("Pause after the current step (Esc cancels)")
            self._set_status("Thinking")
        else:
            self.worker.pause()
            self.send_btn.setText("…")
            self.send_btn.setToolTip("Pausing after the current step")

    def _on_flow_paused(self, paused: bool):
        if paused:
            self._paused = True
            self.send_btn.setText("▶")
            self.send_btn.setToolTip("Resume")
            self._set_status("⏸ Paused")
        else:
            self._paused = False
            self.send_btn.setText("⏸")
            self.send_btn.setToolTip("Pause after the current step (Esc cancels)")
            self._set_status("Thinking")

    def _set_busy(self, busy: bool):
        self.profile_combo.setEnabled(not busy)
        self.stop_btn.setVisible(bool(busy))
        self._esc_shortcut.setEnabled(bool(busy))
        if busy:
            self.send_btn.setText("⏸")
            self.send_btn.setToolTip("Pause after the current step (Esc cancels)")
        else:
            self._paused = False
            self.send_btn.setText("➤")
            self.send_btn.setToolTip("Send (Enter)")

    # -- turn status (Thinking… / Running tool… / live card timer) -----------

    def _set_status(self, text: str):
        """Show ``text`` as the sticky status line under the live card (``""`` clears it). The card
        narrates the steps, but only the status line can say WHY nothing moves (waiting for an approval,
        pausing, stopping), so it is never suppressed while a card exists."""
        self._status = text
        if (text or self._turn_card is not None) and self.worker is not None:
            if not self._status_timer.isActive():
                self._status_timer.start()
        self._render()

    def _flash_status(self, text: str, ms: int = 2500) -> None:
        """A transient status line: shown now, cleared after ``ms`` when nothing replaced it meanwhile."""
        self._set_status(text)

        def clear() -> None:
            if self._status == text and self.worker is None:
                self._set_status("")

        QtCore.QTimer.singleShot(ms, clear)

    def _tick_status(self):
        if self.worker is None and not self._status:
            self._status_timer.stop()
            return
        self._status_dots = (self._status_dots + 1) % 4
        if self._status or self._turn_card is not None:
            self._render()  # dots cycle / Working… elapsed refresh

    def _on_notice(self, text: str):
        if text.startswith("Image input disabled") or text.startswith("Message queued"):
            # model-capability notices never clutter the chat; log only ("Message queued" is the worker's
            # echo of a steer: the "queued" tag on the user bubble already says it)
            App.Console.PrintLog(f"Oscar: {text}\n")
            return
        self._append(Block.TOOL, text)   # Block.html draws the ℹ
        self._render()

    # -- rendering -----------------------------------------------------------

    def _flush_render(self):
        if not self._dirty:
            return
        self._dirty = False
        self._render()

    def _fragment(self, block: Block) -> str:
        """HTML of one block. Settled blocks (no running clock) are cached by their signature, so a
        render during streaming only rebuilds the live card / streaming bubble."""
        if not block.settled():
            return block.html()
        signature = block.signature()
        cached = self._frag_cache.get(block.bid)
        if cached is not None and cached[0] == signature:
            return cached[1]
        fragment = block.html()
        self._frag_cache[block.bid] = (signature, fragment)
        return fragment

    def _on_range_changed(self, _low: int, high: int) -> None:
        """The log's scroll range changed (Qt grows it while it lays a long document out lazily): finish
        what ``_render`` started: stay at the bottom, or return to the position the user had. A value
        different from the one this code set means the user scrolled meanwhile: stop interfering."""
        bar = self.log.verticalScrollBar()
        if self._scroll_mode == "follow":
            if bar.value() >= self._scroll_set - 8:
                bar.setValue(high)
                self._scroll_set = bar.value()
            else:
                self._scroll_mode = ""
        elif self._scroll_mode == "hold":
            if bar.value() == self._scroll_set:
                want = min(self._scroll_target, high)
                if want != bar.value():
                    bar.setValue(want)
                    self._scroll_set = bar.value()
                if high >= self._scroll_target:
                    self._scroll_mode = ""
            else:
                self._scroll_mode = ""

    def _render(self):
        """Rebuild the log. Scroll position: pinned to the bottom stays pinned; a user who scrolled up
        keeps the exact position (``setHtml`` resets it to the top otherwise)."""
        self._dirty = False
        bar = self.log.verticalScrollBar()
        value = bar.value()
        pinned = _scroll_target_for(value, bar.maximum()) is None
        if self.blocks is not self._scroll_blocks:
            # another chat: its newest message is what to show, not the pixel offset the previous conversation
            # was scrolled to
            self._scroll_blocks = self.blocks
            pinned = True
        style = (_muted(), _text_color())
        if style != self._frag_style:  # theme changed: cached colours are stale
            self._frag_cache.clear()
            self._frag_style = style
        parts = [self._fragment(b) for b in self.blocks]
        self._frag_cache = {b.bid: self._frag_cache[b.bid] for b in self.blocks if b.bid in self._frag_cache}
        if self._status:
            dots = "." * self._status_dots
            parts.append(
                f"<div style='color:{_muted()};font-size:small;margin-top:6px;'>"
                f"{html.escape(self._status)}{dots}</div>"
            )
        body = "<br>".join(parts)
        self._scroll_mode = ""
        self.log.setHtml(
            f"<html><body style='font-size:{_font_size()}pt;'>{body}</body></html>"
        )
        self._register_thumbnails()   # the new document needs its images again (see panel_turn)
        if pinned:
            bar.setValue(bar.maximum())
            self._scroll_mode = "follow"
        else:
            bar.setValue(value)
            self._scroll_mode = "hold"
            self._scroll_target = value
        self._scroll_set = bar.value()

    # -- events ---------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self.input and event.type() == QtCore.QEvent.KeyPress:
            if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                if not event.modifiers() & QtCore.Qt.ShiftModifier:
                    self._send()
                    return True
            if event.key() == QtCore.Qt.Key_Escape and self._busy():
                self._stop()
                return True
        return super().eventFilter(obj, event)


# ---------------------------------------------------------------------------
# Dock integration


def show_panel() -> None:
    """Create or raise the Oscar dock panel on the main window."""
    from . import params as _params
    from .cad import dispatch as _dispatch

    _dispatch.init()  # GUI thread here; arms the cross-thread dispatcher
    settings_ui.account_hub()   # the account hub lives on the GUI thread too (it receives the sign-in's on_change)
    mw = Gui.getMainWindow()
    dock = mw.findChild(QtWidgets.QDockWidget, DOCK_OBJECT_NAME)
    if dock is None:
        dock = QtWidgets.QDockWidget("Oscar", mw)
        dock.setObjectName(DOCK_OBJECT_NAME)
        dock.setWidget(OscarPanel())
        # Stored in Oscar's own settings file (FreeCAD's user
        # parameters can be silently dropped on exit — see oscar/params.py).
        mw.addDockWidget(QtCore.Qt.DockWidgetArea(_restored_dock_area()), dock)
        dock.setFloating(_params.get_bool("DockFloating", False))
        dock.setMinimumWidth(280)

        # Keep floating state in sync so a restart restores the same layout.
        dock.dockLocationChanged.connect(
            lambda area=None, *_args: _save_dock_state(mw, dock, area))
        dock.topLevelChanged.connect(lambda floating: _save_floating(bool(floating)))
    dock.show()
    dock.raise_()


_DOCK_AREAS = (1, 2, 4, 8)  # Qt left, right, top, bottom — 2 (right) default


def _dock_area_value(area) -> int:
    """A Qt.DockWidgetArea as the plain int kept in settings (PySide6 enums
    are not int()-able; they carry .value)."""
    try:
        return int(getattr(area, "value", area))
    except (TypeError, ValueError):
        return 0


def _restored_dock_area() -> int:
    area = _params.get_int("DockArea", 2)
    return area if area in _DOCK_AREAS else 2


def _save_dock_state(mw, dock, area=None) -> None:
    """Remember where the dock is. Runs from a Qt signal: a settings file that is locked right now
    (``params.SettingsUnavailable``, nothing written) only costs this one update, never an exception in Qt."""
    value = _dock_area_value(area if area is not None
                             else dockWidgetAreaOf(mw, dock))
    try:
        # a floating dock reports NoDockWidgetArea (0); keep its last docked side
        if value in _DOCK_AREAS:
            _params.set_int("DockArea", value)
        _params.set_bool("DockFloating", dock.isFloating())
    except OSError as e:
        App.Console.PrintLog(f"Oscar: the dock position was not saved: {e}\n")


def _save_floating(floating: bool) -> None:
    try:
        _params.set_bool("DockFloating", floating)
    except OSError as e:   # see _save_dock_state
        App.Console.PrintLog(f"Oscar: the dock state was not saved: {e}\n")


def dockWidgetAreaOf(mw, dock):
    return mw.dockWidgetArea(dock)
