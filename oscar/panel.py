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

Shell (2026-11-16 redesign): one compact header row (logo, chat selector, new /
rename / delete chat, project folder, account badge, overflow menu, settings) over a
QTabWidget with exactly three pages — Chat (transcript, pinned plan strip, heartbeat,
question bar, composer), Parameters and Logs. The two side pages mount their own
modules defensively (``panel_tabs.make_params_tab`` / ``make_timeline_tab``:
``panel_params.ParamsTab`` / ``panel_timeline.TimelineTab``, with built-in fallbacks so a
missing module never crashes); the chat page mounts ``panel_questions.QuestionsBar``
below the transcript the same way (``_mount_questions_bar``). There is no Summary page.

Widgets located by object name (GUI tests): ``oscarLog``, ``oscarPlanStrip``,
``oscarArtifactsStrip``, ``oscarHeartbeat``, ``oscarStopButton``, ``oscarPauseButton``,
``oscarActNowButton``, ``oscarMenuButton`` (header menu with the ``oscarFollowAction`` toggle),
``oscarAccountBadge`` (the OriginShape account badge, menu ``oscarAccountMenu``);
header buttons ``oscarLogoLabel``, ``oscarChatSelector``, ``oscarNewChatButton``,
``oscarRenameChatButton``, ``oscarDeleteChatButton``, ``oscarFolderButton``,
``oscarSettingsButton``; the composer's ``oscarComposer`` / ``oscarSendButton`` /
``oscarCredits`` and the tabs widget ``oscarTabs``.

The OriginShape account: the badge before ⚙ reads *Sign in* / *42 credits* / *0 credits* (warning colour) /
*Waitlist* and carries the account menu; it shows for the account profile or a sign-in. Its state comes from the
process-wide ``settings_ui.AccountHub`` (``on_change`` of the sign-in plus a coalesced background ``/v1/me``), refreshed
when the panel opens, on a profile change, after Settings closes, at the end of a turn and on sign-in; nothing here
touches the network. The send gate uses local facts only: the account profile without a sign-in gets the onboarding
card instead of a turn, a waitlisted account (fresh snapshot) the waitlist card, an account that has not accepted the
current Terms the consent card and prompt (``consent_gui``, which also asks once after a sign-in); everything else goes
to the server, which decides about credits. A failed turn's ``account_problem`` (``AgentWorker``) shows the matching
card instead of the raw provider error, and when the turn's very first request failed the typed message goes back to
the input. A finished turn on the account is recorded on OriginShape once the Terms are accepted
(``panel_turn._record_turn``, ``oscar.records``).

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
from . import consent_gui, panel_chats, panel_markup, paths
from . import panel_theme as _theme
from . import params as _params
from . import settings_sync as _settings_sync
from . import settings_ui
from .agent import AgentWorker
from .auth import consent as _consent
from .panel_anchors import AnchorMixin
from .panel_blocks import (  # noqa: F401 - re-exported: tests and gui_smoke import these from oscar.panel
    _BLOCK_SEQ, _DETAIL_LIMIT, Block, LogBrowser, Step, ThumbCache, _accent, _font_size, _fg, _muted, _text_color)
from .panel_follow import FollowMixin
from .panel_render import (  # noqa: F401 - md_to_html stays importable from oscar.panel
    _account_snapshot, account_balance, classify_file_link, md_to_html, TurnView, account_badge,
    account_card, account_menu, on_trial)
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
        self._rendered_chat_items: list[tuple[int, tuple, str, int]] = []
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
        self._init_tabs()          # the Chat / Parameters / Summary pages live in the tab shell
        self._register_question_surface()
        self._register_ask_surface()
        self._register_netguard_hook()
        self._register_quit_hook()
        self._init_account()
        self._init_settings_sync()
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
        try:   # stop being the question surface so nothing waits on a dying panel
            from . import netguard as _netguard

            surface = getattr(_netguard, "panel_surface", None)
            if surface is not None and getattr(surface, "__self__", None) is self:
                _netguard.panel_surface = None
        except Exception:  # noqa: BLE001
            pass
        try:   # release every ask_user waiter and stop being its surface
            from . import askuser as _askuser
            from . import tools_ask as _tools_ask

            surface = getattr(_tools_ask, "surface", None)
            if surface is not None and getattr(surface, "__self__", None) is self:
                _tools_ask.surface = None
            _askuser.REGISTRY.shutdown()
        except Exception:  # noqa: BLE001
            pass
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

        # Header (shell redesign): one compact row — logo, chat selector, new / rename /
        # delete chat, project folder, account badge, overflow menu, settings. The ⋯ menu
        # keeps every historical action (clear / search / exports …); the hidden buttons
        # stay as attributes so tests and muscle memory keep working.
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(3)
        self.subtitle = QtWidgets.QLabel("")   # the "Oscar x.y is out" link — shown only when it applies
        self.subtitle.setTextFormat(QtCore.Qt.RichText)
        self.subtitle.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.subtitle.linkActivated.connect(lambda _link: self._on_account_action("download"))
        self.subtitle.hide()
        logo = QtGui.QIcon(paths.icon("Oscar.svg"))
        logo_pixmap = logo.pixmap(18, 18) if not logo.isNull() else None
        self.logo_label = QtWidgets.QLabel()
        self.logo_label.setObjectName("oscarLogoLabel")
        if logo_pixmap is not None and not logo_pixmap.isNull():
            self.logo_label.setPixmap(logo_pixmap)
        else:   # no SVG icon plugin (headless): the compact wordmark stands in
            self.logo_label.setText("Oscar")
        self.logo_label.setToolTip(f"Oscar {__version__} — the FreeCAD building copilot")
        header.addWidget(self.logo_label)
        self.chat_combo = QtWidgets.QComboBox()
        self.chat_combo.setObjectName("oscarChatSelector")
        self.chat_combo.setMinimumWidth(90)
        self.chat_combo.setToolTip("Task chats — switch between separate "
                                   "conversations")
        self.chat_combo.activated.connect(self._on_chat_picked)
        header.addWidget(self.chat_combo, 1)
        self.new_chat_btn = QtWidgets.QToolButton()
        self.new_chat_btn.setObjectName("oscarNewChatButton")
        self.new_chat_btn.setText("＋")
        self.new_chat_btn.setToolTip("New chat (fresh task context)")
        self.new_chat_btn.clicked.connect(self._new_chat)
        header.addWidget(self.new_chat_btn)
        self.rename_chat_btn = QtWidgets.QToolButton()
        self.rename_chat_btn.setObjectName("oscarRenameChatButton")
        self.rename_chat_btn.setText("✎")
        self.rename_chat_btn.setToolTip("Rename this chat…")
        self.rename_chat_btn.clicked.connect(lambda: panel_chats.rename(self))
        header.addWidget(self.rename_chat_btn)
        clear_chat_btn = QtWidgets.QToolButton()
        clear_chat_btn.setText("🗑")
        clear_chat_btn.setToolTip("Clear this chat's messages (the chat itself stays)")
        clear_chat_btn.clicked.connect(self._clear_chat)
        clear_chat_btn.hide()
        self.clear_chat_btn = clear_chat_btn
        self.del_chat_btn = QtWidgets.QToolButton()
        self.del_chat_btn.setObjectName("oscarDeleteChatButton")
        self.del_chat_btn.setText("🗑")
        self.del_chat_btn.setToolTip("Delete this chat")
        self.del_chat_btn.clicked.connect(self._delete_chat)
        header.addWidget(self.del_chat_btn)
        self.folder_btn = QtWidgets.QToolButton()
        self.folder_btn.setObjectName("oscarFolderButton")
        self.folder_btn.setText("📁")
        self.folder_btn.setToolTip("Open the project folder of this chat")
        self.folder_btn.clicked.connect(
            lambda: self._on_anchor_text("oscar://artifact/open-dir"))
        header.addWidget(self.folder_btn)

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
        self.header_menu.addSeparator()
        self.header_menu.addAction("Clear this chat…").triggered.connect(
            lambda _checked=False: self._clear_chat())
        self.header_menu.addAction("Delete this chat…").triggered.connect(
            lambda _checked=False: self._delete_chat())
        self.header_menu.addSeparator()
        self.header_menu.addAction("Open project folder").triggered.connect(
            lambda _checked=False: self._on_anchor_text("oscar://artifact/open-dir"))
        self.header_menu.addAction("Exports folder").triggered.connect(
            lambda _checked=False: self._on_anchor_text("oscar://artifact/open-exports"))
        self.header_menu.addSeparator()
        self.header_menu.addAction("Settings…").triggered.connect(
            lambda _checked=False: self._open_settings())
        menu_btn.setMenu(self.header_menu)
        header.addWidget(menu_btn)
        self.settings_btn = QtWidgets.QToolButton()
        self.settings_btn.setObjectName("oscarSettingsButton")
        self.settings_btn.setText("⚙")
        self.settings_btn.setToolTip("Settings — profiles, account, permissions")
        self.settings_btn.clicked.connect(lambda: self._open_settings())
        header.addWidget(self.settings_btn)
        root.addLayout(header)

        # The flow widgets (plan strip, artifacts strip, heartbeat row) are built once;
        # the plan strip is PINNED at the top of the Chat page (the only page that shows
        # it — the Summary page is gone), then the tab shell with exactly one visible
        # page (Chat / Parameters / Logs).
        self._build_flow_widgets()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("oscarTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        chat_page = QtWidgets.QWidget()
        chat_layout = QtWidgets.QVBoxLayout(chat_page)
        chat_layout.setContentsMargins(0, 4, 0, 0)
        chat_layout.setSpacing(4)
        self._chat_page = chat_page

        # Pinned plan strip, then the transcript
        chat_layout.addWidget(self.plan_strip)

        # Chat log
        self.log = LogBrowser()
        self.log.setObjectName("oscarLog")
        self.log.setOpenLinks(False)  # every anchor goes through _on_anchor (web / file links via QDesktopServices)
        self.log.setOpenExternalLinks(False)
        self.log.anchorClicked.connect(self._on_anchor)
        self.log.verticalScrollBar().rangeChanged.connect(self._on_range_changed)
        chat_layout.addWidget(self.log, 1)
        chat_layout.addWidget(self.artifacts_strip)
        chat_layout.addWidget(self.heartbeat_row)
        # The chat question bar (seam: panel_questions.QuestionsBar, below the transcript,
        # above the composer; a hidden placeholder while that module is absent)
        chat_layout.addWidget(self._mount_questions_bar())

        # Pending attachments (chips; hidden while empty)
        self.attach_bar = QtWidgets.QWidget()
        attach_layout = QtWidgets.QHBoxLayout(self.attach_bar)
        attach_layout.setContentsMargins(0, 0, 0, 0)
        attach_layout.setSpacing(4)
        self.attach_bar.hide()
        chat_layout.addWidget(self.attach_bar)

        # Input area: the composer (UI redesign) — one recessed frame that lights its border
        # on focus, holding the auto-expanding text edit and the controls row.
        input_frame = QtWidgets.QFrame()
        input_frame.setObjectName("oscarComposer")
        input_box = QtWidgets.QVBoxLayout(input_frame)
        input_box.setContentsMargins(10, 6, 10, 6)
        input_box.setSpacing(4)
        self.input_frame = input_frame

        self.input = QtWidgets.QPlainTextEdit()
        self.input.setPlaceholderText(
            "Ask Oscar… (Enter to send · Shift+Enter for a new line · Esc stops)"
        )
        self.input.setFixedHeight(48)
        self.input.setMinimumHeight(48)
        self.input.setMaximumHeight(140)
        self.input.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.input.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.input.installEventFilter(self)
        self.input.textChanged.connect(self._grow_input)
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
        # Credit balance of the signed-in account (display only — the badge and Settings
        # keep the full account surface). Hidden when there is nothing to show.
        self.credits_label = QtWidgets.QLabel("")
        self.credits_label.setObjectName("oscarCredits")
        self.credits_label.setStyleSheet(f"color:{_muted()};font-size:small;"
                                         f"font-family:Consolas,monospace;")
        self.credits_label.hide()
        controls.addWidget(self.credits_label)

        # The trailing Send/Stop toggle: one slot, two widgets. Idle → ➤ Send; while a
        # turn runs → ■ Stop (same effect as Esc). Pause/Resume lives in the heartbeat row.
        self.stop_btn = QtWidgets.QPushButton("■")
        self.stop_btn.setObjectName("oscarStopButton")
        self.stop_btn.setFixedWidth(40)
        self.stop_btn.setToolTip("Stop the running turn (Esc)")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.hide()
        controls.addWidget(self.stop_btn)

        self.send_btn = QtWidgets.QPushButton("➤")
        self.send_btn.setObjectName("oscarSendButton")
        self.send_btn.setFixedWidth(40)
        self.send_btn.setToolTip("Send (Enter)")
        self.send_btn.clicked.connect(self._on_flow_button)
        controls.addWidget(self.send_btn)
        input_box.addLayout(controls)
        chat_layout.addWidget(input_frame)

        # Parameters and Logs pages (shell redesign) — _init_tabs mounts them through the
        # seam factories (panel_params / panel_timeline, with built-in fallbacks)
        self._params_page = QtWidgets.QWidget()
        self._logs_page = QtWidgets.QWidget()
        self.tabs.addTab(chat_page, "Chat")
        self.tabs.addTab(self._params_page, "Parameters")
        self.tabs.addTab(self._logs_page, "Logs")
        root.addWidget(self.tabs, 1)

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
            _settings_sync.request("panel", prefer="local")   # the profile in use, on the website too
        profile = _params.active_profile()
        self.model_label.setText(_params.model_display_name(profile))
        self._refresh_account()

    def _init_settings_sync(self) -> None:
        """The AI model settings follow the OriginShape account (``settings_sync``): a sync when the panel starts, and
        the picker re-read when a pull changed the profiles."""
        try:
            settings_ui.account_hub().settings_synced.connect(self._on_settings_synced)
            _settings_sync.request("start")
        except Exception as e:  # noqa: BLE001 - the panel works without the sync
            self._log_panel_error("settings_sync", e)

    def _on_settings_synced(self, payload: str = "") -> None:
        """A sync's news (GUI thread): a pull that changed the profiles shows in the picker and the model label."""
        import json

        try:
            changed = json.loads(payload or "{}").get("event") == "changed"
        except (AttributeError, TypeError, ValueError):
            changed = False
        if changed:
            try:
                self._reload_profiles()
                notice = _params.pending_notice()   # the website picked a profile this computer can't use yet
                if notice:
                    self._flash_status(notice, 12000)
            except Exception as e:  # noqa: BLE001
                self._log_panel_error("settings_sync", e)

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
        # The composer's compact credit chip (display only — the badge/menu hold the real account)
        snapshot = _account_snapshot(state)
        balance, held = account_balance(snapshot)
        if show and balance > 0:
            self.credits_label.setText(f"{balance:,.2f} cr")
            self.credits_label.setToolTip("Credit balance" + (f" ({held:,.2f} held by running requests)" if held else ""))
            self.credits_label.show()
        else:
            self.credits_label.hide()
        # Without the badge (your own provider), a newer Oscar shows here instead: a link to the download page.
        newer = state.get("update") in ("available", "required") and state.get("latest_version")
        self.subtitle.setText(f"<a href='oscar://account/download'>Oscar {html.escape(str(newer))} is out</a>"
                              if newer else "copilot")
        self.subtitle.setVisible(not show)
        self.subtitle.setToolTip(f"A newer Oscar is available (this is {__version__}). Click to get it."
                                 if newer else "")
        changed = self._sync_onboarding(profile, state)
        try:
            changed = consent_gui.sync(self, state) or changed   # the Terms: the card and the prompt, or neither
        except Exception as e:  # noqa: BLE001 - the prompt is never a reason for the panel to fail
            self._log_panel_error("consent", e)
        if changed and render:
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
        ``waitlist``, ``download``, ``refresh``, ``consent`` (the Terms prompt), ``signout``. Pages come from the
        account's own links, never from a link's text."""
        if action in ("signin", "device"):
            self._open_settings(account_action=action)
        elif action == "consent":
            consent_gui.open_dialog(self)
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
        snapshot = settings_ui.account_hub().state.get("snapshot")
        text, actions = account_card(kind, reason=reason, returned=returned, trial=on_trial(snapshot))
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
        """Why a send must not start a turn, from LOCAL facts only (no network): ``no_profile``, ``endpoint`` (a sync
        made the profile or changed its provider or base URL, and the user hasn't confirmed it yet), ``no_key``, ``server``
        (the configured OriginShape server is refused), ``signed_out``, ``waitlisted``, ``outdated`` (this Oscar is
        older than the server's minimum version), ``consent`` (the account has not accepted OriginShape's current
        Terms: never for another provider); ``""`` lets the turn go. The
        server stays authoritative: credits are never judged here (a stale "0" must not block someone who just bought
        some), and a sign-in that is not known yet (the OS keychain is still being read) is left to the turn."""
        if profile is None:
            return "no_profile"
        if _settings_sync.unusable_reason(profile):   # made or redirected on the website: its address is confirmed first
            return "endpoint"
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
        # Believed at any age too: accepting again is harmless, and the card's send refreshes the account.
        if _consent.blocks_chat(snapshot, account_profile=True):
            return "consent"
        return ""

    def _block_send(self, blocker: str) -> None:
        """Show why the send did not start a turn; the typed text and the attachments stay where they are."""
        if blocker == "no_profile":
            self._append(Block.ERROR, "No chat profile is configured — open ⚙ settings and add one.")
            self._render()
        elif blocker == "endpoint":
            self._append(Block.ERROR, _settings_sync.unusable_reason(_params.active_profile()))
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
        elif blocker == "consent":
            consent_gui.show_card(self)
            self._render()
            self._refresh_account(force=True)     # accepted on the website meanwhile? the next send will know

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
        # The latest AI model settings first, for the user's turns and the workspace's alike: a pull older than a
        # minute is renewed, waited for up to two seconds (then the turn goes on with what is here).
        _settings_sync.pull_before_turn()
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
            elif blocker == "consent":   # the user is here and wants to chat: the prompt, at once
                QtCore.QTimer.singleShot(0, lambda: consent_gui.open_dialog(self))
            return
        self._settle_typed_clarification(text, pending)
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
        started = time.time()
        chat = self._current_session()
        turn_index = None
        if chat is not None:
            from . import chats as _chats

            turn_index = _chats.next_turn(chat)   # this turn's index in the chat, saved with it just below
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
                           "account": _params.is_account_profile(profile),
                           # what the turn's record says (panel_turn._record_turn)
                           "chat_id": str(session.get("id") or "") if session is not None else "",
                           "turn": turn_index, "started_at": started,
                           "model": _params.account_model_alias(getattr(profile, "model", "") or "")}
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
        self._turn_asks = []
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
                origin="workspace" if workspace is not None else "freecad",
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
        """The trailing Send button (idle only — while a turn runs the Stop square takes
        its place; pause/resume lives in the heartbeat row)."""
        if self.worker is None:
            self._send()

    def _on_pause_button(self):
        """The heartbeat row's pause/resume toggle: pause after the current step, resume a
        paused turn (the same flow ``Esc``'s stop releases before stopping)."""
        if self.worker is None:
            return
        if self._paused:
            self._paused = False
            self.worker.resume()
            self._set_status("Thinking")
        else:
            self.worker.pause()
            self._set_status("Pausing after the current step…")

    def _on_flow_paused(self, paused: bool):
        self._paused = bool(paused)
        self._update_heartbeat()
        self._set_status("⏸ Paused" if paused else "Thinking")

    def _set_busy(self, busy: bool):
        self.profile_combo.setEnabled(not busy)
        # the Send/Stop toggle: exactly one of the two trailing widgets is shown
        self.stop_btn.setVisible(bool(busy))
        self.send_btn.setVisible(not busy)
        self._esc_shortcut.setEnabled(bool(busy))
        pulse = getattr(self, "activity_pulse", None)
        if pulse is not None:
            if busy:
                if pulse.state() != QtCore.QAbstractAnimation.Running:
                    pulse.start()
            else:
                pulse.stop()
                effect = getattr(self, "activity_effect", None)
                if effect is not None:
                    effect.setOpacity(1.0)
        if not busy:
            self._paused = False

    # -- turn status (Thinking… / Running tool… / live card timer) -----------

    def _set_status(self, text: str):
        """Show ``text`` as the sticky status line under the live card (``""`` clears it). The card
        narrates the steps, but only the status line can say WHY nothing moves (waiting for an approval,
        pausing, stopping), so it is never suppressed while a card exists."""
        self._status = text
        if self.worker is not None:
            refresh = getattr(self, "_update_heartbeat", None)
            if callable(refresh):
                refresh()
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
        """Update only the changed transcript suffix; keep scroll position when the user scrolled up."""
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
        # Chat is the user-facing conversation. Process cards collapse into the turn header; the
        # Build Summary stays in Chat as ONE card at the end of its turn — there is no Summary tab
        # (redesign 2026-11-16) and no other page shows it, so hiding it here loses the model's
        # report and its Continue link entirely.
        visible = [b for b in self.blocks if (b.kind != Block.TOOL or not b.steps)]
        items = [(b.bid, b.signature(), self._fragment(b)) for b in visible]
        self._frag_cache = {b.bid: self._frag_cache[b.bid] for b in visible if b.bid in self._frag_cache}
        if self._status and self.worker is None:
            status = self._status
            dots = "." * self._status_dots
            items.append(
                (0, (status, self._status_dots),
                 f"<div style='color:{_muted()};font-size:small;margin-top:6px;'>"
                 f"{html.escape(status)}{dots}</div>")
            )
        self._scroll_mode = ""
        old = self._rendered_chat_items
        prefix = 0
        while (prefix < len(old) and prefix < len(items)
               and old[prefix][0] == items[prefix][0]
               and old[prefix][1] == items[prefix][1]
               and old[prefix][2] == items[prefix][2]):
            prefix += 1
        cursor = QtGui.QTextCursor(self.log.document())
        if prefix < len(old):
            cursor.setPosition(old[prefix][3])
            cursor.movePosition(QtGui.QTextCursor.End, QtGui.QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            updated = old[:prefix]
        else:
            cursor.movePosition(QtGui.QTextCursor.End)
            updated = list(old)
        for index in range(prefix, len(items)):
            if updated or index:
                cursor.insertHtml("<br>")
            bid, signature, fragment = items[index]
            start = cursor.position()
            cursor.insertHtml(f"<span style='font-size:{_font_size()}pt;'>{fragment}</span>")
            updated.append((bid, signature, fragment, start))
        self._rendered_chat_items = updated
        self._register_thumbnails()   # the new document needs its images again (see panel_turn)
        if pinned:
            bar.setValue(bar.maximum())
            self._scroll_mode = "follow"
        else:
            bar.setValue(value)
            self._scroll_mode = "hold"
            self._scroll_target = value
        self._scroll_set = bar.value()
        self._refresh_side_views()   # the Summary/Log page, when visible, follows the transcript

    # -- events ---------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self.input:
            if event.type() == QtCore.QEvent.KeyPress:
                if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                    # Enter (and Ctrl+Enter, the spec's composer shortcut) send; Shift+Enter is a newline
                    if not event.modifiers() & QtCore.Qt.ShiftModifier:
                        self._send()
                        return True
                if event.key() == QtCore.Qt.Key_Escape and self._busy():
                    self._stop()
                    return True
            if event.type() == QtCore.QEvent.FocusIn:
                self._composer_focus(True)
            elif event.type() == QtCore.QEvent.FocusOut:
                self._composer_focus(False)
        return super().eventFilter(obj, event)

    # -- tabs, parameters, logs (shell redesign) ------------------------------

    def _init_tabs(self) -> None:
        """Mount the Parameters and Logs pages through the seam factories and apply the
        theme. Runs after ``_build_ui``."""
        from . import panel_tabs as _tabs

        self.setProperty("oscarTheme", True)
        self.setStyleSheet(_theme.panel_qss())
        self._open_questions: dict[str, dict] = {}

        params_layout = QtWidgets.QVBoxLayout(self._params_page)
        params_layout.setContentsMargins(0, 6, 0, 0)
        self.params_view = _tabs.make_params_tab(on_apply=self._apply_params)
        try:
            self.params_view.on_refresh = self._refresh_params   # the fallback view's refresh hook
        except Exception:   # noqa: BLE001 - a seam tab that rejects the attribute keeps its own refresh
            pass
        params_layout.addWidget(self.params_view)

        logs_layout = QtWidgets.QVBoxLayout(self._logs_page)
        logs_layout.setContentsMargins(0, 6, 0, 0)
        self.logs_view = _tabs.make_timeline_tab()
        logs_layout.addWidget(self.logs_view)

    def _on_tab_changed(self, index: int) -> None:
        if index:   # a light side page: subtle fade, never the heavy chat document
            _theme.fade_in(self.tabs.widget(index))
        if index == 1:
            self._refresh_params()
        elif index == 2:
            self._refresh_logs()

    def _grow_input(self) -> None:
        """Auto-expanding composer: one line up front, up to ~5 lines, then it scrolls."""
        doc_height = self.input.document().size().height()
        frame = self.input.frameWidth()
        self.input.setFixedHeight(max(48, min(140, int(doc_height + frame + 12))))

    def showEvent(self, event):
        super().showEvent(event)
        # Becoming visible reflows the log (the tab shell's first show does it too) and the
        # reflow's scrollbar jump would read as a user scroll, cancelling follow-the-bottom.
        # Stay glued to the newest content — unless the user had deliberately scrolled up.
        log = getattr(self, "log", None)
        if log is None or getattr(self, "_scroll_mode", "") == "hold":
            return
        bar = log.verticalScrollBar()
        bar.setValue(bar.maximum())
        self._scroll_mode = "follow"
        self._scroll_set = bar.value()

    def _composer_focus(self, focused: bool) -> None:
        frame = getattr(self, "input_frame", None)
        if frame is None:
            return
        frame.setProperty("focused", bool(focused))
        frame.style().unpolish(frame)
        frame.style().polish(frame)

    def _active_doc(self):
        try:
            doc = App.ActiveDocument
        except Exception:   # noqa: BLE001 - headless test harnesses
            doc = None
        return doc

    def _refresh_params(self) -> None:
        """Re-read the active document's parameter sheet (GUI thread; on demand, never polling).

        Two seam contracts exist: the built-in ``panel_tabs.ParamsView`` takes ``load(table)``,
        the redesigned ``panel_params.ParamsTab`` takes ``refresh_from(table)``. Both are fed the
        same ``project_gui.param_table`` output — the panel must not assume one of them."""
        view = getattr(self, "params_view", None)
        if view is None:
            return
        table = {"sheet": "", "rows": []}
        doc = self._active_doc()
        if doc is not None:
            try:
                from . import project_gui as _pgui

                table = _pgui.param_table(doc)
            except Exception as e:   # noqa: BLE001 - a read failure must not break the panel
                self._log_panel_error("params_refresh", e)
        setter = getattr(view, "load", None) or getattr(view, "refresh_from", None)
        if callable(setter):
            setter(table)

    def _apply_params(self, edits: "list[dict]") -> dict:
        """Apply (N) on the Parameters page: the real write path (one undo transaction, dependent
        rebuild, rollback on failure), then the outcome is recorded in the chat."""
        doc = self._active_doc()
        if doc is None:
            self._flash_status("No open document to apply parameters to.")
            return {"ok": False, "text": "no document"}
        try:
            from .tools_kit_params import apply_parameter_edits

            result = apply_parameter_edits(doc.Name, edits)
        except Exception as e:   # noqa: BLE001 - the helper reports failures itself
            self._log_panel_error("params_apply", e)
            result = {"ok": False, "text": f"Apply failed: {type(e).__name__}: {e}"}
        text = str(result.get("text") or ("Applied." if result.get("ok") else "Apply failed."))
        self._append(Block.TOOL, ("⚙ " + text) if result.get("ok") else ("⚠ " + text))
        self._dirty = True
        self._render()
        self._refresh_params()
        return result

    def _refresh_logs(self) -> None:
        """The Logs page, fed through whichever seam the tab exposes.

        ``panel_timeline.TimelineTab`` (the redesign) rebuilds itself from the chat's
        ``Project`` step journal — the authoritative record with the snapshot paths, which is
        what a *visual* timeline needs. The built-in fallback ``panel_tabs.LogsView`` takes the
        panel's in-memory ``load_events(events, thumbs)`` instead. Neither is polled: this runs on
        the tab switch and at the end of a turn, and the rebuild is skipped while the journal has
        not changed (``build_timeline`` hashes neighbouring snapshots, so a no-op must not run at
        streaming frequency)."""
        view = getattr(self, "logs_view", None)
        if view is None:
            return
        loader = getattr(view, "load_events", None)
        if callable(loader):
            loader(self._log_events(), getattr(self, "_thumbs", None))
            return
        refresh = getattr(view, "refresh_from", None)
        if not callable(refresh):
            return
        project = self._project_of(self._current_session())
        if project is None:
            return
        try:
            records, _starts = project.read_steps()
        except Exception as e:   # noqa: BLE001 - an unreadable journal shows the empty state
            self._log_panel_error("logs_refresh", e)
            return
        signature = (str(getattr(project, "root", "") or ""), len(records),
                     str((records[-1].to_line().get("ts") if records else "") or ""))
        if signature == getattr(self, "_logs_signature", None):
            return   # nothing new in the journal: keep the rendered rows
        self._logs_signature = signature
        refresh(project)

    def _refresh_side_views(self) -> None:
        tabs = getattr(self, "tabs", None)
        if tabs is None:
            return
        if tabs.currentIndex() == 1:
            self._refresh_params()
        elif tabs.currentIndex() == 2:
            self._refresh_logs()

    def _log_lines(self) -> "list[str]":
        """Execution events only, derived from the same recorded blocks and steps as the conversation."""
        lines: list[str] = []
        for block in self.blocks:
            if block.kind == Block.ERROR:
                detail = block.text.strip().splitlines()[0] if block.text.strip() else "Operation failed"
                lines.append(f"FAILED  | {detail[:140]}")
            elif block.kind == Block.QUESTION:
                state = block.detail or "pending"
                clarification = any(target.startswith("oscar://clarify/") for target, _ in block.actions)
                if clarification and state == "superseded":
                    continue
                title = block.text.splitlines()[0][:90]
                category = "User input" if clarification else "Approval"
                mark = state.upper()
                lines.append(f"{mark:7} | {category}: {title}"
                             + (f" — {block.reason}" if block.reason else ""))
            elif block.kind == Block.TOOL:
                if not block.steps:
                    if block.text.strip():
                        lines.append(f"EVENT   | {block.text.strip().splitlines()[0][:140]}")
                    continue
                for step in block.steps:
                    if step.is_thought or step.plain:
                        continue
                    skipped = bool(getattr(step.view, "skipped", False)) if step.view is not None else False
                    mark = "SKIPPED" if skipped else (
                        "RUNNING" if step.ok is None else ("SUCCESS" if step.ok else "FAILED"))
                    label = step.summary or step.name or "Operation"
                    duration = f" · {max(0.0, float(step.dur or 0.0)):.1f}s" if step.dur else ""
                    started_at = float(getattr(getattr(step, "view", None), "started_at", 0.0) or 0.0)
                    when = time.strftime("%H:%M:%S", time.localtime(started_at)) + " | " if started_at > 0 else ""
                    lines.append(f"{when}{mark:7} | {step.name or 'operation'} · {label[:100]}{duration}")
                    if step.detail.strip() and (not step.ok or "warn" in step.detail.lower()):
                        lines.append(f"        | {step.detail.strip().splitlines()[0][:140]}")
        return lines

    def _log_events(self) -> list[dict]:
        """Full execution records and the current chat's real workflow milestones."""
        events: list[dict] = []
        session = self._current_session()
        ledger = session.get("ledger") if isinstance(session, dict) else None
        if not isinstance(ledger, dict) and self._turn_view is not None:
            ledger = self._turn_view.plan
        steps = ledger.get("steps") if isinstance(ledger, dict) else None
        if isinstance(steps, list) and steps:
            events.append({"kind": "workflow", "title": "Workflow · " + str(ledger.get("intent") or "current task"),
                           "steps": steps})
        for block in self.blocks:
            if block.kind == Block.ERROR:
                events.append({"status": "FAILED", "tool": "Execution", "title": block.text.strip(),
                               "error": block.text.strip()})
            elif block.kind == Block.QUESTION:
                events.append({"status": (block.detail or "pending").upper(), "tool": "User input",
                               "title": block.text.splitlines()[0] if block.text else "Question",
                               "result": block.reason})
            elif block.kind == Block.TOOL:
                if not block.steps:
                    if block.text.strip():
                        events.append({"status": "EVENT", "tool": "Oscar", "title": block.text.strip()})
                    continue
                for step_index, step in enumerate(block.steps):
                    if step.is_thought or step.plain:
                        continue
                    view = step.view
                    skipped = bool(getattr(view, "skipped", False)) if view is not None else False
                    status = "SKIPPED" if skipped else (
                        "RUNNING" if step.ok is None else ("SUCCESS" if step.ok else "FAILED"))
                    started = float(getattr(view, "started_at", 0.0) or 0.0)
                    timestamp = time.strftime("%H:%M:%S", time.localtime(started)) if started > 0 else ""
                    delta = getattr(view, "delta", {}) if view is not None else {}
                    delta_text = "; ".join(f"{key}: {', '.join(map(str, value))}"
                                            for key, value in delta.items() if value)
                    health = getattr(view, "health", None) if view is not None else None
                    validation = str(health.get("line") or "") if isinstance(health, dict) else ""
                    snapshot = str(getattr(view, "snapshot_path", "") or "") if view is not None else ""
                    events.append({
                        "status": status, "timestamp": timestamp, "tool": step.name,
                        "title": step.summary, "duration": float(step.dur or 0.0),
                        "input": (getattr(view, "args_summary", "") or getattr(view, "args_json", "")
                                  or "" if view is not None else ""),
                        "result": (getattr(view, "text", "") or step.detail or "" if view is not None else step.detail),
                        "delta": delta_text, "validation": validation,
                        "warning": str(health.get("warnings") or "") if isinstance(health, dict) and health.get("warnings") else "",
                        "error": str(health.get("errors") or "") if isinstance(health, dict) and health.get("errors") else "",
                        "snapshot": self._abs_snapshot(snapshot) if snapshot else "",
                        "snapshot_key": str(getattr(view, "call_id", "") or step.call_id) if snapshot else "",
                        "block_id": block.bid, "step_index": step_index,
                    })
        return events

    # -- the in-chat question surface (netguard.panel_surface) -----------------

    def _register_question_surface(self) -> None:
        """Offer in-chat question cards; the modal dialog stays the fallback (netguard decides)."""
        self._open_questions = getattr(self, "_open_questions", {})
        try:
            from . import netguard as _netguard

            _netguard.panel_surface = self._question_surface
        except Exception:   # noqa: BLE001
            pass

    def _question_append(self, text: str) -> Block:
        """A QUESTION card goes to the RUNNING TURN's chat (its own context), or the shown
        chat when nothing runs — the same isolation rule every turn write follows."""
        blocks = self.blocks
        turn_blocks = getattr(self, "_turn_blocks", None)
        if getattr(self, "worker", None) is not None and turn_blocks is not None:
            blocks = turn_blocks
        block = Block(Block.QUESTION, text)
        blocks.append(block)
        return block

    def _question_surface(self, info, choices, remaining_s, cancelled, take_remote, by_remote):
        """``netguard.panel_surface``: render the question as a card and wait. Returns the clicked
        ``Choice.key``, ``""`` (settled unanswered) or ``None`` (cannot show → modal dialog)."""
        if self._current_session() is None:
            return None
        from PySide import QtCore as _qc

        qid = str(info.get("id"))
        title = str(info.get("title") or "Oscar needs your decision")
        body = str(info.get("text") or "")
        block = self._question_append(f"{title}\n{body}")
        block.actions = [(f"oscar://answer/{qid}/{choice.key}", choice.label) for choice in choices]
        block.key = qid
        block.detail = "pending"
        self._dirty = True
        self._render()
        # The bar under the transcript is the quick answer surface; the card above is the record.
        self._show_question_bar(f"{title}\n{body}" if body else title,
                                [choice.label for choice in (choices or [])], approve=True)
        self._bar_question_qid = qid      # the bar now speaks for THIS question (answer routing)

        loop = _qc.QEventLoop(self)
        settled: list[str] = []

        def settle(key: str) -> None:
            if settled:
                return
            settled.append(key)
            loop.quit()

        self._open_questions[qid] = {"settle": settle, "block": block,
                                     "choices": list(choices or [])}

        def _expire() -> None:
            settle("")

        timer = _qc.QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(_expire)
        timer.start(max(1, int(float(remaining_s) * 1000)))

        def _poll() -> None:
            if cancelled():
                settle("")
                return
            remote_choice = take_remote()
            if remote_choice is not None:
                by_remote.append(True)
                settle(remote_choice.key)

        poller = _qc.QTimer(self)
        poller.timeout.connect(_poll)
        poller.start(200)
        try:
            loop.exec_()
        finally:
            timer.stop()
            poller.stop()
            self._open_questions.pop(qid, None)

        key = settled[0] if settled else ""
        self._settle_question(block, key, choices or [], cancelled(), int(float(remaining_s)))
        return key or ""

    def _deliver_question_answer(self, qid: str, key: str) -> None:
        """Settle the open question ``qid`` with ``key`` — the ONE tail every answer path (card
        anchor, QuestionsBar) funnels into, so an answer reaches the waiting worker exactly once."""
        open_q = self._open_questions.pop(qid, None)   # close the double-click window first
        if open_q is None:
            self._flash_status("That question is no longer open.")
            return
        open_q["settle"](key)
        self._dirty = True
        self._render()

    def _on_question_answer(self, spec: str) -> None:
        """`oscar://answer/<id>/<key>` — a click on a QUESTION card's button."""
        qid, _, key = spec.partition("/")
        open_q = self._open_questions.get(qid)
        if open_q is None:
            self._flash_status("That question is no longer open.")
            return
        block = open_q["block"]
        if block not in self.blocks:
            # the card belongs to another chat: chat isolation — answer where it was asked
            self._flash_status("Open this chat to answer that question.")
            return
        choice = next((c for c in open_q["choices"] if c.key == key), None)
        if choice is not None and getattr(choice, "role", "") in ("RejectRole", "NoRole"):
            text, ok = QtWidgets.QInputDialog.getText(
                self, "Reason (optional)", "Tell Oscar why (optional):")
            if ok and text.strip():
                block.reason = text.strip()
        self._deliver_question_answer(qid, key)

    def _question_bar_answer(self, label: str) -> bool:
        """A QuestionsBar option click answers the netguard question the bar was armed with (the
        same waiting worker a card click settles). True = consumed — the label never falls through
        to the chat as a message while a question waits for it."""
        qid = getattr(self, "_bar_question_qid", None)
        if qid is None:
            return False                       # the bar is not showing a netguard question
        open_q = self._open_questions.get(qid)
        if open_q is None:
            self._flash_status("That question is no longer open.")
            return True                        # answered after it settled: report it, never re-send it
        if open_q["block"] not in self.blocks:
            # the question belongs to another chat: chat isolation — answer where it was asked
            self._flash_status("Open this chat to answer that question.")
            return True
        choice = next((c for c in open_q["choices"] if c.label == label), None)
        if choice is None:
            self._flash_status("Pick one of the listed choices.")
            return True                  # not an option: tell the user, do not send it as a message
        self._deliver_question_answer(qid, choice.key)
        return True

    def _question_bar_custom(self, text: str) -> bool:
        """A typed bar answer cannot settle a netguard question (its choices are keyed, not free
        text): say so and consume it — never send it into the chat as if it had been accepted."""
        if getattr(self, "_bar_question_qid", None) is None:
            return False
        self._flash_status("That question needs one of the listed choices.")
        return True

    def _settle_question(self, block, key: str, choices, was_cancelled: bool, timeout_s: int) -> None:
        """Flip the card to its final state (approved / rejected / timeout / cancelled)."""
        if key:
            role = next((getattr(c, "role", "") for c in choices if getattr(c, "key", "") == key), "")
            block.detail = "approved" if role in ("YesRole", "AcceptRole") else "rejected"
        elif was_cancelled:
            block.detail = "cancelled"
        else:
            block.detail = "timeout"
            block.reason = str(timeout_s or 120)
        self._dirty = True
        self._render()
        self._clear_question_bar()   # the question is no longer open

    # -- ask_user decision cards (tools_ask renders through here; the WORKER waits on askuser's registry) ----

    def _register_ask_surface(self) -> None:
        """Become ``tools_ask.surface``: the ask_user tool renders its decision/wizard cards here. The
        card call is render-and-return — the agent worker waits on ``askuser.REGISTRY``, not on it."""
        self._open_asks = getattr(self, "_open_asks", {})
        self._turn_asks = getattr(self, "_turn_asks", [])
        try:
            from . import tools_ask as _tools_ask

            _tools_ask.surface = self._ask_surface
        except Exception:   # noqa: BLE001
            pass

    def _ask_questions(self, request) -> list:
        return request.payload.get("wizard") or [request.payload]

    def _ask_active_index(self, state) -> int:
        return next(i for i in range(len(self._ask_questions(state["request"]))) if i not in state["answers"])

    def _ask_card_text(self, request, active: int) -> str:
        """The card body: the question headline (and wizard progress / answered summary)."""
        questions = self._ask_questions(request)
        lines = []
        if request.wizard:
            lines.append(f"Decision wizard — question {active + 1} of {len(questions)}")
        question = questions[active]
        lines.append(question["title"])
        if question.get("subtitle"):
            lines.append(question["subtitle"])
        for i in range(active):
            lines.append("answered: " + questions[i]["title"])
        return "\n".join(lines)

    def _ask_actions(self, request, active: int):
        return [(f"oscar://ask/{request.id}/{active}/{i}", option["label"])
                for i, option in enumerate(self._ask_questions(request)[active]["options"])]

    def _ask_bar(self, request, active: int) -> None:
        question = self._ask_questions(request)[active]
        title = question["title"] + (f"\n{question['subtitle']}" if question.get("subtitle") else "")
        self._show_question_bar(title, [o["label"] for o in question["options"]])
        self._bar_ask_id = request.id     # the bar now speaks for THIS decision card (answer routing)

    def _ask_surface(self, request) -> None:
        """``tools_ask.surface`` (GUI thread): append the decision card for ``request``, arm the bar and
        the settle poller. Raises when no chat is open or the card belongs to another chat — the tool
        then fails honestly instead of leaving an unanswerable question hanging."""
        session = self._current_session()
        if session is None:
            raise RuntimeError("no chat is open to show the decision card in")
        if str(session.get("id") or "") != request.conversation_id:
            raise RuntimeError("the decision belongs to another chat")
        block = self._question_append(self._ask_card_text(request, 0))
        block.key = request.id
        block.detail = "pending"
        block.actions = self._ask_actions(request, 0)
        state = {"request": request, "block": block, "answers": {}, "done": False}
        self._open_asks[request.id] = state
        self._dirty = True
        self._render()
        self._ask_bar(request, 0)

        from PySide import QtCore as _qc

        def _poll() -> None:
            if request.event.is_set():      # settled elsewhere: timeout, Stop, shutdown
                self._settle_ask(state)

        poller = _qc.QTimer(self)
        poller.timeout.connect(_poll)
        poller.start(200)
        state["poller"] = poller

    def _on_ask_answer(self, spec: str) -> None:
        """`oscar://ask/<id>/<question-index>/<option-index>` — an option click on a decision card."""
        rid, _, rest = spec.partition("/")
        qi_text, _, index_text = rest.partition("/")
        state = self._open_asks.get(rid)
        if state is None or state["done"]:
            self._flash_status("That decision is no longer open.")
            return
        try:
            qi, index = int(qi_text), int(index_text)
        except ValueError:
            return
        questions = self._ask_questions(state["request"])
        if not 0 <= qi < len(questions) or qi in state["answers"]:
            return
        options = questions[qi]["options"]
        if not 0 <= index < len(options):
            return
        state["answers"][qi] = {"answer": options[index]["label"], "index": index}
        self._ask_advance(state, qi)

    def _ask_advance(self, state, answered_qi: int) -> None:
        """Record the per-question answer, move the card to the next unanswered question, or resolve
        the request in the registry when the last one is in (the worker wakes on its Event)."""
        from . import askuser

        request = state["request"]
        questions = self._ask_questions(request)
        nxt = next((i for i in range(len(questions)) if i not in state["answers"]), None)
        if nxt is None:
            state["done"] = True
            session = self._current_session()
            conversation = str(session.get("id") or "") if session else request.conversation_id
            payload = ({"answers": [state["answers"][i] for i in range(len(questions))]}
                       if request.wizard else dict(state["answers"][0]))
            ok, err = askuser.REGISTRY.submit_answer(request.id, payload, conversation_id=conversation,
                                                     run_id=request.run_id)
            if not ok:
                state["done"] = False
                state["answers"].pop(answered_qi, None)   # the refused answer must not block the retry
                self._flash_status(f"Answer could not be submitted ({err}). Please try again.")
                return
            self._settle_ask(state)
            return
        block = state["block"]
        block.text = self._ask_card_text(request, nxt)
        block.actions = self._ask_actions(request, nxt)
        self._dirty = True
        self._render()
        self._ask_bar(request, nxt)

    def _open_ask_of_current_chat(self):
        session = self._current_session()
        chat = str(session.get("id") or "") if session else ""
        if not chat:
            return None
        for state in reversed(list(self._open_asks.values())):
            if not state["done"] and state["request"].conversation_id == chat:
                return state
        return None

    def _tracked_ask_state(self):
        """The decision-card state the bar was armed with: ``(state, err)`` — ``err`` flashes and
        consumes when the tracked card settled meanwhile or belongs to another chat."""
        rid = getattr(self, "_bar_ask_id", None)
        if rid is None:
            if getattr(self, "_bar_question_qid", None) is not None:
                return None, ""    # the bar speaks for a netguard question, never a decision card
            return self._open_ask_of_current_chat(), ""      # untracked: the per-chat fallback
        state = self._open_asks.get(rid)
        if state is None:
            return None, "That decision is no longer open."
        session = self._current_session()
        chat = str(session.get("id") or "") if session else ""
        if chat and state["request"].conversation_id != chat:
            return None, "Open this chat to answer that decision."
        return state, ""

    def _ask_bar_answer(self, label: str) -> bool:
        """A QuestionsBar option click answers the open decision card of THIS chat. True = consumed —
        the label never falls through to the chat as a message while the decision waits."""
        state, err = self._tracked_ask_state()
        if err:
            self._flash_status(err)
            return True
        if state is None:
            return False
        qi = self._ask_active_index(state)
        for i, option in enumerate(self._ask_questions(state["request"])[qi]["options"]):
            if option["label"] == label:
                state["answers"][qi] = {"answer": option["label"], "index": i}
                self._ask_advance(state, qi)
                return True
        if getattr(self, "_bar_ask_id", None) is not None:
            self._flash_status("Pick one of the listed choices.")
            return True                  # the bar speaks for this decision: no fall-through to send
        return False

    def _ask_bar_custom(self, text: str) -> bool:
        """A typed bar answer becomes the custom value of the open decision card (when it accepts
        one). True = consumed — never sent into the chat while the decision waits."""
        state, err = self._tracked_ask_state()
        if err:
            self._flash_status(err)
            return True
        if state is None:
            return False
        qi = self._ask_active_index(state)
        question = self._ask_questions(state["request"])[qi]
        if not (question.get("allow_free_text") or question.get("custom_value")):
            if getattr(self, "_bar_ask_id", None) is not None:
                self._flash_status("That decision takes one of the listed choices, not text.")
                return True              # the bar speaks for this decision: no fall-through to send
            return False
        state["answers"][qi] = {"answer": text, "index": -1, "value": text}
        self._ask_advance(state, qi)
        return True

    def _settle_ask(self, state) -> None:
        """Flip the decision card to its final state (answered summary / cancelled / timeout)."""
        from . import askuser

        request = state["request"]
        poller = state.pop("poller", None)
        if poller is not None:
            poller.stop()
            poller.deleteLater()
        self._open_asks.pop(request.id, None)
        self._turn_asks = getattr(self, "_turn_asks", None) or []
        self._turn_asks.append(request.record())      # persisted with the turn (blocks_meta["asks"])
        block = state["block"]
        if request.state == askuser.ANSWERED:
            answers = request.answer.get("answers") if request.wizard else [request.answer]
            block.detail = "answered"
            block.reason = " | ".join(str(a.get("answer")) for a in answers if a)
        else:
            block.detail = str(request.state)
            if request.state == askuser.TIMEOUT:
                # the honest duration the waiter allowed (the render otherwise assumes 120 s)
                block.reason = str(int(round(request.timeout_s)))
        self._dirty = True
        self._render()
        self._clear_question_bar()

    def _settle_typed_clarification(self, text: str, attachments) -> None:
        """Record a custom answer or a new request on this chat's latest open clarification."""
        if not text.strip() and not attachments:
            return
        block = next((b for b in reversed(self.blocks)
                      if b.kind == Block.QUESTION and b.detail == "clarification_pending"), None)
        if block is None:
            return
        try:
            from .flow_intent import answers_open_question

            answered = answers_open_question(
                text, self._session_ledger(), [str(path or "") for path in attachments])
        except Exception:  # noqa: BLE001 - presentation state must not prevent the user's send
            answered = bool(text.strip() or attachments)
        block.detail = "answered" if answered else "superseded"
        block.reason = text.strip() or ("Attached file" if len(attachments) == 1 else "Attached files")
        self._dirty = True
        self._render()
        self._clear_question_bar()   # a typed answer settled the open question

    def _on_clarification_option(self, spec: str) -> None:
        """Send a selected option as the next user turn in the chat that asked it."""
        qid, _, index_text = spec.partition("/")
        block = next((b for b in self.blocks if b.kind == Block.QUESTION and b.key == qid), None)
        if block is None or block.detail != "clarification_pending":
            self._flash_status("That clarification is no longer open.")
            return
        try:
            index = int(index_text)
            target, answer = block.actions[index]
        except (ValueError, IndexError):
            self._flash_status("That answer option is no longer available.")
            return
        if not target.startswith(f"oscar://clarify/{qid}/") or self._busy():
            self._flash_status("This clarification cannot be answered right now.")
            return
        block.detail = "answered"
        block.reason = answer
        self._render()
        self._clear_question_bar()
        self._send_text(answer)

    # -- the chat question bar (seam: panel_questions.QuestionsBar) -----------

    def _mount_questions_bar(self) -> QtWidgets.QWidget:
        """The strip below the transcript, above the composer: ``panel_questions.QuestionsBar``
        when that module exists, a hidden placeholder while it does not (a missing module never
        crashes the shell). ``answered`` / ``custom_answered`` route into the existing steer
        path: an option click answers the open clarification and sends; a custom answer settles
        the question and sends. The in-transcript question cards are unchanged."""
        try:
            from .panel_questions import QuestionsBar
            bar = QuestionsBar()
        except ImportError:
            bar = QtWidgets.QWidget()
            bar.setObjectName("oscarQuestionsBar")
            bar.hide()
            return bar
        bar.setObjectName("oscarQuestionsBar")
        loading = getattr(bar, "load", None)
        if callable(loading):
            self._questions_show = loading       # present a question in the bar
        clearing = getattr(bar, "reset", None)
        if callable(clearing):
            self._questions_clear = clearing
        answered = getattr(bar, "answered", None)
        if answered is not None and hasattr(answered, "connect"):
            answered.connect(lambda text: self._on_bar_answer(str(text)))
        custom = getattr(bar, "custom_answered", None)
        if custom is not None and hasattr(custom, "connect"):
            custom.connect(lambda text: self._on_bar_custom(str(text)))
        return bar

    def _show_question_bar(self, text: str, options, *, approve: bool = False) -> None:
        """Put an OPEN question in the bar under the transcript (a no-op when the seam is absent).

        ``options`` are the (label) choices; ``text`` may carry a title and body on two lines.
        The transcript card remains the durable record — the bar is the quick way to answer it.
        A new question re-arms the bar: previous answer routing (netguard question / decision
        card) is dropped — the caller re-declares what the bar now speaks for."""
        self._bar_question_qid = None
        self._bar_ask_id = None
        show = getattr(self, "_questions_show", None)
        if not callable(show):
            return
        labels = [str(option) for option in (options or [])][:6]
        try:
            show({"text": str(text or ""), "options": labels, "approve": bool(approve),
                  "allow_custom": True})
        except Exception as e:   # noqa: BLE001 - the transcript cards still work
            self._log_panel_error("question_bar", e)

    def _clear_question_bar(self) -> None:
        self._bar_question_qid = None
        self._bar_ask_id = None
        clear = getattr(self, "_questions_clear", None)
        if callable(clear):
            try:
                clear()
            except Exception as e:   # noqa: BLE001
                self._log_panel_error("question_bar_clear", e)

    def _on_bar_answer(self, text: str) -> None:
        """``QuestionsBar.answered``: the chosen option settles whatever the bar speaks for — the
        open decision card, the open netguard question, or the open clarification (which becomes
        the next user message). An answer for a waiting question is NEVER sent as a message."""
        answer = str(text or "").strip()
        if not answer:
            return
        if self._ask_bar_answer(answer):
            return
        if self._question_bar_answer(answer):
            return
        block = next((b for b in reversed(self.blocks)
                      if b.kind == Block.QUESTION and b.detail == "clarification_pending"), None)
        if block is not None:
            block.detail = "answered"
            block.reason = answer
            self._dirty = True
            self._render()
        self._send_or_steer(answer)

    def _on_bar_custom(self, text: str) -> None:
        """``QuestionsBar.custom_answered``: a typed answer settles the open clarification the
        same way the composer's would, then sends; a waiting question that cannot take free text
        reports that instead of silently sending the text into the chat."""
        answer = str(text or "").strip()
        if not answer:
            return
        if self._ask_bar_custom(answer):
            return
        if self._question_bar_custom(answer):
            return
        self._settle_typed_clarification(answer, [])
        self._send_or_steer(answer)

    def _send_or_steer(self, text: str) -> None:
        """Deliver ``text`` from the question bar: a new turn when idle, the steer path
        (queued bubble) while a turn runs."""
        if self._busy():
            self.input.setPlainText(text)
            self._steer_while_busy()
            return
        self._send_text(text)


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
