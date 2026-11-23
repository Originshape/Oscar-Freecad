"""The panel's visual system (UI redesign 2026-11-15, shell redesign 2026-11-16,
card-chat retheme 2026-11-17): one place for the design tokens, the stylesheet built
from them and the two subtle animations the shell uses.

The card-chat palette (owner spec 2026-11-17): a flat dark engineering look — surfaces
``bg < panel < card``, 1px borders, no gradients or bevels. TEAL is the single
selected/answered/success accent; ATTENTION YELLOW marks decision cards and "working";
danger and link complete the set. ``TOKENS`` is the constant table (a LIGHT variant can
be added later by defining a second dict with the same keys); every value the panel and
its HTML fragments paint comes from here so the widget side and the rich-text side
cannot drift apart. The old palette-following helpers (``panel_blocks._muted`` & co)
delegate here as well.

Radii: 8 px on cards, 6 px on buttons, 4 px on the smallest chips. Font: the system UI
face at 10 pt; technical values use ``MONO``.

Animations are deliberately tiny: 120–180 ms opacity fades (tab switch, a strip or card
appearing), driven by QPropertyAnimation on the Qt event loop — nothing blocks the GUI
thread and no fade ever runs on the chat transcript itself (re-rendering the rich-text
document per frame would fight the streaming throttle).
"""

# The card-chat tokens (owner spec 2026-11-17). Flat surfaces, 1px borders, no shadows/gradients.
# Keys are stable API: panel_blocks / panel_render (_Q_TOKENS mirror) / the widget modules read them.
TOKENS = {
    "bg": "#0f1115",            # the dock's own background — the darkest surface
    "panel": "#151820",         # tab pages
    "bg_card": "#1b1f29",       # elevated cards
    "bg_recessed": "#0f1115",   # sunk surfaces (composer, tables) read as carved out of their parent
    "border": "#272c38",
    "user_bg": "rgba(60,201,192,0.14)",   # the user's bubble: a flat teal wash (HTML background value)
    "accent": "#3cc9c0",        # teal — selected / answered / success
    "success": "#3cc9c0",       # the same teal by design (one positive colour)
    "attention": "#e6b422",     # decision cards, working states
    "warn": "#e6b422",          # legacy key — the same attention yellow
    "danger": "#ef6f61",
    "link": "#6ea8ff",
    "text": "#e6e8ee",
    "text_2": "#8a90a0",        # muted
    "text_3": "#6f7583",        # the faintest tier (queued labels, "+N more")
}

RADIUS = 8           # cards
RADIUS_BTN = 6       # buttons
RADIUS_SM = 4        # the smallest chips
BORDER = f"1px solid {TOKENS['border']}"

SP_XS = 3           # inside compact rows (header buttons, tab bar)
SP = 6              # the panel's default gap

MONO = "Consolas,'Cascadia Mono','Courier New',monospace"   # technical values in chat HTML
FONT_PT = 10        # the system UI face the QSS asks for
FONT_PX = 13        # px equivalent for the HTML fragments (kept: fragments size in pt via the panel)

FADE_MS = 150       # the one animation duration (spec: 120–180 ms, subtle)


def panel_qss() -> str:
    """The stylesheet for the OscarPanel root widget (and everything it contains).

    The root carries ``oscarTheme=true`` (a property, not an objectName — the dock already
    owns the "OscarPanel" name); every selector below is scoped by it, so only the panel's
    own children are styled, never FreeCAD's.
    """
    t = TOKENS
    return f"""
    QWidget[oscarTheme="true"] {{ background: {t['bg']}; color: {t['text']}; font-size: {FONT_PT}pt; }}
    QWidget[oscarTheme="true"] > QWidget {{ background: transparent; }}
    QWidget[oscarTheme="true"] QLabel {{ color: {t['text']}; background: transparent; }}
    QWidget[oscarTheme="true"] QLabel[muted="true"] {{ color: {t['text_2']}; }}
    QWidget[oscarTheme="true"] QComboBox {{
        background: {t['bg_recessed']}; border: {BORDER}; border-radius: {RADIUS_BTN}px;
        padding: 2px 6px; color: {t['text']};
    }}
    QWidget[oscarTheme="true"] QComboBox QAbstractItemView {{
        background: {t['bg_card']}; border: {BORDER}; color: {t['text']};
        selection-background-color: {t['accent']}; selection-color: #0f1115;
    }}
    QWidget[oscarTheme="true"] QToolButton {{
        background: transparent; border: none; border-radius: {RADIUS_BTN}px;
        padding: 2px 5px; color: {t['text_2']};
    }}
    QWidget[oscarTheme="true"] QToolButton:hover {{ background: {t['bg_card']}; color: {t['text']}; }}
    QWidget[oscarTheme="true"] QToolButton:pressed {{ background: {t['border']}; }}
    QWidget[oscarTheme="true"] QToolButton:checked {{ color: {t['text']}; border-bottom: 2px solid {t['accent']}; }}
    QWidget[oscarTheme="true"] QMenu {{
        background: {t['bg_card']}; border: {BORDER}; color: {t['text']}; padding: 4px;
    }}
    QWidget[oscarTheme="true"] QMenu::item {{ padding: 4px 18px 4px 12px; border-radius: {RADIUS_SM}px; }}
    QWidget[oscarTheme="true"] QMenu::item:selected {{ background: {t['accent']}; color: #0f1115; }}
    QWidget[oscarTheme="true"] QMenu::separator {{ height: 1px; background: {t['border']}; margin: 4px 8px; }}
    /* tooltips follow the theme everywhere the panel raises one */
    QToolTip {{ background: {t['bg_card']}; color: {t['text']}; border: {BORDER};
        border-radius: {RADIUS_SM}px; padding: 4px 8px; font-size: {FONT_PT}pt; }}
    /* the composer keeps profile and model selection separate and compact */
    QLabel#oscarLogoLabel {{ padding: 0 2px; }}
    QComboBox#oscarChatSelector {{ padding: 2px 4px; }}
    QToolButton#oscarProfileSelector, QToolButton#oscarModelSelector {{
        min-width: 68px; padding: 3px 5px; color: {t['text']};
        background: {t['bg_recessed']}; border: {BORDER}; border-radius: {RADIUS_BTN}px;
        text-align: left;
    }}
    QToolButton#oscarProfileSelector {{ max-width: 126px; }}
    QToolButton#oscarModelSelector {{ max-width: 140px; }}
    QToolButton#oscarProfileSelector:hover, QToolButton#oscarModelSelector:hover {{
        border-color: {t['accent']};
    }}
    QToolButton#oscarProfileSelector:disabled, QToolButton#oscarModelSelector:disabled {{
        color: {t['text_3']};
    }}
    /* tabs: text-only, flat, 2px teal underline on the active tab */
    QTabWidget#oscarTabs::pane {{ background: {t['panel']}; border: none;
        border-top: 1px solid {t['border']}; top: -1px; }}
    QTabWidget#oscarTabs > QTabBar {{ background: transparent; }}
    QTabWidget#oscarTabs > QTabBar::tab {{
        background: transparent; border: none; border-bottom: 2px solid transparent;
        padding: 5px 10px 4px 10px; margin: 0 2px; color: {t['text_2']};
    }}
    QTabWidget#oscarTabs > QTabBar::tab:hover {{ color: {t['text']}; }}
    QTabWidget#oscarTabs > QTabBar::tab:selected {{
        color: {t['text']}; border-bottom: 2px solid {t['accent']};
    }}
    /* the composer: a recessed frame that lights its border on focus */
    QFrame#oscarComposer {{
        background: {t['bg_recessed']}; border: {BORDER}; border-radius: {RADIUS}px;
    }}
    QFrame#oscarComposer[focused="true"] {{ border: 1px solid {t['accent']}; }}
    QFrame#oscarComposer[sending="true"] {{ border-color: {t['attention']}; }}
    QFrame#oscarComposer QPlainTextEdit {{
        background: transparent; border: none; color: {t['text']}; font-size: {FONT_PT}pt;
    }}
    QWidget[oscarTheme="true"] QPushButton {{
        background: {t['bg_card']}; color: {t['text']}; border: {BORDER};
        border-radius: {RADIUS_BTN}px; padding: 3px 12px;
    }}
    QWidget[oscarTheme="true"] QPushButton:hover {{ border-color: {t['accent']}; }}
    QWidget[oscarTheme="true"] QPushButton#oscarSendButton {{
        background: {t['accent']}; color: #0f1115; border: none; border-radius: {RADIUS_BTN}px;
        padding: 4px 0; font-weight: 600;
    }}
    QWidget[oscarTheme="true"] QPushButton#oscarSendButton:disabled {{
        background: {t['border']}; color: {t['text_2']};
    }}
    QWidget[oscarTheme="true"] QPushButton#oscarStopButton {{
        background: {t['danger']}; color: #0f1115; border: none; border-radius: {RADIUS_BTN}px;
        padding: 4px 0; font-weight: 600;
    }}
    QWidget[oscarTheme="true"] QPushButton#oscarParamsApply {{
        background: {t['accent']}; color: #0f1115; border: none; font-weight: 600;
    }}
    QWidget[oscarTheme="true"] QPushButton#oscarParamsApply:disabled {{
        background: {t['bg_card']}; color: {t['text_3']}; border: {BORDER};
    }}
    /* Parameters table + the Logs browser live on recessed card surfaces */
    QTableWidget#oscarParamsTable {{
        background: {t['bg_recessed']}; border: {BORDER}; border-radius: {RADIUS_SM}px;
        color: {t['text']}; gridline-color: {t['border']};
        selection-background-color: {t['accent']}; selection-color: #0f1115;
        alternate-background-color: {t['bg_card']};
    }}
    QTableWidget#oscarParamsTable QHeaderView::section {{
        background: {t['bg_card']}; color: {t['text_2']}; border: none;
        border-bottom: {BORDER}; padding: 3px 6px; font-weight: 600;
    }}
    QLabel#oscarParamsHint {{ color: {t['text_2']}; padding: 8px; }}
    QLabel#oscarParamsSheet {{ color: {t['text_2']}; font-size: 12px; }}
    QTextBrowser#oscarLog, QTextBrowser#oscarLogView {{
        background: {t['bg_recessed']}; border: {BORDER}; border-radius: {RADIUS_SM}px;
        color: {t['text']};
    }}
    QWidget[oscarTheme="true"] QLineEdit {{
        background: {t['bg_recessed']}; border: {BORDER}; border-radius: {RADIUS_BTN}px;
        padding: 2px 6px; color: {t['text']};
    }}
    QWidget[oscarTheme="true"] QLineEdit:focus {{ border-color: {t['accent']}; }}
    QLineEdit#oscarParamsSearch {{
        background: {t['bg_recessed']}; border: {BORDER}; border-radius: {RADIUS_BTN}px;
        padding: 2px 6px; color: {t['text']};
    }}
    QComboBox#timelineStatusFilter {{ min-width: 94px; font-size: 12px; }}
    QLabel#timeline-turn-header {{ color: {t['text_2']}; font-size: 11px; font-weight: 700;
        padding: 8px 2px 2px 2px; }}
    QLabel#timeline-group-header {{ color: {t['text_3']}; font-size: 11px; font-weight: 600;
        padding: 3px 2px 2px 2px; }}
    /* the credits chip: quiet, monospace, never competing with the send button */
    QLabel#oscarCredits {{ color: {t['text_3']}; font-family: {MONO}; }}
    /* the Logs activity timeline (panel_timeline) — it styles itself by objectName only */
    QScrollArea#timelineScroll {{ background: {t['bg_recessed']}; border: {BORDER};
        border-radius: {RADIUS_SM}px; }}
    QWidget#timelineContent {{ background: {t['bg_recessed']}; }}
    QLabel#timeline-empty {{ color: {t['text_2']}; padding: 8px; }}
    QLabel#timeline-entry-stamp {{ color: {t['text_2']}; font-size: 11px; }}
    QLabel#timeline-entry-explain {{ color: {t['text']}; }}
    QLabel#timeline-entry-result {{ color: {t['text_2']}; font-size: 11px; }}
    QLabel#timeline-entry-error {{ color: {t['danger']}; font-size: 11px; }}
    QFrame#timeline-entry-sep {{ color: {t['border']}; }}
    QToolButton#timeline-entry-toggle, QToolButton#timeline-entry-thumb {{
        background: transparent; border: none; color: {t['text_2']}; padding: 0 4px;
    }}
    QToolButton#timeline-entry-thumb {{ border: 1px solid {t['border']}; border-radius: {RADIUS_SM}px; }}
    QToolButton#timeline-entry-toggle:hover, QToolButton#timeline-entry-thumb:hover {{
        border-color: {t['accent']};
    }}
    QPlainTextEdit#timeline-entry-raw {{
        background: {t['bg']}; border: {BORDER}; border-radius: {RADIUS_SM}px;
        color: {t['text_2']}; font-size: 11px;
    }}
    /* status dots — colour is the only status channel here, so keep them distinct */
    QLabel#timeline-dot-running {{ background: {t['attention']}; border-radius: 5px; }}
    QLabel#timeline-dot-completed {{ background: {t['success']}; border-radius: 5px; }}
    QLabel#timeline-dot-failed {{ background: {t['danger']}; border-radius: 5px; }}
    QLabel#timeline-dot-warning {{ background: {t['warn']}; border-radius: 5px; }}
    QLabel#timeline-dot-skipped {{ background: {t['text_3']}; border-radius: 5px; }}
    /* the in-chat question bar (panel_questions) */
    QWidget#oscarQuestionsBar {{
        background: {t['bg_card']}; border: {BORDER}; border-left: 2px solid {t['accent']};
        border-radius: {RADIUS_SM}px;
    }}
    QLabel#oscarQuestionText {{ color: {t['text']}; font-weight: 600; }}
    QLabel#oscarQuestionRecorded {{ color: {t['success']}; font-size: 11px; }}
    QLineEdit#oscarQuestionCustom {{
        background: {t['bg_recessed']}; border: {BORDER}; border-radius: {RADIUS_BTN}px;
        padding: 2px 6px; color: {t['text']};
    }}
    QPushButton#oscarQuestionAccept {{ border-color: {t['success']}; color: {t['success']}; }}
    QPushButton#oscarQuestionReject {{ border-color: {t['danger']}; color: {t['danger']}; }}
    /* thin flat scrollbars: 8 px, rounded, no trough decoration */
    QWidget[oscarTheme="true"] QScrollBar:vertical {{
        background: transparent; width: 8px; margin: 0;
    }}
    QWidget[oscarTheme="true"] QScrollBar::handle:vertical {{
        background: {t['border']}; border-radius: 4px; min-height: 24px;
    }}
    QWidget[oscarTheme="true"] QScrollBar::handle:vertical:hover {{ background: {t['text_3']}; }}
    QWidget[oscarTheme="true"] QScrollBar:horizontal {{
        background: transparent; height: 8px; margin: 0;
    }}
    QWidget[oscarTheme="true"] QScrollBar::handle:horizontal {{
        background: {t['border']}; border-radius: 4px; min-width: 24px;
    }}
    QWidget[oscarTheme="true"] QScrollBar::handle:horizontal:hover {{ background: {t['text_3']}; }}
    QWidget[oscarTheme="true"] QScrollBar::add-line, QWidget[oscarTheme="true"] QScrollBar::sub-line {{
        background: none; border: none; height: 0; width: 0;
    }}
    QWidget[oscarTheme="true"] QScrollBar::add-page, QWidget[oscarTheme="true"] QScrollBar::sub-page {{
        background: none;
    }}
    """


def fade_in(widget, ms: int = FADE_MS) -> None:
    """Fade ``widget`` from 0 to full opacity over ``ms`` (clamped to the 120–180 ms band).

    A QPropertyAnimation on the Qt event loop — it never blocks the GUI thread. The
    opacity effect is dropped again when the fade ends, so a faded-in widget does not keep
    rendering through an offscreen buffer. Qt is imported here, not at module level, so
    the token table above stays importable by Qt-free tooling.
    """
    if widget is None:
        return
    from PySide import QtCore, QtWidgets

    duration = max(120, min(180, int(ms)))
    previous = getattr(widget, "_oscar_fade", None)
    if previous is not None:
        try:
            previous.stop()
        except RuntimeError:   # the C++ object is already gone
            pass
    try:
        effect = QtWidgets.QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.0)
        anim = QtCore.QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        def _drop_effect() -> None:
            try:
                widget.setGraphicsEffect(None)
            except RuntimeError:
                pass

        anim.finished.connect(_drop_effect)
        widget._oscar_fade = anim
        widget.setGraphicsEffect(effect)
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
    except Exception:   # noqa: BLE001 - an animation is a nicety, never a reason to fail
        pass


def html_card(body: str, *, accent: str = "", padding: str = "6px 10px") -> str:
    """An elevated card div for the chat HTML fragments (flat, 1px border)."""
    t = TOKENS
    return (f"<div style='background:{t['bg_card']};border:1px solid {t['border']};"
            f"border-radius:{RADIUS}px;padding:{padding};margin:4px 0;"
            + (f"border-left:2px solid {accent};" if accent else "")
            + f">{body}</div>")


def mono(text: str) -> str:
    """Wrap a technical value in the mono face inside chat HTML."""
    return f"<span style='font-family:{MONO};'>{text}</span>"
