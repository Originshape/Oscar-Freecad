"""GUI-side conversation objects of the Oscar panel: ``Step``, ``Block``, the log browser and the theme colours.

Split out of ``panel.py`` (which re-exports every name, so ``from oscar.panel import Block, Step``
keeps working). These classes only build Qt rich-text HTML; all pure markdown / row / badge logic
lives in ``panel_render`` (no Qt). The colour helpers read the live Qt palette, so this module needs
QtGui but the only widget it defines is ``LogBrowser``.

A ``Step`` is one process entry of a turn card. A tool step may carry ``view`` (a ``panel_render.StepView``
filled from the worker's ``step_started`` / ``step_finished`` payloads): it is then drawn as the titled rich
row (delta, health, thumbnail, code / output on demand) instead of the legacy one-line row.

``ThumbCache`` decodes each step snapshot once (scaled to 2x the displayed width) and ``LogBrowser`` serves
those images to the document by URL (``es-img://step/<id>``) from ``loadResource``, so a thumbnail survives
every ``setHtml`` of the log (the panel re-renders the whole document on each event).
"""

import html
import itertools
import os
import time
from collections import OrderedDict

import FreeCAD as App
from PySide import QtCore, QtGui, QtWidgets

from .panel_render import (DETAIL_LIMIT, StepView, account_card_html, clip_text, md_to_html, outcome_color,
                           restore_card_html, step_row_html, summary_block_html, thumb_url)

_BLOCK_SEQ = itertools.count(1)

_DETAIL_LIMIT = DETAIL_LIMIT  # characters of tool result shown when expanded (12000, was 4000)


def _font_size() -> int:
    """The chat font size (pt) from the ``OscarFontSize`` user preference (12 by default)."""
    return App.ParamGet("User parameter:BaseApp/Preferences/Units").GetInt("OscarFontSize", 12) or 12


def _accent() -> str:
    color = App.ParamGet("User parameter:BaseApp/Preferences/View").GetUnsigned("DefaultHighlightColor", 0x1E6EE8)
    return f"#{(color >> 16) & 0xFF:02x}{(color >> 8) & 0xFF:02x}{color & 0xFF:02x}"


def _fg() -> str:
    return "palette(text)"


def _muted() -> str:
    """Dimmed gray that works in Qt rich text (the CSS palette(mid) form
    does not reliably apply inside QTextBrowser HTML). Derived from the
    live theme's text color with alpha, so it reads as gray in light and
    dark themes alike."""
    try:
        color = QtGui.QGuiApplication.palette().color(QtGui.QPalette.WindowText)
        return f"rgba({color.red()},{color.green()},{color.blue()},165)"
    except Exception:
        return "#888888"


def _text_color() -> str:
    """Explicit theme text color (same rationale as _muted)."""
    try:
        color = QtGui.QGuiApplication.palette().color(QtGui.QPalette.WindowText)
        return color.name()
    except Exception:
        return "#dddddd"


class Step:
    """One process entry inside a turn's activity card: a tool call, or a
    reasoning round (``is_thought=True``) — rendered inline, gray."""

    def __init__(self, name: str = "", summary: str = "", detail: str = "",
                 ok: bool | None = None, dur: float = 0.0, is_thought: bool = False):
        self.name = name or ("thought" if is_thought else "tool")
        self.summary = summary or self.name
        self.detail = detail
        self.ok = ok          # None while running (tool steps)
        self.dur = dur        # 0.0 while a thought streams
        self.expanded = False
        self.is_thought = is_thought
        self.plain = False    # static inline note (folded pre-tool remark)
        self.live = False     # plain note currently streaming in-card
        self.t0 = 0.0         # live start (thoughts)
        self.view: StepView | None = None   # rich payload of a tool step (step_started / step_finished)
        self.call_id = ""     # the panel's unique key of the call (panel_render.unique_call_id)

    def sig(self) -> tuple:
        """The part of the Block signature that belongs to this step."""
        view = self.view
        return (self.name, self.summary, hash(self.detail), self.ok, self.dur, self.expanded, self.is_thought,
                self.plain, self.live, (view.rev, view.running) if view is not None else None)

    def html(self, bid: int, index: int) -> str:
        if self.plain:
            return (
                f"<div style='font-style:italic;color:{_muted()};"
                f"padding:2px 0 2px 14px;'>{md_to_html(self.summary)}</div>"
            )
        if self.is_thought:
            return self._thought_html(bid, index)
        if self.view is not None:
            return step_row_html(self.view, bid, index, muted=_muted(), expanded=self.expanded)
        if self.ok is None:
            head = f"◌ {self.summary} …"
        else:
            mark = "✓" if self.ok else "✗"
            head = f"{mark} {self.summary}" + (f" · {self.dur:.1f}s" if self.dur else "")
        anchor = (
            f"<a href='step:{bid}:{index}' style='color:{_muted()};"
            f"text-decoration:none;'>{html.escape(head)}</a>"
        )
        # Tool steps render as compact bordered rows (z.ai-style), with the
        # full result folding open inside the row.
        row = (
            "border:1px solid rgba(128,128,128,0.28);border-radius:8px;"
            "padding:4px 8px;margin:2px 0 2px 14px;"
        )
        if self.expanded and self.detail:
            shown = clip_text(self.detail, _DETAIL_LIMIT)  # one ellipsis, never two
            return (
                f"<div style='{row}'>{anchor}"
                f"<div style='white-space:pre-wrap;padding:3px 0 0 4px;'>"
                f"{html.escape(shown)}</div></div>"
            )
        return f"<div style='{row}'>{anchor}</div>"

    def _thought_html(self, bid: int, index: int) -> str:
        if self.dur == 0 and self.t0:
            # streaming: header counting + recent tail, plain indented text
            elapsed = max(0.0, time.monotonic() - self.t0)
            head = f"✻ thinking… {elapsed:.0f}s"
            shown = self.detail[-2500:]
            if len(self.detail) > 2500:
                shown = "…\n" + shown
            return (
                f"<div style='padding:2px 0 2px 14px;'>"
                f"<span style='color:{_muted()};'>{html.escape(head)}</span>"
                f"<div style='font-style:italic;padding:2px 0 0 2px;'>"
                f"{md_to_html(shown)}</div></div>"
            )
        head = f"Thought · {self.dur:.1f}s"
        anchor = (
            f"<a href='step:{bid}:{index}' style='color:{_muted()};"
            f"text-decoration:none;'>▸ {html.escape(head)}</a>"
        )
        body = ""
        if self.expanded and self.detail:
            body = (
                f"<div style='font-style:italic;padding:2px 0 0 2px;'>"
                f"{md_to_html(clip_text(self.detail, _DETAIL_LIMIT))}</div>"
            )
        return f"<div style='padding:2px 0 2px 14px;'>{anchor}{body}</div>"


class Block:
    """One renderable conversation entry."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"        # activity card (turn's process) or a notice line
    ERROR = "error"
    SUMMARY = "summary"  # the Build Summary card of a turn (+ the Continue link after the round cap)
    RESTORE = "restore"  # the card shown when a saved chat is opened again
    ACCOUNT = "account"  # an OriginShape account card: ``detail`` is its kind (panel_render.ACCOUNT_CARD_KINDS)
    RATING = "rating"    # a sampled "How did Oscar do?" card under a turn: ``detail`` is its state (oscar.rating)

    def __init__(self, kind: str, text: str = "", detail: str = "", expanded: bool = False):
        self.kind = kind
        self.text = text
        self.detail = detail
        self.expanded = expanded
        self.t0 = 0.0           # card creation; drives the live Working… timer
        self.steps: list[Step] = []  # TOOL cards only
        self.snapshot: dict = {}     # doc undo depths at turn start
        self.revertible = False      # finished turn that changed documents
        self.reverted = False
        self.revert_enabled = True   # the panel clears it on every card but the latest non-reverted one
        self.workspace_message = ""  # TOOL cards of a turn the OriginShape workspace sent: that message's id (connect)
        self.queued = False          # USER bubble sent while a turn ran: waiting for the next step
        self.steer_key = ""          # queued bubbles: the exact text handed to ``AgentWorker.steer`` (typed text + attachment block)
        self.steer_text = ""         # queued bubbles: the text the user typed (what a re-send shows)
        self.header_kind = ""        # TOOL cards: outcome kind of the finished turn ("ok" | "warn" | "error" | "neutral" | "")
        self.actions: list[tuple[str, str]] = []   # SUMMARY / ACCOUNT blocks: (anchor target, label) links under the card
        self.meta: list[dict] = []   # RESTORE blocks: the persisted turn records (``blocks_meta``)
        self.messages = 0            # RESTORE blocks: number of restored messages
        self.thumbs: set[str] = set()  # RESTORE blocks: thumbnail ids the panel could load
        self.reason = ""             # ACCOUNT onboarding cards: why the account is offered ("signed_out" | "no_key")
        self.key = ""                # RATING cards: the random key of their links (``rating.new_key``)
        self.bid = next(_BLOCK_SEQ)

    @property
    def live(self) -> bool:
        """True while the turn that owns this card runs (``t0`` is set at creation and cleared when the
        turn ends), so the header never says "Done" mid-turn, also between rounds."""
        return bool(self.t0)

    def settled(self) -> bool:
        """True when ``html()`` depends only on the block's fields (no running clock): such a
        fragment can be cached by ``signature()``."""
        if self.kind == Block.TOOL and self.live:
            return False
        return not any(step.is_thought and not step.dur and step.t0 for step in self.steps)

    def signature(self) -> tuple:
        """A cheap value that changes whenever ``html()`` of a settled block would."""
        return (
            self.kind, hash(self.text), hash(self.detail), self.expanded, self.revertible, self.reverted,
            self.revert_enabled, self.queued, self.header_kind, tuple(self.actions), self.messages,
            len(self.meta), len(self.thumbs),
            tuple(s.sig() for s in self.steps),
        )

    def html(self) -> str:
        if self.kind == Block.USER:
            body = md_to_html(self.text)
            if self.queued:
                body += (f"<div style='color:{_muted()};font-size:small;'>"
                         f"queued · Oscar will see it at the next step</div>")
            return (
                f"<table width='100%' cellpadding='0' cellspacing='0'><tr>"
                f"<td align='right'><table cellpadding='8' style="
                f"'background:rgba(128,128,128,0.16);color:{_text_color()};"
                f"border-radius:14px;'><tr>"
                f"<td>{body}</td></tr></table></td></tr></table>"
            )
        if self.kind == Block.ASSISTANT:
            return (
                f"<div style='color:{_text_color()};margin:2px 0 2px 4px;'>"
                f"{md_to_html(self.text)}</div>"
            )
        if self.kind == Block.SUMMARY:
            return summary_block_html(self.text, self.actions, muted=_muted())
        if self.kind == Block.ACCOUNT:
            return account_card_html(self.text, self.actions, kind=self.detail, muted=_muted())
        if self.kind == Block.RATING:
            from .rating import card_html

            return card_html(self.key, self.detail, muted=_muted())
        if self.kind == Block.RESTORE:
            return restore_card_html(self.text, self.messages, self.meta, notice=self.detail, muted=_muted(),
                                     key=str(self.bid), thumbs=self.thumbs)
        if self.kind == Block.TOOL:
            live = self.live
            if live:
                header = f"✻ Working… {max(0.0, time.monotonic() - self.t0):.0f}s"
            else:
                header = self.text or "✻ Done"
            # The header toggles at ANY time — a live turn can be minimized;
            # a new streaming thought re-expands it (handled by the panel).
            caret = "▾" if self.expanded else "▸"
            # warning / error outcomes colour the header (it never reads as a plain "Done")
            color = outcome_color(self.header_kind, _muted()) if (not live and self.header_kind in ("warn", "error")) \
                else _muted()
            anchor = (
                f"<a href='tool:{self.bid}' style='color:{color};"
                f"text-decoration:none;'>{html.escape(f'{caret} {header}')}</a>"
            )
            if self.steps:  # process section: borderless, z.ai-style
                tail = ""
                if not live and self.revertible:
                    if self.reverted:
                        tail = f" <span style='color:{_muted()};'>· reverted ✓</span>"
                    elif self.revert_enabled:
                        tail = (
                            f" · <a href='revert:{self.bid}' style='color:{_muted()};'"
                            f"text-decoration:none;'>↩ revert changes</a>"
                        )
                if self.expanded:
                    steps_html = "".join(
                        step.html(self.bid, i) for i, step in enumerate(self.steps)
                    )
                    return f"<div style='margin:6px 0 2px 0;'>{anchor}{tail}{steps_html}</div>"
                return f"<div style='margin:6px 0 2px 0;'>{anchor}{tail}</div>"
            # plain notice line (ℹ …)
            return (
                f"<div style='color:{_muted()};font-size:small;"
                f"margin:3px 0 3px 4px;'>ℹ {html.escape(self.text)}</div>"
            )
        # error
        return (
            f"<div style='color:#e06c5a;'>⚠ {md_to_html(self.text)}</div>"
        )


class ThumbCache:
    """Decoded step thumbnails keyed by their ``thumb_url`` id: every PNG is decoded ONCE (scaled to
    ``WIDTH`` px, twice the displayed width, for a crisp image on a high-DPI screen) and kept in an LRU of at
    most ``limit`` entries. A file that is missing or cannot be decoded is remembered as failed so it is
    never probed again. GUI thread only."""

    WIDTH = 320

    def __init__(self, limit: int = 96):
        self.limit = max(1, int(limit))
        self._images: "OrderedDict[str, QtGui.QImage]" = OrderedDict()
        self._urls: dict[str, str] = {}   # normalized URL string -> key
        self._failed: set[str] = set()

    @staticmethod
    def url_key(key: str) -> str:
        """The normalized URL string the document asks for (``QUrl.toString`` of ``thumb_url(key)``)."""
        return QtCore.QUrl(thumb_url(key)).toString()

    def has(self, key: str) -> bool:
        return key in self._images

    def failed(self, key: str) -> bool:
        return key in self._failed

    def get(self, key: str) -> "QtGui.QImage | None":
        """The cached image of ``key`` (no I/O, no side effect on a miss)."""
        return self._images.get(key)

    def load(self, key: str, path: str, keep=()) -> "QtGui.QImage | None":
        """The image of ``key``: from the cache, else decoded from ``path`` (``None`` + remembered when the
        file is missing or not an image). Entries named in ``keep`` (the thumbnails the document needs right now)
        are never evicted to make room: the cache then grows past ``limit`` instead of dropping an image the
        very next pass would have to decode again (which never converged)."""
        image = self._images.get(key)
        if image is not None:
            self._images.move_to_end(key)
            return image
        if key in self._failed:
            return None
        if not path or not os.path.isfile(path):
            self._failed.add(key)
            return None
        reader = QtGui.QImageReader(path)
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            self._failed.add(key)
            return None
        if image.width() > self.WIDTH:
            image = image.scaledToWidth(self.WIDTH, QtCore.Qt.SmoothTransformation)
        self._images[key] = image
        self._urls[self.url_key(key)] = key
        while len(self._images) > self.limit:
            victim = next((k for k in self._images if k not in keep and k != key), None)
            if victim is None:
                break
            del self._images[victim]
            self._urls.pop(self.url_key(victim), None)
        return image

    def by_url(self, url: str) -> "QtGui.QImage | None":
        """The cached image whose normalized URL is ``url`` (what ``LogBrowser.loadResource`` is asked for)."""
        key = self._urls.get(url)
        return self._images.get(key) if key is not None else None

    def keys(self) -> list[str]:
        return list(self._images)

    def clear(self) -> None:
        self._images.clear()
        self._urls.clear()
        self._failed.clear()


def _image_resource_type():
    kind = getattr(QtGui.QTextDocument, "ImageResource", None)
    return kind if kind is not None else QtGui.QTextDocument.ResourceType.ImageResource


class LogBrowser(QtWidgets.QTextBrowser):
    """The chat log: a ``QTextBrowser`` that serves the thumbnails of ``thumbs`` for ``es-img://`` URLs.

    ``setHtml`` rebuilds the document (and Qt may drop resources registered with ``addResource`` before it);
    answering from ``loadResource`` makes the images independent of when the layout asks for them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thumbs: ThumbCache | None = None

    def loadResource(self, rtype, url):  # noqa: N802 - Qt override
        if self.thumbs is not None and url.scheme() == "es-img":
            image = self.thumbs.by_url(url.toString())
            if image is not None:
                return image
        return super().loadResource(rtype, url)

    def register_thumbnails(self, thumbs: ThumbCache) -> int:
        """Also register every cached image with the CURRENT document (``QTextDocument.addResource``) after a
        ``setHtml``; returns how many were registered. The ``loadResource`` override is the safety net."""
        self.thumbs = thumbs
        document = self.document()
        kind = _image_resource_type()
        count = 0
        for key in thumbs.keys():
            image = thumbs.get(key)
            if image is not None:
                document.addResource(kind, QtCore.QUrl(thumb_url(key)), image)
                count += 1
        return count

    def has_resource(self, key: str) -> bool:
        """True when the document can resolve the thumbnail ``key`` (what the GUI smoke test asserts after a
        second render)."""
        value = self.document().resource(_image_resource_type(), QtCore.QUrl(thumb_url(key)))
        return isinstance(value, (QtGui.QImage, QtGui.QPixmap)) and not value.isNull()
