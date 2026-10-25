"""Link handling of the Oscar panel, as a mixin of ``OscarPanel``.

Every anchor the chat can contain goes through ``_on_anchor`` (the log browser never opens links itself):

- ``tool:<bid>``, ``step:<bid>:<idx>``, ``revert:<bid>``;
- web / mail / file links, opened with the desktop (``_open_external``: a file link is only OPENED when it is a
  folder or a plain document, anything else is revealed in its folder);
- ``rewind:<bid>:<idx>`` restores the model to just after that step (``panel_rewind``);
- step rows: ``show:<bid>:<idx>`` selects and frames the objects the step added or changed, ``copy:<bid>:<idx>``
  puts the step's script (the full recorded file when there is one) on the clipboard;
- ``plan:toggle`` collapses / expands the long plan, ``continue:`` sends the plan's resume prompt after a
  round cap;
- ``oscar://artifact/<action>[/<arg>]`` (the artifacts strip): ``open-doc``, ``open-notes``, ``open-rebuild``,
  ``open-exports``, ``open-provenance``, ``open-dir``. The target is resolved ONLY against the files the worker
  reported for the current project (``panel._artifacts``), never joined from the link text: a link that comes
  from model text cannot reach any other file;
- ``oscar://rate/<card>/<1-5|later>`` (the sampled rating card, ``rating_gui``): acts only for a card this panel
  offered; the stars open a dialog and nothing is sent until the user presses Send;
- ``oscar://account/<action>`` (the ``ACCOUNT`` cards): ``signin`` and ``device`` open Settings and start that
  sign-in (the user still approves it in the browser), ``buy`` / ``subscribe`` / ``manage`` / ``waitlist`` /
  ``download`` open the account's own pages, ``refresh`` checks the account again; ``oscar://settings`` opens Settings. Which page opens comes
  from the account's configuration, never from the link, and no link does anything destructive (signing out is a menu
  entry only), so these are safe when the model writes them too.

The mixin relies on the host panel for ``blocks``, ``_busy``, ``_send_text``, ``_resume_text``, ``_artifacts``,
``_plan_collapsed``, ``_refresh_plan_strip``, ``_on_notice``, ``_flash_status``, ``_open_settings`` and
``_on_account_action``.
"""

import os

import FreeCAD as App
import FreeCADGui as Gui
from PySide import QtCore, QtGui

from .panel_blocks import Block
from .panel_follow import select_objects
from .panel_render import classify_file_link, parse_artifact_anchor

_ARTIFACT_KINDS = {"open-doc": "fcstd", "open-notes": "notes", "open-rebuild": "rebuild",
                   "open-provenance": "provenance"}
_ACCOUNT_ACTIONS = frozenset({"signin", "device", "buy", "subscribe", "manage", "waitlist", "download", "refresh"})
_MAX_COPY_BYTES = 2_000_000


class AnchorMixin:
    """Anchor methods of the panel (see the module docstring)."""

    def _on_anchor(self, url):
        """Slot of the log browser's ``anchorClicked`` (a ``QUrl``)."""
        target = url.toString()
        if target.startswith("oscar://artifact/"):
            target = url.toString(QtCore.QUrl.FullyEncoded)   # the arg is un-quoted once, by parse_artifact_anchor
        self._on_anchor_target(target)

    def _on_anchor_text(self, target: str) -> None:
        """Slot of a ``QLabel.linkActivated`` (the artifacts strip): same dispatch, the target is a string."""
        self._on_anchor_target(str(target))

    def _on_anchor_target(self, target: str) -> None:
        if target.startswith("oscar://artifact/"):
            self._handle_artifact_anchor(target)
            return
        if target.startswith("oscar://account/") or target.rstrip("/") == "oscar://settings":
            self._handle_account_anchor(target)
            return
        if target.startswith("oscar://rate/"):
            from . import rating_gui

            rating_gui.on_anchor(self, target)
            return
        if target.startswith(("http://", "https://", "file://", "mailto:")):
            self._open_external(target)
            return
        if target == "plan:toggle":
            self._plan_collapsed = not self._plan_collapsed
            self._refresh_plan_strip()
            return
        if target == "continue:":
            self._continue_turn()
            return
        if target.startswith(("show:", "copy:")):
            verb, _, rest = target.partition(":")
            step = self._step_at(rest)
            if verb == "show":
                self._show_step_objects(step)
            else:
                self._copy_step(step)
            return
        if target.startswith("rewind:"):
            self._on_rewind_anchor(target.partition(":")[2])
            return
        if target.startswith("revert:"):
            for block in self.blocks:
                if (target == f"revert:{block.bid}" and block.revertible
                        and not block.reverted and block.revert_enabled):
                    self._revert_turn(block)
                    return
            return
        for block in self.blocks:
            if target == f"tool:{block.bid}" and block.kind == Block.TOOL:
                block.expanded = not block.expanded
                self._render()
                return
            if target.startswith(f"step:{block.bid}:") and block.steps:
                try:
                    index = int(target.rsplit(":", 1)[1])
                    step = block.steps[index]
                except (ValueError, IndexError):
                    return
                step.expanded = not step.expanded
                self._render()
                return

    def _step_at(self, spec: str):
        """The ``Step`` addressed by ``"<bid>:<idx>"`` (``None`` when the card or the index does not exist)."""
        try:
            bid_text, idx_text = spec.split(":", 1)
            bid, idx = int(bid_text), int(idx_text)
        except ValueError:
            return None
        for block in self.blocks:
            if block.bid == bid and block.kind == Block.TOOL:
                return block.steps[idx] if 0 <= idx < len(block.steps) else None
        return None

    # -- external links ------------------------------------------------------------------------------

    def _open_external(self, target: str) -> None:
        """Open a web / mail / file link from the chat with the desktop's default handler (the log
        browser does not open links itself). A ``file://`` link is only OPENED when it is a folder or a
        plain document / image / CAD file; anything else (scripts, applications) is revealed in its
        folder, because the link text comes from the model."""
        url = QtCore.QUrl(target)
        if url.isLocalFile():
            path = url.toLocalFile()
            if not os.path.exists(path):
                self._on_notice(f"File not found: {path}")
                return
            if classify_file_link(path, os.path.isdir(path)) != "open":
                url = QtCore.QUrl.fromLocalFile(os.path.dirname(path))
        if not QtGui.QDesktopServices.openUrl(url):
            self._on_notice(f"Could not open {target}")

    def _open_local(self, path: str) -> None:
        """``_open_external`` for a local path (scripts are revealed, never launched)."""
        self._open_external(QtCore.QUrl.fromLocalFile(path).toString())

    # -- the account cards -------------------------------------------------------------------------------

    def _handle_account_anchor(self, target: str) -> None:
        """``oscar://settings`` and ``oscar://account/<action>``: only the exact actions in ``_ACCOUNT_ACTIONS``; the
        rest of the link (a path, a query) is ignored, and so is anything else."""
        if target.rstrip("/") == "oscar://settings":
            self._open_settings()
            return
        action = target[len("oscar://account/"):].split("?", 1)[0].split("#", 1)[0].strip("/")
        if action not in _ACCOUNT_ACTIONS:
            self._on_notice(f"Unknown account link: {action[:40] or target[:60]}")
            return
        self._on_account_action(action)

    # -- the artifacts strip ---------------------------------------------------------------------------

    def _artifact_target(self, action: str, arg: str) -> str:
        """The absolute path an ``oscar://artifact/<action>[/<arg>]`` link stands for, resolved against the
        files the worker reported (``""`` when nothing matches or the match lies outside the project)."""
        data = getattr(self, "_artifacts", None) or {}
        root = str(data.get("project_dir") or "")
        files = [f for f in (data.get("files") or []) if isinstance(f, dict)
                 and not str(f.get("name", "")).lower().endswith(".fcbak")]
        if action == "open-dir":
            return root
        if action == "open-exports":
            return os.path.join(root, "exports") if root else ""
        kind = _ARTIFACT_KINDS.get(action)
        if kind is None:
            return ""
        group = [f for f in files if f.get("kind") == kind]
        if action == "open-doc":
            wanted = arg or "model.FCStd"
            pick = next((f for f in group if str(f.get("name")) == wanted), None)
        else:
            pick = group[0] if group else None
        path = str(pick.get("path") or "") if pick else ""
        if path and root:
            try:
                if os.path.commonpath([os.path.abspath(root), os.path.abspath(path)]) != os.path.abspath(root):
                    return ""
            except ValueError:
                return ""
        return path

    def _handle_artifact_anchor(self, target: str) -> None:
        action, arg = parse_artifact_anchor(target)
        if action != "open-dir" and action != "open-exports" and action not in _ARTIFACT_KINDS:
            self._on_notice(f"Unknown artifact link: {action or target}")
            return
        path = self._artifact_target(action, arg)
        if not path or not os.path.exists(path):
            self._on_notice(f"File not found: {path or arg or action}")
            return
        if action == "open-doc":
            self._open_document(path)
        else:
            self._open_local(path)

    def _open_document(self, path: str) -> None:
        """Open a project FCStd in the application (never through the desktop: its default handler for
        ``.FCStd`` is another instance of this very app). A document that is already open is only activated."""
        target = os.path.normcase(os.path.abspath(path))
        try:
            for doc in App.listDocuments().values():
                if doc.FileName and os.path.normcase(os.path.abspath(doc.FileName)) == target:
                    try:
                        Gui.setActiveDocument(doc.Name)
                    except Exception:  # noqa: BLE001 - activation is cosmetic
                        pass
                    self._on_notice(f"{doc.Name} is already open.")
                    return
        except Exception:  # noqa: BLE001
            pass
        try:
            from .cad import get_engine

            result = get_engine().import_file(path, "")
        except Exception as e:  # noqa: BLE001
            self._on_notice(f"open failed: {e}")
            return
        if not result.get("success"):
            self._on_notice(f"open failed: {result.get('error')}")
            return
        self._on_notice(f"opened {result.get('document_name', os.path.basename(path))}")
        self._fit_view_after_turn(force=True)

    # -- step rows --------------------------------------------------------------------------------------

    def _show_step_objects(self, step) -> None:
        """``show:`` anchor: select the objects the step added / changed and frame them."""
        view = getattr(step, "view", None)
        names = view.changed_objects() if view is not None else []
        if not names:
            self._on_notice("This step did not add or change any objects.")
            return
        try:
            present = select_objects(App, Gui, names)
            if not present:
                self._on_notice("The objects of this step no longer exist (deleted or rewound).")
                return
            try:
                Gui.SendMsgToActiveView("ViewSelection")
            except Exception:  # noqa: BLE001
                from .cad.views import fit_view

                fit_view()
            self._fit_sig = self._camera_signature()   # an explicit framing: the next follow-fit starts from here
        except Exception as e:  # noqa: BLE001
            self._on_notice(f"Could not show the objects: {e}")

    def _copy_step(self, step) -> None:
        """``copy:`` anchor: the step's script (the recorded file when it exists, it is never clipped) or its
        arguments to the clipboard."""
        view = getattr(step, "view", None)
        text = ""
        if view is not None:
            text = view.code or view.args_json
            path = view.script_path
            if path and os.path.isfile(path) and os.path.getsize(path) <= _MAX_COPY_BYTES:
                try:
                    with open(path, encoding="utf-8") as f:
                        text = f.read() or text
                except (OSError, UnicodeDecodeError):
                    pass
        if not text:
            self._on_notice("Nothing to copy for this step.")
            return
        QtGui.QGuiApplication.clipboard().setText(text)
        self._flash_status(f"Copied {len(text)} characters to the clipboard.")
