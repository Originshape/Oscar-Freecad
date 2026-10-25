"""Oscar chat sessions: multiple conversations, persisted.

Each session is one "task chat": its full provider-message history (the
context the model needs to continue the task) plus metadata. Stored as
JSON under ``<app-support>/Oscar/chats/<id>.json`` with atomic
writes (the ``oscar.params`` pattern — never FreeCAD user parameters,
which are silently dropped on contended exits).

Rendered UI blocks are view state and are NOT persisted — on restore the
panel shows a compact summary card while the model-side history (what
actually matters for continuing the task) is intact.

Persisted-copy trim: oversized screenshots are dropped at save time
(old images are stubbed by context compaction anyway), only the newest
few tool-result images are written, and very long tool results are
clipped. The trim applies to the FILE only: the in-memory history, and so
what the model is sent, is never touched. Everything else round-trips
verbatim, including tool calls/results so a restored chat can keep
tool-looping.

Thread safety: ``save`` / ``persist_round`` may run on the GUI thread
(send, turn end, chat switch) and on the agent worker (after every
provider round) at the same time. One module-level lock serializes the
serialize + write + replace sequence, every write goes to a UNIQUE temp
file, and the history is serialized from a copy.

A chat id is a file name (``YYYYmmdd-HHMMSS-<6 hex>``): ``load`` / ``list_sessions`` take it from the name of the file
they read (never from its content), and ``save`` / ``delete`` refuse anything that is not a plain name, so a damaged or
hand-edited chat file can neither write outside this folder nor alias another chat. Temp files that a crash left behind
are removed by ``list_sessions`` once they are a day old.

``list_sessions`` is metadata-only: each chat has a sidecar
``<id>.meta.json`` (id, title, updated, message count) so the panel never
parses full chat files on the GUI thread. ``is_unused`` says from the chat FILE (never from the sidecar) whether a
stored chat is an empty leftover the panel may delete at start. One damaged chat never takes the panel down: ``load``
and ``list_sessions`` skip an unreadable file (any parse failure), ``load`` keeps the readable messages of a chat with
a damaged one and gives the default of every wrong-typed field.
"""

import dataclasses
import json
import os
import re
import tempfile
import threading
import time
import uuid
from typing import Any
from . import fsretry as _fsretry

from .providers import Message

_IMAGE_DROP_BYTES = 1_500_000  # per-image persistence cap
_KEEP_RESULT_IMAGES = 4        # newest tool-result images written to the file
_RESULT_CLIP_CHARS = 12_000    # tool-result content longer than this is clipped in the file
_RESULT_HEAD = 8_000
_RESULT_TAIL = 3_500

_SAVE_LOCK = threading.Lock()

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}")   # what a chat id may be: it becomes a file name


def chats_dir() -> str:
    """The chats folder (``<FreeCAD user app data>/Oscar/chats``), created on demand. Tests replace this
    function to point the store at a temp folder."""
    try:
        import FreeCAD

        base = os.path.join(FreeCAD.getUserAppDataDir(), "Oscar",
                            "chats")
    except Exception:
        base = os.path.join(os.path.expanduser("~"), ".local", "share",
                            "Oscar", "chats")
    os.makedirs(base, exist_ok=True)
    return base


def new_session(title: str = "") -> dict[str, Any]:
    """A fresh, empty session dict: ``id`` (``YYYYmmdd-HHMMSS-<6 hex>``), ``title`` (``New chat`` by default),
    ``created`` / ``updated``, an empty ``history`` and the optional keys ``project_dir`` / ``ledger`` (None) and
    ``blocks_meta`` (``[]``) that the panel fills."""
    return {
        "id": time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
        "title": title or "New chat",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "history": [],
        "project_dir": None,   # absolute folder of the chat's Project (oscar.project), set by the panel
        "ledger": None,        # the C5 ledger dict, set by the panel from plan_changed / the turn result
        "blocks_meta": [],     # compact JSON list of the rendered cards for the restore card
    }


def _clip_result(content: str) -> str:
    """First 8000 + marker + last 3500 characters of an over-long tool result (persisted copy only)."""
    if len(content) <= _RESULT_CLIP_CHARS:
        return content
    omitted = len(content) - _RESULT_HEAD - _RESULT_TAIL
    return (content[:_RESULT_HEAD]
            + f"\n...({omitted} chars omitted from the saved chat copy)...\n"
            + content[-_RESULT_TAIL:])


def _message_to_dict(m: Message, *, keep_result_images: bool = True,
                     clip_results: bool = False) -> dict[str, Any]:
    """JSON-ready dict of one message (a copy: ``m`` is never modified).

    Images above the 1.5 MB per-image cap are blanked. With ``keep_result_images=False`` every
    tool-result image is blanked (``save`` keeps only the newest few); with ``clip_results=True`` a
    tool-result ``content`` over 12000 characters is clipped. The defaults keep the historic behaviour.
    """
    data = dataclasses.asdict(m)
    for result in data.get("tool_results") or []:
        if not keep_result_images or len(result.get("image_b64") or "") > _IMAGE_DROP_BYTES:
            result["image_b64"] = ""
        if clip_results and isinstance(result.get("content"), str):
            result["content"] = _clip_result(result["content"])
    for i, image in enumerate(data.get("images") or []):
        if len(image or "") > _IMAGE_DROP_BYTES:
            data["images"][i] = ""
    return data


def _message_from_dict(data: dict[str, Any]) -> Message:
    from .providers import ToolCall, ToolResult

    calls = [ToolCall(**{k: v for k, v in c.items()
                         if k in ("id", "name", "arguments", "args_error", "raw_args")})
             for c in data.get("tool_calls") or []]
    results = [ToolResult(**{k: v for k, v in r.items()
                             if k in ("tool_call_id", "name", "content",
                                      "is_error", "image_b64")})
               for r in data.get("tool_results") or []]
    return Message(
        role=data.get("role", "user"),
        content=data.get("content", ""),
        tool_calls=calls,
        tool_results=results,
        images=[b for b in (data.get("images") or []) if b],
        provider_blocks=data.get("provider_blocks") or [],
        synthetic=bool(data.get("synthetic", False)),
        tag=str(data.get("tag", "") or ""),
    )


def _messages_from_list(raw: Any) -> list[Message]:
    """The ``Message`` objects of a stored ``messages`` value. A value that is not a list gives ``[]`` and an entry that
    cannot be decoded (not an object, a tool call without an id, wrong types) is skipped, so one damaged message never
    makes the whole chat unreadable."""
    if not isinstance(raw, list):
        return []
    out: list[Message] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(_message_from_dict(entry))
        except (TypeError, ValueError, AttributeError, KeyError):
            continue
    return out


def _atomic_write_json(path: str, payload: Any) -> None:
    """Write ``payload`` as JSON to ``path`` through a unique temp file in the same directory + os.replace."""
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, default=str)
            f.flush()
        _fsretry.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _sidecar_path(directory: str, session_id: str) -> str:
    return os.path.join(directory, f"{session_id}.meta.json")


def _newest_result_images(history: list[Message]) -> set[tuple[int, int]]:
    """``(message index, result index)`` of the newest tool-result images that are worth persisting."""
    found: list[tuple[int, int]] = []
    for mi, message in enumerate(history):
        for ri, result in enumerate(message.tool_results or []):
            if result.image_b64:
                found.append((mi, ri))
    return set(found[-_KEEP_RESULT_IMAGES:])


def _payload_of(session: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(chat file payload, sidecar meta)`` of ``session``: the persisted-copy trim rules applied to a COPY of the
    history (``session["history"]`` is never modified)."""
    if not isinstance(session, dict) or "history" not in session:
        raise TypeError(f"chats.save / persist_round expect the session dict of chats.new_session(), got "
                        f"{type(session).__name__}; the worker's on_round receives the HISTORY, so pass "
                        "on_round=lambda history: chats.persist_round(session)")
    if not isinstance(session.get("id"), str) or not _SAFE_ID.fullmatch(session["id"]):
        raise ValueError(f"chat id {session.get('id')!r} is not a plain file name; refusing to write outside the chats folder")
    session["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    history = list(session["history"])
    keep = _newest_result_images(history)
    messages = []
    for mi, message in enumerate(history):
        data = _message_to_dict(message, clip_results=True)
        for ri, result in enumerate(data.get("tool_results") or []):
            if result.get("image_b64") and (mi, ri) not in keep:
                result["image_b64"] = ""
        messages.append(data)
    payload: dict[str, Any] = {
        "id": session["id"],
        "title": session["title"],
        "created": session["created"],
        "updated": session["updated"],
        "messages": messages,
    }
    if session.get("project_dir"):
        payload["project_dir"] = session["project_dir"]
    if session.get("ledger"):
        payload["ledger"] = session["ledger"]
    if session.get("blocks_meta"):
        payload["blocks_meta"] = session["blocks_meta"]
    if session.get("named"):
        payload["named"] = True   # the user named the chat (panel_chats.rename): nothing renames it again
    meta = {"id": session["id"], "title": session["title"], "updated": session["updated"],
            "messages": len(messages)}
    return payload, meta


def save(session: dict[str, Any]) -> None:
    """Persist ``session`` atomically (chat file + metadata sidecar).

    Safe to call from any thread, concurrently: ONE module lock is held around serialize + write + replace, so
    two saves never interleave and an older snapshot can never be written AFTER a newer one (the file only
    moves forward), and every write goes to its own temp file. The history is serialized from a copy, so a
    concurrent append cannot raise "list changed size during iteration". Besides id / title / created / updated /
    messages the optional keys ``project_dir``, ``ledger`` and ``blocks_meta`` are stored when set. Only the
    newest 4 tool-result images are written and tool results over 12000 characters are clipped in the FILE;
    ``session["history"]`` itself is never modified (apart from ``session["updated"]``).
    """
    with _SAVE_LOCK:
        payload, meta = _payload_of(session)
        directory = chats_dir()
        _atomic_write_json(os.path.join(directory, f"{session['id']}.json"), payload)
        _atomic_write_json(_sidecar_path(directory, session["id"]), meta)


def persist_round(session: dict[str, Any]) -> None:
    """The same atomic save, called by the agent worker after every provider round and once more at the end of the
    turn. The panel passes ``on_round=lambda history: chats.persist_round(session)`` (the worker calls
    ``on_round(history)`` on its own thread; ``history`` IS ``session["history"]``, so the session is what
    gets saved).

    Cheap enough for that (the persisted copy is bounded by the trim rules of ``save``) and safe while the
    GUI thread saves the same session. A "changed size during iteration" ``RuntimeError`` (the panel replaced a
    ledger / card list while it was serialized) is retried once; any other failure (disk full, locked file)
    propagates to the worker, which logs ``persist_error`` and carries on: a failing save never kills the turn.
    """
    try:
        save(session)
    except RuntimeError:
        save(session)


def _text_field(value: Any, default: str) -> str:
    """``value`` when it is a non-empty string, else ``default`` (a damaged chat file never injects a number or a
    list where the panel expects text)."""
    return value if isinstance(value, str) and value.strip() else default


def load(session_id: str) -> dict[str, Any] | None:
    """The stored session, or None when it is missing or unreadable (also when it holds messages but not one of them can be
    decoded: that is a damaged file, not an empty chat; with at least one readable message the readable ones are kept).

    The result carries ``project_dir`` (None when absent), ``ledger`` (None when absent),
    ``blocks_meta`` (``[]`` when absent) and ``named`` (the user named the chat; False when absent); chat files
    written by older builds load unchanged. Every field is
    type-checked: a hand-edited or damaged file gives the default for a wrong-typed value instead of an error
    later in the panel, and an ``id`` that is not a plain file name (a path, ``..``) is replaced by the name of
    the file it was read from, so ``save`` / ``delete`` can never leave the chats folder.
    """
    if not isinstance(session_id, str) or not _SAFE_ID.fullmatch(session_id):
        return None
    path = os.path.join(chats_dir(), f"{session_id}.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # Unreadable (or so deeply nested that the parser gives up): one bad chat must never take the panel
        # down - it loads every chat at startup - so it is skipped.
        return None
    if not isinstance(data, dict):
        return None
    raw_messages = data.get("messages")
    history = _messages_from_list(raw_messages)
    if isinstance(raw_messages, list) and raw_messages and not history:
        return None   # it holds messages and none can be read: not a chat, and never an empty one that a save could overwrite it with
    project_dir = data.get("project_dir")
    ledger = data.get("ledger")
    meta = data.get("blocks_meta")
    return {
        "id": session_id,   # the file name is the identity: a copied chat file must not alias another chat's file
        "title": _text_field(data.get("title"), "Chat"),
        "created": data.get("created") if isinstance(data.get("created"), str) else "",
        "updated": data.get("updated") if isinstance(data.get("updated"), str) else "",
        "history": history,
        "project_dir": project_dir if isinstance(project_dir, str) and project_dir else None,
        "ledger": ledger if isinstance(ledger, dict) and ledger else None,
        "blocks_meta": meta if isinstance(meta, list) else [],
        "named": data.get("named") is True,
    }


def is_unused(session_id: str) -> bool:
    """True when the stored chat holds nothing worth keeping: its file parses to an object without messages (missing,
    ``null`` or ``[]``), without a plan (``ledger``) and without turn records (``blocks_meta``) - a "New chat" nobody
    wrote in. The panel deletes such a leftover at start. Decided from the chat FILE itself (never from the sidecar,
    which a crash can leave stale); a file that is missing, unreadable, not an object or holds any message (even a
    damaged one) is never "unused": nothing is deleted on a guess."""
    if not isinstance(session_id, str) or not _SAFE_ID.fullmatch(session_id):
        return False
    try:
        with open(os.path.join(chats_dir(), f"{session_id}.json"), encoding="utf-8") as f:
            data = json.load(f)
    except Exception:   # noqa: BLE001 - unreadable: keep it
        return False
    if not isinstance(data, dict):
        return False
    messages = data.get("messages")
    return ((messages is None or messages == [])
            and not data.get("ledger") and not data.get("blocks_meta"))


def recovery_notice(session: dict[str, Any]) -> str:
    """Non-empty when the session ended in the middle of a round.

    That is the case when the last history message is an assistant message with tool calls and no
    following results (a crash or a forced quit while tools were running). The panel shows the text as a
    restore card; an empty string means the history ends cleanly.
    """
    history = session.get("history") or []
    if not history:
        return ""
    last = history[-1]
    calls = getattr(last, "tool_calls", None) or []
    if getattr(last, "role", "") != "assistant" or not calls:
        return ""
    names = ", ".join(dict.fromkeys(c.name for c in calls))
    return (f"Oscar was interrupted in the middle of a step: the last reply asked for {len(calls)} tool "
            f"call(s) ({names}) that never reported back. Check the model before you continue, then say "
            "'continue' to resume from the last completed step.")


def _meta_from_chat_file(directory: str, name: str) -> dict[str, Any] | None:
    """Parse one chat file once, (re)write its sidecar and return the metadata."""
    session_id = name[: -len(".json")]
    try:
        with open(os.path.join(directory, name), encoding="utf-8") as f:
            data = json.load(f)
    except Exception:   # unreadable or malformed (even a parser recursion limit): not listed, never fatal
        return None
    if not isinstance(data, dict):
        return None
    messages = data.get("messages")
    meta = {"id": session_id,
            "title": _text_field(data.get("title"), "Chat"),
            "updated": data.get("updated") if isinstance(data.get("updated"), str) else "",
            "messages": len(messages) if isinstance(messages, list) else 0}
    try:
        _atomic_write_json(_sidecar_path(directory, session_id), meta)
    except OSError:
        pass
    return meta


_STALE_TEMP_S = 24 * 3600   # a write that is still running finishes in milliseconds; a day-old temp file is a crash's leftover


def _drop_stale_temp(path: str) -> None:
    """Remove the temp file of a save that a crash interrupted (best effort; a young file may be a save in progress)."""
    try:
        if time.time() - os.path.getmtime(path) > _STALE_TEMP_S:
            os.remove(path)
    except OSError:
        pass


def list_sessions(limit: int = 40) -> list[dict[str, Any]]:
    """Session metas (id/title/updated/messages), newest first. Reads sidecars only.

    A chat file without a sidecar (an old chat), or one newer than its sidecar, is parsed once and its
    sidecar written; every other entry costs one small file read.
    """
    out: list[dict[str, Any]] = []
    directory = chats_dir()
    try:
        names = os.listdir(directory)
    except OSError:
        return out
    for name in names:
        if name.endswith(".tmp"):
            _drop_stale_temp(os.path.join(directory, name))
            continue
        if not name.endswith(".json") or name.endswith(".meta.json"):
            continue
        sidecar = _sidecar_path(directory, name[: -len(".json")])
        meta = None
        try:
            if os.path.getmtime(sidecar) + 1.0 >= os.path.getmtime(os.path.join(directory, name)):
                with open(sidecar, encoding="utf-8") as f:
                    meta = json.load(f)
        except Exception:
            meta = None
        if not isinstance(meta, dict) or "id" not in meta:
            meta = _meta_from_chat_file(directory, name)
        if meta:
            if not _SAFE_ID.fullmatch(name[:-5]):
                continue   # not a chat id this app ever wrote: never open it, never write next to it
            out.append({"id": name[:-5], "title": _text_field(meta.get("title"), "Chat"),
                        "updated": meta.get("updated") if isinstance(meta.get("updated"), str) else "",
                        "messages": meta.get("messages") if isinstance(meta.get("messages"), int) else 0})
    out.sort(key=lambda s: str(s["updated"] or ""), reverse=True)
    return out[:limit]


def delete(session_id: str) -> None:
    """Remove the chat file and its metadata sidecar (missing files are ignored; an id that is not a plain
    file name removes nothing)."""
    if not isinstance(session_id, str) or not _SAFE_ID.fullmatch(session_id):
        return
    directory = chats_dir()
    for path in (os.path.join(directory, f"{session_id}.json"), _sidecar_path(directory, session_id)):
        try:
            os.remove(path)
        except OSError:
            pass


def title_from_prompt(prompt: str, fallback: str = "New chat") -> str:
    """First line of the first user message, trimmed for a combo entry."""
    text = " ".join((prompt or "").split())
    if not text:
        return fallback
    return text[:48] + ("…" if len(text) > 48 else "")
