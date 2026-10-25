"""Best-effort text from a PDF datasheet with the standard library only (``attachments`` uses it for PDF attachments).

PURE module: no FreeCAD, no Qt, no third-party packages. A datasheet is often the only place a dimension is written down, so an
attached PDF should reach the model as text. Two extractors, tried in this order:

1. ``pdftotext`` (poppler) when it is installed: layout-preserving and handles every font encoding;
2. a built-in reader: it scans the file for content streams (FlateDecode or plain), pulls the strings out of the text-showing
   operators (``Tj``, ``TJ``, ``'``, ``"``) and breaks lines at the positioning operators. It reads documents with simple
   single-byte fonts (most generated datasheets); it does NOT resolve CID fonts (``Identity-H``), embedded custom encodings,
   object streams of text, scanned pages or encrypted files, and says so instead of returning garbage: the result carries a
   ``status`` (``ok`` | ``empty`` | ``garbled`` | ``encrypted`` | ``unreadable`` | ``too_large``) and a ``note`` for the model.

The text is DATA from a document the user attached: callers must present it as untrusted, never as instructions.
"""

import os
import re
import shutil
import subprocess
import time
import zlib
from dataclasses import dataclass

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_STREAM_BYTES = 8 * 1024 * 1024
MAX_STREAMS = 6000
BUILTIN_BUDGET_S = 10.0   # wall-clock budget of the built-in extractor (a hostile or huge PDF must not hold the agent thread)
POPPLER_TIMEOUT_S = 20.0
MAX_PAGES_POPPLER = 60

_PAGE_RE = re.compile(rb"/Type\s*/Page(?![A-Za-z])")
_ESCAPES = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12, ord("("): 40, ord(")"): 41, ord("\\"): 92}
_OTHER_FILTERS = re.compile(rb"/(DCTDecode|LZWDecode|ASCII85Decode|ASCIIHexDecode|JBIG2Decode|CCITTFaxDecode|RunLengthDecode|JPXDecode)")
_WHITESPACE = b" \t\r\n\x0c\x00"
_DELIMS = b"()<>[]{}/%"


@dataclass
class PdfText:
    """Result of ``extract_text``: ``text`` (may be empty), ``pages`` (0 when unknown), ``method`` (``pdftotext`` | ``builtin`` | ``""``),
    ``status`` (see the module docstring), ``note`` (one sentence for the model, ``""`` when everything is fine) and ``truncated``."""

    text: str = ""
    pages: int = 0
    method: str = ""
    status: str = "unreadable"
    note: str = ""
    truncated: bool = False


# --- the content-stream scanner -----------------------------------------------------------------------------

def _literal(data: bytes, i: int) -> tuple[bytes, int]:
    """Parse a literal string starting at ``data[i] == b'('``; returns ``(bytes, index after the closing paren)``."""
    out = bytearray()
    depth = 0
    n = len(data)
    while i < n:
        c = data[i]
        if c == 0x5C and i + 1 < n:                             # backslash escape
            nxt = data[i + 1]
            if nxt in _ESCAPES:
                out.append(_ESCAPES[nxt])
                i += 2
            elif 0x30 <= nxt <= 0x37:                           # up to three octal digits
                j = i + 1
                while j < n and j < i + 4 and 0x30 <= data[j] <= 0x37:
                    j += 1
                out.append(int(data[i + 1:j], 8) & 0xFF)
                i = j
            elif nxt in (10, 13):                               # line continuation
                i += 2 + (1 if nxt == 13 and i + 2 < n and data[i + 2] == 10 else 0)
            else:
                out.append(nxt)
                i += 2
            continue
        if c == 0x28:
            depth += 1
            if depth > 1:
                out.append(c)
        elif c == 0x29:
            depth -= 1
            if depth == 0:
                return bytes(out), i + 1
            out.append(c)
        else:
            out.append(c)
        i += 1
    return bytes(out), n


def _hex(data: bytes, i: int) -> tuple[bytes, int]:
    end = data.find(b">", i)
    end = len(data) if end < 0 else end
    digits = re.sub(rb"[^0-9A-Fa-f]", b"", data[i + 1:end])
    if len(digits) % 2:
        digits += b"0"
    return bytes.fromhex(digits.decode("ascii")), end + 1


def _decode_string(raw: bytes) -> tuple[str, bool]:
    """``(text, looks_like_cid)``: two-byte CID strings (many NULs) cannot be read without the font's CMap."""
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", "replace"), False
    if raw and raw.count(0) * 3 >= len(raw):
        return "", True
    return raw.decode("cp1252", "replace"), False


def scan_content_stream(data: bytes) -> tuple[str, bool]:
    """The text shown by one content stream and whether any string looked like a CID-font string (unreadable)."""
    out: list[str] = []
    stack: list = []
    cid = False
    i, n = 0, len(data)
    while i < n:
        c = data[i]
        if c in _WHITESPACE:
            i += 1
        elif c == 0x25:                                          # comment
            j = data.find(b"\n", i)
            i = n if j < 0 else j + 1
        elif c == 0x28:
            value, i = _literal(data, i)
            stack.append(value)
        elif c == 0x3C and data[i + 1:i + 2] == b"<":           # dictionary (inline image parameters): skip it
            depth, j = 1, i + 2
            while j < n and depth:
                if data.startswith(b"<<", j):
                    depth, j = depth + 1, j + 2
                elif data.startswith(b">>", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            i = j
        elif c == 0x3C:
            value, i = _hex(data, i)
            stack.append(value)
        elif c == 0x5B:                                          # array: strings and numbers up to the closing bracket
            items: list = []
            i += 1
            while i < n and data[i] != 0x5D:
                if data[i] == 0x28:
                    value, i = _literal(data, i)
                    items.append(value)
                elif data[i] == 0x3C and data[i + 1:i + 2] != b"<":
                    value, i = _hex(data, i)
                    items.append(value)
                elif data[i] in b"+-.0123456789":
                    j = i + 1
                    while j < n and data[j] in b"+-.0123456789":
                        j += 1
                    try:
                        items.append(float(data[i:j]))
                    except ValueError:
                        pass
                    i = j
                else:
                    i += 1
            i += 1
            stack.append(items)
        elif c == 0x2F:                                          # name
            j = i + 1
            while j < n and data[j] not in _WHITESPACE and data[j] not in _DELIMS:
                j += 1
            stack.append(data[i:j])
            i = j
        elif c in b"+-.0123456789":
            j = i + 1
            while j < n and data[j] in b"+-.0123456789":
                j += 1
            try:
                stack.append(float(data[i:j]))
            except ValueError:
                stack.append(0.0)
            i = j
        elif c in b"]>{})":
            i += 1
        else:
            j = i + 1
            while j < n and data[j] not in _WHITESPACE and data[j] not in _DELIMS:
                j += 1
            op = data[i:j]
            i = j
            if op in (b"Tj", b"'", b'"'):
                if op != b"Tj":
                    out.append("\n")
                if stack and isinstance(stack[-1], bytes):
                    text, was_cid = _decode_string(stack[-1])
                    out.append(text)
                    cid = cid or was_cid
            elif op == b"TJ":
                for item in stack[-1] if stack and isinstance(stack[-1], list) else []:
                    if isinstance(item, bytes):
                        text, was_cid = _decode_string(item)
                        out.append(text)
                        cid = cid or was_cid
                    elif isinstance(item, float) and item < -180:
                        out.append(" ")
            elif op in (b"T*", b"Tm", b"ET", b"BT"):
                out.append("\n")
            elif op in (b"Td", b"TD"):
                ty = stack[-1] if stack and isinstance(stack[-1], float) else 0.0
                out.append("\n" if abs(ty) > 0.5 else " ")
            stack.clear()
    return "".join(out), cid


def _streams(data: bytes):
    """Yield ``(dictionary bytes, decoded stream bytes)`` for every plain or Flate stream of the file (bounded)."""
    count = 0
    for match in re.finditer(rb"stream\r?\n", data):
        if data[max(0, match.start() - 3):match.start()] == b"end":
            continue
        count += 1
        if count > MAX_STREAMS:
            return
        head = data[max(0, match.start() - 1500):match.start()]
        cut = head.rfind(b" obj")
        head = head[cut + 4:] if cut >= 0 else head
        end = data.find(b"endstream", match.end())
        if end < 0:
            continue
        raw = data[match.end():end]
        if b"/Subtype" in head and b"/Image" in head:
            continue
        if _OTHER_FILTERS.search(head):
            continue
        if b"/FlateDecode" in head or b"/Fl " in head:
            try:
                raw = zlib.decompressobj().decompress(raw, MAX_STREAM_BYTES)
            except zlib.error:
                continue
        yield head, raw


def builtin_text(data: bytes, max_chars: "int | None" = None, budget_s: float = BUILTIN_BUDGET_S, clock=time.monotonic) -> PdfText:
    """The built-in extractor on the bytes of a PDF (see the module docstring for what it cannot read).

    Stops early (``truncated`` set) once ``max_chars`` of text were collected, and when ``budget_s`` seconds of wall clock are
    used up (the note says so); what was read until then is returned."""
    result = PdfText(pages=len(_PAGE_RE.findall(data)), method="builtin")
    chunks, cid_streams, collected = [], 0, 0
    deadline = clock() + max(0.0, float(budget_s))
    out_of_time = False
    for _head, body in _streams(data):
        if max_chars is not None and collected >= max_chars:
            result.truncated = True
            break
        if clock() >= deadline:
            out_of_time = result.truncated = True
            break
        if b"BT" not in body or not (b"Tj" in body or b"TJ" in body):
            continue
        text, cid = scan_content_stream(body)
        if cid:
            cid_streams += 1
        cleaned = "\n".join(line.strip() for line in re.sub(r"[ \t]+", " ", text).splitlines())
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if cleaned:
            chunks.append(cleaned)
            collected += len(cleaned) + 2
    text = "\n\n".join(chunks)
    letters = sum(ch.isalnum() for ch in text)
    shown = sum(not ch.isspace() for ch in text)
    if not text.strip():
        result.status = "garbled" if cid_streams else "empty"
        result.note = ("the text uses embedded font encodings that cannot be read without a PDF library" if cid_streams else
                       "the PDF has no extractable text (a scanned page or a vector drawing)")
    elif shown and letters / shown < 0.45:
        result.status, result.text = "garbled", ""
        result.note = "the text uses embedded font encodings that cannot be read without a PDF library"
    else:
        result.status, result.text = "ok", text
        if cid_streams:
            result.note = "some text uses embedded font encodings and is missing"
        if out_of_time:
            result.note = (result.note + "; " if result.note else "") + f"reading stopped after {budget_s:g} s, the rest of the PDF is not included"
    return result


# --- poppler ----------------------------------------------------------------------------------------------------

def poppler_text(path: str, runner=None) -> "PdfText | None":
    """``pdftotext -layout`` of the file, or ``None`` when poppler is not installed, fails or finds no text."""
    exe = shutil.which("pdftotext") if runner is None else "pdftotext"
    if not exe:
        return None
    run = runner or subprocess.run
    try:
        done = run([exe, "-layout", "-enc", "UTF-8", "-l", str(MAX_PAGES_POPPLER), path, "-"], capture_output=True, timeout=POPPLER_TIMEOUT_S,
                   check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    text = done.stdout.decode("utf-8", "replace") if isinstance(done.stdout, bytes) else str(done.stdout or "")
    text = re.sub(r"[ \t]+\n", "\n", text).replace("\x0c", "\n\n").strip()
    if not text:
        return None
    return PdfText(text=text, method="pdftotext", status="ok")


# --- entry point -------------------------------------------------------------------------------------------------

def extract_text(path: str, *, max_chars: int = 40_000, use_poppler: bool = True, runner=None) -> PdfText:
    """Text of the PDF at ``path``, clipped to ``max_chars`` (``truncated`` says so). Never raises.

    ``status`` is ``ok`` when there is text; otherwise the reason (``empty`` / ``garbled`` / ``encrypted`` / ``unreadable`` /
    ``too_large``) with a ``note`` that tells the model to ask the user for the dimensions it needs.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            head = fh.read(1024)
            if b"%PDF-" not in head:
                return PdfText(status="unreadable", note="the file is not a PDF document")
            if size > MAX_FILE_BYTES:
                found = poppler_text(path, runner) if use_poppler else None
                if found is None:
                    return PdfText(status="too_large", note=f"the PDF is {size / 1e6:.0f} MB, too large to read here")
                return _clip(found, max_chars)
            data = head + fh.read()
    except OSError as exc:
        return PdfText(status="unreadable", note=f"the file cannot be read: {exc}")
    if use_poppler:
        found = poppler_text(path, runner)
        if found is not None:
            found.pages = len(_PAGE_RE.findall(data))
            return _clip(found, max_chars)
    if b"/Encrypt" in data:
        return PdfText(pages=len(_PAGE_RE.findall(data)), status="encrypted", note="the PDF is encrypted; ask the user for an unprotected copy or the dimensions")
    return _clip(builtin_text(data, max_chars), max_chars)


def _clip(result: PdfText, max_chars: int) -> PdfText:
    if len(result.text) > max_chars:
        result.text, result.truncated = result.text[:max_chars], True
    return result
