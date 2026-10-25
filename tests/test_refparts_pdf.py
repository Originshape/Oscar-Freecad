"""``oscar.refparts.pdftext``: best-effort PDF text with the standard library (the datasheet path of ``attachments``).

PURE: the PDFs are built in the tests (hand-written objects, plain or Flate streams); poppler is replaced by a fake runner so the
result never depends on what is installed on the machine.
"""

import os
import subprocess
import sys
import unittest
import zlib
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.refparts import pdftext  # noqa: E402


def pdf(streams, *, flate=True, pages=None, trailer=b"", images=()) -> bytes:
    """A small PDF (no xref table: the extractor scans for streams) with one content stream per ``streams`` item."""
    out = bytearray(b"%PDF-1.4\n")
    number = 1
    count = pages if pages is not None else len(streams)
    for _ in range(count):
        out += b"%d 0 obj\n<< /Type /Page /Parent 99 0 R >>\nendobj\n" % number
        number += 1
    out += b"98 0 obj\n<< /Type /Pages /Count %d >>\nendobj\n" % count
    for body in streams:
        data = zlib.compress(body) if flate else body
        filt = b" /Filter /FlateDecode" if flate else b""
        out += b"%d 0 obj\n<< /Length %d%s >>\nstream\n" % (number, len(data), filt) + data + b"\nendstream\nendobj\n"
        number += 1
    for body in images:
        out += b"%d 0 obj\n<< /Type /XObject /Subtype /Image /Filter /DCTDecode /Length %d >>\nstream\n" % (number, len(body)) + body + b"\nendstream\nendobj\n"
        number += 1
    out += b"trailer\n<< /Root 1 0 R " + trailer + b">>\n%%EOF\n"
    return bytes(out)


def write(data: bytes, name="sheet.pdf") -> str:
    path = os.path.join(isolation.fresh_dir("pdf_"), name)
    open(path, "wb").write(data)
    return path


def text_of(content: bytes, **kwargs) -> pdftext.PdfText:
    return pdftext.extract_text(write(pdf([content])), use_poppler=False, **kwargs)


class ScannerTest(unittest.TestCase):
    def scan(self, content: bytes) -> str:
        return pdftext.scan_content_stream(content)[0]

    def test_tj_and_line_breaks(self):
        text = self.scan(b"BT /F1 12 Tf 72 700 Td (Board 85 x 56 mm) Tj 0 -14 Td (Hole 2.75 mm) Tj ET")
        self.assertEqual([line for line in text.split("\n") if line], ["Board 85 x 56 mm", "Hole 2.75 mm"])

    def test_tj_array_gaps_become_spaces_and_kerning_does_not(self):
        text = self.scan(b"BT [(Mount) -20 (ing) -300 (holes)] TJ ET")
        self.assertIn("Mounting holes", text)

    def test_hex_strings_and_utf16(self):
        self.assertIn("Pi4", self.scan(b"BT <50 69 34> Tj ET"))
        self.assertIn("±", self.scan(b"BT <FEFF00B1> Tj ET"))
        self.assertIn("A", self.scan(b"BT <4> Tj ET") + "A", "an odd number of hex digits must not crash")

    def test_escapes_nesting_and_octal(self):
        text = self.scan(rb"BT (a\(b\)c \\ d\101 (nested) end) Tj ET")
        self.assertIn("a(b)c \\ dA (nested) end", text)
        self.assertIn("x\ny", self.scan(rb"BT (x\ny) Tj ET"))

    def test_quote_operators_start_a_new_line(self):
        text = self.scan(b"BT (one) Tj (two) ' 0 0 (three) \" ET")
        self.assertEqual([l for l in text.split("\n") if l], ["one", "two", "three"])

    def test_comments_dictionaries_and_numbers_are_skipped(self):
        text = self.scan(b"% a comment (not text) Tj\nBT /F1 9 Tf << /MCID 0 >> BDC 1 0 0 1 5 5 Tm (Real) Tj ET")
        self.assertIn("Real", text)
        self.assertNotIn("not text", text)

    def test_same_line_td_is_a_space(self):
        text = self.scan(b"BT (a) Tj 10 0 Td (b) Tj ET")
        self.assertIn("a b", text)

    def test_cid_strings_are_flagged_not_decoded(self):
        text, cid = pdftext.scan_content_stream(b"BT <0041004200430044> Tj ET")
        self.assertTrue(cid)
        self.assertNotIn("A", text)

    def test_garbage_never_raises(self):
        for blob in (b"", b"(((((", b"<<<<<<", b"[ (a", b"BT" * 100, bytes(range(256)) * 4):
            pdftext.scan_content_stream(blob)


class BuiltinExtractionTest(unittest.TestCase):
    def test_flate_and_plain_streams(self):
        for flate in (True, False):
            got = pdftext.extract_text(write(pdf([b"BT (Raspberry Pi 4 Model B: 85 x 56 mm) Tj ET"], flate=flate)), use_poppler=False)
            self.assertEqual((got.status, got.method, got.pages, got.note), ("ok", "builtin", 1, ""), flate)
            self.assertIn("85 x 56 mm", got.text)

    def test_several_pages_are_joined_in_file_order_and_counted(self):
        data = pdf([b"BT (page one) Tj ET", b"BT (page two) Tj ET", b"BT (page three) Tj ET"])
        got = pdftext.extract_text(write(data), use_poppler=False)
        self.assertEqual(got.pages, 3)
        self.assertLess(got.text.index("page one"), got.text.index("page two"))
        self.assertLess(got.text.index("page two"), got.text.index("page three"))

    def test_pages_object_is_not_a_page(self):
        data = pdf([b"BT (x) Tj ET"], pages=2)
        self.assertEqual(pdftext.extract_text(write(data), use_poppler=False).pages, 2)

    def test_truncated_flate_stream_is_still_read_as_far_as_it_goes(self):
        import random

        body = b"BT (first line) Tj ET " + bytes(random.Random(7).choices(b"0123456789 mlh", k=6000))
        packed = zlib.compress(body)
        broken = (b"%%PDF-1.4\n1 0 obj\n<< /Type /Page >>\nendobj\n2 0 obj\n<< /Filter /FlateDecode /Length %d >>\nstream\n" % len(packed)
                  + packed[:len(packed) // 2] + b"\nendstream\nendobj\n")
        got = pdftext.extract_text(write(broken), use_poppler=False)
        self.assertEqual(got.status, "ok")
        self.assertIn("first line", got.text)

    def test_corrupt_stream_is_skipped_and_the_others_survive(self):
        data = pdf([b"BT (good) Tj ET"]) + b"9 0 obj\n<< /Filter /FlateDecode >>\nstream\n\x00\x01not zlib\nendstream\nendobj\n"
        got = pdftext.extract_text(write(data), use_poppler=False)
        self.assertEqual((got.status, got.text.strip()), ("ok", "good"))

    def test_image_and_foreign_filter_streams_are_ignored(self):
        data = pdf([b"BT (caption) Tj ET"], images=(b"BT (hidden) Tj ET",))
        got = pdftext.extract_text(write(data), use_poppler=False)
        self.assertNotIn("hidden", got.text)
        self.assertIn("caption", got.text)

    def test_streams_without_text_operators_are_ignored(self):
        self.assertEqual(text_of(b"0 0 m 100 100 l S").status, "empty")
        self.assertEqual(text_of(b"BT ET").status, "empty")

    def test_vector_drawing_says_so(self):
        got = text_of(b"0 0 m 85 0 l 85 56 l 0 56 l h S")
        self.assertEqual(got.status, "empty")
        self.assertIn("vector drawing", got.note)

    def test_cid_font_text_is_reported_as_garbled_not_returned(self):
        got = text_of(b"BT <0041004200430044> Tj <00450046> Tj ET")
        self.assertEqual((got.status, got.text), ("garbled", ""))
        self.assertIn("font encodings", got.note)

    def test_mostly_symbols_is_garbled(self):
        got = text_of(b"BT (\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0e\x0f) Tj ET")
        self.assertEqual((got.status, got.text), ("garbled", ""))

    def test_partial_cid_text_keeps_the_readable_part_with_a_note(self):
        got = text_of(b"BT (Readable dimensions 85 x 56) Tj <0041004200430044> Tj ET")
        self.assertEqual(got.status, "ok")
        self.assertIn("Readable dimensions", got.text)
        self.assertIn("missing", got.note)

    def test_encrypted_pdf(self):
        got = pdftext.extract_text(write(pdf([b"BT (secret) Tj ET"], trailer=b"/Encrypt 7 0 R ")), use_poppler=False)
        self.assertEqual(got.status, "encrypted")
        self.assertNotIn("secret", got.text)
        self.assertEqual(got.pages, 1)

    def test_not_a_pdf_and_missing_file(self):
        self.assertEqual(pdftext.extract_text(write(b"hello world"), use_poppler=False).status, "unreadable")
        got = pdftext.extract_text("/no/such/file.pdf", use_poppler=False)
        self.assertEqual(got.status, "unreadable")
        self.assertIn("cannot be read", got.note)

    def test_empty_file(self):
        self.assertEqual(pdftext.extract_text(write(b""), use_poppler=False).status, "unreadable")

    def test_text_is_clipped_with_a_flag(self):
        got = text_of(b"BT (" + b"x" * 500 + b") Tj ET", max_chars=100)
        self.assertEqual((len(got.text), got.truncated), (100, True))
        self.assertFalse(text_of(b"BT (short) Tj ET", max_chars=100).truncated)

    def test_huge_files_are_refused(self):
        with mock.patch.object(pdftext, "MAX_FILE_BYTES", 200):
            got = pdftext.extract_text(write(pdf([b"BT (" + b"y" * 400 + b") Tj ET"])), use_poppler=False)
        self.assertEqual(got.status, "too_large")

    def test_stream_count_is_bounded(self):
        data = pdf([b"BT (a) Tj ET"] * 20)
        with mock.patch.object(pdftext, "MAX_STREAMS", 5):
            got = pdftext.extract_text(write(data), use_poppler=False)
        self.assertEqual(got.text.count("a"), 5)

    def test_decompression_bomb_is_capped(self):
        bomb = zlib.compress(b"BT (start) Tj ET " + b" " * (6 * 1024 * 1024))
        with mock.patch.object(pdftext, "MAX_STREAM_BYTES", 1024 * 1024):
            got = pdftext.extract_text(write(pdf([b"x"]) + b"7 0 obj\n<< /Filter /FlateDecode >>\nstream\n" + bomb + b"\nendstream\n"), use_poppler=False)
        self.assertEqual(got.status, "ok")
        self.assertIn("start", got.text)


class PopplerTest(unittest.TestCase):
    def runner(self, stdout=b"Board 85 mm\x0cPage 2 56 mm\n", returncode=0, raises=None):
        calls = []

        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            if raises:
                raise raises
            return SimpleNamespace(returncode=returncode, stdout=stdout)

        run.calls = calls
        return run

    def test_poppler_wins_when_it_has_text(self):
        run = self.runner()
        got = pdftext.extract_text(write(pdf([b"BT (builtin) Tj ET"])), runner=run)
        self.assertEqual((got.status, got.method), ("ok", "pdftotext"))
        self.assertIn("Board 85 mm", got.text)
        self.assertNotIn("\x0c", got.text)
        self.assertEqual(got.pages, 1)
        argv, kwargs = run.calls[0]
        self.assertEqual(argv[:2], ["pdftotext", "-layout"])
        self.assertEqual(argv[-1], "-")
        self.assertEqual(kwargs["timeout"], pdftext.POPPLER_TIMEOUT_S)

    def test_failure_falls_back_to_the_builtin_reader(self):
        for run in (self.runner(returncode=1), self.runner(stdout=b"   \n"), self.runner(raises=OSError("gone")),
                    self.runner(raises=subprocess.TimeoutExpired("pdftotext", 20))):
            got = pdftext.extract_text(write(pdf([b"BT (builtin) Tj ET"])), runner=run)
            self.assertEqual((got.method, got.status), ("builtin", "ok"))

    def test_poppler_may_read_what_the_builtin_reader_cannot(self):
        data = pdf([b"BT <0041004200430044> Tj ET"])
        self.assertEqual(pdftext.extract_text(write(data), use_poppler=False).status, "garbled")
        self.assertEqual(pdftext.extract_text(write(data), runner=self.runner()).status, "ok")

    def test_use_poppler_false_never_calls_it(self):
        run = self.runner()
        pdftext.extract_text(write(pdf([b"BT (x) Tj ET"])), use_poppler=False, runner=run)
        self.assertEqual(run.calls, [])

    def test_big_file_goes_to_poppler_or_is_refused(self):
        with mock.patch.object(pdftext, "MAX_FILE_BYTES", 100):
            data = pdf([b"BT (" + b"z" * 300 + b") Tj ET"])
            self.assertEqual(pdftext.extract_text(write(data), runner=self.runner()).method, "pdftotext")
            self.assertEqual(pdftext.extract_text(write(data), runner=self.runner(returncode=2)).status, "too_large")

    def test_no_poppler_installed(self):
        with mock.patch.object(pdftext.shutil, "which", return_value=None):
            self.assertIsNone(pdftext.poppler_text("/x.pdf"))
            self.assertEqual(pdftext.extract_text(write(pdf([b"BT (x) Tj ET"]))).method, "builtin")

    def test_poppler_text_is_clipped_too(self):
        got = pdftext.extract_text(write(pdf([b"BT (x) Tj ET"])), runner=self.runner(stdout=b"w" * 500), max_chars=50)
        self.assertEqual((len(got.text), got.truncated), (50, True))


class BuiltinBudgetTest(unittest.TestCase):
    """The built-in extractor stops at ``max_chars`` and at its wall-clock budget."""

    @staticmethod
    def many(n):
        return pdf([b"BT (" + (b"word%03d " % i) * 20 + b") Tj ET" for i in range(n)])

    def test_early_exit_at_max_chars_scans_few_streams(self):
        with mock.patch.object(pdftext, "scan_content_stream", wraps=pdftext.scan_content_stream) as scan:
            got = pdftext.builtin_text(self.many(200), max_chars=300)
        self.assertTrue(got.truncated)
        self.assertEqual(got.status, "ok")
        self.assertLess(scan.call_count, 10)

    def test_without_max_chars_everything_is_read(self):
        got = pdftext.builtin_text(self.many(20))
        self.assertFalse(got.truncated)
        self.assertIn("word019", got.text)

    def test_the_wall_clock_budget_stops_the_scan(self):
        ticks = iter([0.0, 0.0, 0.1, 99.0] + [99.0] * 500)
        got = pdftext.builtin_text(self.many(50), budget_s=1.0, clock=lambda: next(ticks))
        self.assertTrue(got.truncated)
        self.assertIn("reading stopped after 1 s", got.note)
        self.assertIn("word000", got.text)
        self.assertNotIn("word049", got.text)

    def test_extract_text_passes_max_chars_through(self):
        got = pdftext.extract_text(write(self.many(200)), max_chars=100, use_poppler=False)
        self.assertEqual((len(got.text), got.truncated), (100, True))


if __name__ == "__main__":
    unittest.main()
