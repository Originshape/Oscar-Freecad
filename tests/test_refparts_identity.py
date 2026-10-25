"""Tests of ``oscar.refparts.identity``: magic-byte sniffing of downloads and the bounding-box / outline identity check.

Pure logic: files are built in memory or in a temp directory; nothing touches the network or FreeCAD.
"""

import io
import math
import os
import struct
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.refparts import catalog, identity  # noqa: E402

STEP_BODY = (b"ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'2;1');\nENDSEC;\nDATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\nENDSEC;\n"
             b"END-ISO-10303-21;\n")
ASCII_STL = b"solid cube\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid cube\n"


def binary_stl(triangles: int = 2, header: bytes = b"binary stl exported by a tool") -> bytes:
    """A syntactically valid binary STL: 80-byte header, uint32 count, 50 bytes per triangle."""
    body = b"".join(struct.pack("<12fH", *([0.0] * 12), 0) for _ in range(triangles))
    return header.ljust(80, b" ") + struct.pack("<I", triangles) + body


def zip_bytes(**members) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


class SniffTest(unittest.TestCase):
    def test_step(self):
        self.assertEqual(identity.sniff_bytes(STEP_BODY), "step")
        self.assertEqual(identity.sniff_bytes(STEP_BODY, len(STEP_BODY)), "step")
        self.assertEqual(identity.sniff_bytes(b"\n\r\n  " + STEP_BODY), "step")                    # leading blank lines
        self.assertEqual(identity.sniff_bytes(b"\xef\xbb\xbf" + STEP_BODY), "step")                # UTF-8 BOM
        self.assertEqual(identity.sniff_bytes(STEP_BODY.replace(b"ISO", b"iso", 1)), "step")       # case-insensitive keyword
        self.assertEqual(identity.sniff_bytes(b"ISO-10303-21\r\nHEADER;"), "step")                 # missing semicolon tolerated

    def test_stl_ascii(self):
        self.assertEqual(identity.sniff_bytes(ASCII_STL), "stl")
        self.assertEqual(identity.sniff_bytes(b"  \nSOLID part\nFACET NORMAL 0 0 1\n"), "stl")
        self.assertEqual(identity.sniff_bytes(b"solid empty\nendsolid empty\n"), "stl")

    def test_a_text_that_merely_starts_with_solid_is_not_an_stl(self):
        self.assertEqual(identity.sniff_bytes(b"solid state drives are fast and cheap\n"), "unknown")

    def test_stl_binary_needs_the_exact_size(self):
        data = binary_stl(3)
        self.assertEqual(identity.sniff_bytes(data, len(data)), "stl")
        self.assertEqual(identity.sniff_bytes(data[:300], len(data)), "stl")                        # only the head is read
        self.assertEqual(identity.sniff_bytes(data, len(data) + 1), "unknown")                      # truncated / padded file
        self.assertEqual(identity.sniff_bytes(data), "unknown")                                     # no size: cannot be sure
        self.assertEqual(identity.sniff_bytes(binary_stl(0), len(binary_stl(0))), "unknown")        # zero triangles is not a mesh

    def test_binary_stl_whose_header_starts_with_solid(self):
        data = binary_stl(2, header=b"solid exported by cad")
        self.assertEqual(identity.sniff_bytes(data, len(data)), "stl")

    def test_zip(self):
        data = zip_bytes(**{"pi4.step": STEP_BODY})
        self.assertEqual(identity.sniff_bytes(data[:4096], len(data)), "zip")
        self.assertEqual(identity.sniff_bytes(b"PK\x05\x06" + b"\x00" * 18), "zip")                # an empty archive
        self.assertEqual(identity.sniff_bytes(b"PK\x07\x08"), "zip")

    def test_pdf(self):
        self.assertEqual(identity.sniff_bytes(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj"), "pdf")
        self.assertEqual(identity.sniff_bytes(b"\x00\x00junk before the header %PDF-1.4 ..."), "pdf")
        self.assertEqual(identity.sniff_bytes(b"\n  %PDF-1.3"), "pdf")

    def test_html(self):
        for body in (b"<!DOCTYPE html><html><body>404</body></html>", b"  \n<html lang='en'>", b"<HTML><HEAD>", b"<head><title>x",
                     b"<meta charset=utf-8><title>Sign in</title>", b'<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml">',
                     b"\xef\xbb\xbf<!doctype html>", b"<!-- cached --><html>", b"<body>Not found"):
            self.assertEqual(identity.sniff_bytes(body), "html", body)

    def test_xml_that_is_not_html_is_unknown(self):
        self.assertEqual(identity.sniff_bytes(b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"/>'), "unknown")

    def test_git_lfs_pointer(self):
        pointer = (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"ab" * 32 + b"\nsize 1234567\n")
        self.assertEqual(identity.sniff_bytes(pointer), "lfs_pointer")
        self.assertEqual(identity.sniff_bytes(pointer, len(pointer)), "lfs_pointer")
        self.assertEqual(identity.sniff_bytes(pointer, 5_000_000), "unknown")                       # a big file is not a pointer
        self.assertEqual(identity.sniff_bytes(b"\xef\xbb\xbf" + pointer), "lfs_pointer")

    def test_brep_and_iges(self):
        self.assertEqual(identity.sniff_bytes(b"DBRep_DrawableShape\n\nCASCADE Topology V1\n"), "brep")
        first = b" " * 72 + b"S" + b"      1"
        self.assertEqual(identity.sniff_bytes(first + b"\n1H,,1H;,...   G      1\n"), "iges")
        self.assertEqual(identity.sniff_bytes(b" " * 72 + b"X" + b"      1\n"), "unknown")

    def test_empty(self):
        self.assertEqual(identity.sniff_bytes(b""), "empty")
        self.assertEqual(identity.sniff_bytes(b"", 0), "empty")
        self.assertEqual(identity.sniff_bytes(None), "empty")
        self.assertEqual(identity.sniff_bytes(b"", 10), "unknown")      # an unreadable head of a non-empty file

    def test_unknown_and_never_raises(self):
        self.assertEqual(identity.sniff_bytes(b"\x00\x01\x02\x03 random bytes"), "unknown")
        self.assertEqual(identity.sniff_bytes(b"abc"), "unknown")
        self.assertEqual(identity.sniff_bytes(bytearray(b"ISO-10303-21;")), "step")
        self.assertIn(identity.sniff_bytes(os.urandom(512), 512), identity.KNOWN_TYPES)
        for head in (b"\xff" * 5, b"\xef\xbb\xbf", b"PK", b"%PDF", b"<"):
            self.assertIn(identity.sniff_bytes(head, len(head)), identity.KNOWN_TYPES)

    def test_every_result_is_a_known_type(self):
        for body in (STEP_BODY, ASCII_STL, b"<html>", b"", b"zzz", b"%PDF-1", b"PK\x03\x04"):
            self.assertIn(identity.sniff_bytes(body, len(body)), identity.KNOWN_TYPES)


class SniffFileTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="identity_files_")

    def write(self, name, data):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def test_the_content_decides_never_the_name(self):
        self.assertEqual(identity.sniff_file(self.write("model.step", b"<!DOCTYPE html><html>Please sign in</html>")), "html")
        self.assertEqual(identity.sniff_file(self.write("page.html", STEP_BODY)), "step")
        self.assertEqual(identity.sniff_file(self.write("mesh.step", ASCII_STL)), "stl")
        self.assertEqual(identity.sniff_file(self.write("part.zip", STEP_BODY)), "step")

    def test_binary_stl_on_disk_uses_the_real_size(self):
        data = binary_stl(100)
        self.assertGreater(len(data), identity.SNIFF_BYTES)           # bigger than the head that is read
        self.assertEqual(identity.sniff_file(self.write("big.stl", data)), "stl")
        self.assertEqual(identity.sniff_file(self.write("cut.stl", data[:-10])), "unknown")

    def test_empty_file_and_lfs_pointer_and_zip(self):
        self.assertEqual(identity.sniff_file(self.write("zero.step", b"")), "empty")
        pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 99\n"
        self.assertEqual(identity.sniff_file(self.write("lfs.step", pointer)), "lfs_pointer")
        self.assertEqual(identity.sniff_file(self.write("a.zip", zip_bytes(**{"x.step": STEP_BODY}))), "zip")

    def test_missing_file_raises_os_error(self):
        with self.assertRaises(OSError):
            identity.sniff_file(os.path.join(self.dir, "nope.step"))


class TypeHelperTest(unittest.TestCase):
    def test_rejection_hints(self):
        self.assertIn("raw.githubusercontent.com", identity.rejection_hint("html"))
        self.assertIn("media.githubusercontent.com", identity.rejection_hint("lfs_pointer"))
        self.assertIn("0 bytes", identity.rejection_hint("empty"))
        for fine in ("step", "stl", "zip", "pdf", "brep", "iges", "unknown", "nonsense"):
            self.assertEqual(identity.rejection_hint(fine), "", fine)

    def test_hints_say_what_to_do(self):
        for kind in ("html", "lfs_pointer", "empty"):
            self.assertIn(";", identity.rejection_hint(kind))

    def test_describe_and_extensions(self):
        self.assertEqual(identity.describe_type("step"), "a STEP file")
        self.assertEqual(identity.describe_type("stl"), "an STL mesh")
        self.assertEqual(identity.describe_type("weird"), "a file of unrecognised type")
        self.assertEqual(identity.canonical_extension("step"), ".step")
        self.assertEqual(identity.canonical_extension("stl"), ".stl")
        self.assertEqual(identity.canonical_extension("zip"), ".zip")
        for none in ("html", "unknown", "empty", "lfs_pointer", "weird"):
            self.assertEqual(identity.canonical_extension(none), "", none)
        self.assertTrue(identity.extension_matches(".STP", "step"))
        self.assertTrue(identity.extension_matches(".3mf", "zip"))
        self.assertTrue(identity.extension_matches(".FCStd", "zip"))
        self.assertFalse(identity.extension_matches(".stl", "step"))
        self.assertFalse(identity.extension_matches("", "step"))
        self.assertFalse(identity.extension_matches(None, "step"))

    def test_cad_types_and_extensions(self):
        self.assertEqual(set(identity.CAD_TYPES), {"step", "stl", "brep", "iges"})
        for ext in (".step", ".stp", ".stl", ".fcstd", ".3mf", ".obj", ".brep", ".iges"):
            self.assertIn(ext, identity.CAD_EXTENSIONS)
        self.assertNotIn(".pdf", identity.CAD_EXTENSIONS)
        for kind in identity.CAD_TYPES:
            self.assertIn(kind, identity.KNOWN_TYPES)


class BboxIdentityTest(unittest.TestCase):
    def test_exact_and_swapped_axes_match(self):
        for measured in ((85.0, 56.0, 17.0), (56.0, 17.0, 85.0), (17.0, 85.0, 56.0)):
            r = identity.bbox_identity(measured, (85.0, 56.0, 17.0))
            self.assertTrue(r.ok)
            self.assertEqual(r.status, "match")
            self.assertEqual(r.measured, (85.0, 56.0, 17.0))
            self.assertEqual(r.expected, (85.0, 56.0, 17.0))
            self.assertIn("plausibility check", r.detail)

    def test_tolerance_boundaries(self):
        # tolerance per side: max(1.5 mm, 8% of the expected side)  -> 6.8 on 85, 4.48 on 56, 1.5 on 17 (8% = 1.36)
        self.assertEqual(identity.bbox_identity((91.7, 56.0, 17.0), (85.0, 56.0, 17.0)).status, "match")
        self.assertEqual(identity.bbox_identity((91.9, 56.0, 17.0), (85.0, 56.0, 17.0)).status, "mismatch")
        self.assertEqual(identity.bbox_identity((85.0, 56.0, 18.4), (85.0, 56.0, 17.0)).status, "match")
        self.assertEqual(identity.bbox_identity((85.0, 56.0, 18.6), (85.0, 56.0, 17.0)).status, "mismatch")

    def test_custom_tolerances(self):
        self.assertEqual(identity.bbox_identity((86.0, 56.0, 17.0), (85.0, 56.0, 17.0), rel_tol=0.0, abs_tol=0.5).status, "mismatch")
        self.assertEqual(identity.bbox_identity((86.0, 56.0, 17.0), (85.0, 56.0, 17.0), rel_tol=0.0, abs_tol=1.5).status, "match")

    def test_three_dimensions_against_a_two_dimensional_outline(self):
        r = identity.bbox_identity((85.2, 56.1, 17.0), (85.0, 56.0))
        self.assertEqual((r.status, r.measured, r.expected), ("match", (85.2, 56.1), (85.0, 56.0)))
        r = identity.bbox_identity((85.0, 56.0), (85.0, 56.0, 17.0))
        self.assertEqual(r.status, "match")

    def test_a_different_part_is_a_mismatch_and_says_so(self):
        r = identity.bbox_identity((65.0, 30.0, 5.0), (85.0, 56.0, 17.0))
        self.assertFalse(r.ok)
        self.assertEqual(r.status, "mismatch")
        self.assertIn("probably a different part", r.detail)
        self.assertIn("65", r.detail)

    def test_unit_mix_ups_are_named(self):
        expected = (85.0, 56.0, 17.0)
        cases = {25.4: "millimetres and read as inches", 1 / 25.4: "inches and read as millimetres",
                 1000.0: "millimetres and read as metres", 0.001: "metres and read as millimetres",
                 10.0: "millimetres and read as centimetres", 0.1: "centimetres and read as millimetres"}
        for factor, words in cases.items():
            r = identity.bbox_identity(tuple(v * factor for v in expected), expected)
            self.assertEqual(r.status, "mismatch", factor)
            self.assertIn(words, r.detail, factor)

    def test_unchecked_inputs_never_block(self):
        for measured in (None, (), (1.0,), (1, 2, 3, 4), (0, 1, 2), (-1, 2, 3), (float("nan"), 1, 2), (math.inf, 1, 2), ("a", "b"),
                         (True, 1.0, 2.0), 7):
            r = identity.bbox_identity(measured, (85.0, 56.0, 17.0))
            self.assertTrue(r.ok, measured)
            self.assertEqual(r.status, "unchecked", measured)
            self.assertIn("measured bounding box", r.detail)
        for expected in (None, (), [], (0, 0, 0), (1.0,)):
            r = identity.bbox_identity((85.0, 56.0, 17.0), expected)
            self.assertTrue(r.ok, expected)
            self.assertEqual(r.status, "unchecked", expected)
            self.assertIn("no expected size", r.detail)

    def test_result_is_immutable(self):
        r = identity.bbox_identity((1, 2, 3), (1, 2, 3))
        with self.assertRaises(Exception):
            r.ok = False


class OutlineIdentityTest(unittest.TestCase):
    def setUp(self):
        self.pi4 = catalog.load_board("rpi4b")

    def test_a_bare_pcb_and_a_full_model_both_match_the_pi4(self):
        for measured in ((85.0, 56.0, 1.4), (56.0, 85.0, 1.4), (87.0, 56.0, 17.0), (89.0, 57.5, 17.0)):
            r = identity.outline_identity(measured, self.pi4)
            self.assertEqual(r.status, "match", measured)
            self.assertTrue(r.ok)
            self.assertEqual(r.expected, (85.0, 56.0))
            self.assertIn("Raspberry Pi 4 Model B", r.detail)
            self.assertIn("plausibility check", r.detail)

    def test_too_small_or_too_big_is_a_mismatch(self):
        for measured in ((65.0, 30.0, 5.0), (100.0, 56.0, 17.0), (85.0, 40.0, 17.0), (200.0, 150.0, 30.0), (30.0, 20.0)):
            r = identity.outline_identity(measured, self.pi4)
            self.assertFalse(r.ok, measured)
            self.assertEqual(r.status, "mismatch", measured)
            self.assertIn("probably a different part", r.detail)

    def test_boards_with_one_footprint_cannot_be_told_apart(self):
        # a Pi 3 B+ model passes as a Pi 4 by size alone: this check is plausibility, and the wording says so
        pi3 = catalog.load_board("rpi3bplus")
        self.assertEqual(identity.outline_identity((85.0, 56.0, 17.4), pi3).status, "match")
        r = identity.outline_identity((85.0, 56.0, 17.4), self.pi4)
        self.assertEqual(r.status, "match")
        self.assertIn("not proof of identity", r.detail)

    def test_a_pi_zero_is_not_a_pi4(self):
        zero = catalog.load_board("rpizero2w")
        self.assertEqual(identity.outline_identity((65.0, 30.0, 5.0), zero).status, "match")
        self.assertEqual(identity.outline_identity((65.0, 30.0, 5.0), self.pi4).status, "mismatch")

    def test_unit_hint_on_the_footprint(self):
        r = identity.outline_identity((85.0 / 25.4, 56.0 / 25.4, 1.0), self.pi4)
        self.assertEqual(r.status, "mismatch")
        self.assertIn("inches and read as millimetres", r.detail)

    def test_unchecked_when_nothing_to_compare(self):
        for board in (None, {}, {"outline": {}}, {"outline": {"length": "x", "width": 56}}, 5):
            r = identity.outline_identity((85.0, 56.0, 1.4), board)
            self.assertTrue(r.ok)
            self.assertEqual(r.status, "unchecked")
            self.assertIn("no outline size", r.detail)
        for measured in (None, (), (1.0,), (0, 1, 2), (float("nan"), 2, 3)):
            r = identity.outline_identity(measured, self.pi4)
            self.assertTrue(r.ok)
            self.assertEqual(r.status, "unchecked")
            self.assertIn("measured bounding box", r.detail)

    def test_a_board_without_connectors_still_checks_the_outline(self):
        bare = {"name": "Bare", "outline": {"length": 50.0, "width": 30.0}}
        self.assertEqual(identity.outline_identity((50.0, 30.0, 1.6), bare).status, "match")
        self.assertEqual(identity.outline_identity((60.0, 30.0, 1.6), bare).status, "mismatch")

    def test_every_shipped_board_matches_its_own_outline(self):
        for item in catalog.list_boards():
            board = catalog.load_board(item["id"])
            o = board["outline"]
            r = identity.outline_identity((o["length"], o["width"], board["pcb"]["thickness"]), board)
            self.assertEqual(r.status, "match", item["id"])


if __name__ == "__main__":
    unittest.main()
