"""Write-location guard of ``save_document(path)`` / ``export_model(directory)`` (oscar/tools_project.py)."""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

import project_support as S  # noqa: E402
from test_project_tools import ToolCase  # noqa: E402
from oscar import tools_project  # noqa: E402


class WriteLocationTest(unittest.TestCase):
    def problem(self, path, **kw):
        return tools_project._write_location_problem(path, None, directory=kw.pop("directory", False), **kw)

    def test_system_and_protected_locations_are_refused(self):
        for path in ("/System/x.FCStd", "/usr/local/x.FCStd", "/etc/x.FCStd", "/Library/x.FCStd", "/Applications/x.FCStd",
                     os.path.expanduser("~/.ssh/x.FCStd"), os.path.expanduser("~/Library/x.FCStd")):
            self.assertTrue(self.problem(path), path)
        self.assertTrue(self.problem("/System", directory=True))
        self.assertTrue(self.problem("/", directory=True))

    def test_ordinary_locations_pass(self):
        self.assertEqual(self.problem(os.path.join(S.sandbox_dir(), "a.FCStd")), "")
        self.assertEqual(self.problem(os.path.expanduser("~/Documents/a.FCStd")), "")
        self.assertEqual(self.problem(S.sandbox_dir(), directory=True), "")

    @unittest.skipIf(sys.platform == "win32", "the /etc symlink scenario is POSIX-specific (no /etc on Windows)")
    def test_a_symlink_into_a_system_folder_is_refused(self):
        link = os.path.join(S.sandbox_dir(), "sneaky")
        os.symlink("/etc", link)
        self.assertIn("system", self.problem(os.path.join(link, "x.FCStd")))

    # -- Windows protected roots (the guard reads the environment at CALL time; fake roots, never the real ones) ----

    @staticmethod
    def _fake_roots(*names):
        return {name: tempfile.mkdtemp(prefix=f"oscar_{name}_", dir=S.sandbox_dir()) for name in names}

    def test_windows_protected_roots_are_refused(self):
        if sys.platform != "win32":
            self.skipTest("Windows environment roots only")
        roots = self._fake_roots("SystemRoot", "ProgramFiles", "ProgramData")
        # the sandbox counts as a temp folder, and temp bypasses the system check by design: make the
        # guard see a DIFFERENT temp dir so the fake roots are judged as ordinary locations
        elsewhere = mock.patch.object(tools_project.tempfile, "gettempdir",
                                      return_value=os.path.join(S.sandbox_dir(), "not-a-root"))
        with mock.patch.dict(os.environ, roots), elsewhere:
            for root in roots.values():
                self.assertIn("system directory", self.problem(os.path.join(root, "x.FCStd")))
                self.assertIn("system directory", self.problem(root, directory=True))
                # case- and separator-insensitive spellings of the same root
                self.assertIn("system directory", self.problem(root.lower().replace("\\", "/") + "/x.FCStd"))
                self.assertIn("system directory", self.problem(root + os.sep, directory=True))

    def test_windows_lookalike_siblings_are_not_refused(self):
        if sys.platform != "win32":
            self.skipTest("Windows environment roots only")
        base = tempfile.mkdtemp(prefix="oscar_SystemRoot_", dir=S.sandbox_dir())
        elsewhere = mock.patch.object(tools_project.tempfile, "gettempdir",
                                      return_value=os.path.join(S.sandbox_dir(), "not-a-root"))
        with mock.patch.dict(os.environ, {"SystemRoot": base}), elsewhere:
            self.assertIn("system directory", self.problem(os.path.join(base, "x.FCStd")))
            self.assertEqual(self.problem(base + "_backup", directory=True), "")
            self.assertEqual(self.problem(os.path.join(base + "_backup", "x.FCStd")), "")

    def test_a_junction_into_a_windows_protected_root_is_refused(self):
        if sys.platform != "win32":
            self.skipTest("junctions are a Windows feature")
        root = self._fake_roots("ProgramData")["ProgramData"]
        link = os.path.join(S.sandbox_dir(), "jroot")
        subprocess.run(["cmd", "/c", "mklink", "/J", link, root], check=True, capture_output=True)
        elsewhere = mock.patch.object(tools_project.tempfile, "gettempdir",
                                      return_value=os.path.join(S.sandbox_dir(), "not-a-root"))
        with mock.patch.dict(os.environ, {"ProgramData": root}), elsewhere:
            self.assertIn("system directory", self.problem(os.path.join(link, "x.FCStd")))

    def test_the_real_machine_windows_roots_are_refused(self):
        if sys.platform != "win32":
            self.skipTest("Windows environment roots only")
        for name in ("SystemRoot", "ProgramFiles", "ProgramData"):
            root = os.environ.get(name, "")
            if root:   # decision-only: nothing is ever written there
                self.assertIn("system directory", self.problem(os.path.join(root, "oscar_probe.FCStd")), name)

    def test_an_existing_non_fcstd_file_is_never_replaced(self):
        target = os.path.join(S.sandbox_dir(), "notes.FCStd")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("precious text")
        self.assertIn("not an .FCStd", self.problem(target, existing_fcstd_only=True))
        self.assertIn("folder", self.problem(S.sandbox_dir()))


class ToolGuardTest(ToolCase):
    def test_save_document_refuses_system_and_foreign_files(self):
        doc = self.solid_doc()
        text, _, is_error = self.call("save_document", doc_name=doc.Name, path="/etc/model.FCStd")
        self.assertTrue(is_error)
        self.assertIn("system directory", text)
        self.assertIn("nothing was written", text)
        self.assertFalse(os.path.exists("/etc/model.FCStd"))
        victim = os.path.join(S.sandbox_dir(), "victim.FCStd")
        with open(victim, "w", encoding="utf-8") as fh:
            fh.write("not a zip")
        text, _, is_error = self.call("save_document", doc_name=doc.Name, path=victim)
        self.assertTrue(is_error)
        self.assertIn("not an .FCStd", text)
        with open(victim, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "not a zip")

    def test_existing_fcstd_inside_the_sandbox_is_still_overwritten(self):
        doc = self.solid_doc()
        target = os.path.join(S.sandbox_dir(), "again.FCStd")
        for _ in range(2):
            text, _, is_error = self.call("save_document", doc_name=doc.Name, path=target)
            self.assertFalse(is_error, text)

    def test_export_model_refuses_a_system_directory(self):
        doc = self.solid_doc()
        text, _, is_error = self.call("export_model", doc_name=doc.Name, directory="/usr/local/oscar_out", formats=["stl"])
        self.assertTrue(is_error)
        self.assertIn("system directory", text)
        self.assertFalse(os.path.exists("/usr/local/oscar_out"))


if __name__ == "__main__":
    unittest.main()
