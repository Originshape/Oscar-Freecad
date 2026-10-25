"""``oscar.fsretry.replace``: atomic replace that rides out Windows sharing violations (testable on every OS)."""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from support import isolation  # noqa: E402

isolation.isolate_process()

from oscar import fsretry  # noqa: E402


class ReplaceTest(unittest.TestCase):
    def test_posix_style_replaces_once_and_never_retries(self):
        with mock.patch.object(fsretry, "_windows", return_value=False), \
                mock.patch("os.replace", side_effect=PermissionError("denied")) as replace, \
                mock.patch("time.sleep") as sleep:
            with self.assertRaises(PermissionError):
                fsretry.replace("a", "b")
        replace.assert_called_once_with("a", "b")
        sleep.assert_not_called()

    def test_windows_retries_a_sharing_violation_until_it_succeeds(self):
        attempts = [PermissionError("[WinError 5]"), PermissionError("[WinError 32]"), None]
        with mock.patch.object(fsretry, "_windows", return_value=True), \
                mock.patch("os.replace", side_effect=attempts) as replace, mock.patch("time.sleep") as sleep:
            fsretry.replace("a", "b")
        self.assertEqual(replace.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [0.01, 0.02])      # a growing delay

    def test_windows_gives_up_after_the_last_attempt_with_the_original_error(self):
        with mock.patch.object(fsretry, "_windows", return_value=True), \
                mock.patch("os.replace", side_effect=PermissionError("locked")) as replace, mock.patch("time.sleep") as sleep:
            with self.assertRaises(PermissionError) as ctx:
                fsretry.replace("a", "b")
        self.assertEqual(str(ctx.exception), "locked")
        self.assertEqual(replace.call_count, fsretry.ATTEMPTS)
        self.assertEqual(sleep.call_count, fsretry.ATTEMPTS - 1)
        self.assertLessEqual(max(c.args[0] for c in sleep.call_args_list), fsretry.MAX_DELAY_S)

    def test_other_errors_are_not_retried(self):
        with mock.patch.object(fsretry, "_windows", return_value=True), \
                mock.patch("os.replace", side_effect=FileNotFoundError("gone")) as replace, mock.patch("time.sleep"):
            with self.assertRaises(FileNotFoundError):
                fsretry.replace("a", "b")
        self.assertEqual(replace.call_count, 1)

    def test_it_really_replaces_a_file(self):
        with tempfile.TemporaryDirectory() as folder:
            src, dst = os.path.join(folder, "a"), os.path.join(folder, "b")
            with open(src, "w") as f:
                f.write("new")
            with open(dst, "w") as f:
                f.write("old")
            fsretry.replace(src, dst)
            with open(dst) as f:
                self.assertEqual(f.read(), "new")
            self.assertFalse(os.path.exists(src))


if __name__ == "__main__":
    unittest.main()
