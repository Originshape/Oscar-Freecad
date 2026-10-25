# SPDX-License-Identifier: Apache-2.0
"""Is there a newer Oscar? (oscar/updates.py): the public ``GET /v1/plugin`` against the fake OriginShape server,
what its answer means for this Oscar, how often it is asked, and the switches that turn it off. Pure python (stdlib +
requests).
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

from oscar import updates  # noqa: E402
from oscar.auth import discovery, session  # noqa: E402
from oscar.auth.transport import AuthError, AuthUnavailable  # noqa: E402
from support.fake_originshape import FakeOriginShape  # noqa: E402


class UpdateCheckTest(unittest.TestCase):
    def setUp(self):
        for reset in (updates.clear_cache, discovery.clear_cache, session.reset_sessions):
            reset()
            self.addCleanup(reset)
        self.fake = FakeOriginShape()
        self.addCleanup(self.fake.close)

    def test_a_newer_oscar_is_found_without_signing_in(self):
        self.fake.plugin = {"latest_version": "0.3.0", "min_version": "0.2.0", "download_url": ""}
        release = updates.fetch(self.fake.url)
        self.assertEqual((release.latest_version, release.min_version), ("0.3.0", "0.2.0"))
        self.assertEqual(release.download_url, self.fake.url + "/download")
        call = self.fake.calls_to("/v1/plugin")[-1]
        self.assertNotIn("authorization", call.headers)                    # public: no token goes with it
        self.assertTrue(call.headers.get("user-agent", "").startswith("Oscar-FreeCAD/"))
        self.assertEqual(release.update("0.2.0"), "available")
        self.assertEqual(release.update("0.1.5"), "required")
        self.assertEqual(release.update("0.3.0"), "")
        self.assertEqual(updates.cached(self.fake.url), release)

    def test_it_is_asked_at_most_once_a_day_per_server(self):
        self.assertTrue(updates.due(self.fake.url))
        updates.fetch(self.fake.url)
        self.assertFalse(updates.due(self.fake.url))
        self.assertTrue(updates.due(self.fake.url, now=updates.cached(self.fake.url).checked_at
                                    + updates.CHECK_EVERY_S + 1))
        self.assertTrue(updates.due("https://other.example.com"))

    def test_a_failed_check_is_not_retried_before_the_next_day(self):
        with self.assertRaises(AuthUnavailable):
            updates.fetch("http://127.0.0.1:9")                             # nothing listens there
        self.assertFalse(updates.due("http://127.0.0.1:9"))
        self.assertIsNone(updates.cached("http://127.0.0.1:9"))

    def test_an_answer_that_is_not_a_release_is_refused(self):
        for body in ({}, {"latest_version": "soon"}, ["0.3.0"], None):
            with self.assertRaises(AuthError):
                updates.parse(body)
        bad_link = updates.parse({"latest_version": "1.0.0", "download_url": "javascript:alert(1)"})
        self.assertEqual((bad_link.latest_version, bad_link.download_url), ("1.0.0", ""))
        with mock.patch.object(self.fake, "plugin", {"latest_version": "x"}):
            with self.assertRaises(AuthError):
                updates.fetch(self.fake.url)

    def test_the_switches(self):
        with mock.patch.dict(os.environ, {updates.ENV: "0"}):
            self.assertFalse(updates.enabled())
        with mock.patch.dict(os.environ, {updates.ENV: "1"}):
            self.assertTrue(updates.enabled())                              # no FreeCAD here: the default is on


if __name__ == "__main__":
    unittest.main()
