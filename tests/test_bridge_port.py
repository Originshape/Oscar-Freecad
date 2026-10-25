"""OSCAR_BRIDGE_PORT: the bridge port override that lets a test instance run beside the user's live app."""

import os
import socket
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import bridge  # noqa: E402
from oscar.cad import server  # noqa: E402


def with_port(value):
    env = {bridge.ENV_PORT: value} if value is not None else {}
    return mock.patch.dict(os.environ, env, clear=False)


class ConfiguredPortTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop(bridge.ENV_PORT, None)

    def test_the_default_is_9876(self):
        self.assertEqual(bridge.configured_port(), 9876)
        self.assertEqual(bridge.DEFAULT_PORT, server.DEFAULT_PORT)

    def test_a_valid_override_is_used(self):
        for value, expected in (("9911", 9911), (" 8080 ", 8080), ("1", 1), ("65535", 65535)):
            with with_port(value):
                self.assertEqual(bridge.configured_port(), expected, value)

    def test_an_invalid_override_falls_back_to_the_default(self):
        for value in ("", "abc", "0", "-5", "65536", "98.76", "9876x", "999999999999"):
            with with_port(value):
                self.assertEqual(bridge.configured_port(), 9876, value)

    def test_it_is_read_at_call_time(self):
        with with_port("9001"):
            self.assertEqual(bridge.configured_port(), 9001)
        self.assertEqual(bridge.configured_port(), 9876)

    def test_the_status_line_names_the_configured_port_while_stopped(self):
        with with_port("9123"), mock.patch.object(server, "is_running", return_value=False):
            self.assertEqual(bridge.status_line(), "stopped (port 9123)")


class ServesOnTheConfiguredPortTest(unittest.TestCase):
    def test_the_server_binds_the_override_and_answers(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free = s.getsockname()[1]
        with with_port(str(free)):
            self.addCleanup(lambda: (server.stop(), time.sleep(0.3)))

            class Engine:
                def ping(self):
                    return True

            message = server.start(Engine(), bridge.configured_port())
            self.assertIn(f"127.0.0.1:{free}", message)
            self.assertEqual(server.port(), free)
            import xmlrpc.client

            self.assertTrue(xmlrpc.client.ServerProxy(f"http://127.0.0.1:{free}").ping())


if __name__ == "__main__":
    unittest.main()
