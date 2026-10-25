"""tests/support/gaps.py - pure helpers for tests that pin a CONTRACT behaviour a production module does not deliver yet."""

import unittest


def known_gap(reason: str):
    """Marks a test that asserts the intended behaviour that the code does not deliver yet (an expected failure
    whose docstring names the defect). When the owner fixes the module the test turns into an "unexpected success": remove
    the decorator and keep the test."""
    def wrap(fn):
        fn.__doc__ = f"KNOWN GAP: {reason}"
        return unittest.expectedFailure(fn)
    return wrap
