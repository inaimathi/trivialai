import unittest

from src.trivialapi.toda import core


class TestBasics(unittest.TestCase):
    def test_basics(self):
        core.Twin
        self.assertTrue(True)
        self.assertIsNone(None)
        self.assertEqual("a", "a")
