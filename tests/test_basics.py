import unittest

from src.trivialai import ollama


class TestBasics(unittest.TestCase):
    def test_basics(self):
        ollama.Ollama
        self.assertTrue(True)
        self.assertIsNone(None)
        self.assertEqual("a", "a")
