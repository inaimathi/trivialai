import unittest

from src.trivialai.util import TransformError, loadch


class TestUtil(unittest.TestCase):
    def test_loadch_valid_json(self):
        """Test loadch with a valid JSON string."""
        valid_resp = '{"key": "value"}'
        result = loadch(valid_resp)
        self.assertEqual(result, {"key": "value"})

    def test_loadch_valid_json_with_code_block(self):
        """Test loadch with a JSON string inside a code block."""
        valid_resp = '```json\n{"key": "value"}\n```'
        result = loadch(valid_resp)
        self.assertEqual(result, {"key": "value"})

    def test_loadch_none_input(self):
        """Test loadch with None as input."""
        with self.assertRaises(TransformError) as context:
            loadch(None)
        self.assertEqual(str(context.exception), "no-message-given")

    def test_loadch_invalid_json(self):
        """Test loadch with an invalid JSON string."""
        invalid_resp = "{key: value}"  # Invalid JSON
        with self.assertRaises(TransformError) as context:
            loadch(invalid_resp)
        self.assertEqual(str(context.exception), "parse-failed")

    def test_loadch_invalid_format_with_code_block(self):
        """Test loadch with an invalid JSON string inside a code block."""
        invalid_resp = "```json\n{key: value}\n```"
        with self.assertRaises(TransformError) as context:
            loadch(invalid_resp)
        self.assertEqual(str(context.exception), "parse-failed")
