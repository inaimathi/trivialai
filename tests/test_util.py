import unittest

from src.trivialai.util import TransformError, loadch, stream_checked


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

    # --- New streaming tests ---

    def _fake_stream(self, parts):
        yield {"type": "start", "provider": "test", "model": "dummy"}
        for p in parts:
            yield {"type": "delta", "text": p}
        yield {"type": "end", "content": "".join(parts)}

    def test_stream_checked_success(self):
        """stream_checked should pass through events and emit a parsed final event on success."""
        parts = ['{"key": ', '"value"', "}"]
        evs = list(stream_checked(self._fake_stream(parts), loadch))

        # Ensure we passed through start/delta/end
        self.assertTrue(any(e.get("type") == "start" for e in evs))
        self.assertTrue(any(e.get("type") == "delta" for e in evs))
        self.assertTrue(any(e.get("type") == "end" for e in evs))

        # Final event
        final = evs[-1]
        self.assertEqual(final.get("type"), "final")
        self.assertTrue(final.get("ok"))
        self.assertEqual(final.get("parsed"), {"key": "value"})

    def test_stream_checked_failure(self):
        """stream_checked should emit a final error if transformFn fails."""
        parts = ["{key: ", "value", "}"]  # invalid JSON
        evs = list(stream_checked(self._fake_stream(parts), loadch))

        final = evs[-1]
        self.assertEqual(final.get("type"), "final")
        self.assertFalse(final.get("ok"))
        self.assertEqual(final.get("error"), "parse-failed")
