# test_util.py
import asyncio
import unittest

from src.trivialai.util import (TransformError, astream_checked_retries,
                                loadch, stream_checked, stream_checked_retries)


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

    # --- Existing streaming tests ---

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

    # --- New: retry-capable streaming tests (sync & async) ---

    def test_stream_checked_retries_retry_then_success(self):
        """First attempt fails to parse; second succeeds."""
        attempts = [
            ["{bad:", "json}"],  # invalid
            ['[{"ok":', " true}]"],  # valid JSON list
        ]
        idx = {"i": 0}

        def factory():
            # pick attempt's parts, then advance
            parts = attempts[min(idx["i"], len(attempts) - 1)]
            idx["i"] += 1
            return self._fake_stream(parts)

        evs = list(stream_checked_retries(factory, loadch, retries=2))

        # should have an attempt-failed message
        self.assertTrue(any(e.get("type") == "attempt-failed" for e in evs))
        final = evs[-1]
        self.assertEqual(final.get("type"), "final")
        self.assertTrue(final.get("ok"))
        self.assertEqual(final.get("attempt"), 2)
        self.assertEqual(final.get("parsed"), [{"ok": True}])

    def test_stream_checked_retries_all_fail(self):
        """All attempts fail; final should indicate failure with attempts count."""
        attempts = [
            ["{bad:", "json}"],
            ["{still:", "bad}"],
        ]
        idx = {"i": 0}

        def factory():
            parts = attempts[min(idx["i"], len(attempts) - 1)]
            idx["i"] += 1
            return self._fake_stream(parts)

        evs = list(stream_checked_retries(factory, loadch, retries=2))
        finals = [e for e in evs if e.get("type") == "final"]
        self.assertEqual(len(finals), 1)
        final = finals[0]
        self.assertFalse(final.get("ok"))
        self.assertEqual(final.get("attempts"), 2)
        self.assertEqual(final.get("error"), "failed-on-2-retries")

    def test_astream_checked_retries_retry_then_success(self):
        """Async variant: first attempt fails; second succeeds."""
        attempts = [
            ["{bad:", "json}"],
            ['{"good":', " 1}"],
        ]
        idx = {"i": 0}

        async def async_stream(parts):
            yield {"type": "start", "provider": "test", "model": "dummy"}
            for p in parts:
                yield {"type": "delta", "text": p}
                await asyncio.sleep(0)  # yield control
            yield {"type": "end", "content": "".join(parts)}

        async def factory():
            parts = attempts[min(idx["i"], len(attempts) - 1)]
            idx["i"] += 1
            return async_stream(parts)

        async def run():
            out = []
            async for ev in astream_checked_retries(factory, loadch, retries=2):
                out.append(ev)
            return out

        evs = asyncio.run(run())
        self.assertTrue(any(e.get("type") == "attempt-failed" for e in evs))
        final = evs[-1]
        self.assertEqual(final.get("type"), "final")
        self.assertTrue(final.get("ok"))
        self.assertEqual(final.get("attempt"), 2)
        self.assertEqual(final.get("parsed"), {"good": 1})
