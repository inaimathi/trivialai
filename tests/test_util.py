# tests/test_util.py
import asyncio
import unittest

from src.trivialai.util import (TransformError, astream_checked,
                                generate_checked, loadch, stream_checked)


class TestUtil(unittest.TestCase):
    # -------- loadch --------

    def test_loadch_valid_json(self):
        valid_resp = '{"key": "value"}'
        result = loadch(valid_resp)
        self.assertEqual(result, {"key": "value"})

    def test_loadch_valid_json_with_code_block(self):
        valid_resp = '```json\n{"key": "value"}\n```'
        result = loadch(valid_resp)
        self.assertEqual(result, {"key": "value"})

    def test_loadch_none_input(self):
        with self.assertRaises(TransformError) as ctx:
            loadch(None)
        self.assertEqual(str(ctx.exception), "no-message-given")

    def test_loadch_invalid_json(self):
        invalid_resp = "{key: value}"  # Invalid JSON
        with self.assertRaises(TransformError) as ctx:
            loadch(invalid_resp)
        self.assertEqual(str(ctx.exception), "parse-failed")

    def test_loadch_invalid_format_with_code_block(self):
        invalid_resp = "```json\n{key: value}\n```"
        with self.assertRaises(TransformError) as ctx:
            loadch(invalid_resp)
        self.assertEqual(str(ctx.exception), "parse-failed")

    # -------- generate_checked (one-shot) --------

    def test_generate_checked_success(self):
        class _Res:
            raw = None
            content = '{"ok": true}'
            scratchpad = None

        def _gen():
            return _Res()

        out = generate_checked(_gen, loadch)
        self.assertEqual(out.content, {"ok": True})
        self.assertIsNone(out.scratchpad)

    def test_generate_checked_failure_raises_transformerror(self):
        class _Res:
            raw = None
            content = "{not:json}"
            scratchpad = None

        def _gen():
            return _Res()

        with self.assertRaises(TransformError) as ctx:
            _ = generate_checked(_gen, loadch)
        self.assertEqual(str(ctx.exception), "parse-failed")

    # -------- streaming (one-shot) --------

    def _fake_stream(self, parts):
        yield {"type": "start", "provider": "test", "model": "dummy"}
        for p in parts:
            yield {"type": "delta", "text": p}
        yield {"type": "end", "content": "".join(parts)}

    def test_stream_checked_success(self):
        parts = ['{"key": ', '"value"', "}"]
        evs = list(stream_checked(self._fake_stream(parts), loadch))

        # passthrough
        self.assertTrue(any(e.get("type") == "start" for e in evs))
        self.assertTrue(any(e.get("type") == "delta" for e in evs))
        self.assertTrue(any(e.get("type") == "end" for e in evs))

        final = evs[-1]
        self.assertEqual(final.get("type"), "final")
        self.assertTrue(final.get("ok"))
        self.assertEqual(final.get("parsed"), {"key": "value"})

    def test_stream_checked_failure(self):
        parts = ["{key: ", "value", "}"]  # invalid JSON
        evs = list(stream_checked(self._fake_stream(parts), loadch))

        final = evs[-1]
        self.assertEqual(final.get("type"), "final")
        self.assertFalse(final.get("ok"))
        self.assertEqual(final.get("error"), "parse-failed")

    # -------- async streaming (one-shot) --------

    async def _async_stream(self, parts):
        yield {"type": "start", "provider": "test", "model": "dummy"}
        for p in parts:
            yield {"type": "delta", "text": p}
            await asyncio.sleep(0)
        yield {"type": "end", "content": "".join(parts)}

    def test_astream_checked_success(self):
        async def run():
            parts = ['{"a":', " 1}"]
            out = []
            async for ev in astream_checked(self._async_stream(parts), loadch):
                out.append(ev)
            return out

        evs = asyncio.run(run())
        final = evs[-1]
        self.assertEqual(final.get("type"), "final")
        self.assertTrue(final.get("ok"))
        self.assertEqual(final.get("parsed"), {"a": 1})

    def test_astream_checked_failure(self):
        async def run():
            parts = ["{bad:", " json}"]
            out = []
            async for ev in astream_checked(self._async_stream(parts), loadch):
                out.append(ev)
            return out

        evs = asyncio.run(run())
        final = evs[-1]
        self.assertEqual(final.get("type"), "final")
        self.assertFalse(final.get("ok"))
        self.assertEqual(final.get("error"), "parse-failed")
