# tests/test_util.py
import asyncio
import unittest

from src.trivialai.util import (TransformError, astream_checked,
                                generate_checked, loadch, loadchmulti,
                                stream_checked)


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
        invalid_resp = "{key: value}"  # Invalid even for JSON5 (bare string value)
        with self.assertRaises(TransformError) as ctx:
            loadch(invalid_resp)
        self.assertEqual(str(ctx.exception), "parse-failed")

    def test_loadch_invalid_format_with_code_block(self):
        invalid_resp = "```json\n{key: value}\n```"
        with self.assertRaises(TransformError) as ctx:
            loadch(invalid_resp)
        self.assertEqual(str(ctx.exception), "parse-failed")

    def test_loadch_passthrough_dict(self):
        obj = {"a": 1}
        # loadch should pass through already-structured dicts unchanged
        result = loadch(obj)
        self.assertIs(result, obj)

    def test_loadch_multiline_string_newlines_escaped(self):
        # Model-style JSON with a multiline string value (illegal JSON, but
        # the lenient loader should fix it by escaping newlines).
        resp = '{"type": "conclusion", ' '"summary": "Line1\nLine2\nLine3"}'
        result = loadch(resp)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["type"], "conclusion")
        summary = result["summary"]
        self.assertIn("Line1", summary)
        self.assertIn("Line2", summary)
        self.assertIn("Line3", summary)
        # Ensure we actually preserved newlines in the parsed value
        self.assertIn("\n", summary)

    # -------- loadchmulti --------

    def test_loadchmulti_list_passthrough(self):
        resp = [{"a": 1}, {"b": 2}]
        out = loadchmulti(resp)
        self.assertEqual(out, resp)

    def test_loadchmulti_tuple_passthrough(self):
        resp = ({"a": 1}, {"b": 2})
        out = loadchmulti(resp)
        self.assertEqual(out, [{"a": 1}, {"b": 2}])

    def test_loadchmulti_dict_wrapped(self):
        resp = {"a": 1}
        out = loadchmulti(resp)
        self.assertEqual(out, [resp])

    def test_loadchmulti_single_json_string(self):
        resp = '{"x": 1}'
        out = loadchmulti(resp)
        self.assertEqual(out, [{"x": 1}])

    def test_loadchmulti_code_blocks_multiple(self):
        resp = '```json\n{"a": 1}\n```\n\n```json\n{"b": 2}\n```'
        out = loadchmulti(resp)
        self.assertEqual(out, [{"a": 1}, {"b": 2}])

    def test_loadchmulti_embedded_multiple_objects(self):
        resp = (
            'Saying: {"type": "tool-call", "tool": "slurp"}\n\n'
            '{"type": "summary", "summary": "done"}'
        )
        out = loadchmulti(resp)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["type"], "tool-call")
        self.assertEqual(out[0]["tool"], "slurp")
        self.assertEqual(out[1]["type"], "summary")
        self.assertEqual(out[1]["summary"], "done")

    def test_loadchmulti_ignores_invalid_objects(self):
        # First brace block is invalid JSON5; second is valid.
        resp = 'noise {not: "json" 123}\n' '{"type": "summary", "summary": "ok"}'
        out = loadchmulti(resp)
        self.assertEqual(out, [{"type": "summary", "summary": "ok"}])

    def test_loadchmulti_no_json_raises(self):
        with self.assertRaises(TransformError) as ctx:
            loadchmulti("just some text, no json here")
        self.assertEqual(str(ctx.exception), "parse-failed")

    def test_loadchmulti_multiline_string_newlines(self):
        resp = (
            '{"type": "conclusion", '
            '"summary": "First line\nSecond line\nThird line"}'
        )
        out = loadchmulti(resp)
        self.assertEqual(len(out), 1)
        obj = out[0]
        self.assertEqual(obj["type"], "conclusion")
        summary = obj["summary"]
        self.assertIn("First line", summary)
        self.assertIn("Second line", summary)
        self.assertIn("Third line", summary)
        self.assertIn("\n", summary)

    def test_loadchmulti_code_block_multiline_string(self):
        json_text = '{"type": "conclusion", ' '"summary": "First line\nSecond line"}'
        resp = f"```json\n{json_text}\n```"
        out = loadchmulti(resp)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["type"], "conclusion")

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
