# test_llm.py
import asyncio
import unittest

from src.trivialai.llm import LLMMixin
from src.trivialai.util import LLMResult, TransformError, loadch


class FakeLLM(LLMMixin):
    """
    A tiny test double that returns predefined contents on each `generate` call.
    Last value is reused if calls exceed provided contents.
    """

    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = 0
        self.model = "fake-model"

    def generate(self, system, prompt, images=None) -> LLMResult:
        idx = min(self.calls, len(self.contents) - 1)
        self.calls += 1
        content = self.contents[idx]
        return LLMResult(raw=None, content=content, scratchpad=None)


class _StubTools:
    """
    Minimal Tools-like stub for tests.
    Enforces list-of-dicts with {"functionName": str, "args": dict}.
    """

    def list(self):
        return [
            {"name": "foo", "description": "stub", "args": {"x": {"type": "integer"}}}
        ]

    def transform_multi(self, resp):
        data = loadch(resp)
        if not isinstance(data, list):
            raise TransformError("result-not-list", raw=data)
        for item in data:
            if (
                not isinstance(item, dict)
                or "functionName" not in item
                or "args" not in item
            ):
                raise TransformError("invalid-tool-subcall", raw=item)
        return data


class TestLLMMixin(unittest.TestCase):
    def test_generate_checked_success(self):
        llm = FakeLLM(['{"x": 1}'])
        res = llm.generate_checked(loadch, "sys", "prompt")
        # content is parsed by transformFn
        self.assertEqual(res.content, {"x": 1})
        self.assertIsNone(res.scratchpad)

    def test_generate_checked_retries_then_success(self):
        # First attempt invalid JSON, second attempt valid
        llm = FakeLLM(["{x: 1}", '{"x": 1}'])
        res = llm.generate_checked(loadch, "sys", "prompt", retries=2)
        self.assertEqual(res.content, {"x": 1})
        self.assertEqual(llm.calls, 2)

    def test_generate_json_strips_code_fence(self):
        llm = FakeLLM(['```json\n{"ok": true}\n```'])
        res = llm.generate_json("sys", "prompt")
        self.assertEqual(res.content, {"ok": True})

    def test_stream_default_emits_start_delta_end(self):
        text = "hello world"
        llm = FakeLLM([text])
        events = list(llm.stream("sys", "prompt"))
        kinds = [e.get("type") for e in events]
        self.assertIn("start", kinds)
        self.assertIn("delta", kinds)
        self.assertIn("end", kinds)
        # concat of deltas == text
        deltas = [e.get("text", "") for e in events if e.get("type") == "delta"]
        self.assertEqual("".join(deltas), text)
        # end contains the full content
        end = next(e for e in events if e.get("type") == "end")
        self.assertEqual(end.get("content"), text)

    def test_stream_json_final_parse(self):
        llm = FakeLLM(['{"a": 1, "b": 2}'])
        events = list(llm.stream_json("sys", "prompt"))
        # last event is final parsed JSON
        final = events[-1]
        self.assertEqual(final.get("type"), "final")
        self.assertTrue(final.get("ok"))
        self.assertEqual(final.get("parsed"), {"a": 1, "b": 2})

    def test_agenerate_default(self):
        llm = FakeLLM(["result text"])

        async def run():
            r = await llm.agenerate("sys", "prompt")
            return r

        res = asyncio.run(run())
        self.assertEqual(res.content, "result text")

    # --- New: tool-call streaming tests (sync & async) ---

    def test_stream_tool_calls_success_no_retry(self):
        tools = _StubTools()
        # Already a valid list of tool calls
        payload = '[{"functionName": "foo", "args": {"x": 1}}]'
        llm = FakeLLM([payload])

        evs = list(llm.stream_tool_calls(tools, prompt="ignored", retries=3))

        # Ensure pass-through events happened
        self.assertTrue(any(e.get("type") == "start" for e in evs))
        self.assertTrue(any(e.get("type") == "end" for e in evs))

        final = evs[-1]
        self.assertEqual(final.get("type"), "final")
        self.assertTrue(final.get("ok"))
        self.assertEqual(final.get("attempt"), 1)
        self.assertEqual(
            final.get("parsed"), [{"functionName": "foo", "args": {"x": 1}}]
        )

    def test_stream_tool_calls_retry_then_success(self):
        tools = _StubTools()
        # First attempt is NOT a list (invalid), second attempt is valid
        bad = '{"functionName": "foo", "args": {"x": 1}}'
        good = '[{"functionName": "foo", "args": {"x": 2}}]'
        llm = FakeLLM([bad, good])

        evs = list(llm.stream_tool_calls(tools, prompt="ignored", retries=2))

        # We should see at least one attempt-failed before the final
        self.assertTrue(any(e.get("type") == "attempt-failed" for e in evs))
        final = evs[-1]
        self.assertEqual(final.get("type"), "final")
        self.assertTrue(final.get("ok"))
        self.assertEqual(final.get("attempt"), 2)
        self.assertEqual(
            final.get("parsed"), [{"functionName": "foo", "args": {"x": 2}}]
        )

    def test_astream_tool_calls_retry_then_success(self):
        tools = _StubTools()
        bad = '{"not": "a list"}'
        good = '[{"functionName": "foo", "args": {"x": 42}}]'
        llm = FakeLLM([bad, good])

        async def run():
            out = []
            async for ev in llm.astream_tool_calls(tools, prompt="ignored", retries=2):
                out.append(ev)
            return out

        evs = asyncio.run(run())

        # Expect an attempt-failed and then a final success
        self.assertTrue(any(e.get("type") == "attempt-failed" for e in evs))
        final = evs[-1]
        self.assertEqual(final.get("type"), "final")
        self.assertTrue(final.get("ok"))
        self.assertEqual(final.get("attempt"), 2)
        self.assertEqual(
            final.get("parsed"), [{"functionName": "foo", "args": {"x": 42}}]
        )
