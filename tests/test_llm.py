import asyncio
import unittest

from src.trivialai.llm import LLMMixin
from src.trivialai.util import LLMResult, loadch


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
