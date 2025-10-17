import asyncio
import unittest

from src.trivialai.chatgpt import ChatGPT


class FakeStreamChatGPT(ChatGPT):
    def __init__(self):
        super().__init__(model="gpt-4o-mini", api_key="sk-test")

    async def astream(self, system, prompt, images=None):
        yield {"type": "start", "provider": "openai", "model": self.model}
        for part in ["Hello", " ", "world"]:
            yield {"type": "delta", "text": part, "scratchpad": ""}
        yield {"type": "end", "content": "Hello world", "scratchpad": None, "tokens": 2}


class TestChatGPT(unittest.TestCase):
    def test_constructor_values(self):
        c = ChatGPT(
            model="gpt-4o-mini",
            api_key="sk-xyz",
            anthropic_version=None,
            max_tokens=None,
            timeout=10.0,
        )
        self.assertEqual(c.model, "gpt-4o-mini")
        self.assertEqual(c.api_key, "sk-xyz")
        self.assertEqual(c.timeout, 10.0)
        self.assertTrue(c.max_tokens >= 1)

    def test_stream_facade_includes_deltas_and_end(self):
        c = FakeStreamChatGPT()
        events = list(c.stream("sys", "prompt"))

        kinds = [e.get("type") for e in events]
        self.assertIn("start", kinds)
        self.assertIn("delta", kinds)
        self.assertIn("end", kinds)

        # All deltas should have scratchpad="" for ChatGPT
        deltas = [e for e in events if e.get("type") == "delta"]
        self.assertTrue(len(deltas) > 0)
        self.assertTrue(all(d.get("scratchpad") == "" for d in deltas))

        # Aggregated content matches end content
        text = "".join(d.get("text", "") for d in deltas)
        end = next(e for e in events if e.get("type") == "end")
        self.assertEqual(text, end.get("content"))
        self.assertIsNone(end.get("scratchpad"))

    def test_agenerate_aggregates_content(self):
        c = FakeStreamChatGPT()

        async def run():
            return await c.agenerate("sys", "prompt")

        res = asyncio.run(run())
        self.assertEqual(res.content, "Hello world")
        self.assertIsNone(res.scratchpad)
