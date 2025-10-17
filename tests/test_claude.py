import asyncio
import unittest

from src.trivialai.claude import Claude


class FakeStreamClaude(Claude):
    """Subclass that overrides astream to avoid any network I/O."""

    def __init__(self):
        super().__init__(model="claude-3-haiku-20240307", api_key="sk-test")

    async def astream(self, system, prompt, images=None):
        yield {"type": "start", "provider": "anthropic", "model": self.model}
        for part in ["Hi", " ", "Claude"]:
            yield {"type": "delta", "text": part, "scratchpad": ""}
        yield {"type": "end", "content": "Hi Claude", "scratchpad": None, "tokens": 2}


class TestClaude(unittest.TestCase):
    def test_constructor_values(self):
        c = Claude(
            model="claude-3-haiku-20240307",
            api_key="sk-xyz",
            anthropic_version=None,
            max_tokens=None,
            timeout=10.0,
        )
        self.assertEqual(c.model, "claude-3-haiku-20240307")
        self.assertEqual(c.api_key, "sk-xyz")
        self.assertEqual(c.timeout, 10.0)
        self.assertTrue(c.max_tokens >= 1)
        self.assertTrue(isinstance(c.version, str))

    def test_stream_facade_includes_deltas_and_end(self):
        c = FakeStreamClaude()
        events = list(c.stream("sys", "prompt"))

        kinds = [e.get("type") for e in events]
        self.assertIn("start", kinds)
        self.assertIn("delta", kinds)
        self.assertIn("end", kinds)

        deltas = [e for e in events if e.get("type") == "delta"]
        self.assertTrue(len(deltas) > 0)
        # All deltas should have scratchpad="" for Claude
        self.assertTrue(all(d.get("scratchpad") == "" for d in deltas))

        text = "".join(d.get("text", "") for d in deltas)
        end = next(e for e in events if e.get("type") == "end")
        self.assertEqual(text, end.get("content"))
        self.assertIsNone(end.get("scratchpad"))

    def test_agenerate_aggregates_content(self):
        c = FakeStreamClaude()

        async def run():
            return await c.agenerate("sys", "prompt")

        res = asyncio.run(run())
        self.assertEqual(res.content, "Hi Claude")
        self.assertIsNone(res.scratchpad)
