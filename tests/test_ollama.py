# tests/test_ollama.py
import asyncio
import unittest

from src.trivialai.ollama import Ollama


class FakeStreamOllama(Ollama):
    """
    Synthetic Ollama subclass that overrides _astream_raw to avoid network I/O.

    We don't test tag-splitting here (that's covered by LLMMixin tests); we only
    verify that:
      - the stream facade passes through deltas correctly, and
      - Ollama.agenerate aggregates content and scratchpad from the stream.
    """

    def __init__(self):
        super().__init__(
            model="fake",
            ollama_server="http://example",
            skip_healthcheck=True,
        )

    async def _astream_raw(self, system, prompt, images=None):
        # Simple, fully-controlled event stream
        yield {"type": "start", "provider": "ollama", "model": self.model}

        # One scratch-only delta, then two text-only deltas
        yield {"type": "delta", "text": "", "scratchpad": "abc"}
        yield {"type": "delta", "text": " Hi", "scratchpad": ""}
        yield {"type": "delta", "text": " there", "scratchpad": ""}

        # End event with final aggregates
        yield {
            "type": "end",
            "content": " Hi there",
            "scratchpad": "abc",
            "tokens": 2,
        }


class TestOllama(unittest.TestCase):
    def test_constructor_normalizes_server(self):
        o = Ollama("mistral", "http://host:11434/", skip_healthcheck=True)
        self.assertEqual(o.server, "http://host:11434")
        self.assertEqual(o.model, "mistral")

    def test_think_tags_configured(self):
        # Ollama should configure think boundaries for LLMMixin helpers
        self.assertEqual(Ollama.THINK_OPEN, "<think>")
        self.assertEqual(Ollama.THINK_CLOSE, "</think>")

    def test_stream_facade_includes_scratchpad_deltas(self):
        o = FakeStreamOllama()
        events = list(o.stream("sys", "prompt"))

        # Ensure start, some deltas, and end
        kinds = [e.get("type") for e in events]
        self.assertIn("start", kinds)
        self.assertIn("delta", kinds)
        self.assertIn("end", kinds)

        # Collect text and scratchpad deltas
        deltas = [e for e in events if e.get("type") == "delta"]
        self.assertTrue(len(deltas) > 0)

        text_chunks = [e["text"] for e in deltas]
        scratch_chunks = [e["scratchpad"] for e in deltas]

        # Some deltas should be purely scratchpad, some purely text
        self.assertIn("", text_chunks)  # scratch-only delta exists
        self.assertIn("", scratch_chunks)  # text-only deltas exist

        # Final aggregates should match end event
        end = next(e for e in events if e.get("type") == "end")
        self.assertEqual("".join(text_chunks), end["content"])
        self.assertEqual("".join(scratch_chunks), end["scratchpad"])

    def test_agenerate_accumulates_both_streams(self):
        o = FakeStreamOllama()

        async def run():
            return await o.agenerate("sys", "prompt")

        res = asyncio.run(run())
        self.assertEqual(res.content, " Hi there")
        self.assertEqual(res.scratchpad, "abc")
