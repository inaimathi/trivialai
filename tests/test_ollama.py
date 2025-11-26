# tests/test_ollama.py
import asyncio
import unittest

from src.trivialai.ollama import Ollama


class FakeStreamOllama(Ollama):
    """
    Synthetic Ollama subclass that overrides `astream` to avoid network I/O.

    We don't test tag-splitting helpers here (that's covered by LLMMixin tests).
    Instead we verify that:

      - the public `.stream(...)` facade yields start / delta / end with
        correctly split text vs scratchpad based on THINK_OPEN/THINK_CLOSE, and
      - `.agenerate(...)` aggregates content and scratchpad from that stream.
    """

    def __init__(self):
        super().__init__(
            model="fake",
            ollama_server="http://example",
            skip_healthcheck=True,
        )

    async def astream(self, system, prompt, images=None):
        """
        Provider-level async stream.

        We simulate a model whose raw output is:

            "<think>abc</think> Hi there"

        broken into three streaming chunks. LLMMixin.stream will:
          - route "abc" into the scratchpad,
          - route " Hi there" into content,
          - and generate the appropriate deltas/end event.
        """
        yield {"type": "start", "provider": "ollama", "model": self.model}

        parts = ["<think>abc</think>", "Hi", " there"]
        for p in parts:
            yield {"type": "delta", "text": p}

        # Base end event; LLMMixin.stream will rewrite `content` and `scratchpad`.
        yield {
            "type": "end",
            "content": "".join(parts),
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

        # Some deltas should be purely scratchpad, some purely text:
        # - from "<think>abc</think>" we get text == "" and scratchpad == "abc"
        # - from " Hi" / " there" we get text, scratchpad == ""
        self.assertIn("", text_chunks)  # scratch-only delta exists
        self.assertIn("", scratch_chunks)  # text-only deltas exist

        # Final aggregates should match end event (rewritten by LLMMixin.stream)
        end = next(e for e in events if e.get("type") == "end")
        self.assertEqual("".join(text_chunks), end["content"])
        self.assertEqual("".join(scratch_chunks), end["scratchpad"])

    def test_agenerate_accumulates_both_streams(self):
        o = FakeStreamOllama()

        async def run():
            return await o.agenerate("sys", "prompt")

        res = asyncio.run(run())
        # LLMMixin.stream should have turned "<think>abc</think> Hi there"
        # into content=" Hi there", scratchpad="abc"
        self.assertEqual(res.content, "Hi there")
        self.assertEqual(res.scratchpad, "abc")
