import asyncio
import unittest

from src.trivialai.ollama import (Ollama, _separate_think_delta,
                                  _split_think_full)


class TestOllamaHelpers(unittest.TestCase):
    def test_separate_think_delta_across_chunks(self):
        chunks = ["<thi", "nk>abc", "123</t", "hink>HEL", "LO"]
        in_think = False
        carry = ""
        public_out = []
        scratch_out = []
        for c in chunks:
            pub, scr, in_think, carry = _separate_think_delta(c, in_think, carry)
            if pub:
                public_out.append(pub)
            if scr:
                scratch_out.append(scr)
        # After final chunk there should be no carry and not inside think
        self.assertFalse(in_think)
        self.assertEqual(carry, "")
        self.assertEqual("".join(scratch_out), "abc123")
        self.assertEqual("".join(public_out), "HELLO")


class FakeStreamOllama(Ollama):
    def __init__(self):
        super().__init__(
            model="fake", ollama_server="http://example", skip_healthcheck=True
        )

    async def astream(self, system, prompt, images=None):
        yield {"type": "start", "provider": "ollama", "model": self.model}
        parts = ["<think>abc", "def</think>", " Hi", " there"]
        in_think = False
        carry = ""
        content_buf = []
        scratch_buf = []
        for part in parts:
            out, scr, in_think, carry = _separate_think_delta(part, in_think, carry)
            if scr:
                scratch_buf.append(scr)
            if out:
                content_buf.append(out)
            if out or scr:
                yield {"type": "delta", "text": out or "", "scratchpad": scr or ""}
        # flush carry if any (shouldn't be in this synthetic case)
        if carry:
            if in_think:
                scratch_buf.append(carry)
                yield {"type": "delta", "text": "", "scratchpad": carry}
            else:
                content_buf.append(carry)
                yield {"type": "delta", "text": carry, "scratchpad": ""}
        yield {
            "type": "end",
            "content": "".join(content_buf),
            "scratchpad": "".join(scratch_buf) or None,
            "tokens": 2,
        }


class TestOllamaClass(unittest.TestCase):
    def test_constructor_normalizes_server(self):
        o = Ollama("mistral", "http://host:11434/", skip_healthcheck=True)
        self.assertEqual(o.server, "http://host:11434")
        self.assertEqual(o.model, "mistral")

    def test_stream_facade_includes_scratchpad_deltas(self):
        o = FakeStreamOllama()
        events = list(o.stream("sys", "prompt"))

        # Ensure start, some deltas, and end
        kinds = [e.get("type") for e in events]
        self.assertIn("start", kinds)
        self.assertIn("delta", kinds)
        self.assertIn("end", kinds)

        # Collect text and scratchpad deltas
        text_chunks = [e["text"] for e in events if e.get("type") == "delta"]
        scratch_chunks = [e["scratchpad"] for e in events if e.get("type") == "delta"]

        # Some deltas should be purely scratchpad, some purely text
        self.assertIn("", text_chunks)  # scratch-only deltas exist
        self.assertIn("", scratch_chunks)  # text-only deltas exist

        # Final aggregates should match end event
        end = next(e for e in events if e.get("type") == "end")
        self.assertEqual("".join(text_chunks).strip(), end["content"].strip())
        self.assertEqual("".join(scratch_chunks), end["scratchpad"])

    def test_agenerate_accumulates_both_streams(self):
        o = FakeStreamOllama()

        async def run():
            return await o.agenerate("sys", "prompt")

        res = asyncio.run(run())
        self.assertEqual(res.content, " Hi there")
        self.assertEqual(res.scratchpad, "abcdef")
