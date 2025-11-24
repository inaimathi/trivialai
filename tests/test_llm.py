# tests/test_llm.py
import asyncio
import unittest

from src.trivialai.llm import LLMMixin, LLMResult


class DummyLLM(LLMMixin):
    """
    Minimal concrete LLM for testing the default LLMMixin behavior.

    - No THINK_OPEN/THINK_CLOSE configured.
    - generate() just returns fixed content/scratchpad.
    """

    def __init__(self, content: str, scratchpad=None):
        self._content = content
        self._scratchpad = scratchpad

    def generate(self, system, prompt, images=None) -> LLMResult:
        # Ignore system/prompt/images; fixed response
        return LLMResult(raw=None, content=self._content, scratchpad=self._scratchpad)


class ThinkLLM(LLMMixin):
    """
    LLM that uses LLMMixin's tag-based helpers for full-response splitting.
    """

    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"

    def __init__(self, full_text: str):
        self._full_text = full_text

    def generate(self, system, prompt, images=None) -> LLMResult:
        content, scratch = self._split_think_full(self._full_text)
        return LLMResult(raw=None, content=content, scratchpad=scratch)


class StreamingThinkLLM(LLMMixin):
    """
    LLM that uses the tag-based helpers in a custom _astream_raw to exercise
    streaming scratchpad splitting via .stream(), not by testing helpers directly.
    """

    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"

    def __init__(self, chunks):
        self._chunks = chunks

    def generate(self, system, prompt, images=None) -> LLMResult:
        # Fallback non-streaming behavior: reuse the helpers on the full text.
        full = "".join(self._chunks)
        content, scratch = self._split_think_full(full)
        return LLMResult(raw=None, content=content, scratchpad=scratch)

    async def _astream_raw(self, system, prompt, images=None):
        # Simulate a streaming provider that returns the chunks in self._chunks
        yield {"type": "start", "provider": "streamingthinkllm", "model": None}

        in_think = False
        carry = ""
        content_buf = []
        scratch_buf = []

        for part in self._chunks:
            out, scr, in_think, carry = self._separate_think_delta(
                part, in_think, carry
            )
            if scr:
                scratch_buf.append(scr)
            if out:
                content_buf.append(out)
            if out or scr:
                yield {"type": "delta", "text": out or "", "scratchpad": scr or ""}

        # Flush any carry that turned out not to be a tag
        if carry:
            if in_think:
                scratch_buf.append(carry)
                yield {"type": "delta", "text": "", "scratchpad": carry}
            else:
                content_buf.append(carry)
                yield {"type": "delta", "text": carry, "scratchpad": ""}

        final_content = "".join(content_buf)
        final_scratch = "".join(scratch_buf) or None
        yield {
            "type": "end",
            "content": final_content,
            "scratchpad": final_scratch,
            "tokens": len(final_content.split()) if final_content else 0,
        }


class TestLLMMixinBasics(unittest.TestCase):
    def test_generate_checked_applies_transform(self):
        llm = DummyLLM("hello world")

        def to_upper(s: str) -> str:
            return s.upper()

        res = llm.generate_checked(to_upper, "sys", "prompt")
        self.assertEqual(res.content, "HELLO WORLD")
        self.assertIsNone(res.scratchpad)

    def test_generate_json_uses_loadch_via_generate_checked(self):
        # We don't need to exercise JSON parsing here; just make sure
        # generate_json delegates and returns an LLMResult.
        llm = DummyLLM('{"foo": "bar"}')
        res = llm.generate_json("sys", "prompt")
        # generate_json wraps loadch, so content should be a dict
        self.assertIsInstance(res.content, dict)
        self.assertEqual(res.content.get("foo"), "bar")
        self.assertIsNone(res.scratchpad)

    def test_agenerate_runs_generate_in_thread_by_default(self):
        llm = DummyLLM("threaded content")

        async def run():
            return await llm.agenerate("sys", "prompt")

        res = asyncio.run(run())
        self.assertEqual(res.content, "threaded content")
        self.assertIsNone(res.scratchpad)

    def test_stream_default_emits_start_delta_end_with_empty_scratchpad(self):
        llm = DummyLLM("hello stream")

        # DualStream: sync iteration is allowed
        events = list(llm.stream("sys", "prompt"))
        kinds = [e.get("type") for e in events]

        self.assertIn("start", kinds)
        self.assertIn("delta", kinds)
        self.assertIn("end", kinds)

        deltas = [e for e in events if e.get("type") == "delta"]
        self.assertTrue(len(deltas) > 0)

        # Default LLMMixin behavior for non-think models:
        # all deltas should have scratchpad == ""
        self.assertTrue(all(d.get("scratchpad") == "" for d in deltas))

        text = "".join(d.get("text", "") for d in deltas)
        end = next(e for e in events if e.get("type") == "end")

        self.assertEqual(text, end.get("content"))
        # DummyLLM returns scratchpad=None
        self.assertIsNone(end.get("scratchpad"))

    def test_stream_checked_emits_final_with_parsed_payload(self):
        llm = DummyLLM("foo bar baz")

        def to_upper(s: str) -> str:
            return s.upper()

        events = list(llm.stream_checked(to_upper, "sys", "prompt"))
        kinds = [e.get("type") for e in events]

        # We should see the streaming events plus a "final"
        self.assertIn("start", kinds)
        self.assertIn("delta", kinds)
        self.assertIn("end", kinds)
        self.assertIn("final", kinds)

        final_ev = next(e for e in events if e.get("type") == "final")
        self.assertTrue(final_ev.get("ok"))
        self.assertEqual(final_ev.get("parsed"), "FOO BAR BAZ")


class TestLLMMixinScratchpadHelpers(unittest.TestCase):
    def test_split_think_full_extracts_scratchpad(self):
        text = "Visible before <think>hidden reasoning</think> visible after"
        llm = ThinkLLM(text)

        res = llm.generate("sys", "prompt")
        # _split_think_full should strip the think block from public content
        self.assertEqual(res.content, "Visible before  visible after".strip())
        self.assertEqual(res.scratchpad, "hidden reasoning")

    def test_split_think_full_without_tags_returns_original(self):
        text = "No think tags here"
        llm = ThinkLLM(text)

        res = llm.generate("sys", "prompt")
        self.assertEqual(res.content, text)
        self.assertIsNone(res.scratchpad)


class TestLLMMixinStreamingScratchpad(unittest.TestCase):
    def test_stream_splits_think_blocks_into_scratchpad_deltas(self):
        # Chunks intentionally break the tags in weird places
        chunks = ["<thi", "nk>abc", "123</t", "hink>HEL", "LO"]
        llm = StreamingThinkLLM(chunks)

        events = list(llm.stream("sys", "prompt"))
        kinds = [e.get("type") for e in events]

        self.assertIn("start", kinds)
        self.assertIn("delta", kinds)
        self.assertIn("end", kinds)

        deltas = [e for e in events if e.get("type") == "delta"]

        # Aggregate the public text vs scratchpad from the delta stream
        text_stream = "".join(d.get("text", "") for d in deltas)
        scratch_stream = "".join(d.get("scratchpad", "") for d in deltas)

        # End event should reflect the same aggregates
        end = next(e for e in events if e.get("type") == "end")
        self.assertEqual(text_stream, end.get("content"))
        self.assertEqual(scratch_stream, end.get("scratchpad"))

        # And they should match the expected split from these chunks
        self.assertEqual(scratch_stream, "abc123")
        self.assertEqual(text_stream, "HELLO")

    def test_agenerate_accumulates_streamed_content_and_scratchpad(self):
        chunks = ["<think>reason</think>", " result"]
        llm = StreamingThinkLLM(chunks)

        async def run():
            return await llm.agenerate("sys", "prompt")

        res = asyncio.run(run())
        # Full text is "<think>reason</think> result"
        self.assertEqual(res.content, "result")
        self.assertEqual(res.scratchpad, "reason")
