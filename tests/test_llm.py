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
        # Use the instance-level helper (new API)
        content, scratch = self.split_think_full(self._full_text)
        return LLMResult(raw=None, content=content, scratchpad=scratch)


class StreamingThinkLLM(LLMMixin):
    """
    LLM that exercises streaming scratchpad splitting via LLMMixin.stream().

    THINK_OPEN/THINK_CLOSE are configured so that:
      - .astream() yields raw chunks (with tags possibly split across chunks)
      - .stream() applies the tag-based splitting to produce text/scratchpad deltas
    """

    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"

    def __init__(self, chunks):
        # e.g. ["<thi", "nk>abc", "123</t", "hink>HEL", "LO"]
        self._chunks = chunks

    def generate(self, system, prompt, images=None) -> LLMResult:
        # Non-streaming fallback: reuse the same helper on the full text.
        full = "".join(self._chunks)
        content, scratch = self.split_think_full(full)
        return LLMResult(raw=None, content=content, scratchpad=scratch)

    async def astream(self, system, prompt, images=None):
        """
        Simulate a streaming provider that yields the raw chunks verbatim.

        LLMMixin.stream() will wrap this, calling self.separate_think_delta()
        to split out THINK_OPEN/THINK_CLOSE regions into scratchpad.
        """
        yield {"type": "start", "provider": "streamingthinkllm", "model": None}
        for part in self._chunks:
            # Raw text, including <think>...</think> fragments across chunks
            yield {"type": "delta", "text": part}
        # Base end event; LLMMixin.stream will rewrite content/scratchpad.
        yield {"type": "end", "content": "".join(self._chunks)}


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

        # BiStream: sync iteration is allowed
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
        # split_think_full should strip the think block from public content
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
