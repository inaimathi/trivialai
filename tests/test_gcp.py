import asyncio
import unittest

from src.trivialai.gcp import _SAFETY_MAP, GCP
from src.trivialai.llm import LLMResult


class DummyInitGCP(GCP):
    """Bypass real credentials and vertex init for tests."""

    def _gcp_creds(self):
        # set dummy creds and project id without calling google.auth
        self.gcp_creds = object()
        self.gcp_project_id = "test-project"

    def vertex_init(self):
        # no-op to avoid touching vertexai.init in tests
        pass


class FakeStreamGCP(DummyInitGCP):
    """Provide a deterministic in-process stream to test LLM surfaces."""

    def __init__(self):
        super().__init__(
            model="gemini-1.5-flash",
            vertex_api_creds="{}",
            region="us-central1",
        )

    async def astream(self, system, prompt, images=None):
        yield {"type": "start", "provider": "gcp", "model": self.model}
        for part in ["Hi", " ", "GCP"]:
            yield {"type": "delta", "text": part, "scratchpad": ""}
        yield {"type": "end", "content": "Hi GCP", "scratchpad": None, "tokens": 2}

    async def agenerate(self, system, prompt, images=None):
        # aggregate from our astream so tests don't invoke real Vertex
        parts = []
        async for ev in self.astream(system, prompt, images):
            if ev.get("type") == "delta":
                parts.append(ev.get("text", ""))
            elif ev.get("type") == "end" and ev.get("content") is not None:
                parts = [ev["content"]]
        return LLMResult(raw=None, content="".join(parts), scratchpad=None)


class TestGCP(unittest.TestCase):
    def test_constructor_builds_safety_settings(self):
        g = DummyInitGCP(
            model="gemini-1.5-flash",
            vertex_api_creds="{}",
            region="us-central1",
        )
        # safety list should cover all categories from the map
        self.assertEqual(len(g.safety_settings), len(_SAFETY_MAP))

    def test_stream_facade_and_agenerate(self):
        g = FakeStreamGCP()

        # Sync stream facade should iterate events
        events = list(g.stream("sys", "prompt"))
        kinds = [e.get("type") for e in events]
        self.assertIn("start", kinds)
        self.assertIn("delta", kinds)
        self.assertIn("end", kinds)

        deltas = [e for e in events if e.get("type") == "delta"]
        text = "".join(d.get("text", "") for d in deltas)
        end = next(e for e in events if e.get("type") == "end")
        self.assertEqual(text, end.get("content"))
        self.assertIsNone(end.get("scratchpad"))

        # Async aggregate via agenerate()
        async def run():
            return await g.agenerate("sys", "prompt")

        res = asyncio.run(run())
        self.assertEqual(res.content, "Hi GCP")
        self.assertIsNone(res.scratchpad)
