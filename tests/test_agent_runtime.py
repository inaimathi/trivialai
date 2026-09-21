import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from src.trivialai.agent.core import Agent
from src.trivialai.agent.prompting import build_agent_prompt
from src.trivialai.agent.toolkit import ToolKit
from src.trivialai.bistream import BiStream
from src.trivialai.llm import LLMMixin


class FakeLLM(LLMMixin):
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.system_prompts = []
        self.user_prompts = []

    def stream(self, system, prompt, images=None):
        self.system_prompts.append(system)
        self.user_prompts.append(prompt)
        decision = self.decisions.pop(0)
        content = decision if isinstance(decision, str) else json.dumps(decision)

        async def events():
            yield {"type": "start", "provider": "fake", "model": "text-only"}
            midpoint = max(1, len(content) // 2)
            yield {"type": "delta", "text": content[:midpoint], "scratchpad": ""}
            yield {"type": "delta", "text": content[midpoint:], "scratchpad": ""}
            yield {
                "type": "end",
                "content": content,
                "scratchpad": None,
                "tokens": len(content),
            }

        return BiStream(events())


class AgentRuntimeTests(unittest.TestCase):
    def make_agent(self, llm, *tools):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Agent(
            llm,
            system="You are a test agent.",
            tools=list(tools),
            name="runtime-test",
            root=Path(tmp.name),
        )

    def test_textual_tool_call_result_then_final(self):
        def search(query: str):
            return [{"title": "hit", "query": query}]

        llm = FakeLLM(
            [
                {"type": "tool-call", "tool": "search", "args": {"query": "foo"}},
                {"type": "final", "content": "done"},
            ]
        )
        events = list(self.make_agent(llm, search).run("Investigate it."))
        self.assertEqual(events[-1], {"type": "final", "content": "done", "steps": 2})
        result = next(ev for ev in events if ev["type"] == "tool-result")
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool_call_id"], "call-1")
        self.assertIn('"type":"tool-result"', llm.system_prompts[1])
        self.assertIn('"query":"foo"', llm.system_prompts[1])
        self.assertEqual(llm.user_prompts, ["Investigate it.", "Investigate it."])

    def test_two_sequential_tool_calls(self):
        seen = []

        def first(value: int):
            seen.append(("first", value))
            return value + 1

        def second(value: int):
            seen.append(("second", value))
            return value * 2

        llm = FakeLLM(
            [
                {"type": "tool-call", "tool": "first", "args": {"value": 2}},
                {"type": "tool-call", "tool": "second", "args": {"value": 3}},
                {"type": "final", "content": "6"},
            ]
        )
        events = list(self.make_agent(llm, first, second).run("Compute."))
        self.assertEqual(seen, [("first", 2), ("second", 3)])
        self.assertEqual(
            [e["tool_call_id"] for e in events if e["type"] == "tool-call"],
            ["call-1", "call-2"],
        )
        self.assertEqual(events[-1]["steps"], 3)

    def test_sync_tool_under_async_consumption(self):
        def add(a: int, b: int):
            return a + b

        llm = FakeLLM(
            [
                {"type": "tool-call", "tool": "add", "args": {"a": 2, "b": 5}},
                {"type": "final", "content": "7"},
            ]
        )
        agent = self.make_agent(llm, add)

        async def consume():
            return [event async for event in agent.run("Add.")]

        events = asyncio.run(consume())
        result = next(e for e in events if e["type"] == "tool-result")
        self.assertEqual(result["result"], 7)

    def test_async_tool_under_sync_consumption(self):
        async def add(a: int, b: int):
            await asyncio.sleep(0)
            return a + b

        llm = FakeLLM(
            [
                {"type": "tool-call", "tool": "add", "args": {"a": 4, "b": 6}},
                {"type": "final", "content": "10"},
            ]
        )
        events = list(self.make_agent(llm, add).run("Add."))
        result = next(e for e in events if e["type"] == "tool-result")
        self.assertEqual(result["result"], 10)

    def test_async_tool_under_async_consumption(self):
        async def add(a: int, b: int):
            await asyncio.sleep(0)
            return a + b

        llm = FakeLLM(
            [
                {"type": "tool-call", "tool": "add", "args": {"a": 4, "b": 6}},
                {"type": "final", "content": "10"},
            ]
        )
        agent = self.make_agent(llm, add)

        async def consume():
            return [event async for event in agent.run("Add.")]

        events = asyncio.run(consume())
        result = next(e for e in events if e["type"] == "tool-result")
        self.assertEqual(result["result"], 10)

    def test_toolkit_call_tool_bridges_async_tool(self):
        async def add(a: int, b: int):
            await asyncio.sleep(0)
            return a + b

        tools = ToolKit(add)
        result = tools.call_tool(
            {"type": "tool-call", "tool": "add", "args": {"a": 1, "b": 2}}
        )
        self.assertEqual(result, 3)

    def test_bad_arguments_are_recoverable(self):
        def typed(value: int):
            return value

        bad_calls = [
            {"type": "tool-call", "tool": "typed", "args": {}},
            {"type": "tool-call", "tool": "typed", "args": {"value": 1, "extra": 2}},
            {"type": "tool-call", "tool": "typed", "args": {"value": "wrong"}},
        ]
        for bad in bad_calls:
            with self.subTest(bad=bad):
                llm = FakeLLM([bad, {"type": "final", "content": "recovered"}])
                events = list(self.make_agent(llm, typed).run("Try it."))
                result = next(e for e in events if e["type"] == "tool-result")
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"]["type"], "tool-error")
                self.assertEqual(events[-1]["content"], "recovered")
                self.assertEqual(len(llm.system_prompts), 2)

    def test_tool_results_are_json_serializable_with_deterministic_fallbacks(self):
        class Opaque:
            pass

        def inspect_values():
            return {
                "path": Path("repo/file.txt"),
                "values": {3, 1, 2},
                "opaque": Opaque(),
            }

        llm = FakeLLM(
            [
                {"type": "tool-call", "tool": "inspect_values", "args": {}},
                {"type": "final", "content": "done"},
            ]
        )
        events = list(self.make_agent(llm, inspect_values).run("Inspect."))
        result = next(e for e in events if e["type"] == "tool-result")
        json.dumps(result)
        self.assertEqual(result["result"]["path"], "repo/file.txt")
        self.assertEqual(result["result"]["values"], [1, 2, 3])
        self.assertEqual(result["result"]["opaque"]["type"], "python-object")
        self.assertTrue(result["result"]["opaque"]["class"].endswith("Opaque"))

    def test_tool_exception_is_recoverable(self):
        def explode():
            raise ValueError("boom")

        llm = FakeLLM(
            [
                {"type": "tool-call", "tool": "explode", "args": {}},
                {"type": "final", "content": "worked around it"},
            ]
        )
        events = list(self.make_agent(llm, explode).run("Try."))
        result = next(e for e in events if e["type"] == "tool-result")
        self.assertFalse(result["ok"])
        self.assertIn("boom", result["error"]["message"])
        self.assertEqual(events[-1]["type"], "final")

    def test_unknown_tool_is_recoverable(self):
        llm = FakeLLM(
            [
                {"type": "tool-call", "tool": "does_not_exist", "args": {}},
                {"type": "final", "content": "fallback"},
            ]
        )
        events = list(self.make_agent(llm).run("Try."))
        result = next(e for e in events if e["type"] == "tool-result")
        self.assertFalse(result["ok"])
        self.assertEqual(result["tool"], "does_not_exist")
        self.assertEqual(events[-1]["type"], "final")

    def test_max_steps_is_terminal(self):
        def ping():
            return "pong"

        llm = FakeLLM(
            [
                {"type": "tool-call", "tool": "ping", "args": {}},
                {"type": "tool-call", "tool": "ping", "args": {}},
            ]
        )
        events = list(
            self.make_agent(llm, ping).run(
                "Loop.", max_steps=2, max_identical_tool_calls=None
            )
        )
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "max-steps-exceeded")
        self.assertEqual(len(llm.system_prompts), 2)

    def test_repeated_identical_tool_calls_are_bounded(self):
        calls = []

        def ping(value: int):
            calls.append(value)
            return value

        decision = {"type": "tool-call", "tool": "ping", "args": {"value": 1}}
        llm = FakeLLM([decision, decision, decision])
        events = list(
            self.make_agent(llm, ping).run(
                "Loop.", max_steps=8, max_identical_tool_calls=2
            )
        )
        self.assertEqual(calls, [1, 1])
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "repeated-tool-call")
        failed = [e for e in events if e["type"] == "tool-result" and not e["ok"]]
        self.assertEqual(len(failed), 1)

    def test_invalid_decision_is_retried_without_consuming_agent_step(self):
        seen = []

        def echo(value: int):
            seen.append(value)
            return value

        llm = FakeLLM(
            [
                "I should call the tool now.",
                {"type": "tool-call", "tool": "echo", "args": {"value": 7}},
                {"type": "final", "content": "done"},
            ]
        )

        events = list(
            self.make_agent(llm, echo).run(
                "Use echo.",
                decision_retries=3,
            )
        )

        self.assertEqual(seen, [7])
        self.assertEqual(events[-1], {"type": "final", "content": "done", "steps": 2})

        failures = [
            event for event in events
            if event.get("type") == "model-attempt-failed"
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["step"], 0)
        self.assertEqual(failures[0]["attempt"], 1)
        self.assertEqual(len(llm.system_prompts), 3)

    def test_prompt_trimming_keeps_tools_and_latest_complete_pair(self):
        def search(query: str):
            """Search the test corpus."""
            return query

        tools = ToolKit(search)
        history = [
            (
                {
                    "type": "tool-call",
                    "tool_call_id": "call-1",
                    "tool": "search",
                    "args": {"query": "OLD_CALL"},
                },
                {
                    "type": "tool-result",
                    "tool_call_id": "call-1",
                    "tool": "search",
                    "ok": True,
                    "result": "OLD_RESULT",
                },
            ),
            (
                {
                    "type": "tool-call",
                    "tool_call_id": "call-2",
                    "tool": "search",
                    "args": {"query": "LATEST_CALL"},
                },
                {
                    "type": "tool-result",
                    "tool_call_id": "call-2",
                    "tool": "search",
                    "ok": True,
                    "result": "LATEST_RESULT",
                },
            ),
        ]
        prompt = build_agent_prompt(
            "SYSTEM_SENTINEL",
            "task",
            tools,
            history=history,
            context_size=1,
            context_summary="SUMMARY_SENTINEL",
        )
        self.assertIn("SYSTEM_SENTINEL", prompt)
        self.assertIn("search", prompt)
        self.assertIn("LATEST_CALL", prompt)
        self.assertIn("LATEST_RESULT", prompt)
        self.assertLess(prompt.index("LATEST_CALL"), prompt.index("LATEST_RESULT"))
        self.assertNotIn("OLD_CALL", prompt)
        self.assertNotIn("OLD_RESULT", prompt)
        self.assertNotIn("SUMMARY_SENTINEL", prompt)


if __name__ == "__main__":
    unittest.main()
