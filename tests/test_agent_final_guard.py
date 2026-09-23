import json
import tempfile
import unittest
from pathlib import Path

from src.trivialai.agent import runtime
from src.trivialai.agent.core import Agent
from src.trivialai.agent.toolkit import ToolKit
from src.trivialai.bistream import BiStream
from src.trivialai.llm import LLMMixin


class FakeLLM(LLMMixin):
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.system_prompts = []

    def stream(self, system, prompt, images=None):
        self.system_prompts.append(system)
        value = self.decisions.pop(0)
        content = value if isinstance(value, str) else json.dumps(value)
        async def events():
            yield {"type":"start", "provider":"fake", "model":"fake"}
            yield {"type":"end", "content":content}
        return BiStream(events())


class DecisionNormalizationTests(unittest.TestCase):
    def test_registered_tool_type_shorthand(self):
        def repo_edit(path: str, old: str, new: str): return new
        tools = ToolKit(repo_edit)
        parsed = runtime._parse_decision(
            '{"type":"repo_edit","path":"x","old":"a","new":"b"}',
            tools=tools,
        )
        self.assertEqual(parsed, {"type":"tool-call","tool":"repo_edit","args":{"path":"x","old":"a","new":"b"}})

    def test_tool_args_shorthand_without_type(self):
        def repo_edit(path: str, old: str, new: str): return new
        tools = ToolKit(repo_edit)
        parsed = runtime._parse_decision(
            '{"tool":"repo_edit","args":{"path":"x","old":"a","new":"b"}}',
            tools=tools,
        )
        self.assertEqual(parsed["type"], "tool-call")
        self.assertEqual(parsed["tool"], "repo_edit")

    def test_unknown_type_remains_invalid(self):
        with self.assertRaises(Exception) as caught:
            runtime._parse_decision('{"type":"banana"}', tools=ToolKit())
        self.assertEqual(str(caught.exception), "invalid-agent-decision-type")


class FinalGuardTests(unittest.TestCase):
    def test_rejected_final_is_fed_back_and_agent_continues(self):
        def finish_work(): return {"published":True}
        def final_check(content, history):
            if not any(call.get("tool") == "finish_work" and result.get("ok") is True for call,result in history):
                return {"code":"not-finished", "message":"Call finish_work before finalizing.", "next_action":"finish_work"}
            return None
        llm = FakeLLM([
            {"type":"final","content":"too early"},
            {"type":"tool-call","tool":"finish_work","args":{}},
            {"type":"final","content":"done"},
        ])
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        agent=Agent(llm, system="test", tools=[finish_work], root=Path(tmp.name), final_check=final_check)
        events=list(agent.run("finish"))
        rejected=[e for e in events if e.get("type")=="final-rejected"]
        self.assertEqual(len(rejected),1)
        self.assertEqual(rejected[0]["error"]["code"],"not-finished")
        self.assertEqual(events[-1]["type"],"final")
        self.assertEqual(events[-1]["steps"],3)
        self.assertIn("not-finished", llm.system_prompts[1])


if __name__ == "__main__": unittest.main()
