# src/trivialai/agent.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from .. import util
from ..bistream import BiStream, repeat_until, sequentially
from ..llm import LLMMixin
from ..ollama import Ollama
from ..tools import ToolKit
from ..vectorstore.core import Whim
from . import prompting

TODO = "TODO"

_DEFAULT_IGNORE = r"(^__pycache__|^node_modules|\.git/|/env-.*|\.egg-info/.*|^venv|^\..*|~$|\.pyc$|Thumbs\.db$|^build[\\/]|^dist[\\/]|^coverage[\\/]|\.log$|\.lock$|\.bak$|\.swp$|\.swo$|\.tmp$|\.temp$|\.class$|^target$|^Cargo\.lock$)"

# core.run_pdb_test("/home/inaimathi/projects/pycronado/")
# res = core.force(_)


def run_pdb_test(path: str):

    def log_event_to_file(ev: dict[str, Any]) -> None:
        line = json.dumps(ev, default=str)
        util.spit("agent_log.ndjson", line + "\n", mode="a")

    qwq = Ollama("qwq:latest", "http://localhost:11435/")
    tools = ToolKit(util.slurp, util.spit)
    files = list(util.deep_ls(path, ignore=_DEFAULT_IGNORE))
    src_files = [f for f in files if f.endswith(".py")]

    system = (
        "You are an autonomous agent that follows instructions carefully and"
        " respects file write permissions enforced by the host."
        " You MUST NOT write outside the directories explicitly "
        "granted write access, except for the agent's own scratchpad."
        "You may write to the scratchpad by calling the `write_own_scratchpad` tool."
        " Your current task is to look over a git repository and find potential bugs."
        " The way you should go about doing this is by using the Hypothesis testing module "
        "to write properties in new test files and then run them. You should propose tests "
        "that point to real, exploitable bugs, rather than tests that might be a result of "
        "style issues. Use any tools you've been handed, and feel free to write to disk where "
        "you've been given permissions. Your target is a git repository, and you don't have "
        "general shell or commit permissions so at worst any mistakes can be easily rolled "
        "back once they're reviewed."
    )
    # agent = Agent(
    #     Ollama("qwq:latest", "http://localhost:11435/"),
    #     system=system,
    #     name="pdb_agent_001",
    # )

    def _check_resp(resp: str) -> dict[str, Any]:
        parsed = util.loadch(resp)
        if parsed.get("type") not in {"summary", "tool-call"}:
            raise util.TransformError("invalid-object-structure")
        if parsed["type"] == "tool-call":
            return tools.check_tool(parsed)
        return parsed

    def _per_file_stream(f: str):
        prompt = (
            f"Find bugs in the repository {path}. The relevant file tree in the repo is {files}. "
            f"You are currently working on file {f}. Your response should be either\n"
            "1. a tool call (in which case the tool will be called, and once it completes, you "
            "   will be given its result to evaluate). In this case it is IMPORTANT that your "
            f"   response be ONLY the tool call structure (a JSON object like {tools.to_tool_shape()}) "
            " and no other commentary.\n"
            "2. a summary of what you've done so far and an approval to move on to the next phase. "
            "   In this case, it is IMPORTANT that your response be ONLY a summary structure "
            "   {type: 'summary', summary: MarkdownString}. The markdown string should be a concise "
            "   summary of your findings so far sufficient for a reviewer to understand and double "
            "   check further on in the process.\n"
        )

        base = qwq.stream_checked(
            _check_resp,
            prompting.build_prompt(system, prompt, tools=tools),
            prompt,
        )

        def _proceed(final_ev: dict[str, Any]):
            parsed = final_ev.get("parsed", {})

            if parsed.get("type") == "tool-call":
                res = tools.call_tool(parsed)

                yield {
                    "type": "trivialai.agent.log",
                    "message": f"Running a tool call {parsed} -> {type(res)}",
                }

                pr = (
                    f"You previously asked me to run the tool call {parsed}. "
                    f"The result of that call is {res}."
                    f"{prompt}\n"
                )

                yield from qwq.stream_checked(
                    _check_resp,
                    prompting.build_prompt(system, pr, tools=tools),
                    pr,
                )

            # For parsed["type"] == "summary" we don't do anything here.
            # repeat_until will notice the summary and stop scheduling more passes.

        # Repeatedly run LLM -> tool -> LLM until we see a summary in parsed["type"]
        return repeat_until(
            base,
            _proceed,
            pred=lambda ev: isinstance(ev, dict) and ev.get("type") == "final",
            stop=lambda final_ev, i: final_ev.get("parsed", {}).get("type")
            == "summary",
            max_iters=10,  # or whatever upper bound you like
        ).tap(
            log_event_to_file,
            ignore=lambda ev: isinstance(ev, dict) and ev.get("type") == "delta",
        )

    # Top-level: flatten per-file streams
    for f in src_files:
        yield from _per_file_stream(f)


def force(event_generator):
    end_events = []
    current_mode = None  # 'thinking', 'saying', or None

    for event in event_generator:
        if event["type"] == "delta":
            scratchpad = event.get("scratchpad", "")
            text = event.get("text", "")

            # Handle scratchpad content
            if scratchpad:
                if current_mode != "thinking":
                    if current_mode is not None:
                        print()  # Add newline when switching modes
                    print("Thinking: ", end="")
                    current_mode = "thinking"
                print(scratchpad, end="")

            # Handle text content
            if text:
                if current_mode != "saying":
                    if current_mode is not None:
                        print()  # Add newline when switching modes
                    print("Saying: ", end="")
                    current_mode = "saying"
                print(text, end="")

        elif event["type"] == "end":
            end_events.append(event)

        else:
            # Non-delta, non-end events (like 'start')
            if current_mode is not None:
                print()  # Add newline when switching from content to standalone
                current_mode = None
            print(event)

    # Add final newline if we were in a content mode
    if current_mode is not None:
        print()

    return end_events


class Agent:
    def __init__(
        self,
        llm: LLMMixin,
        *,
        system: str,
        tools: Optional[ToolKit] = None,
        name: Optional[str] = None,
        root: Optional[Union[str, Path]] = None,
    ):
        self.llm = llm
        self.name = name or "agent-task"
        self.tools = ToolKit() if tools is None else tools

        self.system = system

        root_path = Path(root or f"./agent-{self.name}").expanduser().resolve()
        self.root = root_path
        scratch = root_path / "scratchpad.md"
        scratch.mkdir(parents=True, exist_ok=True)
        self.scratch_path = scratch
        self.log_path: Path = self.root / f"agent-log.ndjson"

        def write_own_scratchpad(text: str) -> None:
            util.spit(self.scratch_path, text, mode="w")

        self.tools.ensure_tool(write_own_scratchpad)

    # @classmethod
    # def from_logs(cls, log_path: Path) -> Agent:
    #     return TODO

    def log(self, ev):
        line = json.dumps(ev, default=str)
        util.spit(self.log_path, line + "\n", mode="a")

    def stream(self, prompt):
        return self.llm.stream(prompting.build_prompt(self.system), prompt).tap(
            self.log,
            ignore=lambda ev: isinstance(ev, dict) and ev.get("type") == "delta",
        )

    def stream_checked(self, check_fn, prompt, retries=5):
        return self.llm.stream_checked(
            check_fn,
            prompting.build_prompt(self.system),
            prompt,
            retries=retries,
        ).tap(
            self.log,
            ignore=lambda ev: isinstance(ev, dict) and ev.get("type") == "delta",
        )
