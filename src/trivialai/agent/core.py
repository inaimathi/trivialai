# src/trivialai/agent.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from .. import util
from ..bistream import BiStream, repeat_until, sequentially
from ..llm import LLMMixin
from ..vectorstore.core import Whim
from . import prompting, toolkit


class Agent:
    def __init__(
        self,
        llm: LLMMixin,
        *,
        system: str,
        tools: Optional(List[Callable[..., Any]]) = None,
        name: Optional[str] = None,
        root: Optional[Union[str, Path]] = None,
    ):
        self.llm = llm
        self.name = name or "agent-task"
        self.tools = toolkit.ToolKit(*([] if tools is None else tools))

        self.system = system

        root_path = Path(root or f"./agent-{self.name}").expanduser().resolve()
        self.root = root_path
        self.root.mkdir(parents=True, exist_ok=True)
        self.scratch_path = self.root / "scratchpad.md"
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

    def build_prompt(self, user_prompt):
        return prompting.build_prompt(self.system, user_prompt, self.tools)

    def tool_shape(self):
        return self.tools.to_tool_shape()

    def check_tool(self, parsed):
        return self.tools.check_tool(parsed)

    def call_tool(self, parsed):
        return self.tools.call_tool(parsed)

    def stream(self, prompt, images: Optional[list] = None) -> BiStream[Dict[str, Any]]:
        return self.llm.stream(self.build_prompt(prompt), prompt, images=images).tap(
            self.log,
            ignore=lambda ev: isinstance(ev, dict) and ev.get("type") == "delta",
        )

    def stream_checked(
        self,
        check_fn: Callable[[str], Any],
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> BiStream[Dict[str, Any]]:
        return self.llm.stream_checked(
            check_fn,
            self.build_prompt(prompt),
            prompt,
            images=images,
            retries=retries,
        ).tap(
            self.log,
            ignore=lambda ev: isinstance(ev, dict) and ev.get("type") == "delta",
        )
