# src/trivialai/agent.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from .. import util
from ..bistream import BiStream, sequentially
from ..llm import LLMMixin
from ..ollama import Ollama
from ..tools import Tools
from ..vectorstore.core import Whim
from . import prompting

TODO = "TODO"

_DEFAULT_IGNORE = r"(^__pycache__|^node_modules|\.git/|/env-.*|\.egg-info/.*|^venv|^\..*|~$|\.pyc$|Thumbs\.db$|^build[\\/]|^dist[\\/]|^coverage[\\/]|\.log$|\.lock$|\.bak$|\.swp$|\.swo$|\.tmp$|\.temp$|\.class$|^target$|^Cargo\.lock$)"

# core.run_pdb_test("/home/inaimathi/projects/pycronado/")
# res = core.force(_)


def _pr(text):
    print(text, end="", flush=True)


def find_tool(tools, name):
    try:
        return [t for t in tools if getattr(t, "__name__") == name][0]
    except IndexError:
        return None


def tool_call(tools, call):
    tool = find_tool(tools, call["tool"])
    return tool(**call["args"])


def run_pdb_test(path: str):
    agent_name = "pdb_agent_001"
    qwq = Ollama("qwq:latest", "http://localhost:11435/")
    tools = [util.slurp, util.spit]
    tool_names = {t.__name__ for t in tools}
    files = list(util.deep_ls(path, ignore=_DEFAULT_IGNORE))
    src_files = [f for f in files if f.endswith(".py")]

    system = (
        "You are an autonomous agent that follows instructions carefully and"
        " respects file write permissions enforced by the host."
        " You MUST NOT write outside the directories explicitly "
        "granted write access, except for the agent's own scratchpad."
        f" Your scratchpad file is ./{agent_name}.md; write to it freely using the `spit` tool."
        " Your current task is to look over a git repository and find potential bugs."
        " The way you should go about doing this is by using the Hypothesis testing module "
        "to write properties in new test files and then run them. You should propose tests "
        "that point to real, exploitable bugs, rather than tests that might be a result of "
        "style issues. Use any tools you've been handed, and feel free to write to disk where "
        "you've been given permissions. Your target is a git repository, and you don't have "
        "general shell or commit permissions so at worst any mistakes can be easily rolled "
        "back once they're reviewed."
    )

    def _check_resp(resp: str) -> dict[str, Any]:
        parsed = util.loadch(resp)
        if parsed.get("type") not in {"summary", "tool-call"}:
            raise util.TransformError("invalid-object-structure")
        if parsed["type"] == "tool-call":
            if parsed["tool"] not in tool_names:
                raise util.TransformError("no-such-tool")
        return parsed

    def _per_file_stream(f: str):
        prompt = (
            f"Find bugs in the repository {path}. The relevant file tree in the repo is {files}. "
            f"You are currently working on file {f}. Your response should be either\n"
            "1. a tool call (in which case the tool will be called, and once it completes, you "
            "   will be given its result to evaluate). In this case it is IMPORTANT that your "
            "   response be ONLY the tool call structure (a JSON object of type "
            "   {type: 'tool-call', tool: ToolName, args: {ParamName: ParamValue}}) and no other commentary.\n"
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
            parsed = final_ev["parsed"]

            if parsed.get("type") == "tool-call":
                res = tool_call(tools, parsed)
                yield (
                    {"type": "trivialai.agent.log", "message": "Running a tool call..."}
                )
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

            if parsed.get("type") == "summary":
                yield {"type": "summary", "summary": parsed["summary"]}

        return sequentially(base, [_proceed])

    for f in src_files[0:1]:
        yield from _per_file_stream(f)


def force(event_generator):
    """
    Process a generator of streaming events and print them with aggregation.

    Args:
        event_generator: A generator yielding events with structure:
            - {'type': 'start', ...}
            - {'type': 'delta', 'text': str, 'scratchpad': str}
            - {'type': 'end', ...}
            - Other event types

    Behavior:
        - Contiguous scratchpad content is aggregated into "Thinking: " blocks
        - Contiguous text content is aggregated into "Saying: " blocks
        - Other events are printed as standalone dictionaries
        - Prints to stdout as events stream in

    Returns:
        list: Array of any 'end' events encountered
    """
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
        name: Optional[str] = None,
        root: Optional[Union[str, Path]] = None,
    ):
        self.llm = llm
        self.name = name or "agent-task"

        root_path = Path(root or ".").expanduser().resolve()
        internal = root_path / "."
        scratch = root_path / ".agent-scratchpad"
        scratch.mkdir(parents=True, exist_ok=True)

        self.log_path: Path = root_path / f"{self.name}.agent"

    @classmethod
    def from_logs(cls, log_path: Path) -> Agent:
        return TODO

    def consider(self, item: Union[str, Path]) -> Agent:
        return TODO

    def consider_with_write_access(self, path: Union[str, Path]) -> Agent:
        return TODO

    def remember_text(self, label: str, text: str) -> Agent:
        return TODO

    def equip(self, tools: Any) -> Agent:
        return TODO

    def plan(self, description: str, *, kind: str = "do") -> Agent:
        return TODO

    def do(self, description: str) -> Agent:
        return TODO

    # --------- Execution API ---------

    def run(self, max_steps: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Execute all planned steps from the current cursor onward,
        optionally limited by `max_steps`.

        Yields a stream of events:
          - agent meta events (type starts with "agent-...")
          - LLM stream events (start/delta/end/error) with extra 'agent_step' key.

        This is intentionally a generator so it composes well with itertools.chain.
        """
        # Figure out which step indices to run
        start = self._next_step_index
        end = len(self.steps)
        if max_steps is not None:
            end = min(end, start + max_steps)

        if start >= end:
            return iter(())  # nothing to do

        # Append to the task log; one line per event
        with self.log_path.open("a", encoding="utf-8") as logf:
            for step_index in range(start, end):
                step = self.steps[step_index]

                ev_start = {
                    "type": "agent-step-start",
                    "agent_step": step_index,
                    "kind": step.kind,
                    "description": step.description,
                }
                yield ev_start
                self._record_event(ev_start, logf)

                if step.kind == "do":
                    for ev in self._run_do_step(step_index, step):
                        yield ev
                        self._record_event(ev, logf)
                elif step.kind == "loop":
                    for ev in self._run_loop_step(step_index, step):
                        yield ev
                        self._record_event(ev, logf)
                else:
                    # Unknown kind; just mark as skipped
                    ev_unknown = {
                        "type": "agent-step-unknown-kind",
                        "agent_step": step_index,
                        "kind": step.kind,
                    }
                    yield ev_unknown
                    self._record_event(ev_unknown, logf)

                ev_end = {
                    "type": "agent-step-end",
                    "agent_step": step_index,
                    "kind": step.kind,
                }
                yield ev_end
                self._record_event(ev_end, logf)

                # advance cursor
                self._next_step_index = step_index + 1

    # --------- Internal helpers ---------

    def _record_event(self, ev: Dict[str, Any], logf) -> None:
        """
        Append event to in-memory history and to the NDJSON log.
        """
        self.history_events.append(ev)
        print(_json_line(ev), file=logf)

    # --- Context building (late-bound) ---

    def _build_system_prompt(self, step_index: int, step: AgentStep) -> str:
        """
        Build the system prompt for a given step from:
          - a generic agent persona,
          - current memory pointers,
          - recent history of events.
        """
        lines: List[str] = []

        lines.append(
            "You are an autonomous agent that follows "
            "instructions carefully and respects file write permissions enforced "
            "by the host. You MUST NOT write outside the directories explicitly "
            "granted write access, except for the agent's own scratchpad."
        )

        # Memory summary
        if self.memory_items:
            lines.append("")
            lines.append("MEMORY POINTERS (read-only unless marked [rw]):")
            for idx, item in enumerate(self.memory_items, start=1):
                if isinstance(item, MemoryFile):
                    mode = "[rw]" if item.writable else "[ro]"
                    lines.append(f"- [{idx}] FILE {mode}: {item.path}")
                elif isinstance(item, MemoryURL):
                    lines.append(f"- [{idx}] URL: {item.url}")
                elif isinstance(item, MemoryText):
                    preview = item.text.replace("\n", " ")
                    if len(preview) > 80:
                        preview = preview[:77] + "..."
                    lines.append(f"- [{idx}] TEXT '{item.label}': {preview}")
        else:
            lines.append("")
            lines.append("MEMORY POINTERS: (none)")

        # Recent history (very compressed)
        if self.history_events:
            lines.append("")
            lines.append("RECENT HISTORY (most recent last):")
            for ev in self.history_events[-8:]:
                t = ev.get("type")
                if t == "delta":
                    txt = str(ev.get("text", "")).replace("\n", " ")
                    if len(txt) > 60:
                        txt = txt[:57] + "..."
                    lines.append(f"- LLM delta: {txt}")
                elif t == "agent-step-start":
                    lines.append(
                        f"- START step {ev.get('agent_step')}: {ev.get('description')}"
                    )
                elif t == "agent-step-end":
                    lines.append(
                        f"- END step {ev.get('agent_step')}: kind={ev.get('kind')}"
                    )
                elif t and t.startswith("agent-"):
                    lines.append(f"- {t}: {ev}")

        lines.append("")
        lines.append(
            "You will now perform the NEXT STEP in the plan based on the user "
            "instruction. If you need to refer to prior actions, rely on the "
            "RECENT HISTORY and MEMORY POINTERS summaries."
        )

        return "\n".join(lines)

    # --- Step executors (v0: plain LLM streaming) ---

    def _run_do_step(
        self,
        step_index: int,
        step: AgentStep,
    ) -> Iterator[Dict[str, Any]]:
        """
        v0: simple 'do' step.
        - Builds a system prompt from memory + history.
        - Streams a single LLM call with step.description as the user prompt.
        - Yields the raw LLM events, tagged with agent_step.
        """
        system = self._build_system_prompt(step_index, step)
        prompt = step.description

        for ev in self.llm.stream(system, prompt):
            ev2 = dict(ev)
            ev2["agent_step"] = step_index
            yield ev2

    def _run_loop_step(
        self,
        step_index: int,
        step: AgentStep,
    ) -> Iterator[Dict[str, Any]]:
        """
        v0 loop semantics:
          - currently behaves like a single 'do' step but is factored out
            so we can later:
              * orchestrate repeated LLM+tool cycles,
              * inspect test results via equipped tools,
              * stop when some condition is met.
        """
        # For now, this just forwards to _run_do_step once.
        # Future: repeated iterations with tool calls and condition checks.
        yield from self._run_do_step(step_index, step)
