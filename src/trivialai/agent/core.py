# src/trivialai/agent.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from ..llm import LLMMixin
from ..tools import Tools


class Agent:
    """
    A lightweight planning/execution wrapper around an LLMMixin instance.

    - Memory: pointers to files/dirs, URLs, and text snippets.
    - Context: built *per step* from memory + event history.
    - Plan: ordered list of AgentStep.
    - Execution:
        * .run()  -> generator of events for all remaining steps
        * .do(s) -> sugar: .plan(s).run(); returns self

    Events yielded by .run() include both:
      - agent meta-events: "agent-step-start", "agent-step-end", etc.
      - raw LLM stream events, augmented with `agent_step` index.
    """

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

    # --------- Construction / builder API ---------

    @classmethod
    def from_llm(
        cls,
        llm: LLMMixin,
        *,
        name: Optional[str] = None,
        root: Optional[Union[str, Path]] = None,
    ) -> Agent:
        return cls(llm, name=name, root=root)

    # --- Memory (pointers) ---

    def consider(self, item: Union[str, Path]) -> Agent:
        """
        Register a read-only memory pointer:
          - str starting with http(s) -> URL
          - other str / Path         -> file or directory path
        """
        if isinstance(item, Path):
            p = item.expanduser().resolve()
            self.memory_items.append(MemoryFile(path=p, writable=False))
        elif isinstance(item, str):
            if item.startswith("http://") or item.startswith("https://"):
                self.memory_items.append(MemoryURL(url=item))
            else:
                p = Path(item).expanduser().resolve()
                self.memory_items.append(MemoryFile(path=p, writable=False))
        else:
            raise TypeError(f"Unsupported memory item type: {type(item)}")
        return self

    def consider_with_write_access(self, path: Union[str, Path]) -> Agent:
        """
        Register a memory pointer and explicitly grant write access
        to that path (file or directory).
        """
        p = Path(path).expanduser().resolve()
        self.memory_items.append(MemoryFile(path=p, writable=True))
        self.scope.allowed_writes.append(p)
        return self

    def remember_text(self, label: str, text: str) -> Agent:
        """
        Register a raw text snippet as a memory item.
        """
        self.memory_items.append(MemoryText(label=label, text=text))
        return self

    # --- Tooling ("equip" instead of "use") ---

    def equip(self, tools: Any) -> Agent:
        """
        Attach tools to this agent.

        Accepts:
          - a Tools instance (merged into self.tools),
          - a single callable (registered via self.tools.define),
          - an iterable of the above.
        """
        # Tools instance
        if isinstance(tools, Tools):
            # Merge into self.tools, respecting existing entries.
            for name, spec in tools._env.items():  # type: ignore[attr-defined]
                if name not in self.tools._env:  # type: ignore[attr-defined]
                    self.tools._env[name] = spec  # type: ignore[attr-defined]
            return self

        # Iterable of things?
        if isinstance(tools, Iterable) and not isinstance(tools, (str, bytes)):
            for t in tools:
                self.equip(t)
            return self

        # Assume single callable
        if callable(tools):
            self.tools.define(tools)
            return self

        raise TypeError(f"Unsupported tools/equipment type: {type(tools)}")

    # --- Planning API ---

    def plan(self, description: str, *, kind: str = "do") -> Agent:
        """
        Add a step to the internal plan. Does not execute it.
        """
        self.steps.append(AgentStep(kind=kind, description=description))
        return self

    def loop(self, description: str) -> Agent:
        """
        Convenience for planning a 'loop' step.
        Semantics are up to _run_loop_step; currently v0 is simple.
        """
        return self.plan(description, kind="loop")

    def do(self, description: str) -> Agent:
        """
        Sugar: plan a 'do' step and immediately execute all pending steps.

        Equivalent to:
          agent.plan(description)
          for _ in agent.run():
              pass
          return agent
        """
        self.plan(description, kind="do")
        for _ in self.run():
            # side-effects: history & log update
            pass
        return self

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
            "You are an autonomous coding and text-editing agent that follows "
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
