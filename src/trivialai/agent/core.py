# trivialai/agent/core.py

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from ..bistream import BiStream
from ..tools import to_llm_snippet
from .prompting import DEFAULT_CONTEXT_SIZE_CHARS, build_prompt

Message = Dict[str, Any]

_EVENT_PREFIX = "trivialai.agent."


def _etype(local: str) -> str:
    """Prefix event type names for streamed events."""
    return _EVENT_PREFIX + local


@dataclass
class Task:
    id: str
    kind: str  # "do" | "loop"
    prompt: str
    until: Optional[str] = None
    max_iter: int = 1
    status: str = "pending"
    result: Optional[Any] = None


class Agent:
    def __init__(
        self,
        llm,
        name: Optional[str] = None,
        log_path: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
        *,
        base_system_prompt: str = "",
        context_size: int = DEFAULT_CONTEXT_SIZE_CHARS,
        memory: Optional[Any] = None,  # VectorStore | Collection | None (duck-typed)
        context_summary: Optional[str] = None,
    ):
        """
        Agent that uses an LLMMixin-compatible LLM with a streaming API.

        Config:
        - base_system_prompt: high-level behaviour / goals for build_prompt
        - context_size: max char length for system prompt built by build_prompt
        - memory: optional VectorStore/Collection for retrieved context
        - context_summary: optional short summary of what has happened so far
        """
        self.llm = llm
        self.name = name or f"agent-{uuid4().hex[:8]}"
        self.log_path = (
            Path(log_path) if log_path else Path(f"./agent_log_{self.name}.jsonl")
        )

        self.base_system_prompt = base_system_prompt
        self.context_size = context_size
        self.memory = memory
        self.context_summary = context_summary

        self.logger = logger or logging.getLogger(f"trivialai.agent.{self.name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self._logs: List[Dict[str, Any]] = []
        self._tasks: List[Task] = []

        # tools: name -> fn; snippets: name -> to_llm_snippet(fn, ...)
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._tool_snippets: Dict[str, Dict[str, Any]] = {}

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("a", encoding="utf-8")

        self.log("agent.created", {"name": self.name, "log_path": str(self.log_path)})

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "agent": self.name,
            "event_type": event_type,
            "payload": payload,
        }
        self._logs.append(event)
        self._log_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._log_file.flush()
        self.logger.info("%s: %s", event_type, payload)

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def from_llm(
        cls,
        llm,
        log_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> "Agent":
        """
        Convenience constructor; forwards extra kwargs to Agent.__init__.
        """
        return cls(llm=llm, name=None, log_path=log_path, **kwargs)

    @classmethod
    def from_logs(cls, log_path: Union[str, Path]) -> "Agent":
        """
        Rebuild basic Agent state from an NDJSON log file.
        You can't reconstruct function objects or memory; you'll need to
        re-equip tools and reattach memory after creating the Agent.
        """
        log_path = Path(log_path)
        events = [json.loads(line) for line in log_path.open("r", encoding="utf-8")]

        meta = next((e for e in events if e["event_type"] == "agent.created"), None)
        name = meta["payload"]["name"] if meta else None

        agent = cls(llm=None, name=name, log_path=log_path)
        agent._logs = events

        for ev in events:
            if ev["event_type"] == "task.enqueued":
                p = ev["payload"]
                agent._tasks.append(
                    Task(
                        id=p["id"],
                        kind=p["kind"],
                        prompt=p["prompt"],
                        until=p.get("until"),
                        max_iter=p.get("max_iter", 1),
                        status=p.get("status", "pending"),
                    )
                )

        return agent

    # -------------------------------------------------------------------------
    # Tools: functions-as-tools (for prompt description now, tool-calls later)
    # -------------------------------------------------------------------------

    def equip(
        self,
        fn: Callable[..., Any],
        *,
        name: Optional[str] = None,
        types: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        is_async: Optional[bool] = None,
    ) -> str:
        """
        Register a Python function as a tool.

        Uses type annotations and docstring by default (via to_llm_snippet),
        but allows overrides.
        """
        snippet = to_llm_snippet(
            fn,
            name=name,
            types=types,
            description=description,
            is_async=is_async,
        )
        tool_name = snippet["name"]

        self._tools[tool_name] = fn
        self._tool_snippets[tool_name] = snippet

        self.log(
            "tool.equipped",
            {
                "name": tool_name,
                "description": snippet["description"],
                "args": snippet["args"],
                "async": snippet["async"],
            },
        )
        return tool_name

    def unequip(self, tool: Union[str, Callable[..., Any]]) -> None:
        if callable(tool):
            to_remove = [name for name, fn in self._tools.items() if fn is tool]
        else:
            to_remove = [tool]

        for name in to_remove:
            if name in self._tools:
                self._tools.pop(name, None)
                self._tool_snippets.pop(name, None)
                self.log("tool.unequipped", {"name": name})

    async def _dispatch_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool function with the given args.

        Handles sync vs async, and logs results. This isn't wired into the
        streaming loop yet, but is ready for future auto tool-calls.
        """
        fn = self._tools[name]
        snip = self._tool_snippets[name]
        is_async = bool(snip.get("async"))

        self.log("tool.called", {"name": name, "args": args})

        try:
            if is_async:
                result = await fn(**args)  # type: ignore[misc]
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: fn(**args))
        except Exception as e:
            self.log("tool.error", {"name": name, "args": args, "error": repr(e)})
            raise

        self.log("tool.result", {"name": name, "args": args, "result": result})
        return result

    # -------------------------------------------------------------------------
    # Tasks
    # -------------------------------------------------------------------------

    def do(self, prompt: str) -> str:
        task = Task(id=uuid4().hex, kind="do", prompt=prompt)
        self._tasks.append(task)
        self.log(
            "task.enqueued",
            {"id": task.id, "kind": task.kind, "prompt": task.prompt, "max_iter": 1},
        )
        return task.id

    def loop(self, prompt: str, until: str, max_iter: int = 5) -> str:
        task = Task(
            id=uuid4().hex,
            kind="loop",
            prompt=prompt,
            until=until,
            max_iter=max_iter,
        )
        self._tasks.append(task)
        self.log(
            "task.enqueued",
            {
                "id": task.id,
                "kind": task.kind,
                "prompt": task.prompt,
                "until": until,
                "max_iter": max_iter,
            },
        )
        return task.id

    # -------------------------------------------------------------------------
    # Run: single BiStream interface (sync OR async)
    # -------------------------------------------------------------------------

    def run(self) -> BiStream[Dict[str, Any]]:
        """
        Run all planned tasks and return a BiStream of agent events.

        Usage:

            # sync
            for ev in agent.run():
                ...

            # async
            async for ev in agent.run():
                ...
        """

        async def _agen():
            for task in self._tasks:
                if task.status == "done":
                    continue

                task.status = "running"
                self.log("task.started", {"id": task.id, "kind": task.kind})
                yield {
                    "type": _etype("task.started"),
                    "task_id": task.id,
                    "kind": task.kind,
                }

                if task.kind == "do":
                    async for ev in self._run_single_stream(task):
                        yield ev
                elif task.kind == "loop":
                    async for ev in self._run_loop_stream(task):
                        yield ev
                else:
                    err = ValueError(f"Unknown task kind {task.kind}")
                    self.log("task.failed", {"id": task.id, "error": repr(err)})
                    yield {
                        "type": _etype("task.failed"),
                        "task_id": task.id,
                        "error": repr(err),
                    }
                    task.status = "failed"
                    continue

                task.status = "done"
                self.log(
                    "task.done",
                    {"id": task.id, "kind": task.kind, "result": task.result},
                )
                yield {
                    "type": _etype("task.done"),
                    "task_id": task.id,
                    "kind": task.kind,
                    "result": task.result,
                }

        return BiStream.ensure(_agen())

    # -------------------------------------------------------------------------
    # Internal: LLM streaming helpers
    # -------------------------------------------------------------------------

    async def _llm_stream_once(
        self,
        *,
        task_id: str,
        phase: str,
        user_prompt: str,
        iteration: Optional[int] = None,
    ):
        """
        Single LLM call (streaming) as an async generator of events.

        Yields:
          - {"type": "trivialai.agent.llm.start", ...}
          - {"type": "trivialai.agent.llm.delta", ...}
          - {"type": "trivialai.agent.llm.end", ...}
        """
        system = build_prompt(
            base_system_prompt=self.base_system_prompt,
            user_prompt=user_prompt,
            tools=list(self._tool_snippets.values()),
            context_size=self.context_size,
            memory=self.memory,
            context_summary=self.context_summary,
        )

        log_payload: Dict[str, Any] = {
            "task_id": task_id,
            "phase": phase,
            "system_len": len(system),
            "prompt": user_prompt,
        }
        if iteration is not None:
            log_payload["iteration"] = iteration

        self.log("llm.call", log_payload)

        stream = self.llm.stream(system, user_prompt, images=None)  # BiStream

        content_parts: List[str] = []
        scratch_parts: List[str] = []
        final_content: Optional[str] = None
        final_scratch: Optional[str] = None

        async for ev in stream:
            ev_type = ev.get("type")
            if ev_type == "start":
                out = {
                    "type": _etype("llm.start"),
                    "task_id": task_id,
                    "phase": phase,
                    "iteration": iteration,
                    "provider": ev.get("provider"),
                    "model": ev.get("model"),
                }
                self.log(
                    "llm.stream.start",
                    {
                        "task_id": task_id,
                        "phase": phase,
                        "iteration": iteration,
                        "provider": out["provider"],
                        "model": out["model"],
                    },
                )
                yield out

            elif ev_type == "delta":
                text = (ev.get("text") or "") if isinstance(ev.get("text"), str) else ""
                scratch = (
                    (ev.get("scratchpad") or "")
                    if isinstance(ev.get("scratchpad"), str)
                    else ""
                )
                if text:
                    content_parts.append(text)
                if scratch:
                    scratch_parts.append(scratch)

                yield {
                    "type": _etype("llm.delta"),
                    "task_id": task_id,
                    "phase": phase,
                    "iteration": iteration,
                    "text": text,
                    "scratchpad": scratch,
                }

            elif ev_type == "end":
                if "content" in ev and ev["content"] is not None:
                    final_content = ev["content"]
                if "scratchpad" in ev and ev["scratchpad"] is not None:
                    final_scratch = ev["scratchpad"]

        if final_content is not None:
            content = final_content
        else:
            content = "".join(content_parts)

        if final_scratch is not None:
            scratch = final_scratch
        else:
            scratch = "".join(scratch_parts) if scratch_parts else None

        self.log(
            "llm.final",
            {
                "task_id": task_id,
                "phase": phase,
                "iteration": iteration,
                "content": content,
                "scratchpad": scratch,
            },
        )
        yield {
            "type": _etype("llm.end"),
            "task_id": task_id,
            "phase": phase,
            "iteration": iteration,
            "content": content,
            "scratchpad": scratch,
        }

    # -------------------------------------------------------------------------
    # Agent loops: "do" and "loop" in streaming form
    # -------------------------------------------------------------------------

    async def _run_single_stream(self, task: Task):
        """
        Single-step task:
        - One streaming LLM call
        - Task.result set to final content
        """
        last_content: Optional[str] = None

        async for ev in self._llm_stream_once(
            task_id=task.id, phase="do", user_prompt=task.prompt
        ):
            if ev["type"] == _etype("llm.end"):
                last_content = ev.get("content") or ""
            yield ev

        task.result = last_content or ""
        yield {
            "type": _etype("task.result"),
            "task_id": task.id,
            "result": task.result,
        }

    async def _run_loop_stream(self, task: Task):
        """
        Iterative task:
        - For each iteration:
            * stream a "progress" LLM call
            * stream an "until-check" LLM call
        - At the end, stream a "final summary" call
        - Task.result is the final summary content
        """
        iteration_summaries: List[str] = []

        for iteration in range(1, task.max_iter + 1):
            # Progress phase
            self.log(
                "loop.iteration.start",
                {"task_id": task.id, "iteration": iteration},
            )
            yield {
                "type": _etype("loop.iteration.start"),
                "task_id": task.id,
                "iteration": iteration,
            }

            progress_prompt = (
                "You are working on the following task:\n\n"
                f"{task.prompt}\n\n"
                f"This is iteration {iteration} of at most {task.max_iter}. "
                "Describe the concrete actions you would take or have taken, "
                "and what you achieved in this iteration."
            )

            if iteration_summaries:
                prev = "\n\n".join(
                    f"Iteration {i+1}: {s}" for i, s in enumerate(iteration_summaries)
                )
                progress_prompt += (
                    "\n\nHere is a summary of previous iterations:\n\n" f"{prev}"
                )

            iter_content: Optional[str] = None
            async for ev in self._llm_stream_once(
                task_id=task.id,
                phase="loop-iteration",
                user_prompt=progress_prompt,
                iteration=iteration,
            ):
                if ev["type"] == _etype("llm.end"):
                    iter_content = ev.get("content") or ""
                yield ev

            iter_summary = iter_content or ""
            iteration_summaries.append(iter_summary)
            self.log(
                "loop.iteration.summary",
                {
                    "task_id": task.id,
                    "iteration": iteration,
                    "content": iter_summary,
                },
            )
            yield {
                "type": _etype("loop.iteration.summary"),
                "task_id": task.id,
                "iteration": iteration,
                "content": iter_summary,
            }

            # Until-check phase
            if task.until:
                summaries_text = "\n\n".join(
                    f"Iteration {i+1}: {s}" for i, s in enumerate(iteration_summaries)
                )
                check_prompt = (
                    "You are evaluating whether the task is complete.\n\n"
                    f"Task description:\n{task.prompt}\n\n"
                    f"Work done so far:\n{summaries_text}\n\n"
                    "Now, answer STRICTLY 'yes' or 'no':\n"
                    f"{task.until!r}"
                )

                answer_raw: Optional[str] = None
                async for ev in self._llm_stream_once(
                    task_id=task.id,
                    phase="loop-until-check",
                    user_prompt=check_prompt,
                    iteration=iteration,
                ):
                    if ev["type"] == _etype("llm.end"):
                        answer_raw = ev.get("content") or ""
                    yield ev

                answer_norm = (answer_raw or "").strip().lower()
                self.log(
                    "loop.until.check",
                    {"task_id": task.id, "iteration": iteration, "answer": answer_norm},
                )
                yield {
                    "type": _etype("loop.until.check"),
                    "task_id": task.id,
                    "iteration": iteration,
                    "answer": answer_norm,
                }

                if answer_norm.startswith("yes"):
                    self.log(
                        "loop.stopped",
                        {"task_id": task.id, "iteration": iteration},
                    )
                    yield {
                        "type": _etype("loop.stopped"),
                        "task_id": task.id,
                        "iteration": iteration,
                    }
                    break

        # Final summary phase
        summaries_text = "\n\n".join(
            f"Iteration {i+1}:\n{s}" for i, s in enumerate(iteration_summaries)
        )
        summary_prompt = (
            "Summarize the work done across all iterations into a concise report "
            "for the user, focusing on what was attempted, what worked, and what "
            "remains to be done (if anything).\n\n"
            f"{summaries_text}"
        )

        final_summary: Optional[str] = None
        async for ev in self._llm_stream_once(
            task_id=task.id,
            phase="loop-final-summary",
            user_prompt=summary_prompt,
            iteration=None,
        ):
            if ev["type"] == _etype("llm.end"):
                final_summary = ev.get("content") or ""
            yield ev

        task.result = final_summary or ""
        self.log(
            "loop.final.summary",
            {"task_id": task.id, "content": task.result},
        )
        yield {
            "type": _etype("loop.final.summary"),
            "task_id": task.id,
            "content": task.result,
        }
        yield {
            "type": _etype("task.result"),
            "task_id": task.id,
            "result": task.result,
        }
