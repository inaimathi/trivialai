import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from ..tools import to_llm_snippet  # your helper

Message = Dict[str, Any]


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
    ):
        self.llm = llm
        self.name = name or f"agent-{uuid4().hex[:8]}"
        self.log_path = (
            Path(log_path) if log_path else Path(f"./agent_log_{self.name}.jsonl")
        )

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

    # --- logging -------------------------------------------------------------

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

    # --- constructors --------------------------------------------------------

    @classmethod
    def from_llm(cls, llm, log_path: Optional[Union[str, Path]] = None) -> "Agent":
        return cls(llm=llm, name=None, log_path=log_path)

    @classmethod
    def from_logs(cls, log_path: Union[str, Path]) -> "Agent":
        """
        Rebuild basic Agent state from an NDJSON log file.
        You can't reconstruct function objects; you'll need to re-equip tools
        after creating the Agent.
        """
        log_path = Path(log_path)
        events = [json.loads(line) for line in log_path.open("r", encoding="utf-8")]

        meta = next((e for e in events if e["event_type"] == "agent.created"), None)
        name = meta["payload"]["name"] if meta else None

        agent = cls(llm=None, name=name, log_path=log_path)
        agent._logs = events

        # Tasks can be reconstructed; tools must be re-equipped by the caller.
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

    # --- tools: functions-as-tools ------------------------------------------

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
            # find by identity
            to_remove = [name for name, fn in self._tools.items() if fn is tool]
        else:
            to_remove = [tool]

        for name in to_remove:
            if name in self._tools:
                self._tools.pop(name, None)
                self._tool_snippets.pop(name, None)
                self.log("tool.unequipped", {"name": name})

    def _tool_specs_for_llm(self) -> List[Dict[str, Any]]:
        """
        Convert tool snippets into the tool schema expected by the LLM.

        Here I assume an OpenAI-like JSON schema. If your trivialai LLM
        wrapper expects a different shape, adapt this one function.
        """
        specs: List[Dict[str, Any]] = []
        for name, snip in self._tool_snippets.items():
            args_schema = snip["args"]  # already JSON-schema-like

            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": snip["description"],
                        "parameters": {
                            "type": "object",
                            "properties": args_schema,
                            # You can get fancier: required from annotations / defaults
                            "required": list(args_schema.keys()),
                        },
                    },
                }
            )
        return specs

    def _dispatch_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool function with the given args.

        Handles sync vs async, and logs results. Any path/permission checks
        should be enforced inside the tool function itself.
        """
        fn = self._tools[name]
        snip = self._tool_snippets[name]
        is_async = bool(snip.get("async"))

        self.log("tool.called", {"name": name, "args": args})

        try:
            if is_async:
                # naive: if there's already a running loop you'll want a helper
                result = asyncio.run(fn(**args))
            else:
                result = fn(**args)
        except Exception as e:
            self.log("tool.error", {"name": name, "args": args, "error": repr(e)})
            raise

        self.log("tool.result", {"name": name, "args": args, "result": result})
        return result

    # --- tasks ---------------------------------------------------------------

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

    def run(self) -> None:
        for task in self._tasks:
            if task.status == "done":
                continue
            task.status = "running"
            self.log("task.started", {"id": task.id, "kind": task.kind})
            try:
                if task.kind == "do":
                    result = self._run_single(task)
                elif task.kind == "loop":
                    result = self._run_loop(task)
                else:
                    raise ValueError(f"Unknown task kind {task.kind}")
                task.status = "done"
                task.result = result
                self.log("task.done", {"id": task.id, "result": result})
            except Exception as e:
                task.status = "failed"
                self.log("task.failed", {"id": task.id, "error": repr(e)})

    # --- agent loops ---------------------------------------------------------

    def _run_single(self, task: Task) -> Any:
        messages: List[Message] = [
            {
                "role": "system",
                "content": (
                    "You are a tool-using agent. Use tools when needed; "
                    "when you have enough information, answer the user."
                ),
            },
            {"role": "user", "content": task.prompt},
        ]

        tools_for_llm = self._tool_specs_for_llm()

        for step in range(1, 32):
            self.log(
                "llm.call",
                {
                    "task_id": task.id,
                    "step": step,
                    "messages_len": len(messages),
                    "tools": list(self._tools.keys()),
                },
            )
            result = self.llm.generate(messages=messages, tools=tools_for_llm)

            tool_calls = getattr(result, "tool_calls", None)
            if tool_calls:
                for call in tool_calls:
                    tool_name = call["name"]
                    args = call["arguments"]
                    tool_result = self._dispatch_tool(tool_name, args)
                    messages.append(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(tool_result),
                        }
                    )
                # loop; let LLM react to tool results
                continue
            else:
                content = result.content
                messages.append({"role": "assistant", "content": content})
                self.log("llm.final", {"task_id": task.id, "content": content})
                return content

        # exceeded steps
        return {"messages": messages, "truncated": True}

    def _run_loop(self, task: Task) -> Any:
        history: List[Message] = [
            {
                "role": "system",
                "content": (
                    "You are an iterative agent. At each iteration, use tools to "
                    "make progress on the task. Avoid repeating work."
                ),
            },
            {"role": "user", "content": task.prompt},
        ]
        tools_for_llm = self._tool_specs_for_llm()

        for iteration in range(1, task.max_iter + 1):
            self.log(
                "loop.iteration.start", {"task_id": task.id, "iteration": iteration}
            )
            messages = list(history)

            # inner tool-using loop
            for step in range(1, 32):
                self.log(
                    "llm.call",
                    {
                        "task_id": task.id,
                        "iteration": iteration,
                        "step": step,
                        "messages_len": len(messages),
                        "tools": list(self._tools.keys()),
                    },
                )
                result = self.llm.generate(messages=messages, tools=tools_for_llm)

                tool_calls = getattr(result, "tool_calls", None)
                if tool_calls:
                    for call in tool_calls:
                        tool_name = call["name"]
                        args = call["arguments"]
                        tool_result = self._dispatch_tool(tool_name, args)
                        messages.append(
                            {
                                "role": "tool",
                                "name": tool_name,
                                "content": json.dumps(tool_result),
                            }
                        )
                    continue
                else:
                    content = result.content
                    messages.append({"role": "assistant", "content": content})
                    history.extend(messages[len(history) :])
                    self.log(
                        "loop.iteration.summary",
                        {
                            "task_id": task.id,
                            "iteration": iteration,
                            "content": content,
                        },
                    )
                    break

            # stopping condition check
            stop_messages = history + [
                {
                    "role": "user",
                    "content": (
                        "Considering all the work so far, answer STRICTLY 'yes' or 'no':\n"
                        f"Is the following condition now satisfied?\n\n{task.until!r}"
                    ),
                }
            ]
            stop_result = self.llm.generate(messages=stop_messages, tools=None)
            stop_text = stop_result.content.strip().lower()
            self.log(
                "loop.until.check",
                {"task_id": task.id, "iteration": iteration, "answer": stop_text},
            )
            if stop_text.startswith("yes"):
                self.log(
                    "loop.stopped",
                    {"task_id": task.id, "iteration": iteration},
                )
                break

        # final summary
        final_prompt = (
            "Summarize the work done across all iterations into a concise report."
        )
        summary_messages = history + [{"role": "user", "content": final_prompt}]
        final_result = self.llm.generate(messages=summary_messages, tools=None)
        self.log(
            "loop.final.summary", {"task_id": task.id, "content": final_result.content}
        )
        return final_result.content
