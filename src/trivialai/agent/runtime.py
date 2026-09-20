from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import enum
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

from ..bistream import BiStream
from ..util import TransformError
from . import prompting
from .toolkit import ToolKit


def _stable_type_name(value: Any) -> str:
    cls = type(value)
    module = getattr(cls, "__module__", "")
    qualname = getattr(cls, "__qualname__", getattr(cls, "__name__", "object"))
    return f"{module}.{qualname}" if module and module != "builtins" else qualname


def json_safe(value: Any, _seen: Optional[set[int]] = None) -> Any:
    """Return a deterministic JSON-safe representation of a Python value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"type": "float", "value": str(value)}
    if isinstance(value, enum.Enum):
        return json_safe(value.value, _seen)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }

    if _seen is None:
        _seen = set()
    track = isinstance(value, (Mapping, Sequence, set, frozenset)) or dataclasses.is_dataclass(value)
    value_id = id(value)
    if track:
        if value_id in _seen:
            return {"type": "cycle", "class": _stable_type_name(value)}
        _seen.add(value_id)

    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: json_safe(getattr(value, field.name), _seen)
                for field in dataclasses.fields(value)
            }
        if isinstance(value, Mapping):
            rows = []
            for key, item in value.items():
                safe_key = _json_key(key)
                rows.append((safe_key, json_safe(item, _seen)))
            rows.sort(key=lambda row: row[0])
            return {key: item for key, item in rows}
        if isinstance(value, (list, tuple)):
            return [json_safe(item, _seen) for item in value]
        if isinstance(value, (set, frozenset)):
            items = [json_safe(item, _seen) for item in value]
            return sorted(items, key=_canonical_json)
        if hasattr(value, "_asdict") and callable(value._asdict):
            return json_safe(value._asdict(), _seen)
        return {"type": "python-object", "class": _stable_type_name(value)}
    finally:
        if track:
            _seen.discard(value_id)


def _json_key(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return _canonical_json(json_safe(value))
    if isinstance(value, enum.Enum):
        return _json_key(value.value)
    if isinstance(value, (dt.datetime, dt.date, dt.time, Path)):
        return str(value)
    return f"<{_stable_type_name(value)}>"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _error_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or type(exc).__name__


def _tool_result(
    *,
    tool_call_id: str,
    tool: Any,
    ok: bool,
    result: Any = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "type": "tool-result",
        "tool_call_id": tool_call_id,
        "tool": tool,
        "ok": ok,
    }
    if ok:
        event["result"] = json_safe(result)
    else:
        event["error"] = {"type": "tool-error", "message": error or "tool call failed"}
    return event


def _protocol_error(message: str, *, step: Optional[int] = None, code: str = "agent-protocol-error") -> Dict[str, Any]:
    event: Dict[str, Any] = {"type": "error", "message": message, "code": code}
    if step is not None:
        event["step"] = step
    return event


def _parse_decision(content: Any) -> Dict[str, Any]:
    if not isinstance(content, str):
        raise TransformError("agent-decision-not-text")
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise TransformError("invalid-agent-decision-json") from exc
    if not isinstance(parsed, dict):
        raise TransformError("invalid-agent-decision")

    decision_type = parsed.get("type")
    if decision_type == "tool-call":
        return parsed
    if decision_type == "final":
        if set(parsed) != {"type", "content"} or not isinstance(parsed.get("content"), str):
            raise TransformError("invalid-final-decision")
        return parsed
    raise TransformError("invalid-agent-decision-type")


def _normalized_model_event(event: Dict[str, Any], step: int) -> Dict[str, Any]:
    kind = event.get("type")
    mapped = {
        "start": "model-start",
        "delta": "delta",
        "end": "model-end",
    }.get(kind, kind)
    normalized = dict(event)
    normalized["type"] = mapped
    normalized["step"] = step
    return normalized


def _call_signature(decision: Dict[str, Any]) -> str:
    return _canonical_json(
        json_safe({"tool": decision.get("tool"), "args": decision.get("args")})
    )


def run_agent(
    *,
    llm: Any,
    base_system_prompt: str,
    tools: ToolKit,
    task: str,
    name: str = "agent-task",
    max_steps: int = 16,
    max_identical_tool_calls: Optional[int] = 3,
    context_size: Optional[int] = None,
    memory: Any = None,
    context_summary: Optional[str] = None,
    images: Optional[list] = None,
) -> BiStream[Dict[str, Any]]:
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero")
    if max_identical_tool_calls is not None and max_identical_tool_calls <= 0:
        raise ValueError("max_identical_tool_calls must be greater than zero or None")

    async def _run():
        history = []
        identical_counts: Dict[str, int] = {}
        call_number = 0

        yield {
            "type": "agent-start",
            "name": name,
            "task": task,
            "max_steps": max_steps,
        }

        for step in range(max_steps):
            system_prompt = prompting.build_agent_prompt(
                base_system_prompt,
                task,
                tools,
                history=history,
                context_size=context_size,
                memory=memory,
                context_summary=context_summary,
            )

            completed = None
            saw_error = False
            try:
                model_stream = llm.stream(system_prompt, task, images=images)
                async for raw_event in model_stream:
                    if not isinstance(raw_event, dict):
                        yield _protocol_error(
                            "model stream emitted a non-object event",
                            step=step,
                            code="invalid-model-event",
                        )
                        return
                    event = _normalized_model_event(raw_event, step)
                    yield event
                    kind = raw_event.get("type")
                    if kind == "end":
                        completed = raw_event.get("content")
                    elif kind == "error":
                        saw_error = True
                        return
            except Exception as exc:
                yield _protocol_error(
                    _error_message(exc),
                    step=step,
                    code="model-stream-error",
                )
                return

            if saw_error:
                return
            if completed is None:
                yield _protocol_error(
                    "model stream ended without an end event",
                    step=step,
                    code="model-stream-incomplete",
                )
                return

            try:
                decision = _parse_decision(completed)
            except TransformError as exc:
                yield _protocol_error(_error_message(exc), step=step)
                return

            if decision["type"] == "final":
                yield {
                    "type": "final",
                    "content": decision["content"],
                    "steps": step + 1,
                }
                return

            call_number += 1
            tool_call_id = f"call-{call_number}"
            tool_call = dict(decision)
            tool_call["type"] = "tool-call"
            tool_call["step"] = step
            tool_call["tool_call_id"] = tool_call_id
            yield tool_call

            signature = _call_signature(decision)
            identical_counts[signature] = identical_counts.get(signature, 0) + 1
            if (
                max_identical_tool_calls is not None
                and identical_counts[signature] > max_identical_tool_calls
            ):
                result_event = _tool_result(
                    tool_call_id=tool_call_id,
                    tool=decision.get("tool"),
                    ok=False,
                    error=(
                        "repeated identical tool call limit exceeded "
                        f"({max_identical_tool_calls})"
                    ),
                )
                result_event["step"] = step
                yield result_event
                yield _protocol_error(
                    result_event["error"]["message"],
                    step=step,
                    code="repeated-tool-call",
                )
                return

            try:
                result = await tools.acall_tool(decision)
                result_event = _tool_result(
                    tool_call_id=tool_call_id,
                    tool=decision.get("tool"),
                    ok=True,
                    result=result,
                )
            except Exception as exc:
                result_event = _tool_result(
                    tool_call_id=tool_call_id,
                    tool=decision.get("tool"),
                    ok=False,
                    error=_error_message(exc),
                )

            result_event["step"] = step
            yield result_event

            history.append(
                (
                    {k: v for k, v in tool_call.items() if k != "step"},
                    {k: v for k, v in result_event.items() if k != "step"},
                )
            )

        yield _protocol_error(
            f"agent exceeded max_steps ({max_steps}) without a final decision",
            code="max-steps-exceeded",
        )

    return BiStream(_run())
