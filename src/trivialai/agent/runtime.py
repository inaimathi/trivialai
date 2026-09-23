# src/trivialai/agent/runtime.py
from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import enum
import inspect
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

from .. import util
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


def _tool_error_payload(error: Any) -> Dict[str, Any]:
    if isinstance(error, BaseException):
        public = getattr(error, "public_dict", None)
        if callable(public):
            try:
                payload = public()
            except Exception:
                payload = None
            if isinstance(payload, Mapping):
                out = dict(json_safe(payload))
                out.setdefault("type", "tool-error")
                out.setdefault("message", _error_message(error))
                return out

        return {
            "type": "tool-error",
            "message": _error_message(error),
        }

    if isinstance(error, Mapping):
        out = dict(json_safe(error))
        out.setdefault("type", "tool-error")
        out.setdefault("message", "tool call failed")
        return out

    return {
        "type": "tool-error",
        "message": str(error or "tool call failed"),
    }


def _tool_result(
    *,
    tool_call_id: str,
    tool: Any,
    ok: bool,
    result: Any = None,
    error: Any = None,
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
        event["error"] = _tool_error_payload(error)
    return event


def _clip_raw(value: Any, limit: int = 8000) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _protocol_error(
    message: str,
    *,
    step: Optional[int] = None,
    code: str = "agent-protocol-error",
    **details: Any,
) -> Dict[str, Any]:
    event: Dict[str, Any] = {"type": "error", "message": message, "code": code}
    if step is not None:
        event["step"] = step
    for key, value in details.items():
        if value is not None:
            event[key] = value
    return event


def _parse_decision(
    content: Any,
    *,
    tools: Optional[ToolKit] = None,
) -> Dict[str, Any]:
    """Parse and normalize one agent-loop decision.

    Canonical decisions remain ``tool-call`` and ``final``. For registered
    tools only, tolerate the two common collapsed envelopes produced by small
    models::

        {"type":"repo_edit","path":"x","old":"a","new":"b"}
        {"tool":"repo_edit","args":{"path":"x","old":"a","new":"b"}}

    Unknown type values remain protocol errors.
    """
    if not isinstance(content, str):
        raise TransformError("agent-decision-not-text", raw=content)
    try:
        parsed = util.loadch(content)
    except TransformError:
        raise
    except Exception as exc:
        raise TransformError("invalid-agent-decision-json", raw=content) from exc
    if not isinstance(parsed, dict):
        raise TransformError("invalid-agent-decision", raw=content)

    decision_type = parsed.get("type")
    if decision_type == "tool-call":
        if (
            not isinstance(parsed.get("tool"), str)
            or not parsed.get("tool")
            or not isinstance(parsed.get("args"), dict)
        ):
            raise TransformError("invalid-tool-call-decision", raw=content)
        return {"type":"tool-call", "tool":parsed["tool"], "args":parsed["args"]}

    if decision_type == "final":
        if not isinstance(parsed.get("content"), str):
            raise TransformError("invalid-final-decision", raw=content)
        return {"type":"final", "content":parsed["content"]}

    if tools is not None and tools.has_tool(decision_type):
        nested_args = parsed.get("args")
        if isinstance(nested_args, dict) and set(parsed).issubset(
            {"type", "args", "tool_call_id", "step", "attempt"}
        ):
            args = nested_args
        else:
            args = {
                key: value for key, value in parsed.items()
                if key not in {"type", "tool_call_id", "step", "attempt"}
            }
        return {"type":"tool-call", "tool":decision_type, "args":args}

    if (
        decision_type is None
        and tools is not None
        and tools.has_tool(parsed.get("tool"))
        and isinstance(parsed.get("args"), dict)
    ):
        return {"type":"tool-call", "tool":parsed["tool"], "args":parsed["args"]}

    raise TransformError("invalid-agent-decision-type", raw=content)


def _final_check_error_payload(value: Any) -> Optional[Dict[str, Any]]:
    if value is None or value is True:
        return None
    if value is False:
        return {"type":"final-check-error", "message":"The final decision is not yet allowed."}
    if isinstance(value, str):
        return {"type":"final-check-error", "message":value}
    if isinstance(value, Mapping):
        payload = dict(json_safe(value))
        if payload.get("ok") is True:
            return None
        payload.pop("ok", None)
        payload.setdefault("type", "final-check-error")
        payload.setdefault("message", "The final decision is not yet allowed.")
        return payload
    return {"type":"final-check-error", "message":str(value)}


async def _check_final_decision(
    final_check: Any,
    content: str,
    history: Sequence,
) -> Optional[Dict[str, Any]]:
    if final_check is None:
        return None
    result = final_check(content, list(history))
    if inspect.isawaitable(result):
        result = await result
    return _final_check_error_payload(result)


def _normalized_model_event(
    event: Dict[str, Any],
    step: int,
    *,
    attempt: Optional[int] = None,
) -> Dict[str, Any]:
    kind = event.get("type")
    mapped = {
        "start": "model-start",
        "delta": "delta",
        "end": "model-end",
    }.get(kind, kind)
    normalized = dict(event)
    normalized["type"] = mapped
    normalized["step"] = step
    if attempt is not None:
        normalized["attempt"] = attempt
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
    decision_retries: int = 3,
    max_identical_tool_calls: Optional[int] = 3,
    final_check: Any = None,
    context_size: Optional[int] = None,
    memory: Any = None,
    context_summary: Optional[str] = None,
    images: Optional[list] = None,
) -> BiStream[Dict[str, Any]]:
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero")
    if decision_retries <= 0:
        raise ValueError("decision_retries must be greater than zero")
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
            "decision_retries": decision_retries,
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
            checked_final = None
            last_failed_raw = None
            last_failed_error = None
            current_attempt = 1
            saw_error = False

            try:
                parse_decision = lambda content: _parse_decision(
                    content,
                    tools=tools,
                )
                model_stream = llm.stream_checked(
                    parse_decision,
                    system_prompt,
                    task,
                    images=images,
                    retries=decision_retries,
                )

                async for raw_event in model_stream:
                    if not isinstance(raw_event, dict):
                        yield _protocol_error(
                            "model stream emitted a non-object event",
                            step=step,
                            code="invalid-model-event",
                        )
                        return

                    kind = raw_event.get("type")

                    # LLMMixin.stream_checked() emits this between failed
                    # structured-output attempts. Keep it visible and bounded so
                    # callers can diagnose weak-model protocol failures.
                    if kind == "attempt-failed":
                        attempt = raw_event.get("attempt")
                        last_failed_raw = raw_event.get("raw")
                        last_failed_error = raw_event.get("error")
                        yield {
                            "type": "model-attempt-failed",
                            "step": step,
                            "attempt": attempt,
                            "error": last_failed_error,
                            "raw": _clip_raw(last_failed_raw),
                        }
                        if isinstance(attempt, int):
                            current_attempt = attempt + 1
                        else:
                            current_attempt += 1
                        continue

                    # This is stream_checked()'s transform result, not a provider
                    # model event. The provider's ordinary "end" event immediately
                    # before it contains the text we validated.
                    if kind == "final":
                        checked_final = raw_event
                        continue

                    event = _normalized_model_event(
                        raw_event,
                        step,
                        attempt=current_attempt,
                    )
                    yield event

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

            if checked_final is None:
                yield _protocol_error(
                    "checked model stream ended without a final validation event",
                    step=step,
                    code="model-stream-incomplete",
                    raw=_clip_raw(completed),
                )
                return

            if checked_final.get("ok") is not True:
                yield _protocol_error(
                    (
                        checked_final.get("last_error")
                        or last_failed_error
                        or checked_final.get("error")
                        or "invalid-agent-decision"
                    ),
                    step=step,
                    code="agent-protocol-error",
                    attempts=checked_final.get("attempts", decision_retries),
                    raw=_clip_raw(
                        last_failed_raw
                        if last_failed_raw is not None
                        else completed
                    ),
                )
                return

            if completed is None:
                yield _protocol_error(
                    "model stream ended without an end event",
                    step=step,
                    code="model-stream-incomplete",
                )
                return

            # stream_checked() has just validated this same content. Re-parsing
            # keeps the runtime independent of the checked stream's internal
            # result-key naming.
            try:
                decision = _parse_decision(
                    completed,
                    tools=tools,
                )
            except TransformError as exc:
                yield _protocol_error(
                    _error_message(exc),
                    step=step,
                    code="agent-protocol-error",
                    raw=_clip_raw(completed),
                )
                return

            if decision["type"] == "final":
                try:
                    rejection = await _check_final_decision(
                        final_check,
                        decision["content"],
                        history,
                    )
                except Exception as exc:
                    yield _protocol_error(
                        _error_message(exc),
                        step=step,
                        code="final-check-error",
                    )
                    return

                if rejection is not None:
                    yield {
                        "type": "final-rejected",
                        "content": decision["content"],
                        "error": rejection,
                        "step": step,
                    }
                    history.append(
                        (
                            {"type":"final", "content":decision["content"]},
                            {"type":"final-check", "ok":False, "error":rejection},
                        )
                    )
                    continue

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
                    error=exc,
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
