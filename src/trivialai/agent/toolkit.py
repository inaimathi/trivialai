# src/trivialai/agent/toolkit.py
from __future__ import annotations

import asyncio
import inspect
import json
import threading
from collections.abc import Mapping as ABCMapping
from collections.abc import Sequence as ABCSequence
from typing import (Any, Callable, Dict, Literal, Optional, Tuple, Union,
                    get_args, get_origin)

from .. import util
from ..util import TransformError


class ToolCallError(TransformError):
    """
    Structured tool-call validation error.

    `message` intentionally remains the historical short error code so callers
    that catch TransformError and inspect `.message` remain compatible.
    `public_dict()` carries the actionable model-facing diagnostics.
    """

    def __init__(
        self,
        code: str,
        *,
        detail: Optional[str] = None,
        tool: Optional[str] = None,
        expected: Optional[list[str]] = None,
        received: Optional[list[str]] = None,
        missing: Optional[list[str]] = None,
        unexpected: Optional[list[str]] = None,
        argument: Optional[str] = None,
        expected_type: Optional[str] = None,
        received_type: Optional[str] = None,
        available_tools: Optional[list[str]] = None,
        raw: Any = None,
    ) -> None:
        super().__init__(code, raw=raw)
        self.code = code
        self.detail = detail or code
        self.tool = tool
        self.expected = expected
        self.received = received
        self.missing = missing
        self.unexpected = unexpected
        self.argument = argument
        self.expected_type = expected_type
        self.received_type = received_type
        self.available_tools = available_tools

    def public_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": "tool-error",
            "code": self.code,
            "message": self.detail,
        }
        for key in (
            "tool",
            "expected",
            "received",
            "missing",
            "unexpected",
            "argument",
            "expected_type",
            "received_type",
            "available_tools",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


def _run_awaitable_sync(awaitable):
    """
    Resolve an awaitable for the synchronous `call_tool` API.

    If no event loop is running in this thread, use asyncio.run directly.
    If a loop is already running, drive the awaitable in a helper thread so
    sync callers still get a normal return value without trying to nest loops.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_value(awaitable))

    result = []
    error = []

    def runner():
        try:
            result.append(asyncio.run(_await_value(awaitable)))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if error:
        raise error[0]
    return result[0]


async def _await_value(awaitable):
    return await awaitable


class ToolKit:
    """
    Small wrapper around a set of callables that:
    - describes them to an LLM (to_tool_prompt)
    - describes the JSON shape of a tool call (to_tool_shape)
    - validates tool calls (check_tool)
    - executes sync or async tools (call_tool / acall_tool)
    """

    TOOL_CALL_TYPE = "tool-call"

    def __init__(self, *tools: Callable[..., Any]) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._tool_summaries: Dict[str, Dict[str, Any]] = {}
        for fn in tools:
            name = getattr(fn, "__name__", None)
            if not name:
                raise ValueError(f"Tool {fn!r} has no __name__")
            if name in self._tools:
                raise ValueError(f"Duplicate tool name: {name!r}")
            self._tools[name] = fn
            self._tool_summaries[name] = to_summary(fn)

    def __bool__(self) -> bool:
        return bool(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def has_tool(self, name: Any) -> bool:
        """Return whether `name` identifies a registered model-facing tool."""
        return isinstance(name, str) and name in self._tools

    def tool_names(self) -> list[str]:
        """Return registered tool names in deterministic order."""
        return sorted(self._tools)

    # ---------- Public API ----------

    def add_tool(self, fn: Callable[..., Any], *, name: Optional[str] = None) -> None:
        tool_name = name or getattr(fn, "__name__", None)
        if not tool_name:
            raise ValueError(f"Tool {fn!r} has no __name__ and no explicit name")
        if tool_name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool_name!r}")

        self._tools[tool_name] = fn
        self._tool_summaries[tool_name] = to_summary(fn, name=tool_name)

    def ensure_tool(
        self,
        fn: Callable[..., Any],
        *,
        name: Optional[str] = None,
    ) -> None:
        tool_name = name or getattr(fn, "__name__", None)
        if not tool_name:
            raise ValueError(f"Tool {fn!r} has no __name__ and no explicit name")
        self._tools[tool_name] = fn
        self._tool_summaries[tool_name] = to_summary(fn, name=tool_name)

    def remove_tool(self, name: str) -> None:
        try:
            del self._tools[name]
            del self._tool_summaries[name]
        except KeyError:
            raise KeyError(f"No such tool: {name!r}") from None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "tools": [self._tool_summaries[name] for name in sorted(self._tools.keys())]
        }

    def to_tool_shape(self) -> Dict[str, Any]:
        tool_names = ", ".join(sorted(self._tools.keys()))
        return {
            "type": self.TOOL_CALL_TYPE,
            "tool": f"<one of: {tool_names}>",
            "args": {"<param>": "<value>", "...": "..."},
        }

    def to_tool_prompt(self) -> str:
        if len(self._tools) == 0:
            return ""

        summary = self.to_summary()
        shape_json = json.dumps(self.to_tool_shape(), indent=2)

        lines: list[str] = [
            "You have access to the following tools.",
            "",
            (
                "When you want to call a tool, respond with a single JSON object "
                "of the following form, and NOTHING else:"
            ),
            "",
            shape_json,
            "",
            "Available tools:",
            "",
        ]

        for row in summary["tools"]:
            sig = row.get("signature")
            if not sig:
                sig = _format_signature(self._tools[row["name"]])

            desc = (row.get("description") or "").strip()
            lines.append(sig)
            if desc:
                lines.append(f"  {desc.splitlines()[0]}")
            lines.append("")

        lines.append(
            "Use the signature exactly. Required arguments must be present, and "
            "do not invent argument names that are not shown."
        )
        lines.append("")
        lines.append(
            "If you do not need to call a tool, respond normally instead of "
            "emitting a tool-call JSON object."
        )

        return "\n".join(lines)

    # ---------- Validation + execution ----------

    def _argument_shape(self, fn: Callable[..., Any]) -> Dict[str, Any]:
        sig = inspect.signature(fn)
        params = sig.parameters
        accepted = [
            name
            for name, param in params.items()
            if param.kind
            in (
                param.POSITIONAL_OR_KEYWORD,
                param.KEYWORD_ONLY,
            )
        ]
        required = [
            name
            for name, param in params.items()
            if param.kind
            in (
                param.POSITIONAL_OR_KEYWORD,
                param.KEYWORD_ONLY,
            )
            and param.default is inspect._empty
        ]
        return {
            "signature": sig,
            "params": params,
            "accepted": accepted,
            "required": required,
            "has_varkw": any(
                param.kind == param.VAR_KEYWORD for param in params.values()
            ),
        }

    def check_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a textual tool call.

        Validation remains strict, but failures are now structured and
        actionable: missing/unexpected argument names, expected arguments and
        received arguments are all exposed through ToolCallError.public_dict().
        """
        if not isinstance(tool_call, dict):
            raise ToolCallError(
                "invalid-object-structure",
                detail="Tool call must be a JSON object.",
                received_type=type(tool_call).__name__,
                raw=tool_call,
            )

        if tool_call.get("type") != self.TOOL_CALL_TYPE:
            raise ToolCallError(
                "invalid-tool-call-type",
                detail=f"Tool call type must be {self.TOOL_CALL_TYPE!r}.",
                raw=tool_call,
            )

        tool_name = tool_call.get("tool")
        if not isinstance(tool_name, str) or not tool_name:
            raise ToolCallError(
                "invalid-tool-name",
                detail="Tool call must include a non-empty string `tool` name.",
                available_tools=sorted(self._tools),
                raw=tool_call,
            )

        fn = self._tools.get(tool_name)
        if fn is None:
            raise ToolCallError(
                "no-such-tool",
                detail=f"Unknown tool {tool_name!r}.",
                tool=tool_name,
                available_tools=sorted(self._tools),
                raw=tool_call,
            )

        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            raise ToolCallError(
                "invalid-tool-args",
                detail=f"Arguments for {tool_name} must be a JSON object.",
                tool=tool_name,
                expected=sorted(self._argument_shape(fn)["accepted"]),
                received_type=type(args).__name__,
                raw=tool_call,
            )

        shape = self._argument_shape(fn)
        expected = sorted(shape["accepted"])
        received = sorted(args)

        unexpected = sorted(set(args) - set(shape["accepted"]))
        if unexpected and not shape["has_varkw"]:
            names = ", ".join(repr(name) for name in unexpected)
            raise ToolCallError(
                "unexpected-tool-arg",
                detail=(
                    f"Unexpected argument"
                    f"{'s' if len(unexpected) != 1 else ''} for {tool_name}: "
                    f"{names}. Expected arguments: {', '.join(expected) or '(none)'}."
                ),
                tool=tool_name,
                expected=expected,
                received=received,
                unexpected=unexpected,
                raw=tool_call,
            )

        missing = sorted(name for name in shape["required"] if name not in args)
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ToolCallError(
                "missing-tool-arg",
                detail=(
                    f"Missing required argument"
                    f"{'s' if len(missing) != 1 else ''} for {tool_name}: "
                    f"{names}. Expected arguments: {', '.join(expected) or '(none)'}."
                ),
                tool=tool_name,
                expected=expected,
                received=received,
                missing=missing,
                raw=tool_call,
            )

        annotations = getattr(fn, "__annotations__", {}) or {}
        for name, expected_type in annotations.items():
            if name == "return" or name not in args:
                continue
            value = args[name]
            if not self._type_ok(value, expected_type):
                formatted = _format_type(expected_type)
                raise ToolCallError(
                    "invalid-tool-arg-type",
                    detail=(
                        f"Argument {name!r} for {tool_name} must be {formatted}; "
                        f"received {type(value).__name__}."
                    ),
                    tool=tool_name,
                    expected=expected,
                    received=received,
                    argument=name,
                    expected_type=formatted,
                    received_type=type(value).__name__,
                    raw=tool_call,
                )

        return tool_call

    def call_tool(self, tool_call: Dict[str, Any]) -> Any:
        """
        Validate and execute a tool from synchronous code.

        Async tools are bridged to a result transparently.
        """
        checked = self.check_tool(tool_call)
        tool_name = checked["tool"]
        args = checked.get("args", {})
        fn = self._tools[tool_name]

        try:
            result = fn(**args)
        except TypeError as exc:
            raise ToolCallError(
                "tool-call-failed",
                detail=f"Tool {tool_name} rejected the supplied arguments: {exc}",
                tool=tool_name,
                expected=sorted(self._argument_shape(fn)["accepted"]),
                received=sorted(args),
                raw=tool_call,
            ) from exc

        if inspect.isawaitable(result):
            return _run_awaitable_sync(result)
        return result

    async def acall_tool(self, tool_call: Dict[str, Any]) -> Any:
        """
        Validate and execute a tool from asynchronous code.

        Sync tools return normally; awaitable results are awaited.
        """
        checked = self.check_tool(tool_call)
        tool_name = checked["tool"]
        args = checked.get("args", {})
        fn = self._tools[tool_name]

        try:
            result = fn(**args)
        except TypeError as exc:
            raise ToolCallError(
                "tool-call-failed",
                detail=f"Tool {tool_name} rejected the supplied arguments: {exc}",
                tool=tool_name,
                expected=sorted(self._argument_shape(fn)["accepted"]),
                received=sorted(args),
                raw=tool_call,
            ) from exc

        if inspect.isawaitable(result):
            return await result
        return result

    # ---------- Internal: best-effort type compatibility ----------

    @staticmethod
    def _type_ok(value: Any, annotation: Any) -> bool:
        if annotation is Any or annotation is inspect._empty:
            return True

        if isinstance(annotation, str):
            ann_str = annotation.strip()
            builtin_map = {
                "int": int,
                "str": str,
                "float": float,
                "bool": bool,
                "dict": dict,
                "list": list,
                "tuple": tuple,
                "set": set,
            }
            if ann_str in builtin_map:
                return isinstance(value, builtin_map[ann_str])

            if ann_str.startswith("Optional[") and ann_str.endswith("]"):
                inner = ann_str[len("Optional[") : -1].strip()
                if value is None:
                    return True
                return ToolKit._type_ok(value, inner)

            # `from __future__ import annotations` can leave richer types as
            # strings. Stay permissive rather than rejecting valid calls.
            return True

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is Union:
            return any(ToolKit._type_ok(value, branch) for branch in args)

        if origin is Literal:
            return value in args

        if origin in (list, ABCSequence):
            return isinstance(value, list)

        if origin in (dict, ABCMapping):
            return isinstance(value, dict)

        if isinstance(annotation, type):
            return isinstance(value, annotation)

        return True


def to_summary(
    fn: Callable[..., Any],
    *,
    name: Optional[str] = None,
    types: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    is_async: Optional[bool] = None,
) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        sig = None

    annotations = getattr(fn, "__annotations__", {}) or {}
    if types is not None:
        raw_schema: Dict[str, Any] = types
    else:
        raw_schema = {}
        if sig is not None:
            for pname, param in sig.parameters.items():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                raw_schema[pname] = annotations.get(pname, Any)
        else:
            raw_schema = {
                key: value for key, value in annotations.items() if key != "return"
            }

    norm_schema = {
        argument: _to_schema(annotation) for argument, annotation in raw_schema.items()
    }

    return {
        "name": name or fn.__name__,
        "type": raw_schema,
        "args": norm_schema,
        "description": description or (fn.__doc__ or ""),
        "async": bool(
            is_async if is_async is not None else inspect.iscoroutinefunction(fn)
        ),
        "signature": _format_signature(
            fn,
            sig=sig,
            annotations=annotations,
        ),
    }


def _format_type(annotation: Any) -> str:
    if annotation is inspect._empty:
        return "Any"

    if isinstance(annotation, str):
        return annotation

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union and args:
        non_none = [branch for branch in args if branch is not type(None)]
        if len(non_none) == 1 and len(args) == 2 and type(None) in args:
            return f"Optional[{_format_type(non_none[0])}]"
        return " | ".join(_format_type(branch) for branch in args)

    if isinstance(annotation, type):
        if annotation.__module__ == "builtins":
            return annotation.__name__
        return f"{annotation.__module__}.{annotation.__qualname__}"

    text = str(annotation)
    if text.startswith("typing."):
        text = text[len("typing.") :]
    return text


def _format_signature(
    fn,
    *,
    sig: inspect.Signature | None = None,
    annotations: dict[str, Any] | None = None,
) -> str:
    if sig is None:
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            return f"{getattr(fn, '__name__', '<fn>')}(...)"

    if annotations is None:
        annotations = getattr(fn, "__annotations__", {}) or {}

    parts: list[str] = []
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        type_str = _format_type(annotations.get(param.name, inspect._empty))
        if param.default is inspect._empty:
            parts.append(f"{param.name}: {type_str}")
        else:
            parts.append(f"{param.name}: {type_str} = {param.default!r}")

    ret = annotations.get("return", inspect._empty)
    ret_str = _format_type(ret) if ret is not inspect._empty else "Any"
    return f"{getattr(fn, '__name__', '<fn>')}" f"({', '.join(parts)}) -> {ret_str}"


def _to_schema(annotation: Any) -> Dict[str, Any]:
    if annotation is inspect._empty or annotation is Any:
        return {"type": "Any"}

    if isinstance(annotation, str):
        return {"type": annotation}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union and args:
        return {"type": _format_type(annotation)}

    if origin is Literal:
        return {
            "type": "Literal",
            "enum": list(args),
        }

    if origin in (list, ABCSequence):
        item = args[0] if args else Any
        return {
            "type": f"List[{_format_type(item)}]",
            "items": _to_schema(item),
        }

    if origin in (dict, ABCMapping):
        key = args[0] if len(args) > 0 else Any
        value = args[1] if len(args) > 1 else Any
        return {
            "type": f"Dict[{_format_type(key)}, {_format_type(value)}]",
            "keys": _to_schema(key),
            "values": _to_schema(value),
        }

    if origin is tuple or origin is Tuple:
        if not args:
            return {"type": "Tuple[Any, ...]"}
        if len(args) == 2 and args[1] is Ellipsis:
            item = args[0]
            return {
                "type": f"Tuple[{_format_type(item)}, ...]",
                "items": _to_schema(item),
            }
        return {
            "type": ("Tuple[" + ", ".join(_format_type(item) for item in args) + "]"),
            "items": [_to_schema(item) for item in args],
        }

    if isinstance(annotation, type):
        return {"type": _format_type(annotation)}

    return {"type": _format_type(annotation)}
