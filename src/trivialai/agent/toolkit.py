# toolkit.py
from __future__ import annotations

import inspect
import json
from collections.abc import Mapping as ABCMapping
from collections.abc import Sequence as ABCSequence
from typing import (Any, Callable, Dict, Literal, Optional, Tuple, Union,
                    get_args, get_origin)

from .. import util


class ToolKit:
    """
    Small wrapper around a set of callables that:

    - describes them to an LLM (to_tool_prompt)
    - describes the JSON shape of a tool call (to_prompt_type)
    - validates tool calls (check_tool)
    - executes tool calls (call_tool)
    """

    TOOL_CALL_TYPE = "tool-call"  # must match what you tell the model to emit

    def __init__(self, *tools: Callable[..., Any]) -> None:
        """
        Initialize with a list of tool functions, e.g.:

            tk = ToolKit(util.slurp, util.spit, util.tree, os.listdir)
        """
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
        """
        Truthy if this toolkit has any tools, falsy if it is empty.
        """
        return bool(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    # ---------- Public API ----------
    def add_tool(self, fn: Callable[..., Any], *, name: Optional[str] = None) -> None:
        """
        Register a new tool function after initialization.

        - If `name` is provided, use that as the tool name.
        - Otherwise, use `fn.__name__`.

        Raises:
            ValueError if:
              * the function has no usable name AND no explicit `name` is given
              * a tool with the same name already exists
        """
        tool_name = name or getattr(fn, "__name__", None)
        if not tool_name:
            # This is the "anonymous function with no explicit name" case
            raise ValueError(f"Tool {fn!r} has no __name__ and no explicit name")

        if tool_name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool_name!r}")

        self._tools[tool_name] = fn
        self._tool_summaries[tool_name] = to_summary(fn, name=tool_name)

    def ensure_tool(
        self, fn: Callable[..., Any], *, name: Optional[str] = None
    ) -> None:
        """
        Upsert a tool function.

        - If a tool with this name already exists, overwrite it.
        - If not, add it.

        Name resolution is the same as in `add_tool`:
        - If `name` is provided, use it.
        - Otherwise, use `fn.__name__`.

        Raises:
            ValueError if the function has no usable name and no explicit `name`.
        """
        tool_name = name or getattr(fn, "__name__", None)
        if not tool_name:
            raise ValueError(f"Tool {fn!r} has no __name__ and no explicit name")

        self._tools[tool_name] = fn
        self._tool_summaries[tool_name] = to_summary(fn, name=tool_name)

    def remove_tool(self, name: str) -> None:
        """
        Unregister an existing tool by name.

        Raises KeyError if the tool is not present.
        """
        try:
            del self._tools[name]
            del self._tool_summaries[name]
        except KeyError:
            raise KeyError(f"No such tool: {name!r}") from None

    def to_summary(self) -> Dict[str, Any]:
        """
        Return a machine-usable summary of tools:

            {"tools": [ {name, description, args, ...}, ... ]}
        """
        return {
            "tools": [self._tool_summaries[name] for name in sorted(self._tools.keys())]
        }

    def to_tool_shape(self) -> Dict:
        tool_names = ", ".join(sorted(self._tools.keys()))
        return {
            "type": self.TOOL_CALL_TYPE,
            "tool": f"<one of: {tool_names}>",
            "args": {"<param>": "<value>", "...": "..."},
        }

    def to_tool_prompt(self) -> str:
        """
        Return a complete text block you can stitch into a system prompt,
        using compact Python-ish signatures like:

            spit(file_path: str, content: str, mode: Optional[str] = None) -> None
        """
        if len(self._tools) == 0:
            return ""
        summary = self.to_summary()
        shape = self.to_tool_shape()
        shape_json = json.dumps(shape, indent=2)

        lines: list[str] = []

        lines.append("You have access to the following tools.")
        lines.append("")
        lines.append(
            "When you want to call a tool, respond with a single JSON object "
            "of the following form, and NOTHING else:"
        )
        lines.append("")
        lines.append(shape_json)
        lines.append("")
        lines.append("Available tools:")
        lines.append("")

        for t in summary["tools"]:
            sig = t.get("signature")
            if not sig:
                # Fallback if something omitted it
                fn = self._tools[t["name"]]
                sig = _format_signature(fn)

            desc = (t.get("description") or "").strip()

            lines.append(sig)
            if desc:
                # single-line, indented description
                first_line = desc.splitlines()[0]
                lines.append(f"  {first_line}")
            lines.append("")

        lines.append(
            "If you do not need to call a tool, respond normally instead of "
            "emitting a tool-call JSON object."
        )

        return "\n".join(lines)

    # ---------- Validation + execution ----------

    def check_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a tool-call dict:

        - structure matches the envelope from to_prompt_type()
        - tool exists
        - args is a dict
        - no unexpected args (unless tool has **kwargs)
        - all required args are present
        - best-effort type checks against annotations

        On failure: raise util.TransformError
        On success: return the (possibly normalized) tool_call dict.
        """
        if not isinstance(tool_call, dict):
            raise util.TransformError("invalid-object-structure")

        if tool_call.get("type") != self.TOOL_CALL_TYPE:
            raise util.TransformError("invalid-tool-call-type")

        tool_name = tool_call.get("tool")
        if not isinstance(tool_name, str):
            raise util.TransformError("invalid-tool-name")

        fn = self._tools.get(tool_name)
        if fn is None:
            raise util.TransformError("no-such-tool")

        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            raise util.TransformError("invalid-tool-args")

        sig = inspect.signature(fn)
        params = sig.parameters
        has_varkw = any(p.kind == p.VAR_KEYWORD for p in params.values())

        # 1) Unexpected args (unless function accepts **kwargs)
        allowed_kw = {
            name
            for name, p in params.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        unexpected = set(args) - allowed_kw
        if unexpected and not has_varkw:
            raise util.TransformError("unexpected-tool-arg")

        # 2) Missing required args
        missing = [
            name
            for name, p in params.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            and p.default is inspect._empty
            and name not in args
        ]
        if missing:
            raise util.TransformError("missing-tool-arg")

        # 3) Best-effort type checks based on annotations
        ann = getattr(fn, "__annotations__", {}) or {}
        for name, expected in ann.items():
            if name == "return":
                continue
            if name not in args:
                continue
            value = args[name]
            if not self._type_ok(value, expected):
                raise util.TransformError("invalid-tool-arg-type")

        return tool_call

    def call_tool(self, tool_call: Dict[str, Any]) -> Any:
        """
        Validate and then execute the tool call.

        - Runs check_tool() (may raise TransformError).
        - Calls the underlying function with **args.
        - Returns whatever the tool returns.
        """
        checked = self.check_tool(tool_call)
        tool_name = checked["tool"]
        args = checked.get("args", {})

        fn = self._tools[tool_name]

        # If you want async support, you can detect coroutine functions here
        # and run them with your _run_coro_sync helper.
        try:
            return fn(**args)
        except TypeError as e:
            # In case Python's own call-time checking finds something we missed
            raise util.TransformError("tool-call-failed") from e

    # ---------- Internal: best-effort type compatibility ----------

    @staticmethod
    def _type_ok(value: Any, annotation: Any) -> bool:
        """
        Best-effort runtime compatibility check:

        - Any / empty -> always OK
        - Union[...] / Optional[...] -> any branch OK
        - Literal[...] -> value must be one of the literals
        - Sequence / list[...] -> value must be a list
        - Mapping / dict[...] -> value must be a dict
        - Plain class -> isinstance(value, annotation)
        - Simple string annotations like "int", "str", "Optional[str]" handled specially
        - Everything else -> permissive True (don't over-reject)
        """
        if annotation is Any or annotation is inspect._empty:
            return True

        # Handle string annotations (from `from __future__ import annotations`)
        if isinstance(annotation, str):
            ann_str = annotation.strip()

            # Simple builtin type names
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

            # Optional[T] in string form
            if ann_str.startswith("Optional[") and ann_str.endswith("]"):
                inner_str = ann_str[len("Optional[") : -1].strip()
                if value is None:
                    return True
                return ToolKit._type_ok(value, inner_str)

            # If it's some complex string we don't recognize, be permissive
            return True

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is Union:
            return any(ToolKit._type_ok(value, a) for a in args)

        if origin is Literal:
            return value in args

        if origin in (list, ABCSequence):
            return isinstance(value, list)

        if origin in (dict, ABCMapping):
            return isinstance(value, dict)

        if isinstance(annotation, type):
            return isinstance(value, annotation)

        # Fallback – be permissive
        return True


def to_summary(
    fn: Callable[..., Any],
    *,
    name: Optional[str] = None,
    types: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    is_async: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Build the LLM-facing tool descriptor for a function.

    - name: override tool name (defaults to fn.__name__)
    - types: override parameter annotations dict (defaults to fn.__annotations__ without 'return')
    - description: override description (defaults to fn.__doc__ or "")
    - is_async: override async flag (defaults to inspect.iscoroutinefunction(fn))
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        sig = None

    ann = getattr(fn, "__annotations__", {}) or {}

    if types is not None:
        raw_schema: Dict[str, Any] = types
    else:
        raw_schema = {}

        if sig is not None:
            # Use the signature for parameter *names*,
            # but annotations (if present) for types.
            for pname, param in sig.parameters.items():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                raw_schema[pname] = ann.get(pname, Any)
        else:
            # Fallback: just use annotations (may miss some params).
            raw_schema = {k: v for k, v in ann.items() if k != "return"}

    norm_schema: Dict[str, Any] = {arg: _to_schema(t) for arg, t in raw_schema.items()}

    signature_str = _format_signature(fn, sig=sig, annotations=ann)

    return {
        "name": name or fn.__name__,
        "type": raw_schema,
        "args": norm_schema,
        "description": description or (fn.__doc__ or ""),
        "async": bool(
            is_async if is_async is not None else inspect.iscoroutinefunction(fn)
        ),
        "signature": signature_str,
    }


def _format_type(ann: Any) -> str:
    """
    Best-effort pretty-printer for type annotations.

    - Optional[str] instead of Union[str, NoneType]
    - Strip leading 'typing.'
    - Builtins by bare name
    - Fallback to str(...)
    """
    if ann is inspect._empty:
        return "Any"

    # Forward-ref / string annotation
    if isinstance(ann, str):
        return ann

    origin = get_origin(ann)
    args = get_args(ann)

    # Optional[T]
    if origin is Union and args:
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1 and len(args) == 2 and type(None) in args:  # noqa: E721
            return f"Optional[{_format_type(non_none[0])}]"
        # General Union
        return " | ".join(_format_type(a) for a in args)

    # Normal classes
    if isinstance(ann, type):
        if ann.__module__ == "builtins":
            return ann.__name__
        return f"{ann.__module__}.{ann.__qualname__}"

    # Fallback
    s = str(ann)
    if s.startswith("typing."):
        s = s[len("typing.") :]
    return s


def _format_signature(
    fn,
    *,
    sig: inspect.Signature | None = None,
    annotations: dict[str, Any] | None = None,
) -> str:
    """
    Produce a compact Python-ish signature string:

        spit(file_path: str, content: str, mode: Optional[str] = None) -> None
    """
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
            # for LLM tools, *args/**kwargs are almost never useful
            continue

        name = param.name
        ann = annotations.get(name, inspect._empty)
        type_str = _format_type(ann)

        if param.default is inspect._empty:
            parts.append(f"{name}: {type_str}")
        else:
            default_repr = repr(param.default)
            parts.append(f"{name}: {type_str} = {default_repr}")

    ret_ann = annotations.get("return", inspect._empty)
    ret_str = _format_type(ret_ann) if ret_ann is not inspect._empty else "Any"

    fn_name = getattr(fn, "__name__", "<fn>")
    return f"{fn_name}({', '.join(parts)}) -> {ret_str}"


def _to_schema(ann: Any) -> Dict[str, Any]:
    """
    Convert a Python type annotation into a lightweight, JSON-serializable
    schema dict.

    This is intentionally minimal and geared toward documentation / prompts:
    - Always includes a human-readable "type" string (via _format_type).
    - Adds a bit of structure for common containers and Literal.
    """
    # No annotation / fully dynamic
    if ann is inspect._empty or ann is Any:
        return {"type": "Any"}

    # Forward-ref / string annotation
    if isinstance(ann, str):
        return {"type": ann}

    origin = get_origin(ann)
    args = get_args(ann)

    # Optional / Union[...] (we just pretty-print it)
    if origin is Union and args:
        return {"type": _format_type(ann)}

    # Literal[...] -> record enum values
    if origin is Literal:
        return {
            "type": "Literal",
            "enum": list(args),
        }

    # List[T] / Sequence[T]
    if origin in (list, ABCSequence):
        item_ann = args[0] if args else Any
        return {
            "type": f"List[{_format_type(item_ann)}]",
            "items": _to_schema(item_ann),
        }

    # Dict[K, V] / Mapping[K, V]
    if origin in (dict, ABCMapping):
        key_ann = args[0] if len(args) > 0 else Any
        val_ann = args[1] if len(args) > 1 else Any
        return {
            "type": f"Dict[{_format_type(key_ann)}, {_format_type(val_ann)}]",
            "keys": _to_schema(key_ann),
            "values": _to_schema(val_ann),
        }

    # Tuple[...] (fixed-length or variadic)
    if origin is tuple or origin is Tuple:
        if not args:
            return {"type": "Tuple[Any, ...]"}
        if len(args) == 2 and args[1] is Ellipsis:
            # Tuple[T, ...]
            item_ann = args[0]
            return {
                "type": f"Tuple[{_format_type(item_ann)}, ...]",
                "items": _to_schema(item_ann),
            }
        # Tuple[T1, T2, ...]
        return {
            "type": f"Tuple[{', '.join(_format_type(a) for a in args)}]",
            "items": [_to_schema(a) for a in args],
        }

    # Plain class / builtin / custom type
    if isinstance(ann, type):
        return {"type": _format_type(ann)}

    # Fallback: just stringify via _format_type
    return {"type": _format_type(ann)}
