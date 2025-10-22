# tools.py
from __future__ import annotations

import asyncio
import inspect
from threading import Thread
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    get_args, get_origin)

from . import util
from .util import TransformError


def _run_coro_sync(coro):
    """
    Run a coroutine from synchronous code.
    - If no loop is running: use asyncio.run(coro).
    - If a loop IS running in this thread: spin up a worker thread with its own loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # no loop: safe to run directly
        return asyncio.run(coro)

    # We are inside a running loop -> run in a separate thread
    result_box: Dict[str, Any] = {}
    error_box: Dict[str, BaseException] = {}

    def _runner():
        try:
            result_box["value"] = asyncio.run(coro)
        except BaseException as e:  # propagate any exception to caller
            error_box["error"] = e

    t = Thread(target=_runner, daemon=True)
    t.start()
    t.join()

    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")


class Tools:
    def __init__(self, extras: Optional[Dict[str, Any]] = None):
        assert (extras is None) or (type(extras) is dict)
        self.extras: Optional[Dict[str, Any]] = extras
        self._env: Dict[str, Dict[str, Any]] = {}

    # ---------- Internal helpers ----------

    def _intern(
        self,
        name: str,
        schema: Dict[str, Any],
        description: str,
        fn: Callable[..., Any],
    ) -> bool:
        if name in self._env:
            return False
        self._env[name] = {
            "name": name,
            "type": schema,  # raw annotations (original behavior preserved)
            "description": description,
            "function": fn,
            "is_async": inspect.iscoroutinefunction(fn),
        }
        return True

    def _define_function(
        self,
        fn: Callable[..., Any],
        name: Optional[str] = None,
        type: Optional[Dict[str, Any]] = None,  # noqa: A002
        description: Optional[str] = None,
    ) -> bool:
        assert (
            fn.__annotations__ or type
        ), "either annotate the function or pass in a type dictionary for its inputs"
        assert (
            fn.__doc__ or description
        ), "either document the function or pass in a description"
        schema: Dict[str, Any] = type or {
            k: v for k, v in fn.__annotations__.items() if k != "return"
        }
        desc = description or (fn.__doc__ or "")
        return self._intern(name or fn.__name__, schema, desc, fn)

    # ---------- Public API ----------

    def define(
        self,
        fn: Optional[Callable[..., Any]] = None,
        *,
        name: Optional[str] = None,
        type: Optional[Dict[str, Any]] = None,  # noqa: A002
        description: Optional[str] = None,
    ):
        """
        Register a function. Can be used as:
          - tools.define(func)
          - tools.define(func, name="...", description="...")
          - @tools.define(name="...", description="...")
        Works for BOTH sync and async functions.
        """
        if fn is None:

            def decorator(f: Callable[..., Any]):
                self._define_function(f, name, type, description)
                return f

            return decorator
        return self._define_function(fn, name, type, description)

    def list(self) -> List[Dict[str, Any]]:
        """
        List registered tools in an LLM-friendly shape.

        Returns entries like:
        {
          "name": "...",
          "description": "...",
          "type": {...original python annotations...},   # for back-compat
          "args": { "param": {"type":"string"|...,"items":...,"nullable":...}, ... },
          "async": true|false
        }
        """
        out: List[Dict[str, Any]] = []
        for k, v in self._env.items():
            raw_schema: Dict[str, Any] = v["type"]
            norm_schema: Dict[str, Any] = {
                arg: _to_schema(t) for arg, t in raw_schema.items()
            }
            out.append(
                {
                    "name": k,
                    "type": raw_schema,
                    "args": norm_schema,
                    "description": v["description"],
                    "async": bool(v.get("is_async", False)),
                }
            )
        return out

    def validate(self, tool_call: Dict[str, Any]) -> bool:
        """
        Validation rules:
          - function exists
          - all required parameters present
          - no unknown parameters
        Optional/defaulted params are optional.
        """
        if not (
            isinstance(tool_call, dict)
            and "functionName" in tool_call
            and "args" in tool_call
        ):
            return False
        func_name = tool_call["functionName"]
        if func_name not in self._env:
            return False

        fn = self._env[func_name]["function"]
        required, optional = _param_specs(fn)
        args = tool_call["args"] if isinstance(tool_call["args"], dict) else {}

        if not set(required).issubset(args.keys()):
            return False
        if not set(args.keys()).issubset(set(required) | set(optional)):
            return False
        return True

    def transform(self, resp: Any) -> Dict[str, Any]:
        parsed = util.loadch(resp)
        if self.validate(parsed):
            return parsed
        raise util.TransformError("invalid-tool-call", raw=resp)

    def transform_multi(self, resp: Any) -> List[Dict[str, Any]]:
        parsed = util.loadch(resp)
        if type(parsed) is not list:
            raise util.TransformError("result-not-list", raw=parsed)
        for call in parsed:
            if not self.validate(call):
                raise util.TransformError("invalid-tool-subcall", raw=call)
        return parsed

    def lookup(self, tool_call: Dict[str, Any]) -> Callable[..., Any]:
        name = tool_call.get("functionName")
        if name not in self._env:
            raise TransformError("tool-not-found", raw=tool_call)
        return self._env[name]["function"]

    # ---------- Sync execution (now supports async tools safely) ----------

    def raw_call(self, tool_call: Dict[str, Any]) -> Any:
        """
        Execute without merging extras. Supports both sync and async tools.
        """
        if not self.validate(tool_call):
            raise TransformError("invalid-tool-call", raw=tool_call)
        fn = self.lookup(tool_call)
        res = fn(**tool_call["args"])
        if inspect.isawaitable(res):
            return _run_coro_sync(res)
        return res

    def call_with_extras(
        self,
        extras: Dict[str, Any],
        tool_call: Dict[str, Any],
        *,
        override: bool = True,
    ) -> Any:
        """
        Execute with extras merged into args.
        If override=True (default), extras override user args.
        If override=False, user args override extras.
        Supports both sync and async tools.
        """
        if not self.validate(tool_call):
            raise TransformError("invalid-tool-call", raw=tool_call)
        merged_args = (
            {**tool_call["args"], **extras}
            if override
            else {**extras, **tool_call["args"]}
        )
        fn = self.lookup(tool_call)
        res = fn(**merged_args)
        if inspect.isawaitable(res):
            return _run_coro_sync(res)
        return res

    def call(self, tool_call: Dict[str, Any]) -> Any:
        """
        Execute a tool call, merging self.extras (if provided).
        Supports both sync and async tools.
        """
        if not self.validate(tool_call):
            raise TransformError("invalid-tool-call", raw=tool_call)
        if self.extras is not None:
            return self.call_with_extras(self.extras, tool_call)
        return self.raw_call(tool_call)

    # ---------- Async execution (native await) ----------

    async def araw_call(self, tool_call: Dict[str, Any]) -> Any:
        """
        Async variant of raw_call(). Awaits coroutine tools, returns sync results directly.
        """
        if not self.validate(tool_call):
            raise TransformError("invalid-tool-call", raw=tool_call)
        fn = self.lookup(tool_call)
        res = fn(**tool_call["args"])
        if inspect.isawaitable(res):
            return await res
        return res

    async def acall_with_extras(
        self,
        extras: Dict[str, Any],
        tool_call: Dict[str, Any],
        *,
        override: bool = True,
    ) -> Any:
        """
        Async variant of call_with_extras(). Awaits coroutine tools, returns sync results directly.
        """
        if not self.validate(tool_call):
            raise TransformError("invalid-tool-call", raw=tool_call)
        merged_args = (
            {**tool_call["args"], **extras}
            if override
            else {**extras, **tool_call["args"]}
        )
        fn = self.lookup(tool_call)
        res = fn(**merged_args)
        if inspect.isawaitable(res):
            return await res
        return res

    async def acall(self, tool_call: Dict[str, Any]) -> Any:
        """
        Async variant of call(). Merges self.extras (if provided).
        """
        if not self.validate(tool_call):
            raise TransformError("invalid-tool-call", raw=tool_call)
        if self.extras is not None:
            return await self.acall_with_extras(self.extras, tool_call)
        return await self.araw_call(tool_call)


# ---------- Schema/validation helpers (unchanged) ----------


def _param_specs(fn: Callable[..., Any]) -> Tuple[List[str], List[str]]:
    import inspect as _inspect

    sig = _inspect.signature(fn)
    required: List[str] = []
    optional: List[str] = []
    for name, p in sig.parameters.items():
        if p.kind in (p.POSITIONAL_ONLY, p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is _inspect._empty:
            required.append(name)
        else:
            optional.append(name)
    return required, optional


def _to_schema(t: Any) -> Dict[str, Any]:
    origin = get_origin(t)

    if origin is Union:
        args = list(get_args(t))
        nullable = any(a is type(None) for a in args)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            base = _to_schema(non_none[0])
            base["nullable"] = True if nullable else base.get("nullable", False)
            return base
        return {"anyOf": [_to_schema(a) for a in non_none], "nullable": nullable}

    if origin in (list, List):
        (item_t,) = get_args(t) or (str,)
        return {"type": "array", "items": _to_schema(item_t)}
    if origin in (dict, Dict):
        key_t, val_t = (get_args(t) or (str, Any))[:2]
        return {
            "type": "object",
            "additionalProperties": _to_schema(val_t),
            "keys": str(getattr(key_t, "__name__", key_t)),
        }

    if t in (str, int, float, bool):
        prim = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }[t]
        return {"type": prim}
    if t is Any:
        return {"type": "any"}

    name = getattr(t, "__name__", None) or str(t)
    return {"type": name}
