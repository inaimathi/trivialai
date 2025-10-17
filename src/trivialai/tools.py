from __future__ import annotations

import inspect
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    get_args, get_origin)

from . import util
from .util import TransformError


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
        }
        return True

    def _define_function(
        self,
        fn: Callable[..., Any],
        name: Optional[str] = None,
        type: Optional[
            Dict[str, Any]
        ] = None,  # noqa: A002  (keep param name for back-compat)
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
        Returns True/False when called directly with a function.
        Returns the function itself when used as a decorator.
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
          "args": { "param": {"type":"string"|...,"items":...,"nullable":...}, ... }
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
                    "type": raw_schema,  # original, not super LLM-friendly, kept for compatibility
                    "args": norm_schema,  # nicer normalized schema
                    "description": v["description"],
                }
            )
        return out

    def validate(self, tool_call: Dict[str, Any]) -> bool:
        """
        Validation rules:
          - function exists
          - all required parameters present
          - no unknown parameters
        Optional/defaulted params are, well, optional.
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

        # required present
        if not set(required).issubset(args.keys()):
            return False
        # no unknowns
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

    def raw_call(self, tool_call: Dict[str, Any]) -> Any:
        """
        Execute without merging extras. Raises TransformError on invalid calls.
        """
        if not self.validate(tool_call):
            raise TransformError("invalid-tool-call", raw=tool_call)
        fn = self.lookup(tool_call)
        return fn(**tool_call["args"])

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
        """
        if not self.validate(tool_call):
            raise TransformError("invalid-tool-call", raw=tool_call)
        merged_args = (
            {**tool_call["args"], **extras}
            if override
            else {**extras, **tool_call["args"]}
        )
        return self.raw_call(
            {"functionName": tool_call["functionName"], "args": merged_args}
        )

    def call(self, tool_call: Dict[str, Any]) -> Any:
        """
        Execute a tool call, merging self.extras (if provided).
        Raises TransformError on invalid calls.
        """
        if not self.validate(tool_call):
            raise TransformError("invalid-tool-call", raw=tool_call)
        if self.extras is not None:
            return self.call_with_extras(self.extras, tool_call)
        return self.raw_call(tool_call)


# ---------- Schema/validation helpers ----------


def _param_specs(fn: Callable[..., Any]) -> Tuple[List[str], List[str]]:
    """
    Return (required, optional) param names for a function.
    Required = no default and in standard keyword position.
    """
    sig = inspect.signature(fn)
    required: List[str] = []
    optional: List[str] = []
    for name, p in sig.parameters.items():
        if p.kind in (p.POSITIONAL_ONLY, p.VAR_POSITIONAL, p.VAR_KEYWORD):
            # tools are called via kwargs; ignore *args/**kwargs/pos-only
            continue
        if p.default is inspect._empty:
            required.append(name)
        else:
            optional.append(name)
    return required, optional


def _to_schema(t: Any) -> Dict[str, Any]:
    """
    Convert Python/typing type hints to a lightweight JSON-ish schema fragment.
    """
    origin = get_origin(t)

    # Optional[T] or Union[T, None]
    if origin is Union:
        args = list(get_args(t))
        nullable = any(a is type(None) for a in args)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            base = _to_schema(non_none[0])
            base["nullable"] = True if nullable else base.get("nullable", False)
            return base
        return {
            "anyOf": [_to_schema(a) for a in non_none],
            "nullable": nullable,
        }

    # Collections
    if origin in (list, List):
        (item_t,) = get_args(t) or (str,)
        return {"type": "array", "items": _to_schema(item_t)}
    if origin in (dict, Dict):
        args = get_args(t) or (str, Any)
        key_t, val_t = args[0], args[1]
        return {
            "type": "object",
            "additionalProperties": _to_schema(val_t),
            "keys": str(getattr(key_t, "__name__", key_t)),
        }

    # Primitives / simple classes
    if t in (str, int, float, bool):
        return {"type": t.__name__}
    if t is Any:
        return {"type": "any"}

    # typing.Any or unknown types fallback
    name = getattr(t, "__name__", None) or str(t)
    return {"type": name}
