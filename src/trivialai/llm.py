# llm.py
from __future__ import annotations

from asyncio import to_thread
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Optional

from .async_utils import aiter_to_iter
from .util import LLMResult, TransformError
from .util import astream_checked_retries as _astream_checked_retries
from .util import generate_checked, loadch
from .util import stream_checked as _stream_checked_util
from .util import stream_checked_retries as _stream_checked_retries


class LLMMixin:
    """
    Mixin surface for LLMs.
    Subclasses should override `generate(...)`. Optionally override `agenerate(...)`
    and/or `astream(...)` for true async / incremental streaming.
    """

    # ---- Synchronous APIs (existing) ----

    def generate_checked(
        self,
        transformFn: Callable[[str], Any],
        system: str,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> LLMResult:
        if images is not None:
            fn = lambda: self.generate(system, prompt, images=images)
        else:
            fn = lambda: self.generate(system, prompt)
        return generate_checked(fn, transformFn, retries=retries)

    def generate_json(
        self,
        system: str,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> LLMResult:
        return self.generate_checked(
            loadch, system, prompt, images=images, retries=retries
        )

    def generate_tool_call(
        self,
        tools: Any,
        system: str,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> LLMResult:
        sysprompt = (
            "You are a computer specialist. Your job is translating client requests "
            "into tool calls. Your client has sent a request to use a tool; return the "
            "function call corresponding to the request and no other commentary. "
            'Return a value of type `{"functionName" :: string, "args" :: {arg_name: arg value}}`. '
            f"You have access to the tools: {tools.list()}. {system}"
        )
        return self.generate_checked(
            tools.transform, sysprompt, prompt, images=images, retries=retries
        )

    def generate_many_tool_calls(
        self,
        tools: Any,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> LLMResult:
        sysprompt = (
            "You are a computer specialist. Your job is translating client requests into tool calls. "
            "Your client has sent a request to use some number of tools; return a list of function calls "
            "corresponding to the request and no other commentary. "
            'Return a value of type `[{"functionName" :: string, "args" :: {arg_name: arg value}}]`. '
            f"You have access to the tools: {tools.list()}."
        )
        return self.generate_checked(
            tools.transform_multi, sysprompt, prompt, images=images, retries=retries
        )

    # Subclasses must provide this. Keeps sync compatibility.
    def generate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        raise NotImplementedError

    # ---- Async & streaming APIs (existing) ----

    async def agenerate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        """
        Default async implementation: runs sync `generate` in a thread.
        Subclasses with native async clients should override.
        """
        return await to_thread(self.generate, system, prompt, images)

    async def astream(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Default async streaming: not truly incremental.
        Subclasses should override for token/fragment streaming.
        """
        res = await self.agenerate(system, prompt, images)
        content = res.content or ""
        yield {
            "type": "start",
            "provider": self.__class__.__name__.lower(),
            "model": getattr(self, "model", None),
        }
        if content:
            yield {"type": "delta", "text": content}
        yield {
            "type": "end",
            "content": content,
            "scratchpad": res.scratchpad,
            "tokens": len(content.split()) if isinstance(content, str) else None,
        }

    def stream(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Sync facade over `astream` so you can iterate in a plain REPL / sync server.
        """
        return aiter_to_iter(self.astream(system, prompt, images))

    def stream_checked(
        self,
        transformFn: Callable[[str], Any],
        system: str,
        prompt: str,
        images: Optional[list] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Pass-through stream events, then emit a final parsed event using transformFn(full_text).
        """
        return _stream_checked_util(self.stream(system, prompt, images), transformFn)

    def stream_json(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Streaming helper that parses the final accumulated text as JSON.
        """
        return self.stream_checked(loadch, system, prompt, images)

    # ---- NEW: streaming tool-calls (retries live in util) ----

    async def astream_tool_calls(
        self,
        tools: Any,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Async streaming wrapper that defers attempt/retry logic to util.astream_checked_retries.
        Emits passthrough events from each attempt; on parse failure emits {"type":"attempt-failed", ...};
        on success emits {"type":"final","ok":true,...}; finally emits {"type":"final","ok":false,...} if all fail.
        """
        sysprompt = (
            "You are a computer specialist. Your job is translating client requests into tool calls. "
            "Your client has sent a request to use some number of tools; return a list of function calls "
            "corresponding to the request and no other commentary. "
            'Return a value of type `[{"functionName" :: string, "args" :: {arg_name: arg value}}]`. '
            f"You have access to the tools: {tools.list()}."
        )

        # IMPORTANT: yield from the async generator, don't return it.
        agen = _astream_checked_retries(
            lambda: self.astream(sysprompt, prompt, images),
            tools.transform_multi,
            retries=retries,
        )
        async for ev in agen:
            yield ev

    def stream_tool_calls(
        self,
        tools: Any,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> Iterator[Dict[str, Any]]:
        """
        Sync wrapper around util.stream_checked_retries for tool-call lists.
        """
        sysprompt = (
            "You are a computer specialist. Your job is translating client requests into tool calls. "
            "Your client has sent a request to use some number of tools; return a list of function calls "
            "corresponding to the request and no other commentary. "
            'Return a value of type `[{"functionName" :: string, "args" :: {arg_name: arg value}}]`. '
            f"You have access to the tools: {tools.list()}."
        )
        return _stream_checked_retries(
            lambda: self.stream(sysprompt, prompt, images),
            tools.transform_multi,
            retries=retries,
        )
