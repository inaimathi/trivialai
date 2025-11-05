# llm.py
from __future__ import annotations

from asyncio import to_thread
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Optional

from .async_utils import aiter_to_iter
from .util import GenerationError, LLMResult, TransformError
from .util import astream_checked as _astream_checked_once
from .util import generate_checked as _generate_checked_once
from .util import loadch
from .util import stream_checked as _stream_checked_once


def _stream_with_retries(
    stream_factory: Callable[[], Iterator[Dict[str, Any]]],
    transformFn: Callable[[str], Any],
    *,
    retries: int,
) -> Iterator[Dict[str, Any]]:
    """
    Run util.stream_checked once per attempt; preserve passthrough events;
    on failure, emit attempt-failed; on success, augment final with attempt.
    """
    last_error_msg = None
    for attempt in range(1, max(1, retries) + 1):
        final_ev: Optional[Dict[str, Any]] = None
        for ev in _stream_checked_once(stream_factory(), transformFn):
            if ev.get("type") == "final":
                final_ev = ev
            else:
                yield ev

        if final_ev and final_ev.get("ok"):
            out = dict(final_ev)
            out["attempt"] = attempt
            yield out
            return
        else:
            last_error_msg = (final_ev or {}).get("error")
            yield {
                "type": "attempt-failed",
                "attempt": attempt,
                "error": last_error_msg,
                "raw": (final_ev or {}).get("raw"),
            }

    yield {
        "type": "final",
        "ok": False,
        "error": f"failed-on-{retries}-retries",
        "attempts": retries,
        "last_error": last_error_msg,
    }


async def _astream_with_retries(
    stream_factory: Callable[[], AsyncIterator[Dict[str, Any]]],
    transformFn: Callable[[str], Any],
    *,
    retries: int,
) -> AsyncIterator[Dict[str, Any]]:
    last_error_msg = None
    for attempt in range(1, max(1, retries) + 1):
        final_ev: Optional[Dict[str, Any]] = None
        async for ev in _astream_checked_once(stream_factory(), transformFn):
            if ev.get("type") == "final":
                final_ev = ev
            else:
                yield ev

        if final_ev and final_ev.get("ok"):
            out = dict(final_ev)
            out["attempt"] = attempt
            yield out
            return
        else:
            last_error_msg = (final_ev or {}).get("error")
            yield {
                "type": "attempt-failed",
                "attempt": attempt,
                "error": last_error_msg,
                "raw": (final_ev or {}).get("raw"),
            }

    yield {
        "type": "final",
        "ok": False,
        "error": f"failed-on-{retries}-retries",
        "attempts": retries,
        "last_error": last_error_msg,
    }


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
        """
        Retry at the caller level. Each attempt calls util.generate_checked once.
        """
        last_exc: Optional[TransformError] = None
        for _ in range(max(1, retries)):

            def _gen():
                return self.generate(system, prompt, images=images)

            try:
                return _generate_checked_once(_gen, transformFn)
            except TransformError as e:
                last_exc = e
                continue
        raise GenerationError(
            f"failed-on-{retries}-retries", raw=getattr(last_exc, "raw", None)
        )

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
        return aiter_to_iter(self.astream(system, prompt, images))

    def stream_checked(
        self,
        transformFn: Callable[[str], Any],
        system: str,
        prompt: str,
        images: Optional[list] = None,
    ) -> Iterator[Dict[str, Any]]:
        return _stream_checked_once(self.stream(system, prompt, images), transformFn)

    def stream_json(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> Iterator[Dict[str, Any]]:
        return self.stream_checked(loadch, system, prompt, images)

    # ---- Streaming tool-calls with caller-managed retries ----

    async def astream_tool_calls(
        self,
        tools: Any,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> AsyncIterator[Dict[str, Any]]:
        sysprompt = (
            "You are a computer specialist. Your job is translating client requests into tool calls. "
            "Your client has sent a request to use some number of tools; return a list of function calls "
            "corresponding to the request and no other commentary. "
            'Return a value of type `[{"functionName" :: string, "args" :: {arg_name: arg value}}]`. '
            f"You have access to the tools: {tools.list()}."
        )
        async for ev in _astream_with_retries(
            lambda: self.astream(sysprompt, prompt, images),
            tools.transform_multi,
            retries=retries,
        ):
            yield ev

    def stream_tool_calls(
        self,
        tools: Any,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> Iterator[Dict[str, Any]]:
        sysprompt = (
            "You are a computer specialist. Your job is translating client requests into tool calls. "
            "Your client has sent a request to use some number of tools; return a list of function calls "
            "corresponding to the request and no other commentary. "
            'Return a value of type `[{"functionName" :: string, "args" :: {arg_name: arg value}}]`. '
            f"You have access to the tools: {tools.list()}."
        )
        return _stream_with_retries(
            lambda: self.stream(sysprompt, prompt, images),
            tools.transform_multi,
            retries=retries,
        )
