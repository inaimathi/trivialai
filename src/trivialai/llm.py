# llm.py
from __future__ import annotations

from asyncio import to_thread
from typing import Any, AsyncIterator, Callable, Dict, Optional

from .async_utils import DualStream, dual_stream
from .util import GenerationError, LLMResult, TransformError
from .util import astream_checked as _astream_checked_once
from .util import generate_checked as _generate_checked_once
from .util import loadch


async def _astream_with_retries(
    stream_factory: Callable[[], AsyncIterator[Dict[str, Any]]],
    transformFn: Callable[[str], Any],
    *,
    retries: int,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Async retry helper for streaming.

    - Calls util.astream_checked(stream_factory(), transformFn) once per attempt.
    - Yields passthrough events.
    - Emits a final event with retry metadata.
    """
    last_error_msg: Optional[str] = None

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

        # failure on this attempt
        last_error_msg = (final_ev or {}).get("error")
        yield {
            "type": "attempt-failed",
            "attempt": attempt,
            "error": last_error_msg,
            "raw": (final_ev or {}).get("raw"),
        }

    # Exhausted retries
    yield {
        "type": "final",
        "ok": False,
        "error": f"failed-on-{retries}-retries",
        "attempts": retries,
        "last_error": last_error_msg,
    }


class LLMMixin:
    """
    LLM mixin with:

      - sync core:
          generate, generate_checked, generate_json, generate_tool_call, ...

      - async helpers:
          agenerate (default: run generate() in a thread)

      - streaming surfaces:
          stream, stream_checked, stream_json, stream_tool_calls

        All streaming methods are implemented once as async generators,
        and exposed as DualStream so they work in both sync and async code:

          # async
          async for ev in llm.stream(...): ...
          async for ev in llm.stream_json(...): ...

          # sync (REPL/tests)
          for ev in llm.stream(...): ...
          for ev in llm.stream_json(...): ...
    """

    # ---- Core text generation ------------------------------------------------

    def generate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        """
        Subclasses MUST override this with a synchronous implementation.

        Returns an LLMResult(raw, content, scratchpad).
        """
        raise NotImplementedError

    async def agenerate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        """
        Default async implementation: runs sync `generate` in a thread.

        Subclasses with native async clients (like Ollama using httpx.AsyncClient)
        may override this to avoid thread hopping.
        """
        return await to_thread(self.generate, system, prompt, images)

    def generate_checked(
        self,
        transformFn: Callable[[str], Any],
        system: str,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> LLMResult:
        """
        Retry at the caller level. Each attempt calls util.generate_checked once,
        which in turn calls self.generate(...) and applies transformFn.
        """
        last_exc: Optional[TransformError] = None
        for _ in range(max(1, retries)):

            def _gen() -> LLMResult:
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
        """
        Convenience wrapper for JSON responses.
        """
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

    # ---- Base async streaming primitive -------------------------------------

    async def _astream_raw(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Default streaming implementation in terms of agenerate().

        Subclasses that support true incremental streaming SHOULD override this
        with a real async streaming implementation.

        Yields NDJSON-style events:
          - {"type":"start", "provider":..., "model":...}
          - {"type":"delta", "text": ...}
          - {"type":"end", "content":..., "scratchpad":..., "tokens": int}
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

    # ---- Dual-mode streaming surfaces ---------------------------------------

    @dual_stream
    async def stream(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Dual-mode stream interface.

        - Async:
            async for ev in llm.stream(system, prompt): ...

        - Sync:
            for ev in llm.stream(system, prompt): ...

        Default behavior is to call `_astream_raw`, which in turn uses `agenerate`.
        Subclasses can override `_astream_raw` for true incremental streaming.
        """
        async for ev in self._astream_raw(system, prompt, images):
            yield ev

    @dual_stream
    async def stream_checked(
        self,
        transformFn: Callable[[str], Any],
        system: str,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Retry-aware streaming wrapper that uses util.astream_checked under the hood.

        Exposed as dual-mode:

          - async for ev in llm.stream_checked(...): ...
          - for ev in llm.stream_checked(...): ...
        """
        async for ev in _astream_with_retries(
            lambda: self.stream(system, prompt, images),
            transformFn,
            retries=retries,
        ):
            yield ev

    @dual_stream
    async def stream_json(
        self,
        system: str,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        JSON convenience wrapper over stream_checked with retries.
        """
        async for ev in self.stream_checked(
            loadch, system, prompt, images=images, retries=retries
        ):
            yield ev

    @dual_stream
    async def stream_tool_calls(
        self,
        tools: Any,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streaming tool-call interface.

        Yields:
          - passthrough events from the underlying LLM stream
          - attempt-failed / final events from the checked parser
        """
        sysprompt = (
            "You are a computer specialist. Your job is translating client requests into tool calls. "
            "Your client has sent a request to use some number of tools; return a list of function calls "
            "corresponding to the request and no other commentary. "
            'Return a value of type `[{"functionName" :: string, "args" :: {arg_name: arg value}}]`. '
            f"You have access to the tools: {tools.list()}."
        )

        async for ev in self.stream_checked(
            tools.transform_multi,
            sysprompt,
            prompt,
            images=images,
            retries=retries,
        ):
            yield ev
