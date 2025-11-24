# llm.py
from __future__ import annotations

from asyncio import to_thread
from typing import Any, AsyncIterator, Callable, Dict, Optional, Tuple

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

    Scratchpad support:

      - Class-level tags THINK_OPEN / THINK_CLOSE define scratchpad boundaries.
      - If these are set (e.g., "<think>" / "</think>"), helper methods
        `_split_think_full` and `_separate_think_delta` can be used by
        subclasses to separate public content from scratchpad, both in
        full-response and streaming modes.

      - If tags are left as None (default), no tag-based splitting is applied
        by the helpers (they become no-ops).
    """

    # ---- Scratchpad configuration (override per subclass if desired) --------

    # Example for models with think blocks:
    #   THINK_OPEN = "<think>"
    #   THINK_CLOSE = "</think>"
    THINK_OPEN: Optional[str] = None
    THINK_CLOSE: Optional[str] = None

    # ---- Scratchpad helpers -------------------------------------------------

    @classmethod
    def _has_scratchpad_tags(cls) -> bool:
        return bool(cls.THINK_OPEN) and bool(cls.THINK_CLOSE)

    @classmethod
    def _split_think_full(cls, text: str) -> Tuple[str, Optional[str]]:
        """
        Given a full response string, remove a single THINK_OPEN...THINK_CLOSE
        section (if configured and present) and return (public_text, scratchpad).

        If tags are not configured or no complete pair is found, returns
        (text, None).
        """
        if not cls._has_scratchpad_tags() or not text:
            return text, None

        open_tag = cls.THINK_OPEN or ""
        close_tag = cls.THINK_CLOSE or ""

        start = text.find(open_tag)
        if start == -1:
            return text, None
        end = text.find(close_tag, start + len(open_tag))
        if end == -1:
            return text, None

        scratch = text[start + len(open_tag) : end].strip()
        content = (text[:start] + text[end + len(close_tag) :]).strip()
        return content, (scratch or None)

    @classmethod
    def _separate_think_delta(
        cls,
        delta: str,
        in_think: bool,
        carry: str,
    ) -> Tuple[str, str, bool, str]:
        """
        Streaming-safe splitter that handles THINK_OPEN...THINK_CLOSE across
        chunk boundaries.

        Args:
          delta: current chunk
          in_think: whether we're inside THINK_OPEN…THINK_CLOSE
          carry: trailing fragment from previous chunk that *might* be a tag prefix

        Returns:
          (public_out, scratch_out, new_in_think, new_carry)

        If tags are not configured, returns (delta+carry, "", False, "").
        """
        if not cls._has_scratchpad_tags():
            # No tag semantics: everything is public text.
            full = (carry or "") + (delta or "")
            return full, "", False, ""

        open_tag = cls.THINK_OPEN or ""
        close_tag = cls.THINK_CLOSE or ""
        max_len = max(len(open_tag), len(close_tag))

        text = (carry or "") + (delta or "")
        i = 0
        out_pub: list[str] = []
        out_scr: list[str] = []

        def _maybe_partial_tag_at(idx: int) -> bool:
            remaining = text[idx:]
            if len(remaining) >= max_len:
                return False
            return open_tag.startswith(remaining) or close_tag.startswith(remaining)

        while i < len(text):
            # Full tags
            if text.startswith(open_tag, i):
                in_think = True
                i += len(open_tag)
                continue
            if text.startswith(close_tag, i):
                in_think = False
                i += len(close_tag)
                continue

            # Partial tag at buffer end? keep it in carry for the next call
            if _maybe_partial_tag_at(i):
                carry = text[i:]
                break

            # Normal emission
            ch = text[i]
            if in_think:
                out_scr.append(ch)
            else:
                out_pub.append(ch)
            i += 1
            carry = ""  # we've consumed any previous carry

        # If we consumed all text, ensure carry is empty
        if i >= len(text):
            carry = ""

        return ("".join(out_pub), "".join(out_scr), in_think, carry)

    # ---- Core text generation -----------------------------------------------

    def generate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        """
        Subclasses MUST override this with a synchronous implementation.

        Returns an LLMResult(raw, content, scratchpad).

        Subclasses that want tag-based scratchpad splitting for full responses
        can use `_split_think_full` here, e.g.:

            raw = provider_call(...)
            content, scratch = self._split_think_full(raw_text)
            return LLMResult(raw, content, scratch)
        """
        raise NotImplementedError

    async def agenerate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        """
        Default async implementation: runs sync `generate` in a thread.

        Subclasses with native async clients may override this to avoid thread hopping.
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
        with a real async streaming implementation that can use
        `_separate_think_delta` per chunk.

        Default behavior (no incremental streaming):

          - calls agenerate()
          - emits a single delta with text=res.content and scratchpad=""
          - emits an end event with content=res.content, scratchpad=res.scratchpad
        """
        res = await self.agenerate(system, prompt, images)
        content = res.content or ""
        yield {
            "type": "start",
            "provider": self.__class__.__name__.lower(),
            "model": getattr(self, "model", None),
        }
        if content:
            # For generic models (e.g., ChatGPT/Claude) that don't have
            # a separate scratchpad stream, we standardize on scratchpad=""
            # for deltas. Models with think tags will typically override
            # _astream_raw and use _separate_think_delta instead.
            yield {"type": "delta", "text": content, "scratchpad": ""}

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
        Subclasses can override `_astream_raw` for true incremental streaming and
        can use `_separate_think_delta` to manage scratchpad tokens.
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
