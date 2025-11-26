# llm.py
from __future__ import annotations

import re
from asyncio import to_thread
from typing import Any, AsyncIterator, Callable, Dict, Optional, Tuple

from .bistream import BiStream
from .util import GenerationError, LLMResult, TransformError, astream_checked
from .util import generate_checked as _generate_checked_once
from .util import loadch

# --- Module-level helpers for <think> parsing (default tags) ---

_DEFAULT_THINK_OPEN = "<think>"
_DEFAULT_THINK_CLOSE = "</think>"


def _split_think_full_with_tags(
    resp: str, open_tag: str, close_tag: str
) -> Tuple[str, Optional[str]]:
    """
    Given a full response string, remove a single open_tag...close_tag section (if present)
    and return (public_text, scratchpad_text). If none present, scratchpad is None.
    """
    pattern = re.compile(
        re.escape(open_tag) + r"(.*?)" + re.escape(close_tag), re.DOTALL
    )
    m = pattern.search(resp)
    if not m:
        return resp, None
    scratch = m.group(1).strip()
    content = (resp[: m.start()] + resp[m.end() :]).strip()
    return content, (scratch or None)


def _split_think_full(resp: str) -> Tuple[str, Optional[str]]:
    """
    Default helper using <think>...</think> tags.
    (Kept for back-compat / tests that import it directly.)
    """
    return _split_think_full_with_tags(resp, _DEFAULT_THINK_OPEN, _DEFAULT_THINK_CLOSE)


def _separate_think_delta_with_tags(
    delta: str,
    in_think: bool,
    carry: str,
    open_tag: str,
    close_tag: str,
) -> Tuple[str, str, bool, str]:
    """
    Streaming-safe splitter that handles open_tag...close_tag across chunk boundaries.

    Args:
      delta: current chunk
      in_think: whether we're inside <think>…</think>-style region
      carry: trailing fragment from previous chunk that *might* be a tag prefix

    Returns:
      (public_out, scratch_out, new_in_think, new_carry)
    """
    text = carry + delta
    i = 0
    out_pub: list[str] = []
    out_scr: list[str] = []

    max_tag_len = max(len(open_tag), len(close_tag))

    def _maybe_partial_tag_at(idx: int) -> bool:
        # If remaining text at idx is a prefix of an open/close tag, we need more bytes.
        remaining = text[idx:]
        if len(remaining) >= max_tag_len:
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


def _separate_think_delta(
    delta: str, in_think: bool, carry: str
) -> Tuple[str, str, bool, str]:
    """
    Default helper using <think>...</think> tags.
    (Kept for back-compat / tests that import it directly.)
    """
    return _separate_think_delta_with_tags(
        delta, in_think, carry, _DEFAULT_THINK_OPEN, _DEFAULT_THINK_CLOSE
    )


async def _astream_with_retries(
    stream_factory: Callable[[], Any],
    transformFn: Callable[[str], Any],
    *,
    retries: int,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Async core for streaming with retries.

    For each attempt:
      - Run `astream_checked` on the stream
      - Pass through non-final events
      - Inspect the final event:
          * if ok=True, emit final+attempt and stop
          * else emit attempt-failed and retry

    After exhausting retries, emit a final failure event.
    """
    last_error_msg: Optional[str] = None

    for attempt in range(1, max(1, retries) + 1):
        final_ev: Optional[Dict[str, Any]] = None

        # Normalize stream to BiStream so we can safely consume it async.
        stream = BiStream.ensure(stream_factory())

        async for ev in astream_checked(stream, transformFn):
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

    # Exhausted retries
    yield {
        "type": "final",
        "ok": False,
        "error": f"failed-on-{retries}-retries",
        "attempts": retries,
        "last_error": last_error_msg,
    }


def _stream_with_retries(
    stream_factory: Callable[[], Any],
    transformFn: Callable[[str], Any],
    *,
    retries: int,
) -> BiStream[Dict[str, Any]]:
    """
    Sync/async-friendly wrapper around _astream_with_retries.

    Returns a BiStream so callers can:
      - `for ev in _stream_with_retries(...): ...`
      - `async for ev in _stream_with_retries(...): ...`
    """
    return BiStream(_astream_with_retries(stream_factory, transformFn, retries=retries))


class LLMMixin:
    """
    Mixin surface for LLMs.

    Subclasses MUST implement:
      - generate(system, prompt, images=None) -> LLMResult

    Subclasses MAY override:
      - astream(...) for true incremental streaming
      - agenerate(...) if they want a custom async aggregation strategy

    Scratchpad / think-block handling:

      - Set THINK_OPEN / THINK_CLOSE at class definition time *or* per instance:

          class Ollama(LLMMixin, ...):
              THINK_OPEN = "<think>"
              THINK_CLOSE = "</think>"

          # or
          o = Ollama(..., open_think="<!--think-->", close_think="<!--/think-->")

      - If THINK_* is configured (on the instance), streaming output from `.stream(...)` will:
          * strip think blocks from user-visible 'text'
          * accumulate them into 'scratchpad'
      - If THINK_* is NOT configured, streaming output will:
          * leave text untouched
          * emit scratchpad="" on deltas and scratchpad=None on end.
    """

    THINK_OPEN: Optional[str] = None
    THINK_CLOSE: Optional[str] = None

    # ---- Instance-level helpers for think/scratchpad parsing ----

    def _get_think_tags(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Look up think tags on the instance first (allow per-instance overrides),
        then fall back to class attributes.
        """
        open_tag = getattr(self, "THINK_OPEN", None)
        close_tag = getattr(self, "THINK_CLOSE", None)
        return open_tag, close_tag

    def _think_enabled(self) -> bool:
        open_tag, close_tag = self._get_think_tags()
        return bool(open_tag and close_tag)

    def split_think_full(self, resp: str) -> Tuple[str, Optional[str]]:
        """
        Split a full response string into (public_text, scratchpad) based on
        the instance THINK_* configuration.
        """
        open_tag, close_tag = self._get_think_tags()
        if not (open_tag and close_tag):
            return resp, None
        return _split_think_full_with_tags(resp, open_tag, close_tag)

    def separate_think_delta(
        self, delta: str, in_think: bool, carry: str
    ) -> Tuple[str, str, bool, str]:
        """
        Split a delta chunk into (public, scratchpad) based on THINK_* config.

        If THINK_* is not set, returns (delta_with_carry, "", in_think, "").
        """
        open_tag, close_tag = self._get_think_tags()
        if not (open_tag and close_tag):
            # No scratchpad semantics: entire delta is public.
            text = carry + delta
            return text, "", in_think, ""
        return _separate_think_delta_with_tags(
            delta, in_think, carry, open_tag, close_tag
        )

    # ---- Synchronous APIs ----

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
        """
        Convenience wrapper: decode the model output as JSON via loadch.
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
        """
        Single tool-call generation: return a single validated tool invocation.
        """
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
        """
        Multiple tool-call generation: return a list of validated tool invocations.
        """
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

    # ---- Async full-generate built on streaming ----

    async def agenerate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        """
        Default async implementation: aggregates over the streaming API.

        Subclasses that override `astream` automatically benefit here; callers
        can still `await llm.agenerate(...)` without caring about streaming details.
        """
        content_parts: list[str] = []
        scratch_parts: list[str] = []

        async for ev in self.stream(system, prompt, images):
            t = ev.get("type")
            if t == "delta":
                text = ev.get("text", "") or ""
                scratch = ev.get("scratchpad", "") or ""
                if text:
                    content_parts.append(text)
                if scratch:
                    scratch_parts.append(scratch)
            elif t == "end":
                if "content" in ev and ev["content"] is not None:
                    content_parts = [ev["content"]]
                if "scratchpad" in ev and ev["scratchpad"] is not None:
                    scratch_parts = [ev["scratchpad"]]

        content = "".join(content_parts)
        scratch = "".join(scratch_parts) if scratch_parts else None
        return LLMResult(raw=None, content=content, scratchpad=scratch)

    # ---- Provider-level async streaming core ----

    async def astream(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Default provider-level async streaming implementation: runs sync `generate`
        in a thread and yields a single delta.

        Subclasses with native async/streaming APIs should override *this* to yield
        base events:

          - {"type":"start", ...}
          - {"type":"delta", "text": "..."}  # may include THINK tags
          - {"type":"end", "content": "..."} # optional content

        THINK parsing is applied by `stream(...)`, not here.
        """
        res = await to_thread(self.generate, system, prompt, images)
        content = res.content or ""

        yield {
            "type": "start",
            "provider": self.__class__.__name__.lower(),
            "model": getattr(self, "model", None),
        }
        if content:
            yield {
                "type": "delta",
                "text": content,
            }
        yield {
            "type": "end",
            "content": content,
        }

    # ---- Unified streaming APIs (BiStream-based, with THINK parsing) ----

    def stream(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> BiStream[Dict[str, Any]]:
        """
        Unified streaming API, with scratchpad (<think>…) parsing handled here.

        Returns a BiStream so callers can:
          - use it synchronously:   for ev in llm.stream(...):
          - use it asynchronously:  async for ev in llm.stream(...):
        """

        async def _wrapped() -> AsyncIterator[Dict[str, Any]]:
            in_think = False
            carry = ""
            content_buf: list[str] = []
            scratch_buf: list[str] = []

            async for ev in self.astream(system, prompt, images):
                t = ev.get("type")

                if t == "delta":
                    raw_text = ev.get("text", "") or ""
                    out, scr, in_think, carry = self.separate_think_delta(
                        raw_text, in_think, carry
                    )
                    out = out or ""
                    scr = scr or ""

                    new_ev = dict(ev)
                    new_ev["text"] = out
                    new_ev["scratchpad"] = scr

                    if out:
                        content_buf.append(out)
                    if scr:
                        scratch_buf.append(scr)

                    # For THINK-enabled models we might drop purely tag-only deltas;
                    # for non-THINK we still pass through whatever we got.
                    if out or scr or not self._think_enabled():
                        yield new_ev

                elif t == "end":
                    # Flush any leftover carry as final text/scratch
                    if carry:
                        extra_pub, extra_scr, in_think, carry = (
                            self.separate_think_delta(carry, in_think, "")
                        )
                        if extra_pub:
                            content_buf.append(extra_pub)
                        if extra_scr:
                            scratch_buf.append(extra_scr)

                    final_content = "".join(content_buf) or ev.get("content") or ""
                    final_scratch = "".join(scratch_buf) or None

                    new_ev = dict(ev)
                    new_ev["content"] = final_content
                    new_ev["scratchpad"] = final_scratch

                    yield new_ev

                else:
                    # passthrough for "start" / "error" / etc
                    yield ev

        return BiStream(_wrapped())

    def stream_checked(
        self,
        transformFn: Callable[[str], Any],
        system: str,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> BiStream[Dict[str, Any]]:
        """
        Streaming with transform + retries, returned as a BiStream.
        """
        return _stream_with_retries(
            lambda: self.stream(system, prompt, images),
            transformFn,
            retries=retries,
        )

    def stream_json(
        self,
        system: str,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> BiStream[Dict[str, Any]]:
        """
        Streaming JSON convenience wrapper over stream_checked with retries.
        """
        return self.stream_checked(
            loadch, system, prompt, images=images, retries=retries
        )

    # ---- Streaming tool-calls (BiStream-based) ----

    def stream_tool_calls(
        self,
        tools: Any,
        prompt: str,
        images: Optional[list] = None,
        retries: int = 5,
    ) -> BiStream[Dict[str, Any]]:
        """
        Stream a list of tool calls with retries, as a BiStream of events.
        """
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
