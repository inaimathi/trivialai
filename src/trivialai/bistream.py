# bistream.py
from __future__ import annotations

import asyncio
import inspect
import queue
import sys
import threading
import time
from collections.abc import AsyncIterable as ABCAsyncIterable
from collections.abc import AsyncIterator as ABCAsyncIterator
from collections.abc import Iterable as ABCIterable
from collections.abc import Iterator as ABCIterator
from typing import (TYPE_CHECKING, Any, Callable, Generic, Optional, TextIO,
                    TypeVar, Union, cast)

from .log import getLogger

T = TypeVar("T")
U = TypeVar("U")
I = TypeVar("I")
O = TypeVar("O")

__loop: asyncio.AbstractEventLoop | None = None
__thread: threading.Thread | None = None
__loop_lock = threading.Lock()

_QUEUE_MAXSIZE = 64
_SYNC_ASYNC_WARN_THRESHOLD = 1.0

logger = getLogger("trivialai.bistream")

# ---- typing helpers ------------------------------------------------------

if TYPE_CHECKING:
    # NOTE: this block is *not executed at runtime*, so it's safe to use
    # unions and forward references here without tripping import-time errors.
    StreamLikeAny = "BiStream[Any] | ABCIterable[Any] | ABCAsyncIterable[Any]"
    ThenFn = Callable[[], StreamLikeAny | None] | Callable[[Any], StreamLikeAny | None]
else:
    # Runtime: keep it permissive; we do our own signature dispatch.
    ThenFn = Callable[..., Any]  # type: ignore[assignment]


class Dual(Generic[T]):
    """
    Dual-mode stream source: provides both sync and async iteration.

    - __iter__ returns a fresh sync iterator
    - __aiter__ returns a fresh async iterator

    BiStream can wrap this and remain mode-preserving for combinators.
    """

    def __init__(
        self,
        gen_fn: Callable[[], ABCIterator[T]],
        agen_fn: Callable[[], ABCAsyncIterator[T]],
    ):
        self._gen_fn = gen_fn
        self._agen_fn = agen_fn

    def __iter__(self) -> ABCIterator[T]:
        return self._gen_fn()

    def __aiter__(self) -> ABCAsyncIterator[T]:
        return self._agen_fn()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure a dedicated background event loop running in a daemon thread.

    This loop is used to drive async iterators when they are consumed
    from synchronous code.
    """
    global __loop, __thread
    if __loop and __loop.is_running():
        return __loop

    with __loop_lock:
        if __loop and __loop.is_running():
            return __loop

        loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        __thread = threading.Thread(
            target=_runner,
            name="trivialai-bg-loop",
            daemon=True,
        )
        __thread.start()
        __loop = loop
        return loop


def aiter_to_iter(agen: ABCAsyncIterator[T]) -> ABCIterator[T]:
    """
    Bridge AsyncIterator -> Iterator, driven by the background loop.

    IMPORTANT:
      - Never blocks the background loop thread indefinitely.
      - The returned iterator has .close() to cancel the pump early.
    """
    q: "queue.Queue[object]" = queue.Queue(maxsize=_QUEUE_MAXSIZE)
    sentinel = object()

    async def _q_put(obj: object) -> None:
        # Cooperative backpressure: never block the event loop thread.
        while True:
            try:
                q.put_nowait(obj)
                return
            except queue.Full:
                # Yield so cancellation can be delivered and other tasks can run.
                await asyncio.sleep(0.005)

    async def _pump() -> None:
        try:
            async for item in agen:
                await _q_put(item)
        except asyncio.CancelledError:
            # Graceful termination.
            pass
        except Exception as e:
            await _q_put(e)
        finally:
            await _q_put(sentinel)

    loop = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(_pump(), loop)

    class SyncFromAsync(ABCIterator[T]):
        def __init__(self) -> None:
            self._closed = False

        def __iter__(self) -> "SyncFromAsync":
            return self

        def __next__(self) -> T:
            if self._closed:
                raise StopIteration

            item = q.get()
            if item is sentinel:
                self.close()
                raise StopIteration

            if isinstance(item, Exception):
                self.close()
                raise item

            return cast(T, item)

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True

            if not fut.done():
                fut.cancel()

            # Unblock a waiting consumer if possible.
            try:
                q.put_nowait(sentinel)
            except queue.Full:
                pass

    return SyncFromAsync()


def _sync_iter_to_async(sync_iter: ABCIterator[T]) -> ABCAsyncIterator[T]:
    """
    Async wrapper over a synchronous iterator.

    NOTE:
      - This calls next(sync_iter) in the event loop thread.
      - If next() blocks, the event loop is blocked.
      - We log once when a single next() call exceeds the threshold.
    """

    async def _agen() -> ABCAsyncIterator[T]:
        warned = False
        while True:
            start = time.monotonic()
            try:
                item = next(sync_iter)
            except StopIteration:
                return
            dt = time.monotonic() - start
            if (not warned) and dt > _SYNC_ASYNC_WARN_THRESHOLD:
                warned = True
                logger.warning(
                    "BiStream: sync iterator %r blocked the event loop for %.3fs "
                    "when consumed asynchronously. This likely indicates a "
                    "mis-specified producer that should be async further up the stack.",
                    sync_iter,
                    dt,
                )
            yield item

    return _agen()


def _call_then(
    fn: "ThenFn", done: Any
) -> "BiStream[Any] | ABCIterable[Any] | ABCAsyncIterable[Any] | None":
    """
    Call a then-function either as fn() or fn(done), depending on its signature.

    We try to avoid the "catch TypeError then retry" pattern, because that can
    mask real user errors (TypeError inside fn).
    """
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
    except (TypeError, ValueError):
        # Builtins / odd callables: best effort -> assume it accepts done.
        return cast(Callable[[Any], Any], fn)(done)

    # If it has *args, it's safe to pass done.
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return cast(Callable[[Any], Any], fn)(done)

    pos = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required_pos = [p for p in pos if p.default is p.empty]

    if len(required_pos) == 0:
        return cast(Callable[[], Any], fn)()

    return cast(Callable[[Any], Any], fn)(done)


def _chain_on(
    src: "BiStream[Any] | ABCIterable[Any] | ABCAsyncIterable[Any]",
    then: "ThenFn",
) -> "BiStream[Any]":
    """
    Mode-preserving "then" driven by termination, not by inspecting events.

    Drains `src`, yielding all events unchanged. When `src` terminates:
      - calls `then(...)` exactly once
      - then yields all events from the follow-up stream (if any)

    `then` may be:
      - a 0-arg callable: then()
      - a 1-arg callable: then(done)

    `done` is:
      - for sync iterators: StopIteration.value (usually None unless a generator `return`s a value)
      - for async iterators: first StopAsyncIteration arg if present, else None
    """
    upstream = BiStream.ensure(src)

    def _gen() -> ABCIterator[Any]:
        it = iter(upstream)
        done: Any = None

        while True:
            try:
                ev = next(it)
            except StopIteration as e:
                done = getattr(e, "value", None)
                break
            else:
                yield ev

        follow = _call_then(then, done)
        if follow is None:
            return

        for ev2 in BiStream.ensure(follow):
            yield ev2

    async def _agen() -> ABCAsyncIterator[Any]:
        it = upstream.__aiter__()
        done: Any = None

        while True:
            try:
                ev = await it.__anext__()
            except StopAsyncIteration as e:
                done = e.args[0] if e.args else None
                break
            else:
                yield ev

        follow = _call_then(then, done)
        if follow is None:
            return

        async for ev2 in BiStream.ensure(follow):
            yield ev2

    return BiStream(Dual(_gen, _agen))


def branch(
    src: "BiStream[I] | ABCIterable[I] | ABCAsyncIterable[I]",
    mk_stream: Callable[[I], "BiStream[O] | ABCIterable[O] | ABCAsyncIterable[O]"],
) -> "FanOut[I, O]":
    return FanOut(BiStream.ensure(src), mk_stream)


class FanOut(Generic[I, O]):
    def __init__(
        self,
        src: "BiStream[I]",
        mk_stream: Callable[[I], "BiStream[O] | ABCIterable[O] | ABCAsyncIterable[O]"],
        *,
        prefix: "BiStream[Any] | None" = None,
        passthrough_prefix: bool = True,
    ):
        self._src = src
        self._mk = mk_stream
        self._prefix = prefix
        self._passthrough_prefix = passthrough_prefix

    def sequence(self) -> "BiStream[Any]":
        prefix = self._prefix
        items = self._src
        mk = self._mk
        passthrough = self._passthrough_prefix

        def _gen() -> ABCIterator[Any]:
            if prefix is not None:
                for ev in prefix:
                    if passthrough:
                        yield ev
            for item in items:
                b = BiStream.ensure(mk(item))
                for ev in b:
                    yield ev

        async def _agen() -> ABCAsyncIterator[Any]:
            if prefix is not None:
                async for ev in prefix:
                    if passthrough:
                        yield ev
            async for item in items:
                b = BiStream.ensure(mk(item))
                async for ev in b:
                    yield ev

        return BiStream(Dual(_gen, _agen))

    def interleave(self, *, concurrency: int = 0) -> "BiStream[Any]":
        prefix = self._prefix
        items = self._src
        mk = self._mk
        passthrough = self._passthrough_prefix
        limit = concurrency

        async def _agen() -> ABCAsyncIterator[Any]:
            # 1) Drain prefix first (streaming through, no buffering)
            if prefix is not None:
                async for ev in prefix:
                    if passthrough:
                        yield ev

            # 2) Interleave merge across branches
            active: dict[int, ABCAsyncIterator[Any]] = {}
            tasks: dict[asyncio.Task[Any], int] = {}
            next_id = 0

            src_it = items.__aiter__()
            src_exhausted = False

            async def _start_one_branch() -> bool:
                nonlocal next_id, src_exhausted
                if src_exhausted:
                    return False
                try:
                    item = await src_it.__anext__()
                except StopAsyncIteration:
                    src_exhausted = True
                    return False

                it = BiStream.ensure(mk(item)).__aiter__()
                bid = next_id
                next_id += 1
                active[bid] = it
                t = asyncio.create_task(it.__anext__())
                tasks[t] = bid
                return True

            def _room() -> bool:
                return (limit <= 0) or (len(active) < limit)

            while _room():
                if not await _start_one_branch():
                    break

            try:
                while tasks:
                    done, _ = await asyncio.wait(
                        tasks.keys(),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for t in done:
                        bid = tasks.pop(t)
                        it = active.get(bid)

                        try:
                            ev = t.result()
                        except StopAsyncIteration:
                            active.pop(bid, None)
                            while _room():
                                if not await _start_one_branch():
                                    break
                            continue
                        except Exception:
                            for ot in list(tasks.keys()):
                                ot.cancel()
                            raise
                        else:
                            yield ev
                            if it is not None:
                                nt = asyncio.create_task(it.__anext__())
                                tasks[nt] = bid
            finally:
                for t in list(tasks.keys()):
                    t.cancel()
                if tasks:
                    await asyncio.gather(*tasks.keys(), return_exceptions=True)

        def _gen() -> ABCIterator[Any]:
            it = aiter_to_iter(_agen())
            try:
                for ev in it:
                    yield ev
            finally:
                close = getattr(it, "close", None)
                if callable(close):
                    close()

        return BiStream(Dual(_gen, _agen))

    # ---- friendly “must fan-in” errors ----

    def _need_fan_in(self, name: str) -> None:
        raise RuntimeError(
            f"You called {name}() on a FanOut. FanOut yields branch streams, not events. "
            f"Call .sequence() or .interleave() first to fan-in to a BiStream."
        )

    def then(self, *a: Any, **k: Any) -> Any:  # type: ignore[no-untyped-def]
        self._need_fan_in("then")

    def tap(self, *a: Any, **k: Any) -> Any:  # type: ignore[no-untyped-def]
        self._need_fan_in("tap")

    def repeat_until(self, *a: Any, **k: Any) -> Any:  # type: ignore[no-untyped-def]
        self._need_fan_in("repeat_until")


def repeat_until(
    src: "BiStream[Any] | ABCIterable[Any] | ABCAsyncIterable[Any]",
    step: Callable[[Any], "BiStream[Any] | ABCIterable[Any] | ABCAsyncIterable[Any]"],
    *,
    stop: Callable[[Any], bool],
    max_iters: int = 10,
) -> "BiStream[Any]":
    """
    Mode-preserving repeat-until with early-exit (event-only stop).

    For up to max_iters iterations:
      - stream events from current stream, yielding them
      - if stop(ev) becomes True:
          * yield that ev
          * stop immediately (do NOT run step)
          * close the underlying iterator (best-effort)
      - otherwise, on natural termination:
          * driver = StopIteration.value / StopAsyncIteration.args[0] if present,
                     else last yielded event
          * if driver is None: stop
          * else: current stream = step(driver) and continue

    Also closes the underlying iterator on any exception / consumer abort.
    """
    upstream = BiStream.ensure(src)

    def _close_sync(it: Any) -> None:
        close = getattr(it, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    async def _close_async(it: Any) -> None:
        aclose = getattr(it, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:
                pass

    def _gen() -> ABCIterator[Any]:
        stream: "BiStream[Any]" = upstream

        for _ in range(max_iters):
            it = iter(stream)
            last_ev: Any | None = None
            done: Any | None = None
            hit_stop = False
            exc: BaseException | None = None

            try:
                while True:
                    try:
                        ev = next(it)
                    except StopIteration as e:
                        done = getattr(e, "value", None)
                        break

                    last_ev = ev
                    yield ev

                    if stop(ev):
                        hit_stop = True
                        break

            except BaseException as e:
                exc = e
                raise

            finally:
                # Close on early-exit OR any exception/abort. (Normal exhaustion: no need.)
                if hit_stop or exc is not None:
                    _close_sync(it)

            if hit_stop:
                break

            driver = done if done is not None else last_ev
            if driver is None:
                break

            stream = BiStream.ensure(step(driver))

    async def _agen() -> ABCAsyncIterator[Any]:
        stream: "BiStream[Any]" = upstream

        for _ in range(max_iters):
            it = stream.__aiter__()
            last_ev: Any | None = None
            done: Any | None = None
            hit_stop = False
            exc: BaseException | None = None

            try:
                while True:
                    try:
                        ev = await it.__anext__()
                    except StopAsyncIteration as e:
                        done = e.args[0] if e.args else None
                        break

                    last_ev = ev
                    yield ev

                    if stop(ev):
                        hit_stop = True
                        break

            except BaseException as e:
                exc = e
                raise

            finally:
                if hit_stop or exc is not None:
                    await _close_async(it)

            if hit_stop:
                break

            driver = done if done is not None else last_ev
            if driver is None:
                break

            stream = BiStream.ensure(step(driver))

    return BiStream(Dual(_gen, _agen))


def tap(
    src: "BiStream[T] | ABCIterable[T] | ABCAsyncIterable[T]",
    cb: Callable[[T], Any],
    *,
    focus: Optional[Callable[[T], bool]] = None,
    ignore: Optional[Callable[[T], bool]] = None,
) -> "BiStream[T]":
    """
    Mode-preserving tap: runs cb(ev) for selected events but yields unchanged events.
    """
    upstream = BiStream.ensure(src)

    def _want(ev: T) -> bool:
        if focus is not None and ignore is not None:
            return focus(ev) and not ignore(ev)
        if focus is not None:
            return focus(ev)
        if ignore is not None:
            return not ignore(ev)
        return True

    def _gen() -> ABCIterator[T]:
        for ev in upstream:
            if _want(ev):
                cb(ev)
            yield ev

    async def _agen() -> ABCAsyncIterator[T]:
        async for ev in upstream:
            if _want(ev):
                cb(ev)
            yield ev

    return BiStream(Dual(_gen, _agen))


class BiStream(Generic[T], ABCIterator[T], ABCAsyncIterator[T]):
    """
    Bidirectional stream wrapper.

    Goal:
      - `for x in BiStream(...)` works in sync contexts (REPL, tests).
      - `async for x in BiStream(...)` works in async contexts (handlers).
      - Combinators should be mode-preserving by returning Dual sources.
      - If a sync producer is consumed async and blocks, we warn (no threads).
    """

    def __init__(self, src: "ABCIterable[T] | ABCAsyncIterable[T] | BiStream[T]"):
        self._mode: str | None = None

        if isinstance(src, BiStream):
            # Share the same underlying source and iterators (consumption state).
            self._src = src._src
            self._sync_iter = src._sync_iter
            self._async_iter = src._async_iter
            self._has_sync = src._has_sync
            self._has_async = src._has_async
            self._mode = src._mode
            return

        self._src = src
        self._sync_iter: ABCIterator[T] | None = None
        self._async_iter: ABCAsyncIterator[T] | None = None

        self._has_sync = isinstance(src, ABCIterable)
        self._has_async = isinstance(src, ABCAsyncIterable)

        if not (self._has_sync or self._has_async):
            raise TypeError("BiStream source is neither iterable nor async iterable")

    def _set_mode(self, mode: str) -> None:
        if self._mode is None:
            self._mode = mode
            return
        if self._mode != mode:
            raise RuntimeError(
                f"BiStream is already being consumed in {self._mode} mode; "
                f"cannot also use it in {mode} mode."
            )

    @classmethod
    def ensure(
        cls,
        src: "BiStream[T] | ABCIterable[T] | ABCAsyncIterable[T]",
    ) -> "BiStream[T]":
        if isinstance(src, BiStream):
            return src
        return cls(src)

    # ---- sync side ----

    def __iter__(self) -> "BiStream[T]":
        self._set_mode("sync")
        if self._sync_iter is None:
            if self._has_sync:
                self._sync_iter = iter(cast(ABCIterable[T], self._src))
            else:
                # Async-only source; bridge it now.
                assert self._has_async
                if self._async_iter is None:
                    self._async_iter = cast(ABCAsyncIterable[T], self._src).__aiter__()
                self._sync_iter = aiter_to_iter(self._async_iter)
        return self

    def __next__(self) -> T:
        self._set_mode("sync")
        if self._sync_iter is None:
            self.__iter__()
        assert self._sync_iter is not None
        return next(self._sync_iter)

    # ---- async side ----

    def __aiter__(self) -> "BiStream[T]":
        self._set_mode("async")
        if self._async_iter is None:
            if self._has_async:
                self._async_iter = cast(ABCAsyncIterable[T], self._src).__aiter__()
            else:
                # Sync-only source; adapt (may block loop; we warn inside wrapper).
                assert self._has_sync
                if self._sync_iter is None:
                    self._sync_iter = iter(cast(ABCIterable[T], self._src))
                self._async_iter = _sync_iter_to_async(self._sync_iter)
        return self

    async def __anext__(self) -> T:
        self._set_mode("async")
        if self._async_iter is None:
            self.__aiter__()
        assert self._async_iter is not None
        return await self._async_iter.__anext__()

    # ---- combinators ----

    def then(
        self,
        fn: "ThenFn",
    ) -> "BiStream[Any]":
        return cast("BiStream[Any]", _chain_on(self, fn))

    def branch(
        self: "BiStream[Any]",
        items: "ABCIterable[I] | ABCAsyncIterable[I] | BiStream[I]",
        per_item: Callable[[I], "BiStream[O] | ABCIterable[O] | ABCAsyncIterable[O]"],
        *,
        passthrough_prefix: bool = True,
    ) -> "FanOut[I, O]":
        # “gated” branch: prefix is this stream; items are fanned out afterwards
        return FanOut(
            BiStream.ensure(items),
            per_item,
            prefix=self,
            passthrough_prefix=passthrough_prefix,
        )

    def tap(
        self,
        cb: Callable[[T], Any],
        *,
        focus: Optional[Callable[[T], bool]] = None,
        ignore: Optional[Callable[[T], bool]] = None,
    ) -> "BiStream[T]":
        return tap(self, cb, focus=focus, ignore=ignore)

    def map(self: "BiStream[I]", fn: Callable[[I], O]) -> "BiStream[O]":
        upstream = self

        def _gen() -> ABCIterator[O]:
            for x in upstream:
                yield fn(x)

        async def _agen() -> ABCAsyncIterator[O]:
            async for x in upstream:
                yield fn(x)

        return BiStream(Dual(_gen, _agen))

    def mapcat(
        self: "BiStream[I]",
        fn: Callable[[I], "BiStream[O] | ABCIterable[O] | ABCAsyncIterable[O]"],
        *,
        concurrency: int = 0,
    ) -> "BiStream[O]":
        fo = branch(self, fn)
        merged = (
            fo.sequence()
            if concurrency <= 0
            else fo.interleave(concurrency=concurrency)
        )
        return cast("BiStream[O]", merged)


def is_type(type_name: str) -> Callable[[Any], bool]:
    return lambda ev: isinstance(ev, dict) and ev.get("type") == type_name


def force(
    src: "BiStream[dict] | ABCIterable[dict] | ABCAsyncIterable[dict]",
    *,
    keep: Optional[Union[set[str], str]] = None,
    out: TextIO | None = None,
) -> list[dict]:
    """
    REPL helper: consume a (bi)stream of event dicts, pretty-print deltas,
    and return a list of selected events.

    NOTE: Always consumes synchronously (async producers are driven via background loop).
    """
    stream = BiStream.ensure(src)
    if out is None:
        out = sys.stdout
    if isinstance(keep, str):
        keep = {keep}
    if keep is None:
        keep = {"end", "final"}

    end_events: list[dict] = []
    current_mode: Optional[str] = None  # "thinking", "saying", or None

    for event in stream:
        if isinstance(event, dict):
            etype = event.get("type")

            if etype == "delta":
                scratchpad = event.get("scratchpad") or ""
                text = event.get("text") or ""

                if scratchpad:
                    if current_mode != "thinking":
                        if current_mode is not None:
                            print(file=out)
                        print("Thinking: ", end="", file=out)
                        current_mode = "thinking"
                    print(scratchpad, end="", file=out)

                if text:
                    if current_mode != "saying":
                        if current_mode is not None:
                            print(file=out)
                        print("Saying: ", end="", file=out)
                        current_mode = "saying"
                    print(text, end="", file=out)

                continue

            if etype in keep:
                end_events.append(event)

            if etype in {"end", "final"}:
                continue

        if current_mode is not None:
            print(file=out)
            current_mode = None
        print(event, file=out)

    if current_mode is not None:
        print(file=out)

    return end_events
