# bistream.py
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import AsyncIterable as ABCAsyncIterable
from collections.abc import AsyncIterator as ABCAsyncIterator
from collections.abc import Iterable as ABCIterable
from collections.abc import Iterator as ABCIterator
from functools import reduce
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar

from .log import getLogger

T = TypeVar("T")
U = TypeVar("U")

__loop: asyncio.AbstractEventLoop | None = None
__thread: threading.Thread | None = None
__loop_lock = threading.Lock()

# Soft backpressure for async->sync bridge. If the sync consumer is slower
# than the async producer, the background pump will eventually block on
# q.put() instead of letting the queue grow without bound.
_QUEUE_MAXSIZE = 64

# If a sync iterator is used asynchronously and a single iteration takes
# longer than this, we log a warning once per iterator.
_SYNC_ASYNC_WARN_THRESHOLD = 1  # seconds

logger = getLogger("trivialai.bistream")


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
    Generic bridge from AsyncIterator -> Iterator, using the background loop.

    Pattern:
      - Spin a single coroutine that pumps items into a Queue.
      - Sync side pulls from that Queue.

    Notes:
      - Uses a bounded Queue for soft backpressure: if the sync consumer
        is slower than the async producer, the pump coroutine will block
        on q.put(), which in turn pauses the background loop thread.
      - The returned iterator has a .close() method that cancels the pump
        early, so callers that stop consuming can avoid draining the
        underlying async iterator to exhaustion.
    """
    q: "queue.Queue[object]" = queue.Queue(maxsize=_QUEUE_MAXSIZE)
    sentinel = object()

    async def _pump() -> None:
        try:
            async for item in agen:
                # This is a blocking call in the background loop thread,
                # providing backpressure if the sync consumer is slow.
                q.put(item)
        except asyncio.CancelledError:
            # Treat cancellation as graceful termination: no error surfaced
            # to the sync consumer, just end of stream.
            pass
        except Exception as e:
            # Surface "real" errors to the sync consumer.
            q.put(e)
        finally:
            # Always signal completion (normal or error).
            q.put(sentinel)

    loop = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(_pump(), loop)

    class SyncFromAsync(ABCIterator[T]):
        """
        Synchronous iterator view over an async iterator.

        - Blocks on q.get() to wait for new items.
        - Raises StopIteration when the async side completes or is cancelled.
        - Raises the original exception if the async side fails.
        - Exposes .close() to cancel the pump early.
        """

        def __init__(self) -> None:
            self._closed = False

        def __iter__(self) -> SyncFromAsync:
            return self

        def __next__(self) -> T:
            if self._closed:
                raise StopIteration

            item = q.get()
            if item is sentinel:
                # Ensure we don't leak the pump task.
                self.close()
                raise StopIteration

            if isinstance(item, Exception):
                # Cancel the pump (if it's still running) and re-raise.
                self.close()
                raise item

            return item  # type: ignore[return-value]

        def close(self) -> None:
            """
            Cancel the underlying pump coroutine and mark this iterator closed.

            Safe to call multiple times and safe to call after exhaustion.
            """
            if self._closed:
                return
            self._closed = True

            # Cancel the pump if still in flight.
            if not fut.done():
                fut.cancel()

            # Try to unblock any pending q.get() by ensuring a sentinel is present.
            try:
                q.put_nowait(sentinel)
            except queue.Full:
                # If the queue is already full, the pump's finally block will
                # already have queued a sentinel (or will eventually).
                pass

    return SyncFromAsync()


def _chain_on(
    src: BiStream[Any] | ABCIterable[Any] | ABCAsyncIterable[Any],
    then: Callable[
        [Any],
        BiStream[Any] | ABCIterable[Any] | ABCAsyncIterable[Any] | None,
    ],
    pred: Optional[Callable[[Any], bool]] = None,
) -> BiStream[Any]:
    stream = BiStream.ensure(src)

    if pred is None:

        def pred(ev: Any) -> bool:  # default
            return isinstance(ev, dict) and ev.get("type") == "final"

    def _gen():
        final_ev: Any | None = None

        # Drain src synchronously
        for ev in stream:
            if pred(ev):
                final_ev = ev
            yield ev

        # Chain follow-up if any
        if final_ev is not None:
            follow = then(final_ev)
            if follow is None:
                return
            follow_stream = BiStream.ensure(follow)
            for ev2 in follow_stream:
                yield ev2

    return BiStream(_gen())


def sequentially(
    src: BiStream[Any] | ABCIterable[Any] | ABCAsyncIterable[Any],
    thens: Iterable[
        Callable[
            [Any],
            BiStream[Any] | ABCIterable[Any] | ABCAsyncIterable[Any] | None,
        ]
    ],
    pred: Optional[Callable[[Any], bool]] = None,
) -> BiStream[Any]:
    """
    Apply a sequence of `then`-steps to a stream, in order.

    Roughly:

        sequentially(src, [f1, f2, f3]) ==
            _chain_on(_chain_on(_chain_on(src, f1), f2), f3)
    """
    return reduce(lambda acc, f: _chain_on(acc, f, pred=pred), thens, src)


class BiStream(Generic[T], ABCIterator[T], ABCAsyncIterator[T]):
    """
    Bidirectional stream wrapper.

    Exactly one underlying iterator, but we only adapt async->sync
    lazily when someone actually uses the sync interface.
    """

    def __init__(
        self,
        src: ABCIterable[T] | ABCAsyncIterable[T] | BiStream[T],
    ):
        # Mode is per-instance: "sync", "async" or None (not yet consumed).
        self._mode: str | None = None

        if isinstance(src, BiStream):
            # Reuse underlying iterators (sharing consumption state).
            self._src: ABCIterable[T] | ABCAsyncIterable[T] = src._src
            self._sync_iter: ABCIterator[T] | None = src._sync_iter
            self._async_iter: ABCAsyncIterator[T] | None = src._async_iter
            self._mode = src._mode
            return

        self._src = src

        async def _sync_to_async(
            sync_iter: ABCIterator[T],
        ) -> ABCAsyncIterator[T]:
            """
            Async wrapper over a synchronous iterator.

            NOTE:
              - This runs `next(sync_iter)` directly in the event loop
                thread, so if it blocks, your event loop is blocked.
            """
            last = time.monotonic()
            warned = False

            for item in sync_iter:
                now = time.monotonic()
                delta = now - last
                last = now

                if not warned and delta > _SYNC_ASYNC_WARN_THRESHOLD:
                    warned = True
                    logger.warning(
                        "BiStream: sync iterator %r blocked the event loop for "
                        "%.3fs when consumed asynchronously. This likely "
                        "indicates a mis-specified producer that should be "
                        "async further up the stack.",
                        sync_iter,
                        delta,
                    )

                yield item

        if isinstance(src, ABCAsyncIterable):
            base_async = src.__aiter__()  # type: ignore[assignment]
            self._async_iter: ABCAsyncIterator[T] | None = base_async
            # LAZY: don't build the sync adapter yet
            self._sync_iter: ABCIterator[T] | None = None
        elif isinstance(src, ABCIterable):
            base_sync = iter(src)
            self._sync_iter = base_sync
            self._async_iter = _sync_to_async(base_sync)
        else:
            raise TypeError("BiStream source is neither iterable nor async iterable")

    def _set_mode(self, mode: str) -> None:
        """
        Enforce that a BiStream instance is consumed in a single mode
        ("sync" or "async") for its entire lifetime.
        """
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
        src: BiStream[T] | ABCIterable[T] | ABCAsyncIterable[T],
    ) -> BiStream[T]:
        if isinstance(src, BiStream):
            return src
        return cls(src)

    # ---- sync side ----

    def __iter__(self) -> BiStream[T]:
        self._set_mode("sync")
        if self._sync_iter is None:
            # We were built from an async source; make the bridge *now*.
            assert self._async_iter is not None
            self._sync_iter = aiter_to_iter(self._async_iter)
        return self

    def __next__(self) -> T:
        self._set_mode("sync")
        if self._sync_iter is None:
            # Allow next() without an explicit iter() first.
            assert self._async_iter is not None
            self._sync_iter = aiter_to_iter(self._async_iter)
        return next(self._sync_iter)

    # ---- async side ----

    def __aiter__(self) -> BiStream[T]:
        self._set_mode("async")
        return self

    async def __anext__(self) -> T:
        self._set_mode("async")
        assert self._async_iter is not None
        return await self._async_iter.__anext__()

    def then(
        self: BiStream[T],
        fn: Callable[[T], BiStream[U] | ABCIterable[U] | ABCAsyncIterable[U] | None],
        pred: Optional[Callable[[T], bool]] = None,
    ) -> BiStream[T | U]:
        return _chain_on(self, fn, pred=pred)
