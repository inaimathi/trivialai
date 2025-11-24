# async_utils.py
from __future__ import annotations

import asyncio
import queue
import threading
from functools import wraps
from typing import (Any, AsyncIterator, Awaitable, Callable, Generic, Iterator,
                    TypeVar)

__loop: asyncio.AbstractEventLoop | None = None
__thread: threading.Thread | None = None


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure there is a single background event loop running in a daemon thread.

    Used to:
      - run coroutines from sync code when we're already inside an event loop
      - drive async generators for sync iteration (aiter_to_iter / DualStream.__iter__)
    """
    global __loop, __thread
    if __loop and __loop.is_running():
        return __loop

    loop = asyncio.new_event_loop()
    __loop = loop

    def _runner() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_runner, name="trivialai-bg-loop", daemon=True)
    t.start()
    __thread = t
    return loop


def run_async(coro: Awaitable[Any]) -> Any:
    """
    Run a coroutine from ANY context, returning its result.

    - If called from outside an event loop: uses asyncio.run(coro).
    - If called from inside an event loop: posts coro to the background loop
      and blocks the current thread until the result is available.

    This lets sync APIs call async internals safely.
    """
    try:
        # If this succeeds, we're already in an event loop.
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop -> safe to run directly in this thread.
        return asyncio.run(coro)

    # Already in a loop -> bounce to background loop thread.
    bg = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, bg)
    return fut.result()


def aiter_to_iter(agen: AsyncIterator[Any]) -> Iterator[Any]:
    """
    Bridge an async generator into a sync generator safely.

    - Schedules a pumping coroutine onto the background loop.
    - The coroutine pushes items into a Queue.
    - The caller consumes them synchronously with `for ... in ...`.

    NOTE: Errors in the async generator are surfaced as {"type":"error",...}
    events rather than raised exceptions, to keep the event-stream contract.
    """
    q: "queue.Queue[Any]" = queue.Queue()
    sentinel = object()

    async def _pump() -> None:
        try:
            async for item in agen:
                q.put(item)
        except Exception as e:  # noqa: BLE001
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(sentinel)

    bg = _ensure_loop()
    asyncio.run_coroutine_threadsafe(_pump(), bg)

    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item


# ---- Dual-stream abstraction ------------------------------------------------
T = TypeVar("T")


class DualStream(Generic[T]):
    """
    Wrapper around an async generator factory.

    - In async code:
        async for ev in dual_stream_obj: ...
      uses the native async generator on the current event loop.

    - In sync code:
        for ev in dual_stream_obj: ...
      uses aiter_to_iter(...) and the background loop.

    This lets you expose a single `.stream(...)` surface that works in both
    sync REPL/tests and async handlers.
    """

    def __init__(self, agen_factory: Callable[[], AsyncIterator[T]]) -> None:
        self._agen_factory = agen_factory

    def __aiter__(self) -> AsyncIterator[T]:
        # Async path: return a fresh async generator each time.
        return self._agen_factory()

    def __iter__(self) -> Iterator[T]:
        # Sync path: drive the async generator via the background loop.
        return aiter_to_iter(self._agen_factory())


F = TypeVar("F", bound=Callable[..., AsyncIterator[T]])


def dual_stream(fn: F) -> Callable[..., DualStream[T]]:
    """
    Decorator for async generator functions.

    Given an async generator:

        async def foo(...)-> AsyncIterator[T]: ...

    you can decorate it:

        @dual_stream
        async def foo(...)-> AsyncIterator[T]: ...

    and get a function that returns a DualStream[T] instead:

        s = foo(...)

        # async context
        async for ev in s: ...

        # sync context (REPL/tests)
        for ev in s: ...

    This is the key building block for "async-first core, sync-friendly API".
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> DualStream[T]:
        return DualStream(lambda: fn(*args, **kwargs))

    return wrapper
