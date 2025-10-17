# async_utils.py
import asyncio
import queue
import threading
from typing import Any, AsyncIterator, Callable, Iterator

__loop = None
__thread = None


def _ensure_loop():
    global __loop, __thread
    if __loop and __loop.is_running():
        return __loop
    __loop = asyncio.new_event_loop()

    def _runner():
        asyncio.set_event_loop(__loop)
        __loop.run_forever()

    __thread = threading.Thread(target=_runner, name="trivialai-bg-loop", daemon=True)
    __thread.start()
    return __loop


def run_async(coro):
    """Run a coroutine from ANY context, returning its result."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # no running loop -> safe to run directly
        return asyncio.run(coro)
    # already in a loop -> bounce to background loop thread
    bg = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, bg)
    return fut.result()


def aiter_to_iter(agen: AsyncIterator[Any]) -> Iterator[Any]:
    """Bridge an async generator into a sync generator safely."""
    q: "queue.Queue[Any]" = queue.Queue()
    sentinel = object()

    async def _pump():
        try:
            async for item in agen:
                q.put(item)
        except Exception as e:
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
