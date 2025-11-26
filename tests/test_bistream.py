# tests/test_bistream.py
import asyncio
import logging
import time
import unittest

from src.trivialai import bistream


class TestAiterToIterSyncSide(unittest.TestCase):
    def test_happy_path_async_to_sync(self):
        async def agen():
            for i in range(3):
                yield i

        it = bistream.aiter_to_iter(agen())
        self.assertEqual(list(it), [0, 1, 2])

    def test_exception_propagation_from_async(self):
        class Boom(Exception):
            pass

        async def agen():
            yield 1
            raise Boom("kaboom")

        it = bistream.aiter_to_iter(agen())
        self.assertEqual(next(it), 1)
        with self.assertRaises(Boom):
            next(it)

    def test_cancelled_error_treated_as_graceful_termination(self):
        async def agen():
            yield 1
            raise asyncio.CancelledError()

        it = bistream.aiter_to_iter(agen())
        self.assertEqual(next(it), 1)
        # Next should not raise CancelledError, just StopIteration
        with self.assertRaises(StopIteration):
            next(it)

    def test_close_is_idempotent_and_does_not_raise(self):
        async def agen():
            # Small stream; we won't consume it all
            for i in range(10):
                yield i

        it = bistream.aiter_to_iter(agen())
        # Consume a couple of items
        self.assertEqual(next(it), 0)
        self.assertEqual(next(it), 1)

        # close should be callable and idempotent
        it.close()
        it.close()  # should not raise

        # After close, iterator should behave as exhausted
        with self.assertRaises(StopIteration):
            next(it)


class TestBiStreamSyncSources(unittest.TestCase):
    def test_sync_source_sync_consumption(self):
        src = [1, 2, 3]
        bs = bistream.BiStream(src)
        self.assertEqual(list(bs), [1, 2, 3])

    def test_bistream_ensure_idempotent(self):
        bs1 = bistream.BiStream([1, 2])
        bs2 = bistream.BiStream.ensure(bs1)
        self.assertIs(bs1, bs2)

    def test_bistream_from_bistream_shares_consumption(self):
        src = [1, 2, 3]
        bs1 = bistream.BiStream(src)

        # Consume one item from the first BiStream
        self.assertEqual(next(bs1), 1)

        # Create a second BiStream from the first
        bs2 = bistream.BiStream(bs1)

        # The second should see only the remaining items
        self.assertEqual(list(bs2), [2, 3])

        # The original should now be exhausted
        with self.assertRaises(StopIteration):
            next(bs1)


class TestBiStreamAsyncSources(unittest.TestCase):
    def test_async_source_sync_consumption_via_aiter_to_iter(self):
        async def agen():
            for i in range(3):
                yield i

        bs = bistream.BiStream(agen())
        # Sync iteration over async source
        self.assertEqual(list(bs), [0, 1, 2])


class TestBiStreamModes(unittest.IsolatedAsyncioTestCase):
    async def test_sync_source_async_consumption(self):
        src = [1, 2, 3]
        bs = bistream.BiStream(src)

        out = [x async for x in bs]
        self.assertEqual(out, [1, 2, 3])

    async def test_async_source_async_consumption(self):
        async def agen():
            for i in range(3):
                yield i

        bs = bistream.BiStream(agen())
        out = [x async for x in bs]
        self.assertEqual(out, [0, 1, 2])

    async def test_mode_guard_sync_then_async_raises(self):
        src = [1, 2, 3]
        bs = bistream.BiStream(src)

        # Consume once synchronously
        self.assertEqual(next(bs), 1)

        # Async consumption should now fail with a RuntimeError
        with self.assertRaises(RuntimeError):
            async for _ in bs:  # pragma: no cover (loop body shouldn't run)
                pass

    async def test_mode_guard_async_then_sync_raises(self):
        async def agen():
            for i in range(2):
                yield i

        bs = bistream.BiStream(agen())

        # Consume once asynchronously
        out = []
        async for x in bs:
            out.append(x)
            break
        self.assertEqual(out, [0])

        # Synchronous consumption should now fail with a RuntimeError
        with self.assertRaises(RuntimeError):
            next(bs)

    async def test_next_direct_sets_mode_and_conflicts_with_async(self):
        src = [1, 2, 3]
        bs = bistream.BiStream(src)

        # Call next() directly without iter(); this should still set mode=sync
        self.assertEqual(next(bs), 1)

        with self.assertRaises(RuntimeError):
            async for _ in bs:  # pragma: no cover
                pass


class TestBiStreamSyncToAsyncWarning(unittest.IsolatedAsyncioTestCase):
    async def test_sync_to_async_blocking_logs_warning(self):
        # Make the threshold tiny so the test runs quickly
        old_threshold = bistream._SYNC_ASYNC_WARN_THRESHOLD
        try:
            # 1ms threshold
            bistream._SYNC_ASYNC_WARN_THRESHOLD = 0.001

            class SlowIter:
                def __init__(self, count=2, delay=0.002):
                    self.count = count
                    self.delay = delay
                    self._i = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    if self._i >= self.count:
                        raise StopIteration
                    self._i += 1
                    # Simulate blocking work (very short)
                    time.sleep(self.delay)
                    return self._i

                def __repr__(self):
                    return f"<SlowIter count={self.count} delay={self.delay}>"

            src = SlowIter()
            bs = bistream.BiStream(src)

            # Capture warnings from the module logger
            with self.assertLogs(bistream.logger, level=logging.WARNING) as cm:
                # Consume asynchronously; this will hit _sync_to_async
                out = [x async for x in bs]

            self.assertEqual(out, [1, 2])
            # Assert that a warning about blocking sync iterator was logged
            joined = "\n".join(cm.output)
            self.assertIn("BiStream: sync iterator", joined)
            self.assertIn("blocked the event loop", joined)
        finally:
            bistream._SYNC_ASYNC_WARN_THRESHOLD = old_threshold


if __name__ == "__main__":
    unittest.main()
