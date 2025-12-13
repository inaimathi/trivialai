# tests/test_bistream.py
import asyncio
import logging
import time
import unittest

from src.trivialai import bistream


class TestDual(unittest.TestCase):
    def test_dual_returns_fresh_iterators(self):
        calls = {"gen": 0, "agen": 0}

        def gen():
            calls["gen"] += 1
            yield 1

        async def agen():
            calls["agen"] += 1
            yield 2

        d = bistream.Dual(gen, agen)

        self.assertEqual(list(d), [1])
        self.assertEqual(list(d), [1])
        self.assertEqual(calls["gen"], 2)

        async def run():
            out1 = [x async for x in d]
            out2 = [x async for x in d]
            return out1, out2

        out1, out2 = asyncio.run(run())
        self.assertEqual(out1, [2])
        self.assertEqual(out2, [2])
        self.assertEqual(calls["agen"], 2)


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
        with self.assertRaises(StopIteration):
            next(it)

    def test_close_is_idempotent_and_does_not_raise(self):
        async def agen():
            for i in range(10):
                yield i

        it = bistream.aiter_to_iter(agen())
        self.assertEqual(next(it), 0)
        self.assertEqual(next(it), 1)

        it.close()
        it.close()

        with self.assertRaises(StopIteration):
            next(it)


class TestIsType(unittest.TestCase):
    def test_isType_matches_dict_type(self):
        p = bistream.isType("final")
        self.assertTrue(p({"type": "final"}))
        self.assertFalse(p({"type": "delta"}))
        self.assertFalse(p("final"))
        self.assertFalse(p({"notype": "final"}))


class TestBiStreamBasics(unittest.TestCase):
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

        self.assertEqual(next(bs1), 1)

        bs2 = bistream.BiStream(bs1)
        self.assertEqual(list(bs2), [2, 3])

        with self.assertRaises(StopIteration):
            next(bs1)

    def test_async_source_sync_consumption_via_bridge(self):
        async def agen():
            for i in range(3):
                yield i

        bs = bistream.BiStream(agen())
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
        bs = bistream.BiStream([1, 2, 3])

        self.assertEqual(next(bs), 1)

        with self.assertRaises(RuntimeError):
            async for _ in bs:  # pragma: no cover
                pass

    async def test_mode_guard_async_then_sync_raises(self):
        async def agen():
            for i in range(2):
                yield i

        bs = bistream.BiStream(agen())

        out = []
        async for x in bs:
            out.append(x)
            break
        self.assertEqual(out, [0])

        with self.assertRaises(RuntimeError):
            next(bs)

    async def test_next_direct_sets_mode_and_conflicts_with_async(self):
        bs = bistream.BiStream([1, 2, 3])

        self.assertEqual(next(bs), 1)

        with self.assertRaises(RuntimeError):
            async for _ in bs:  # pragma: no cover
                pass


class TestBiStreamSyncToAsyncWarning(unittest.IsolatedAsyncioTestCase):
    async def test_sync_to_async_blocking_logs_warning(self):
        old_threshold = bistream._SYNC_ASYNC_WARN_THRESHOLD
        try:
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
                    time.sleep(self.delay)
                    return self._i

                def __repr__(self):
                    return f"<SlowIter count={self.count} delay={self.delay}>"

            bs = bistream.BiStream(SlowIter())

            with self.assertLogs(bistream.logger, level=logging.WARNING) as cm:
                out = [x async for x in bs]

            self.assertEqual(out, [1, 2])
            joined = "\n".join(cm.output)
            self.assertIn("BiStream: sync iterator", joined)
            self.assertIn("blocked the event loop", joined)
        finally:
            bistream._SYNC_ASYNC_WARN_THRESHOLD = old_threshold


class TestThenChain(unittest.TestCase):
    def test_then_chains_on_sync_termination_and_passes_done(self):
        def gen():
            yield 1
            yield 2
            return "DONE"

        seen = {"done": None}

        def follow(done):
            seen["done"] = done
            return [done, 3]

        out = list(bistream.BiStream(gen()).then(follow))
        self.assertEqual(out, [1, 2, "DONE", 3])
        self.assertEqual(seen["done"], "DONE")

    def test_then_chains_on_async_termination_and_passes_done(self):
        # Use an async-iterator class so we can attach a StopAsyncIteration arg
        class AIter:
            def __init__(self):
                self.i = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.i == 0:
                    self.i += 1
                    return "a"
                raise StopAsyncIteration("DONE")

        async def run():
            seen = {"done": None}

            def follow(done):
                seen["done"] = done
                return ["x", done]

            bs = bistream.BiStream(AIter()).then(follow)
            out = [x async for x in bs]
            return out, seen["done"]

        out, done = asyncio.run(run())
        self.assertEqual(out, ["a", "x", "DONE"])
        self.assertEqual(done, "DONE")


class TestTap(unittest.TestCase):
    def test_tap_sync_calls_callback_and_preserves_events(self):
        seen = []
        src = [1, 2, 3]
        bs = bistream.tap(src, lambda ev: seen.append(ev), focus=lambda x: x % 2 == 1)
        out = list(bs)
        self.assertEqual(out, [1, 2, 3])
        self.assertEqual(seen, [1, 3])

    def test_tap_async_calls_callback_and_preserves_events(self):
        async def agen():
            for i in range(4):
                yield i

        async def run():
            seen = []
            bs = bistream.tap(
                agen(), lambda ev: seen.append(ev), ignore=lambda x: x == 2
            )
            out = [x async for x in bs]
            return out, seen

        out, seen = asyncio.run(run())
        self.assertEqual(out, [0, 1, 2, 3])
        self.assertEqual(seen, [0, 1, 3])


class TestRepeatUntil(unittest.TestCase):
    def test_repeat_until_uses_done_as_driver_preferentially(self):
        calls = {"driver": None, "count": 0}

        def base():
            yield "base1"
            return "DRIVER"

        def step(driver):
            calls["driver"] = driver
            calls["count"] += 1

            def gen2():
                yield "step1"
                yield "STOP"

            return gen2()

        out = list(
            bistream.repeat_until(
                base(),
                step,
                stop=lambda ev: ev == "STOP",
                max_iters=5,
            )
        )
        self.assertEqual(out, ["base1", "step1", "STOP"])
        self.assertEqual(calls["driver"], "DRIVER")
        self.assertEqual(calls["count"], 1)

    def test_repeat_until_early_stop_prevents_step(self):
        calls = {"count": 0}

        def base():
            yield "x"
            yield "STOP"
            yield "y"  # should never be seen

        def step(_driver):
            calls["count"] += 1
            return ["should-not-run"]

        out = list(
            bistream.repeat_until(
                base(),
                step,
                stop=lambda ev: ev == "STOP",
                max_iters=5,
            )
        )
        self.assertEqual(out, ["x", "STOP"])
        self.assertEqual(calls["count"], 0)


class TestFanOutBranchAndFanIn(unittest.TestCase):
    def test_top_level_branch_sequence(self):
        fan = bistream.branch([10, 20], lambda x: [f"{x}-a", f"{x}-b"])
        out = list(fan.sequence())
        self.assertEqual(out, ["10-a", "10-b", "20-a", "20-b"])

    def test_fanout_misuse_errors(self):
        fan = bistream.branch([1], lambda x: [x])
        with self.assertRaises(RuntimeError):
            fan.then(lambda _: [1])
        with self.assertRaises(RuntimeError):
            fan.tap(lambda _: None)
        with self.assertRaises(RuntimeError):
            fan.repeat_until(lambda _: [1])

    def test_gated_bistream_branch_sequence_passes_prefix_then_branches(self):
        prefix = bistream.BiStream(["p1", "p2"])
        fan = prefix.branch([1, 2], lambda x: [f"b{x}"])
        out = list(fan.sequence())
        self.assertEqual(out, ["p1", "p2", "b1", "b2"])

    def test_gated_bistream_branch_interleave_contains_all_and_prefix_first(self):
        prefix = bistream.BiStream(["p1", "p2"])

        def per_item(x):
            return [f"{x}-a", f"{x}-b"]

        out = list(prefix.branch([1, 2, 3], per_item).interleave(concurrency=0))
        self.assertEqual(out[:2], ["p1", "p2"])
        # Remaining may be in any order; just ensure all are present
        rest = out[2:]
        self.assertEqual(
            set(rest),
            {f"{x}-{s}" for x in (1, 2, 3) for s in ("a", "b")},
        )

    def test_interleave_concurrency_one_is_sequential_for_sync_branches(self):
        # With concurrency=1 and sync branches, interleave should behave like sequence.
        fan = bistream.branch([1, 2], lambda x: [f"{x}-a", f"{x}-b"])
        out = list(fan.interleave(concurrency=1))
        self.assertEqual(out, ["1-a", "1-b", "2-a", "2-b"])


class TestInterleaveAsyncBehavior(unittest.IsolatedAsyncioTestCase):
    async def test_interleave_async_consumption_contains_all(self):
        async def per_item(x):
            yield f"{x}-a"
            await asyncio.sleep(0.001 if x % 2 == 0 else 0.002)
            yield f"{x}-b"

        # Top-level branch over a sync list, per_item returns async iterable
        fan = bistream.branch([1, 2, 3], per_item)
        merged = fan.interleave(concurrency=2)

        out = [ev async for ev in merged]
        self.assertEqual(set(out), {f"{x}-{s}" for x in (1, 2, 3) for s in ("a", "b")})
