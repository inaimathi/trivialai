class BiStream(Generic[T], ABCIterator[T], ABCAsyncIterator[T]):
    """
    Bidirectional stream wrapper.

    Wraps:
      - a synchronous Iterable[T]
      - an AsyncIterable[T]
      - another BiStream[T]

    and acts as BOTH:
      - Iterator[T]        -> usable with `for`, `next()`, list(), itertools, etc.
      - AsyncIterator[T]   -> usable with `async for`.

    Semantics:
      - Single-shot: once consumed, it's exhausted.
      - A BiStream instance must be consumed EITHER synchronously OR
        asynchronously, not both. Attempting to mix modes raises at runtime.
      - Exactly ONE canonical underlying iterator, chosen by source type:
          * async source   -> one AsyncIterator; sync side adapts via aiter_to_iter
          * sync source    -> one Iterator; async side is an async wrapper
      - If constructed from another BiStream, we reuse its underlying
        iterators (sharing partial-consumption progress).
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
            self._sync_iter: ABCIterator[T] = src._sync_iter
            self._async_iter: ABCAsyncIterator[T] = src._async_iter
            self._mode = src._mode
            return

        self._src = src

        async def _sync_to_async(
            sync_iter: ABCIterator[T],
        ) -> ABCAsyncIterator[T]:
            # NOTE: This assumes that `next(sync_iter)` is not doing any
            # heavy blocking work. If it is, using BiStream in async mode
            # will block the event loop. That tradeoff is accepted and
            # documented; callers should implement truly async producers
            # for non-blocking behaviour.
            for item in sync_iter:
                yield item

        if isinstance(src, ABCAsyncIterable):
            base_async = src.__aiter__()  # type: ignore[assignment]
            self._async_iter = base_async
            self._sync_iter = aiter_to_iter(base_async)
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
        """
        Idempotent constructor: if src is already a BiStream, return it;
        otherwise wrap it.

        Prefer this to calling BiStream(...) directly in most code, so that
        you don't accidentally double-wrap a BiStream.
        """
        if isinstance(src, BiStream):
            return src
        return cls(src)

    # ---- sync side ----

    def __iter__(self) -> BiStream[T]:
        self._set_mode("sync")
        return self

    def __next__(self) -> T:
        # Ensure the mode is set and consistent even if someone calls next()
        # directly without going through iter().
        self._set_mode("sync")
        return next(self._sync_iter)

    # ---- async side ----

    def __aiter__(self) -> BiStream[T]:
        self._set_mode("async")
        return self

    async def __anext__(self) -> T:
        # Ensure the mode is set and consistent even if someone calls __anext__()
        # directly without going through __aiter__().
        self._set_mode("async")
        return await self._async_iter.__anext__()
