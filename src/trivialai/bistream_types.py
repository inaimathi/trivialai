# bistream_types.py
from __future__ import annotations

from collections.abc import AsyncIterable as ABCAsyncIterable
from collections.abc import Iterable as ABCIterable
from typing import TYPE_CHECKING, Any, Callable, TypeVar

# ---- shared TypeVars (import these from bistream.py) ----------------------

T = TypeVar("T")
U = TypeVar("U")
I = TypeVar("I")
O = TypeVar("O")

# ---- small reusable type helpers -----------------------------------------

Predicate = Callable[[T], bool]

if TYPE_CHECKING:
    # Type-checker-only: safe to build real unions and refer to BiStream.
    from .bistream import BiStream

    StreamLike = BiStream[T] | ABCIterable[T] | ABCAsyncIterable[T]
    StreamLikeAny = StreamLike[Any]

    ThenFn = Callable[[], StreamLikeAny | None] | Callable[[Any], StreamLikeAny | None]
else:
    # Runtime: keep these permissive placeholders.
    StreamLike = object  # type: ignore[assignment]
    StreamLikeAny = object  # type: ignore[assignment]
    ThenFn = Callable[..., Any]  # type: ignore[assignment]

__all__ = [
    "T",
    "U",
    "I",
    "O",
    "Predicate",
    "StreamLike",
    "StreamLikeAny",
    "ThenFn",
]
