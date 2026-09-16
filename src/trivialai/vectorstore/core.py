# src/trivialai/vectorstore/core.py
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..embedding.core import Embedder
from . import base
from .chromadb import ChromaDB

Vector = List[float]
Metadata = Dict[str, Any]
SearchResult = Dict[str, Any]


class Whim(base.Collection):
    """
    Ephemeral, single-collection vector store that quacks like a Collection.

    Whim deliberately operates through the normal vector-store adapter rather
    than talking directly to the backing database implementation.
    """

    def __init__(
        self,
        embedder: Embedder,
        name: Optional[str] = None,
        extra_metadata: Optional[Dict[str, str]] = None,
    ):
        if name is None:
            name = str(uuid.uuid4())

        self.name = name
        self._embedder = embedder
        self.extra_metadata = dict(extra_metadata or {})

        self._store = ChromaDB()
        self._collection = self._store.createCollection(
            name=name,
            embedder=embedder,
        )

    def embedding(
        self,
        thing: Any,
        metadata: Optional[Metadata] = None,
    ) -> Vector:
        return self._collection.embedding(
            thing,
            metadata=metadata,
        )

    def insert(
        self,
        thing: Any,
        metadata: Optional[Metadata] = None,
    ) -> Vector:
        return self._collection.insert(
            thing,
            metadata=metadata,
        )

    def upsert(
        self,
        thing: Any,
        metadata: Optional[Metadata] = None,
    ) -> Vector:
        return self._collection.upsert(
            thing,
            metadata=metadata,
        )

    def insert_vector(
        self,
        thing: Any,
        vector: Vector,
        metadata: Optional[Metadata] = None,
    ) -> None:
        self._collection.insert_vector(
            thing,
            vector,
            metadata=metadata,
        )

    def lookup(
        self,
        *,
        id: Optional[str] = None,
        vector: Optional[Vector] = None,
    ) -> SearchResult:
        return self._collection.lookup(
            id=id,
            vector=vector,
        )

    def queryBy(
        self,
        content: Optional[Any] = None,
        vector: Optional[Vector] = None,
        maxResults: Optional[int] = None,
        maxTokens: Optional[int] = None,
        filter: Optional[Callable[[Metadata], bool]] = None,
    ) -> List[SearchResult]:
        return self._collection.queryBy(
            content=content,
            vector=vector,
            maxResults=maxResults,
            maxTokens=maxTokens,
            filter=filter,
        )

    def __len__(self) -> int:
        return len(self._collection)

    def deleteBy(
        self,
        content: Optional[Any] = None,
        vector: Optional[Vector] = None,
        id: Optional[str] = None,
    ) -> bool:
        return self._collection.deleteBy(
            content=content,
            vector=vector,
            id=id,
        )

    def iter_rows(
        self,
        batch_size: int = 256,
    ) -> Iterable[SearchResult]:
        return self._collection.iter_rows(
            batch_size=batch_size,
        )

    def doc_id(self, text: str) -> str:
        return self._collection.doc_id(text)

    def insert_many(
        self,
        items: List[Any],
        metadatas: Optional[List[Optional[Metadata]]] = None,
    ) -> List[Vector]:
        if metadatas is None:
            metadatas = [None] * len(items)

        if len(items) != len(metadatas):
            raise ValueError(
                "insert_many() requires the same number of items and metadatas"
            )

        return [
            self.insert(item, metadata=metadata)
            for item, metadata in zip(items, metadatas)
        ]

    def reset(self) -> None:
        """
        Delete every row in this Whim through the Collection API.
        """
        ids = [row["id"] for row in self.iter_rows()]

        for row_id in ids:
            self.deleteBy(id=row_id)

    def _other_embedder_config(
        self,
        other: base.Collection,
    ) -> Optional[Dict[str, Any]]:
        """
        Best-effort embedder-config discovery.

        Collection does not currently expose embedder configuration as part
        of its public interface. ChromaCollection and Whim both retain their
        embedder as `_embedder`, so preserve the existing strict validation
        where that information is available.

        Unknown Collection implementations simply cannot be checked here.
        """
        embedder = getattr(other, "_embedder", None)
        if embedder is None:
            return None

        to_config = getattr(embedder, "to_config", None)
        if not callable(to_config):
            return None

        return to_config()

    def consume(
        self,
        other: base.Collection,
        batch_size: int = 256,
        meta_filter: Optional[Callable[[Metadata], bool]] = None,
        extra_meta: Optional[Metadata] = None,
        strict_embedder: bool = True,
    ) -> None:
        """
        Ingest all rows from `other` without re-embedding.

        - Does NOT mutate `other`.
        - Reuses embeddings returned by `other.iter_rows()`.
        - By default, enforces embedder-config equality where the source
          Collection exposes enough information to do so.
        - IDs in this Whim are generated normally from document content.
        """
        if strict_embedder:
            other_config = self._other_embedder_config(other)

            if other_config is not None and other_config != self._embedder.to_config():
                raise ValueError(
                    f"Embedder mismatch: cannot consume from "
                    f"'{other.name}' into Whim '{self.name}'"
                )

        for row in other.iter_rows(batch_size=batch_size):
            meta = row.get("meta") or {}

            if meta_filter and not meta_filter(meta):
                continue

            vector = row.get("vector")
            if vector is None:
                raise ValueError(
                    f"Cannot consume row {row.get('id')!r} from "
                    f"'{other.name}': source row has no embedding"
                )

            merged_meta = {
                **meta,
                **(extra_meta or {}),
            }

            self.insert_vector(
                row["value"],
                vector,
                metadata=merged_meta,
            )
