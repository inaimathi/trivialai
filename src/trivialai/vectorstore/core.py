from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from ..embedding.core import Embedder
from . import base
from .chromadb import ChromaCollection

Vector = List[float]
Metadata = Dict[str, Any]
SearchResult = Dict[str, Any]


class Whim(ChromaCollection):
    """
    Ephemeral, in-memory, single-collection vector store that quacks like a Collection.
    Intended for short-lived “scratchpad” retrieval tasks.
    """

    def __init__(
        self,
        embedder: Embedder,
        name: Optional[str] = None,
        extra_metadata: Optional[Dict[str, str]] = None,
    ):
        if name is None:
            name = str(uuid.uuid4())
        client = chromadb.EphemeralClient()

        meta = {
            "embedder_config": json.dumps(embedder.to_config()),
            **(extra_metadata or {}),
        }
        collection = client.get_or_create_collection(name=name, metadata=meta)

        self._client = client
        super().__init__(name, collection, embedder)

    # Optional convenience: bulk add
    def insert_many(
        self, items: List[Any], metadatas: Optional[List[Metadata]] = None
    ) -> List[Vector]:
        docs = [str(x) for x in items]
        ids = [self.doc_id(d) for d in docs]
        embeds = [self.embedding(d) for d in docs]
        mds = metadatas if metadatas is not None else [None] * len(docs)
        self._collection.add(
            documents=docs,
            metadatas=mds,
            ids=ids,
            embeddings=embeds,
        )
        return embeds

    def reset(self) -> None:
        self._collection.delete(where={})

    def consume(
        self,
        other: base.Collection,
        batch_size: int = 256,
        meta_filter: Optional[Callable[[Metadata], bool]] = None,
        extra_meta: Optional[Metadata] = None,
        strict_embedder: bool = True,
    ) -> None:
        """
        Ingest all rows from `other` into this Whim, without re-embedding.

        - Does NOT mutate `other`.
        - Reuses embeddings from `other.iter_rows`.
        - By default, enforces embedder-config equality if `other` is a ChromaCollection.
        - IDs in this Whim are generated via self.doc_id(value).
        """
        if strict_embedder and isinstance(other, ChromaCollection):
            if other._embedder.to_config() != self._embedder.to_config():
                raise ValueError(
                    f"Embedder mismatch: cannot consume from '{other.name}' into Whim '{self.name}'"
                )

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Metadata] = []
        vecs: List[Vector] = []

        def flush():
            if not ids:
                return
            self._collection.add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=vecs,
            )
            ids.clear()
            docs.clear()
            metas.clear()
            vecs.clear()

        for row in other.iter_rows(batch_size=batch_size):
            meta = row.get("meta") or {}
            if meta_filter and not meta_filter(meta):
                continue

            doc = str(row["value"])
            vid = self.doc_id(doc)
            merged_meta = {**meta, **(extra_meta or {})}

            ids.append(vid)
            docs.append(doc)
            metas.append(merged_meta)
            vecs.append(row["vector"])

            if len(ids) >= batch_size:
                flush()

        flush()
