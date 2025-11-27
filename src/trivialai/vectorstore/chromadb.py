import json
import os
from typing import Any, Callable, Dict, List, Optional

import chromadb

from ..embedding.core import Embedder
from . import base

Vector = List[float]
Metadata = Dict[str, Any]
SearchResult = Dict[str, Any]


class ChromaDB(base.VectorStore):
    def __init__(self, config: Optional[str] = None):
        if config and os.path.isdir(config):
            self.client = chromadb.PersistentClient(path=config)
        else:
            self.client = chromadb.EphemeralClient()

    def listCollections(self) -> List[str]:
        return [c.name for c in self.client.list_collections()]

    def createCollection(self, name: str, embedder: Embedder) -> "ChromaCollection":
        embedder_config_str = json.dumps(embedder.to_config())
        collection = self.client.get_or_create_collection(
            name=name, metadata={"embedder_config": embedder_config_str}
        )
        return ChromaCollection(name, collection, embedder)

    def getCollection(self, name: str) -> "ChromaCollection":
        collection = self.client.get_collection(name=name)

        metadata = collection.metadata or {}
        embedder_config_str = metadata.get("embedder_config")

        if not embedder_config_str:
            raise ValueError(
                f"No embedder config found in metadata for collection '{name}'"
            )

        try:
            config = json.loads(embedder_config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid embedder config in collection metadata: {e}")

        embedder = Embedder.from_config(config)
        return ChromaCollection(name, collection, embedder)


class ChromaCollection(base.Collection):
    def __init__(self, name: str, collection: Any, embedder: Embedder):
        self.name = name
        self._collection = collection
        self._embedder = embedder

    def embedding(self, thing: Any, metadata: Optional[Metadata] = None) -> Vector:
        return self._embedder(thing)

    def insert(self, thing: Any, metadata: Optional[Metadata] = None) -> Vector:
        text = str(thing)
        vector = self.embedding(thing)
        self._collection.add(
            documents=[text],
            metadatas=[metadata] if metadata is not None else None,
            ids=[self.doc_id(text)],
            embeddings=[vector],
        )
        return vector

    def lookup(self, vector: Vector) -> Optional[SearchResult]:
        results = self._collection.query(
            query_embeddings=[vector],
            n_results=1,
            include=["embeddings", "documents", "metadatas", "distances"],
        )
        if not results["documents"][0]:
            return None

        top_vec = results["embeddings"][0][0]
        if vector == top_vec:
            doc = results["documents"][0][0]
            meta = results["metadatas"][0][0]
            distance = results["distances"][0][0]
            return {
                "value": doc,
                "vector": top_vec,
                "meta": meta,
                "distance": distance,
                "id": self.doc_id(doc),  # derive from content
            }
        return None

    def queryBy(
        self,
        content: Optional[Any] = None,
        vector: Optional[Vector] = None,
        maxResults: Optional[int] = None,
        maxTokens: Optional[int] = None,
        filter: Optional[Callable[[Metadata], bool]] = None,
    ) -> List[SearchResult]:
        n = maxResults or 25

        if content is not None:
            query_vec = self.embedding(content)
        elif vector is not None:
            query_vec = vector
        else:
            return []

        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=n,
            include=["embeddings", "documents", "metadatas", "distances"],
        )

        matches: List[SearchResult] = []
        docs = results["documents"][0]
        vecs = results["embeddings"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc, vec, meta, score in zip(docs, vecs, metas, dists):
            if filter and not filter(meta):
                continue
            matches.append(
                {
                    "value": doc,
                    "vector": vec,
                    "meta": meta,
                    "distance": score,
                    # derive ID from content via your default doc_id
                    "id": self.doc_id(doc),
                }
            )

        return matches

    def __len__(self) -> int:
        return self._collection.count()

    def deleteBy(
        self,
        content: Optional[Any] = None,
        vector: Optional[Vector] = None,
    ) -> bool:
        if content is not None:
            text = str(content)
            self._collection.delete(ids=[self.doc_id(text)])
            return True
        elif vector is not None:
            result = self.lookup(vector)
            if result:
                self._collection.delete(ids=[result["id"]])
                return True
        return False

    def iter_rows(self, batch_size: int = 256):
        offset = 0
        while True:
            batch = self._collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "embeddings", "metadatas"],
            )

            docs = batch.get("documents")
            if docs is None:
                docs = []
            if not docs:
                break

            raw_metas = batch.get("metadatas")
            if raw_metas is None:
                metas = [{} for _ in docs]
            else:
                metas = raw_metas

            raw_vecs = batch.get("embeddings")
            if raw_vecs is None:
                vecs = [None for _ in docs]
            else:
                vecs = raw_vecs

            for doc, vec, meta in zip(docs, vecs, metas):
                yield {
                    "id": self.doc_id(doc),  # canonical ID from content
                    "value": doc,
                    "vector": vec,
                    "meta": meta or {},
                }

            offset += len(docs)
