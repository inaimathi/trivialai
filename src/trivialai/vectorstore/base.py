import hashlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..embedding.core import Embedder

Vector = List[float]
Metadata = Dict[str, str]
SearchResult = Dict[str, Any]


class VectorStore(ABC):
    @abstractmethod
    def __init__(self, config: Optional[str]):
        pass

    @abstractmethod
    def listCollections(self) -> List[str]:
        pass

    @abstractmethod
    def getCollection(self, name: str) -> "Collection":
        pass

    @abstractmethod
    def createCollection(self, name: str, embedder: Embedder) -> "Collection":
        pass


class Collection(ABC):
    @abstractmethod
    def __init__(self, name: str, collection: Any):
        pass

    @abstractmethod
    def embedding(self, thing: Any, metadata: Optional[Metadata] = None) -> Vector:
        pass

    @abstractmethod
    def insert(self, thing: Any, metadata: Optional[Metadata] = None) -> Vector:
        pass

    @abstractmethod
    def lookup(self, vector: Vector) -> Optional[SearchResult]:
        pass

    @abstractmethod
    def queryBy(
        self,
        content: Optional[Any] = None,
        vector: Optional[Vector] = None,
        maxResults: Optional[int] = None,
        maxTokens: Optional[int] = None,
        filter: Optional[Callable[[Metadata], bool]] = None,
    ) -> List[SearchResult]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def deleteBy(
        self,
        content: Optional[Any] = None,
        vector: Optional[Vector] = None,
    ) -> bool:
        pass

    @abstractmethod
    def iter_rows(self, batch_size: int = 256) -> Iterable[SearchResult]:
        pass

    def doc_id(self, text: str) -> str:
        """
        Compute a stable ID for a given document text.

        Default: SHA-256 hash of the text. Applications that care about
        provenance can encode it into the text; backends that want
        different semantics can override this method.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
