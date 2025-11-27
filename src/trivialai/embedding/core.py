from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import httpx
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

Vector = List[float]
Metadata = Dict[str, Any]


class Embedder(ABC):
    _registry: Dict[str, Type["Embedder"]] = {}

    @abstractmethod
    def __call__(self, thing: Any, metadata: Optional[Metadata] = None) -> Vector: ...

    @abstractmethod
    def to_config(self) -> Dict[str, Any]: ...

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Embedder":
        kind = config.get("type")
        if not kind:
            raise ValueError("Embedder config missing 'type'")
        subclass = cls._registry.get(kind)
        if not subclass:
            raise ValueError(f"Unknown embedder type: {kind}")
        cfg = dict(config)  # shallow copy
        cfg.pop("type", None)
        return subclass(**cfg)

    @classmethod
    def register(cls, kind: str):
        """Decorator to register a new Embedder subclass."""

        def decorator(subclass: Type["Embedder"]):
            cls._registry[kind] = subclass
            return subclass

        return decorator


@Embedder.register("ollama")
class OllamaEmbedder(Embedder):
    def __init__(
        self,
        server: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        retries: int = 3,
        timeout: Optional[float] = 30.0,
    ):
        self.server = server
        self.model = model
        self.retries = retries
        self.timeout = timeout

    @retry(
        stop=stop_after_attempt(3),  # keep parity with previous behavior
        wait=wait_exponential(multiplier=0.5),
        retry=retry_if_exception_type((httpx.HTTPError, RuntimeError)),
        reraise=True,
    )
    def __call__(self, thing: Any, metadata: Optional[Metadata] = None) -> Vector:
        prompt = str(thing)
        data: Dict[str, Any] = {"model": self.model, "prompt": prompt}
        url = f"{self.server.rstrip('/')}/api/embeddings"

        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(url, json=data)

        if res.status_code == 200:
            body = res.json()
            # Expecting {"embedding": [floats...]}
            embedding = body.get("embedding")
            if not isinstance(embedding, list):
                raise ValueError("Embedding response missing 'embedding' list")
            return embedding  # type: ignore[return-value]
        elif res.status_code >= 500:
            if "exceeds the context length" in res.text:
                raise ValueError("Embedding failed: chunk too large for context")
            # trigger tenacity retry for transient server errors
            raise RuntimeError(f"Ollama server error: {res.status_code}")
        else:
            raise ValueError(f"Embedding request failed: {res.status_code} {res.text}")

    def to_config(self) -> Dict[str, Any]:
        return {
            "type": "ollama",
            "server": self.server,
            "model": self.model,
            "retries": self.retries,
            "timeout": self.timeout,
        }
