from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import requests
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

Vector = List[float]
Metadata = dict


class Embedder(ABC):
    _registry: Dict[str, Type["Embedder"]] = {}

    @abstractmethod
    def __call__(self, thing: Any, metadata: Optional[Metadata] = None) -> Vector: ...

    @abstractmethod
    def to_config(self) -> dict: ...

    @classmethod
    def from_config(cls, config: dict) -> "Embedder":
        kind = config.get("type")
        if not kind:
            raise ValueError("Embedder config missing 'type'")
        subclass = cls._registry.get(kind)
        if not subclass:
            raise ValueError(f"Unknown embedder type: {kind}")
        config = dict(config)  # shallow copy
        config.pop("type", None)
        return subclass(**config)

    @classmethod
    def register(cls, kind: str):
        """Decorator to register a new Embedder subclass"""

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
    ):
        self.server = server
        self.model = model
        self.retries = retries

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5),
        retry=retry_if_exception_type((requests.RequestException, RuntimeError)),
        reraise=True,
    )
    def __call__(self, thing: Any, metadata: Optional[Metadata] = None) -> Vector:
        prompt = str(thing)
        data = {"model": self.model, "prompt": prompt}
        url = f"{self.server}/api/embeddings"

        res = requests.post(url, json=data)
        if res.status_code == 200:
            return res.json()["embedding"]
        elif res.status_code >= 500:
            raise RuntimeError(f"Ollama server error: {res.status_code}")
        else:
            raise ValueError(f"Embedding request failed: {res.status_code} {res.text}")

    def to_config(self) -> dict:
        return {
            "type": "ollama",
            "server": self.server,
            "model": self.model,
            "retries": self.retries,
        }
