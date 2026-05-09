from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from .core import Embedder, Metadata, Vector

_DEFAULT_MODEL = "amazon.titan-embed-text-v2:0"
_DEFAULT_REGION = "us-east-1"
_DEFAULT_DIMENSIONS = 1024


@Embedder.register("bedrock")
class BedrockEmbedder(Embedder):
    """
    Embedder backed by Amazon Bedrock's Titan Text Embeddings V2 model.

    Auth: bearer token via `aws_bearer_token` (typically sourced from
    the AWS_BEARER_TOKEN_BEDROCK environment variable).

    The Bedrock invoke-model REST endpoint is used directly so we don't
    pull in boto3 as a hard dependency.
    """

    def __init__(
        self,
        aws_bearer_token: str,
        region: str = _DEFAULT_REGION,
        model: str = _DEFAULT_MODEL,
        dimensions: int = _DEFAULT_DIMENSIONS,
        normalize: bool = True,
        retries: int = 3,
        timeout: Optional[float] = 30.0,
    ):
        self.aws_bearer_token = aws_bearer_token
        self.region = region
        self.model = model
        self.dimensions = dimensions
        self.normalize = normalize
        self.retries = retries
        self.timeout = timeout

    @property
    def _endpoint(self) -> str:
        return (
            f"https://bedrock-runtime.{self.region}.amazonaws.com"
            f"/model/{self.model}/invoke"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5),
        retry=retry_if_exception_type((httpx.HTTPError, RuntimeError)),
        reraise=True,
    )
    def __call__(self, thing: Any, metadata: Optional[Metadata] = None) -> Vector:
        prompt = str(thing)

        body = {
            "inputText": prompt,
            "dimensions": self.dimensions,
            "normalize": self.normalize,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.aws_bearer_token}",
        }

        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(self._endpoint, json=body, headers=headers)

        if res.status_code == 200:
            response_body = res.json()
            embedding = response_body.get("embedding")
            if not isinstance(embedding, list):
                raise ValueError("Bedrock embedding response missing 'embedding' list")
            return embedding

        if res.status_code >= 500:
            raise RuntimeError(f"Bedrock server error: {res.status_code}")

        raise ValueError(
            f"Bedrock embedding request failed: {res.status_code} {res.text}"
        )

    def to_config(self) -> Dict[str, Any]:
        return {
            "type": "bedrock",
            "aws_bearer_token": self.aws_bearer_token,
            "region": self.region,
            "model": self.model,
            "dimensions": self.dimensions,
            "normalize": self.normalize,
            "retries": self.retries,
            "timeout": self.timeout,
        }
