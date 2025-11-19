# bedrock.py
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult


class Bedrock(LLMMixin, FilesystemMixin):
    """
    Amazon Bedrock client using the Converse / ConverseStream APIs.

    Streaming event schema (matches ChatGPT / Ollama NDJSON shape):
      - {"type":"start", "provider":"bedrock", "model": "..."}
      - {"type":"delta", "text":"...", "scratchpad": ""}   # no <think> support here
      - {"type":"end", "content":"...", "scratchpad": None, "tokens": int | None}
      - {"type":"error", "message":"..."}

    Notes
    -----
    - `model_id` can be a foundation model id *or* an inference profile id.
    - `images` is accepted for interface parity but currently ignored.
    """

    def __init__(
        self,
        model_id: str,
        *,
        region: str = "us-east-1",
        max_tokens: Optional[int] = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        aws_profile: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        additional_model_fields: Optional[Dict[str, Any]] = None,
    ):
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.additional_model_fields = additional_model_fields or {}

        # ---- Build boto3 Session with optional explicit credentials ----
        session_kwargs: Dict[str, Any] = {}

        # If explicit creds are provided, prefer them over profile.
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
            if aws_session_token:
                session_kwargs["aws_session_token"] = aws_session_token
        elif aws_profile:
            session_kwargs["profile_name"] = aws_profile

        session = boto3.Session(**session_kwargs)
        self._client = session.client("bedrock-runtime", region_name=region)

        # Precompute a base inferenceConfig; we can tweak per-call if needed.
        inference_config: Dict[str, Any] = {}
        if max_tokens is not None:
            inference_config["maxTokens"] = max_tokens
        if temperature is not None:
            inference_config["temperature"] = temperature
        if top_p is not None:
            inference_config["topP"] = top_p
        self._inference_config = inference_config or None

    # ---- Sync full-generate (compat with LLMMixin) ----

    def generate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        """
        Non-streaming generate via bedrock-runtime.converse().
        """
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]
        system_blocks: Optional[List[Dict[str, str]]] = (
            [{"text": system}] if system else None
        )

        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if self._inference_config is not None:
            kwargs["inferenceConfig"] = self._inference_config
        if self.additional_model_fields:
            kwargs["additionalModelRequestFields"] = self.additional_model_fields

        try:
            resp = self._client.converse(**kwargs)
        except (BotoCoreError, ClientError) as e:
            return LLMResult(raw=e, content=None, scratchpad=None)

        content_blocks = resp.get("output", {}).get("message", {}).get("content", [])

        text_parts: List[str] = []
        for block in content_blocks:
            txt = block.get("text")
            if isinstance(txt, str):
                text_parts.append(txt)

        content_text = "".join(text_parts).strip() if text_parts else None
        return LLMResult(raw=resp, content=content_text, scratchpad=None)

    # ---- Sync streaming via ConverseStream ----

    def stream(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        True streaming via bedrock-runtime.converse_stream().
        """
        yield {
            "type": "start",
            "provider": "bedrock",
            "model": self.model_id,
        }

        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]
        system_blocks: Optional[List[Dict[str, str]]] = (
            [{"text": system}] if system else None
        )

        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if self._inference_config is not None:
            kwargs["inferenceConfig"] = self._inference_config
        if self.additional_model_fields:
            kwargs["additionalModelRequestFields"] = self.additional_model_fields

        content_buf: List[str] = []
        usage_tokens: Optional[int] = None

        try:
            response = self._client.converse_stream(**kwargs)
        except (BotoCoreError, ClientError) as e:
            yield {"type": "error", "message": str(e)}
            return

        stream = response.get("stream")
        if stream is None:
            yield {
                "type": "error",
                "message": "Bedrock converse_stream() returned no stream",
            }
            return

        for event in stream:
            cbd = event.get("contentBlockDelta")
            if cbd:
                delta = cbd.get("delta") or {}
                txt = delta.get("text")
                if txt:
                    content_buf.append(txt)
                    yield {"type": "delta", "text": txt, "scratchpad": ""}

            md = event.get("metadata")
            if md:
                usage = md.get("usage") or {}
                usage_tokens = (
                    usage.get("outputTokens")
                    or usage.get("totalTokens")
                    or usage_tokens
                )

        final_content = "".join(content_buf)
        yield {
            "type": "end",
            "content": final_content,
            "scratchpad": None,
            "tokens": (
                usage_tokens
                if usage_tokens is not None
                else (len(final_content.split()) if final_content else 0)
            ),
        }
