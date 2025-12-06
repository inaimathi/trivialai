# bedrock.py
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult

logger = logging.getLogger(__name__)


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
        # Throttling / 429 handling
        retry_on_throttle: bool = True,
        throttle_max_attempts: int = 3,
        throttle_base_delay: float = 1.0,
        throttle_max_delay: float = 16.0,
    ):
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.additional_model_fields = additional_model_fields or {}

        self.retry_on_throttle = retry_on_throttle
        self.throttle_max_attempts = max(0, throttle_max_attempts)
        self.throttle_base_delay = max(0.0, throttle_base_delay)
        self.throttle_max_delay = max(self.throttle_base_delay, throttle_max_delay)

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

        # Error codes we treat as throttling / Too Many Requests.
        self._throttle_error_codes = (
            "ThrottlingException",
            "TooManyRequestsException",
            "TooManyRequests",
            "RequestLimitExceeded",
            "RateLimitExceeded",
        )

    # ---- Internal helpers -------------------------------------------------

    def _build_kwargs(self, system: str, prompt: str) -> Dict[str, Any]:
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

        return kwargs

    def _is_throttling_error(self, exc: BaseException) -> bool:
        """
        Best-effort detection of 429 / throttling errors from Bedrock.
        """
        if isinstance(exc, ClientError):
            err = exc.response.get("Error", {})  # type: ignore[assignment]
            code = (err.get("Code") or "").strip()
            msg = (err.get("Message") or "").strip()
            if any(code == c or code.endswith(c) for c in self._throttle_error_codes):
                return True
            if "too many requests" in msg.lower():
                return True

        # Fallback: search message text
        text = str(exc).lower()
        if "too many requests" in text or "throttlingexception" in text:
            return True
        return False

    def _converse_with_backoff(self, kwargs: Dict[str, Any]):
        """
        Synchronous converse() with local retry/backoff for throttling.
        Safe to use from sync code (generate). For async usage we rely on
        astream() instead.
        """
        delay = self.throttle_base_delay
        attempts_left = self.throttle_max_attempts

        while True:
            try:
                return self._client.converse(**kwargs)
            except (BotoCoreError, ClientError) as e:
                if not (
                    self.retry_on_throttle
                    and self._is_throttling_error(e)
                    and attempts_left > 0
                ):
                    raise

                logger.warning(
                    "Bedrock.converse throttled (%s); retrying in %.2fs (attempts left: %d)",
                    e,
                    delay,
                    attempts_left,
                )
                attempts_left -= 1
                time.sleep(delay)
                delay = min(delay * 2, self.throttle_max_delay)

    # ---- Sync full-generate (compat with LLMMixin) ----

    def generate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        """
        Non-streaming generate via bedrock-runtime.converse(), with
        local retry/backoff on throttling errors.
        """
        kwargs = self._build_kwargs(system, prompt)

        try:
            resp = self._converse_with_backoff(kwargs)
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

    # ---- Async streaming via ConverseStream (provider-level) ----

    async def astream(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Provider-level async streaming wrapper around bedrock-runtime.converse_stream().

        We run the blocking Bedrock client in a worker thread and do local
        retry/backoff on throttling errors there, so the asyncio event loop
        never blocks.
        """
        # Emit a start event immediately.
        yield {
            "type": "start",
            "provider": "bedrock",
            "model": self.model_id,
        }

        kwargs = self._build_kwargs(system, prompt)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Dict[str, Any] | object] = asyncio.Queue()
        sentinel = object()

        def _worker() -> None:
            """
            Runs in a background thread, pushes normalized events into the
            asyncio queue using call_soon_threadsafe.
            """
            delay = self.throttle_base_delay
            attempts_left = self.throttle_max_attempts

            content_buf: List[str] = []
            usage_tokens: Optional[int] = None

            try:
                # Local retry loop specifically for throttling.
                while True:
                    try:
                        response = self._client.converse_stream(**kwargs)
                        break
                    except (BotoCoreError, ClientError) as e:
                        if not (
                            self.retry_on_throttle
                            and self._is_throttling_error(e)
                            and attempts_left > 0
                        ):
                            # Non-throttling error or out of retries: surface and bail.
                            loop.call_soon_threadsafe(
                                queue.put_nowait,
                                {
                                    "type": "error",
                                    "message": str(e),
                                },
                            )
                            return

                        logger.warning(
                            "Bedrock.converse_stream throttled (%s); retrying in %.2fs (attempts left: %d)",
                            e,
                            delay,
                            attempts_left,
                        )
                        attempts_left -= 1
                        time.sleep(delay)
                        delay = min(delay * 2, self.throttle_max_delay)

                stream = response.get("stream")
                if stream is None:
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {
                            "type": "error",
                            "message": "Bedrock converse_stream() returned no stream",
                        },
                    )
                    return

                for event in stream:
                    cbd = event.get("contentBlockDelta")
                    if cbd:
                        delta = cbd.get("delta") or {}
                        txt = delta.get("text")
                        if txt:
                            content_buf.append(txt)
                            loop.call_soon_threadsafe(
                                queue.put_nowait,
                                {"type": "delta", "text": txt},
                            )

                    md = event.get("metadata")
                    if md:
                        usage = md.get("usage") or {}
                        token_count = usage.get("outputTokens") or usage.get(
                            "totalTokens"
                        )
                        if token_count is not None:
                            usage_tokens = token_count

                final_content = "".join(content_buf)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "type": "end",
                        "content": final_content,
                        "scratchpad": None,
                        "tokens": (
                            usage_tokens
                            if usage_tokens is not None
                            else (len(final_content.split()) if final_content else 0)
                        ),
                    },
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.exception("Unexpected error in Bedrock astream worker: %s", e)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "type": "error",
                        "message": str(e),
                    },
                )
            finally:
                # Always signal completion so the async side can finish.
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        # Kick off the worker thread.
        loop.run_in_executor(None, _worker)

        # Drain the queue until the sentinel is seen.
        while True:
            ev = await queue.get()
            if ev is sentinel:
                break
            yield ev
