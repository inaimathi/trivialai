from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult


class ChatGPT(LLMMixin, FilesystemMixin):
    """
    OpenAI Chat Completions client with sync/async + NDJSON-style streaming.

    Streaming event schema:
      - {"type":"start", "provider":"openai", "model":"..."}
      - {"type":"delta", "text":"...", "scratchpad": ""}   # ChatGPT doesn't expose <think>, keep empty
      - {"type":"end", "content":"...", "scratchpad": None, "tokens": int}
      - {"type":"error", "message":"..."}
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        anthropic_version: Optional[
            str
        ] = None,  # kept for signature compatibility; unused
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = 300.0,
    ):
        self.max_tokens = max_tokens or 4096
        self.version = anthropic_version or "2023-06-01"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    # ---- Sync full-generate (compat) ----
    def generate(self, system: str, prompt: str) -> LLMResult:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            # Keep stream False for sync path
        }
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=body,
            )

        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"]
            return LLMResult(res, content, None)
        return LLMResult(res, None, None)

    # ---- Async full-generate built on top of streaming ----
    async def agenerate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        content_parts: list[str] = []
        async for ev in self.astream(system, prompt, images):
            if ev.get("type") == "delta":
                content_parts.append(ev.get("text") or "")
            elif ev.get("type") == "end":
                if ev.get("content") is not None:
                    content_parts = [ev["content"]]
        return LLMResult(raw=None, content="".join(content_parts), scratchpad=None)

    # ---- True async streaming ----
    async def astream(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streams via OpenAI Chat Completions (`stream: true`).
        Emits NDJSON-style events as documented in the class docstring.
        """
        yield {"type": "start", "provider": "openai", "model": self.model}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # NOTE: We accept `images` for API parity but don't translate them here.
        # If you later want vision messages, convert `messages` content to a list
        # of parts with {"type":"text",...} and {"type":"image_url",...}.
        body: Dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            # Optionally include max_tokens:
            # "max_tokens": self.max_tokens,
        }

        content_buf: list[str] = []

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=body,
                ) as resp:
                    if resp.status_code != 200:
                        yield {
                            "type": "error",
                            "message": f"OpenAI HTTP {resp.status_code}",
                        }
                        return

                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        piece = delta.get("content") or ""
                        if piece:
                            content_buf.append(piece)
                            yield {"type": "delta", "text": piece, "scratchpad": ""}

            except httpx.HTTPError as e:
                yield {"type": "error", "message": str(e)}
                return

        final_content = "".join(content_buf)
        yield {
            "type": "end",
            "content": final_content,
            "scratchpad": None,
            "tokens": len(final_content.split()) if final_content else 0,
        }
