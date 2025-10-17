from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult


class Claude(LLMMixin, FilesystemMixin):
    """
    Anthropic Messages client with sync/async + NDJSON-style streaming.

    Streaming event schema:
      - {"type":"start", "provider":"anthropic", "model":"..."}
      - {"type":"delta", "text":"...", "scratchpad": ""}   # Claude doesn't expose <think> here
      - {"type":"end", "content":"...", "scratchpad": None, "tokens": int}
      - {"type":"error", "message":"..."}
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        anthropic_version: Optional[str] = None,
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
            "X-Api-Key": self.api_key,
            "anthropic-version": self.version,
        }
        body: Dict[str, Any] = {
            "system": system,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=body
            )

        if res.status_code == 200:
            j = res.json()
            # Typical shape: {"content":[{"type":"text","text":"..."}], ...}
            try:
                text = j["content"][0]["text"]
            except Exception:
                return LLMResult(res, None, None)
            return LLMResult(res, text, None)
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
        Streams via Anthropic Messages API (`stream: true` SSE).
        Emits NDJSON-style events as documented above.
        """
        yield {"type": "start", "provider": "anthropic", "model": self.model}

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "anthropic-version": self.version,
        }

        # NOTE: We accept `images` to match the LLMMixin signature.
        # If you add vision later, convert to content blocks with image sources.
        body: Dict[str, Any] = {
            "system": system,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        content_buf: list[str] = []

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=body,
                ) as resp:
                    if resp.status_code != 200:
                        yield {
                            "type": "error",
                            "message": f"Anthropic HTTP {resp.status_code}",
                        }
                        return

                    # Anthropic sends SSE lines with optional "event:" and "data:".
                    # We only need JSON payloads carried in "data:" lines.
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            # ignore "event:" lines (message_start, content_block_start, etc.)
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        ev_type = obj.get("type")
                        if ev_type == "content_block_delta":
                            delta = obj.get("delta", {})
                            if delta.get("type") == "text_delta":
                                piece = delta.get("text") or ""
                                if piece:
                                    content_buf.append(piece)
                                    yield {
                                        "type": "delta",
                                        "text": piece,
                                        "scratchpad": "",
                                    }
                        elif ev_type == "message_stop":
                            break
                        else:
                            # ignore other event types (message_start, content_block_start, etc.)
                            pass

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
