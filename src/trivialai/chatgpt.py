# src/trivialai/chatgpt.py
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult


def _is_text_chat_model(model_id: str) -> bool:
    """
    Best-effort filtering of OpenAI's /v1/models catalogue down to models
    useful to this text-only Chat Completions adapter.

    /v1/models does not expose a clean input/output-modality capability
    field, so exclude known non-text/specialized families and retain the
    GPT / ChatGPT / o-series model families.
    """
    value = (model_id or "").strip().lower()

    if not value:
        return False

    excluded = (
        "embedding",
        "moderation",
        "realtime",
        "transcribe",
        "transcription",
        "whisper",
        "tts",
        "audio",
        "dall-e",
        "image",
        "sora",
        "computer-use",
    )

    if any(fragment in value for fragment in excluded):
        return False

    if value.startswith(
        (
            "gpt-",
            "chatgpt-",
        )
    ):
        return True

    # Reasoning models use names such as o1, o3-mini, o4-mini, etc.
    if len(value) >= 2 and value[0] == "o" and value[1].isdigit():
        return True

    return False


class ChatGPT(LLMMixin, FilesystemMixin):
    """
    OpenAI Chat Completions client with sync/async + NDJSON-style streaming.

    Streaming event schema:
      - {"type":"start", "provider":"openai", "model":"..."}
      - {"type":"delta", "text":"...", "scratchpad": ""}
      - {"type":"end", "content":"...", "scratchpad": None, "tokens": int}
      - {"type":"error", "message":"..."}

    `model=None` is supported for discovery-only instances. Call `models()`
    to list available text/chat models, then construct a generation instance
    with the selected model.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        anthropic_version: Optional[
            str
        ] = None,  # kept for signature compatibility; unused
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = 300.0,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.max_tokens = max_tokens or 4096
        self.version = anthropic_version or "2023-06-01"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    # ---- Model discovery ----
    def models(self) -> Dict[str, list]:
        """
        Return text/chat models visible to this API key.

        OpenAI's Models API also returns embeddings, audio, image,
        moderation, realtime, and other specialized models. This adapter
        implements text Chat Completions only, so those are filtered out.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        with httpx.Client(timeout=self.timeout) as client:
            res = client.get(
                "https://api.openai.com/v1/models",
                headers=headers,
            )

        res.raise_for_status()

        body = res.json()

        text_models = []

        for row in body.get("data") or []:
            if not isinstance(row, dict):
                continue

            model_id = str(row.get("id") or "")

            if not _is_text_chat_model(model_id):
                continue

            entry = dict(row)

            # The generic trivialai consumer can use `label` when present.
            entry["label"] = model_id

            text_models.append(entry)

        text_models.sort(key=lambda row: str(row.get("id") or "").lower())

        return {
            "text": text_models,
        }

    # ---- Sync full-generate (compat) ----
    def generate(
        self,
        system: str,
        prompt: str,
    ) -> LLMResult:
        if not self.model:
            raise ValueError("model is not set; " "select a model before generation")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }

        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=body,
            )

        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"]

            return LLMResult(
                res,
                content,
                None,
            )

        return LLMResult(
            res,
            None,
            None,
        )

    # ---- Async full-generate built on top of streaming ----
    async def agenerate(
        self,
        system: str,
        prompt: str,
        images: Optional[list] = None,
    ) -> LLMResult:
        content_parts: list[str] = []

        async for ev in self.astream(
            system,
            prompt,
            images,
        ):
            if ev.get("type") == "delta":
                content_parts.append(ev.get("text") or "")

            elif ev.get("type") == "end":
                if ev.get("content") is not None:
                    content_parts = [ev["content"]]

        return LLMResult(
            raw=None,
            content="".join(content_parts),
            scratchpad=None,
        )

    # ---- True async streaming ----
    async def astream(
        self,
        system: str,
        prompt: str,
        images: Optional[list] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streams via OpenAI Chat Completions (`stream: true`).

        Emits NDJSON-style events as documented in the class docstring.
        """
        if not self.model:
            yield {
                "type": "error",
                "message": ("model is not set; " "select a model before generation"),
            }
            return

        yield {
            "type": "start",
            "provider": "openai",
            "model": self.model,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # NOTE: We accept `images` for API parity but don't translate them
        # here. The application currently exposes text modality only.
        body: Dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
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
                            "message": ("OpenAI HTTP " f"{resp.status_code}"),
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

                        delta = (
                            choices[0].get(
                                "delta",
                                {},
                            )
                            or {}
                        )

                        piece = delta.get("content") or ""

                        if piece:
                            content_buf.append(piece)

                            yield {
                                "type": "delta",
                                "text": piece,
                                "scratchpad": "",
                            }

            except httpx.HTTPError as e:
                yield {
                    "type": "error",
                    "message": str(e),
                }
                return

        final_content = "".join(content_buf)

        yield {
            "type": "end",
            "content": final_content,
            "scratchpad": None,
            "tokens": (len(final_content.split()) if final_content else 0),
        }
