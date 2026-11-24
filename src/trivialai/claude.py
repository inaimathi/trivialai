# src/trivialai/claude.py
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .filesystem import FilesystemMixin
from .image import encode_image_input
from .llm import LLMMixin, LLMResult

# Image formats the Messages API accepts; anything else is sent as PNG.
_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


def _user_content(prompt: str, images: Optional[list] = None) -> Any:
    """
    The user turn's content: the bare prompt, or — when images are given —
    one image block per image, in order, followed by the prompt.
    """
    if not images:
        return prompt

    blocks: list[Dict[str, Any]] = []
    for img in images:
        media_type, data = encode_image_input(img, allowed=_IMAGE_MEDIA_TYPES)
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }
        )
    blocks.append({"type": "text", "text": prompt})
    return blocks


class Claude(LLMMixin, FilesystemMixin):
    """
    Anthropic Messages client with sync/async + NDJSON-style streaming.

    Streaming event schema:
      - {"type":"start", "provider":"anthropic", "model":"..."}
      - {"type":"delta", "text":"...", "scratchpad": ""}
      - {"type":"end", "content":"...", "scratchpad": None, "tokens": int}
      - {"type":"error", "message":"..."}

    `model=None` is supported for discovery-only instances. Call `models()`
    to list models visible to the supplied Anthropic API key, then construct
    a generation instance with the selected model.

    Vision: `generate` / `stream` (and the JSON / checked variants) take
    `images=[...]`, a list of Pictures, bytes, file paths, URLs or PIL
    images.  They are sent ahead of the prompt, in order.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        anthropic_version: Optional[str] = None,
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
        Return all Claude models visible to this API key.

        Anthropic's model listing is cursor-paginated. Fetch the complete
        catalogue so callers don't silently lose models when the account
        exposes more than one page.
        """
        headers = {
            "X-Api-Key": self.api_key,
            "anthropic-version": self.version,
        }

        models = []

        after_id = None

        with httpx.Client(timeout=self.timeout) as client:
            while True:
                params: Dict[
                    str,
                    Any,
                ] = {
                    "limit": 1000,
                }

                if after_id:
                    params["after_id"] = after_id

                res = client.get(
                    "https://api.anthropic.com/v1/models",
                    headers=headers,
                    params=params,
                )

                res.raise_for_status()

                body = res.json()

                page = body.get("data") or []

                for row in page:
                    if not isinstance(
                        row,
                        dict,
                    ):
                        continue

                    model_id = row.get("id") or ""

                    if not model_id:
                        continue

                    entry = dict(row)

                    entry["label"] = row.get("display_name") or model_id

                    models.append(entry)

                if not body.get("has_more"):
                    break

                next_after = body.get("last_id") or (
                    page[-1].get("id")
                    if page
                    and isinstance(
                        page[-1],
                        dict,
                    )
                    else None
                )

                # Avoid looping forever if the service sends a malformed
                # pagination response.
                if not next_after or next_after == after_id:
                    break

                after_id = next_after

        return {
            "text": models,
        }

    def _build_body(
        self,
        system: str,
        prompt: str,
        images: Optional[list] = None,
        *,
        stream: bool = False,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "system": system,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": _user_content(prompt, images),
                }
            ],
        }
        if stream:
            body["stream"] = True
        return body

    # ---- Sync full-generate (compat) ----
    def generate(
        self,
        system: str,
        prompt: str,
        images: Optional[list] = None,
    ) -> LLMResult:
        if not self.model:
            raise ValueError("model is not set; " "select a model before generation")

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "anthropic-version": self.version,
        }

        body = self._build_body(system, prompt, images)

        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=body,
            )

        if res.status_code == 200:
            j = res.json()

            try:
                text = j["content"][0]["text"]

            except Exception:
                return LLMResult(
                    res,
                    None,
                    None,
                )

            return LLMResult(
                res,
                text,
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
        Streams via Anthropic Messages API (`stream: true` SSE).

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
            "provider": "anthropic",
            "model": self.model,
        }

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "anthropic-version": self.version,
        }

        try:
            # Off the event loop: preparing images can read files or URLs.
            body = await asyncio.to_thread(
                self._build_body, system, prompt, images, stream=True
            )
        except Exception as e:
            # An unreadable image is reported the way other failures are.
            yield {
                "type": "error",
                "message": f"could not prepare images: {e}",
            }
            return

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
                            "message": ("Anthropic HTTP " f"{resp.status_code}"),
                        }
                        return

                    # Anthropic sends SSE lines with optional event: and
                    # data: lines. The JSON payloads we need are carried
                    # by data:.
                    async for line in resp.aiter_lines():
                        if not line:
                            continue

                        if not line.startswith("data:"):
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
                            delta = (
                                obj.get(
                                    "delta",
                                    {},
                                )
                                or {}
                            )

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
