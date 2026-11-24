# src/trivialai/chatgpt.py
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from .filesystem import FilesystemMixin
from .image import ImageMixin, Picture, encode_image_input
from .llm import LLMMixin, LLMResult

DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Image formats Chat Completions accepts; anything else is sent as PNG.
_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


def _user_content(prompt: str, images: Optional[list] = None) -> Any:
    """
    The user turn's content: the bare prompt, or — when images are given —
    one image_url part per image (as a data URL), in order, followed by
    the prompt.
    """
    if not images:
        return prompt

    parts: list[Dict[str, Any]] = []
    for img in images:
        media_type, data = encode_image_input(img, allowed=_IMAGE_MEDIA_TYPES)
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            }
        )
    parts.append({"type": "text", "text": prompt})
    return parts


# Input formats the Images API accepts; anything else is sent as PNG.
_IMAGE_INPUT_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})


def _image_part(obj: Any) -> Dict[str, Any]:
    """An Images API input reference: ``{"image_url": "data:..."}``."""
    media_type, data = encode_image_input(obj, allowed=_IMAGE_INPUT_MEDIA_TYPES)
    return {"image_url": f"data:{media_type};base64,{data}"}


def _is_image_model(model_id: str) -> bool:
    """Best-effort: OpenAI image-generation model families."""
    value = (model_id or "").strip().lower()
    return value.startswith(("gpt-image", "chatgpt-image", "dall-e"))


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


class ChatGPT(LLMMixin, ImageMixin, FilesystemMixin):
    """
    OpenAI client: text (Chat Completions) and images (Images API), with
    sync/async + NDJSON-style streaming.

    `model` targets text generation and `image_model` targets image
    generation; the two are independent, and either may be left unset on
    an instance that only does the other.

    Text streaming event schema:
      - {"type":"start", "provider":"openai", "model":"..."}
      - {"type":"delta", "text":"...", "scratchpad": ""}
      - {"type":"end", "content":"...", "scratchpad": None, "tokens": int}
      - {"type":"error", "message":"..."}

    `model=None` is supported for discovery-only instances. Call `models()`
    to list available text/chat and image models, then construct a
    generation instance with the selected model.

    Vision: `generate` / `stream` (and the JSON / checked variants) take
    `images=[...]`, a list of Pictures, bytes, file paths, URLs or PIL
    images.  They are sent ahead of the prompt, in order.

    Images: `generate_image(prompt)` calls /images/generations;
    `generate_image(prompt, image=...)` calls /images/edits with one image
    or an ordered list of up to 16 reference images.  See `generate_image`.

    `base_url` points the client at any OpenAI-compatible server.
    """

    PROVIDER = "openai"
    MAX_REFERENCE_IMAGES = 16

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        anthropic_version: Optional[
            str
        ] = None,  # kept for signature compatibility; unused
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = 300.0,
        image_model: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.max_tokens = max_tokens or 4096
        self.version = anthropic_version or "2023-06-01"
        self.api_key = api_key
        self.model = model
        self.image_model = image_model
        self.timeout = timeout
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    # ---- Model discovery ----
    def models(self) -> Dict[str, list]:
        """
        Return the text/chat and image models visible to this API key::

            {"text": [...], "image": [...]}

        OpenAI's Models API also returns embeddings, audio, moderation,
        realtime, and other specialized models; those are filtered out.
        """
        with httpx.Client(timeout=self.timeout) as client:
            res = client.get(
                self._url("/models"),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )

        res.raise_for_status()

        body = res.json()

        text_models = []
        image_models = []

        for row in body.get("data") or []:
            if not isinstance(row, dict):
                continue

            model_id = str(row.get("id") or "")

            if _is_image_model(model_id):
                bucket = image_models
            elif _is_text_chat_model(model_id):
                bucket = text_models
            else:
                continue

            entry = dict(row)

            # The generic trivialai consumer can use `label` when present.
            entry["label"] = model_id

            bucket.append(entry)

        def _by_id(row: Dict[str, Any]) -> str:
            return str(row.get("id") or "").lower()

        text_models.sort(key=_by_id)
        image_models.sort(key=_by_id)

        return {
            "text": text_models,
            "image": image_models,
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
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": _user_content(prompt, images),
                },
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

        body = self._build_body(system, prompt, images)

        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(
                self._url("/chat/completions"),
                headers=self._headers(),
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
            "provider": self.PROVIDER,
            "model": self.model,
        }

        headers = self._headers()

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
                    self._url("/chat/completions"),
                    headers=headers,
                    json=body,
                ) as resp:
                    if resp.status_code != 200:
                        yield {
                            "type": "error",
                            "message": f"{self.PROVIDER} HTTP {resp.status_code}",
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

    # ------------------------------------------------------------------
    # Image generation (Images API)
    # ------------------------------------------------------------------

    def _build_image_request(
        self,
        prompt: str,
        references: List[Picture],
        model: str,
        options: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (path, JSON body).  No references: /images/generations.
        With references: /images/edits, which takes them as an ordered
        `images` list and an optional `mask`.
        """
        options = dict(options)
        mask = options.pop("mask", None)
        body: Dict[str, Any] = {"model": model, "prompt": prompt, **options}

        if not references:
            if mask is not None:
                raise ValueError("mask= needs an input image to apply to")
            return "/images/generations", body

        body["images"] = [_image_part(pic) for pic in references]
        if mask is not None:
            body["mask"] = _image_part(mask)
        return "/images/edits", body

    def _parse_image_response(
        self,
        payload: Any,
        *,
        image_index: int,
        output_format: Optional[str],
        metadata: Dict[str, Any],
    ) -> Picture:
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list) or not items:
            keys = sorted(payload) if isinstance(payload, dict) else payload
            raise RuntimeError(
                f"{self.PROVIDER} image model {metadata.get('model')!r} returned "
                f"no images. Response keys: {keys}"
            )
        if not (0 <= image_index < len(items)):
            raise IndexError(
                f"image_index={image_index} is out of range "
                f"({len(items)} image(s) returned)."
            )

        item = items[image_index] or {}
        if item.get("b64_json"):
            data = base64.b64decode(item["b64_json"])
        elif item.get("url"):
            # Legacy models answer with a short-lived URL instead of bytes.
            data = Picture.from_url(item["url"]).bytes()
        else:
            raise RuntimeError(
                f"{self.PROVIDER} image response item has neither b64_json nor "
                f"url: {sorted(item)}"
            )

        # SVG cannot be recognized from magic bytes, so trust what was asked for.
        media_type = item.get("media_type")
        fmt = output_format or payload.get("output_format")
        if media_type is None and fmt == "svg":
            media_type = "image/svg+xml"

        meta = dict(metadata)
        meta["num_images"] = len(items)
        if item.get("revised_prompt"):
            meta["revised_prompt"] = item["revised_prompt"]
        usage = payload.get("usage")
        if isinstance(usage, dict):
            meta["usage"] = usage
            if isinstance(usage.get("cost"), (int, float)):
                meta["cost_usd"] = usage["cost"]

        return Picture.from_bytes(data, media_type=media_type, metadata=meta)

    def generate_image(
        self,
        prompt: str,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Picture:
        """
        Generate or edit an image.

        Parameters
        ----------
        prompt:
            Text description, or the instruction for an edit.
        image:
            ``None`` for text-to-image.  Otherwise one input image or an
            ordered list of up to 16 reference images; the prompt can
            refer to them as "image 1", "image 2", ...
        model:
            Override ``self.image_model`` for this call.
        mask:
            Optional image (same size as the first input image) whose
            fully transparent pixels mark the area to edit.
        image_index:
            Which result to return when ``n`` > 1 (default 0).
        size, quality, background, output_format, n, ...:
            Any other keyword is sent as a field of the request body, e.g.
            ``background="transparent"``, ``size="1024x1024"``,
            ``quality="low"``.

        The returned Picture's ``metadata`` carries the provider, model,
        mode, prompt, number of references and, when reported, ``usage``.
        """
        effective_model = model or self.image_model
        if effective_model is None:
            raise ValueError(
                "image_model is not set; cannot run image generation. "
                "Pass image_model= to the constructor."
            )

        if kwargs.get("stream"):
            raise ValueError(
                "stream=True is not supported by generate_image; use imagestream()."
            )

        image_index = int(kwargs.pop("image_index", 0))
        references = self._coerce_image_inputs(image, what=effective_model)
        path, body = self._build_image_request(
            prompt, references, effective_model, kwargs
        )

        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(self._url(path), headers=self._headers(), json=body)

        if res.status_code != 200:
            raise RuntimeError(
                f"{self.PROVIDER} {path} failed: HTTP {res.status_code} "
                f"body={res.text[:500]!r}"
            )

        return self._parse_image_response(
            res.json(),
            image_index=image_index,
            output_format=body.get("output_format"),
            metadata={
                "provider": self.PROVIDER,
                "model": effective_model,
                "mode": "img2img" if references else "txt2img",
                "prompt": prompt,
                "references": len(references),
            },
        )
