# src/trivialai/gemini.py
"""
Google Gemini adapter for trivialai — unified text + image generation.

Replaces both the old ``gemini.py`` (text-only) and ``nanobanana.py``
(image-only) modules.  Both capabilities share a single ``genai.Client``
and the same ``generate_content`` endpoint; ``response_modalities=["IMAGE"]``
is the only meaningful wire-level difference between a text call and an image
call.  There is no reason to split them across two files.

Also replaces the deprecated ``gcp.py`` module which used
``vertexai.generative_models`` (deprecated June 2025, removal June 2026).

Auth modes
----------
1. **Gemini Developer API** (AI Studio key) — simplest::

       Gemini(api_key="AIza...")

2. **Vertex AI via service-account JSON** (file path or raw JSON string)::

       Gemini(vertex_api_creds="/path/to/sa.json",
              project="my-project", region="us-central1")

3. **Vertex AI via Application Default Credentials**::

       Gemini(project="my-project", region="us-central1", use_vertexai=True)

Model parameters
----------------
``model``
    Text / reasoning model.  Defaults to ``"gemini-3-flash-preview"``.
    Set to ``None`` if you only need image generation.

``image_model``
    Image generation model.  Defaults to ``"gemini-3.1-flash-image-preview"``
    (Nano Banana 2 — fast, high-fidelity).
    Set to ``None`` if you only need text generation.

Known image model strings::

    "gemini-3.1-flash-image-preview"   # Nano Banana 2  (default)
    "gemini-3-pro-image-preview"        # Nano Banana Pro
    "gemini-2.5-flash-image"            # Nano Banana v1

Text generation features
------------------------
* True async token-level streaming via ``astream`` / ``stream``.
* Thinking / extended reasoning: thought tokens arrive as ``part.thought=True``
  parts and are routed to ``LLMResult.scratchpad``; no tag parsing needed.
  Control via ``thinking_budget`` (token budget, ``0`` = off, ``None`` = model
  default).
* Image *input* (img2txt): pass ``images=`` to ``generate`` / ``stream``.
* Safety settings::

      Gemini(safety_settings={"hate_speech": "none", "harassment": "low"})

Image generation features
-------------------------
* txt2img and img2img through the same ``generate_content`` call.
* Image *output* parsed from ``part.inline_data`` response parts.
* Async streaming with synthetic progress event before the blocking call.
* ``image_index`` kwarg selects which image when multiple are returned.
* ``aspect_ratio`` and other ``GenerateContentConfig`` kwargs forwarded as-is.

Model discovery
---------------
::

    g = Gemini()
    info = g.models()
    # {"text": [{"name": ..., "display_name": ..., "actions": [...]}, ...],
    #  "image": [...]}

    print(g.text_model_names())   # ["models/gemini-3-flash-preview", ...]
    print(g.image_model_names())  # ["models/gemini-3.1-flash-image-preview", ...]

The split is based on whether the model's base name (after ``models/``) ends
with ``-image`` or contains ``-image-`` — Google's consistent naming convention
for image-output models — with a fallback to listing all ``generateContent``
models as text-capable.

Dependencies
------------
    pip install google-genai pillow
    pip install google-auth     # only needed for service-account auth
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from io import BytesIO
from typing import Any, AsyncIterator, Dict, List, Optional

from google import genai
from google.genai import types as genai_types
from google.oauth2.service_account import Credentials as SACredentials

from .filesystem import FilesystemMixin
from .image import ImageMixin, Picture
from .llm import LLMMixin, LLMResult

# ---------------------------------------------------------------------------
# Default models
# ---------------------------------------------------------------------------

_DEFAULT_TEXT_MODEL = "gemini-3-flash-preview"
_DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"

# Used as a sentinel to distinguish "thinking_budget not passed" from None.
_UNSET = object()


# ---------------------------------------------------------------------------
# Safety-setting helpers
# ---------------------------------------------------------------------------

_HARM_CATEGORY_MAP: Dict[str, str] = {
    "dangerous_content": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "harassment": "HARM_CATEGORY_HARASSMENT",
    "hate_speech": "HARM_CATEGORY_HATE_SPEECH",
    "sexually_explicit": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "civic_integrity": "HARM_CATEGORY_CIVIC_INTEGRITY",
}

_HARM_THRESHOLD_MAP: Dict[str, str] = {
    "off": "BLOCK_NONE",
    "none": "BLOCK_NONE",
    "low_and_above": "BLOCK_LOW_AND_ABOVE",
    "low": "BLOCK_LOW_AND_ABOVE",
    "medium_and_above": "BLOCK_MEDIUM_AND_ABOVE",
    "medium": "BLOCK_MEDIUM_AND_ABOVE",
    "high_and_above": "BLOCK_HIGH_AND_ABOVE",
    "only_high": "BLOCK_HIGH_AND_ABOVE",
    "high": "BLOCK_HIGH_AND_ABOVE",
}


def _build_safety_settings(
    safety_settings: Dict[str, Any],
) -> List[genai_types.SafetySetting]:
    unknown_cats = set(safety_settings) - set(_HARM_CATEGORY_MAP)
    if unknown_cats:
        raise ValueError(
            f"Unknown harm categories: {sorted(unknown_cats)}. "
            f"Valid: {sorted(_HARM_CATEGORY_MAP)}"
        )
    out: List[genai_types.SafetySetting] = []
    for cat_key, threshold_val in safety_settings.items():
        threshold_key = (
            str(threshold_val).lower().strip() if threshold_val is not None else "none"
        )
        if threshold_key not in _HARM_THRESHOLD_MAP:
            raise ValueError(
                f"Unknown threshold {threshold_val!r} for {cat_key!r}. "
                f"Valid: {sorted(_HARM_THRESHOLD_MAP)}"
            )
        out.append(
            genai_types.SafetySetting(
                category=_HARM_CATEGORY_MAP[cat_key],
                threshold=_HARM_THRESHOLD_MAP[threshold_key],
            )
        )
    return out


# Apply BLOCK_NONE to every category by default — callers can tighten as needed.
_DEFAULT_SAFETY: Dict[str, Any] = {k: "none" for k in _HARM_CATEGORY_MAP}


# ---------------------------------------------------------------------------
# Image-to-Part coercion (input images for img2txt and img2img)
# ---------------------------------------------------------------------------


def _to_part(obj: Any) -> genai_types.Part:
    """
    Convert an image-like object into a ``genai_types.Part``.

    Accepts: ``Picture``, ``bytes``/``bytearray``, file path (str/Path),
    or a PIL ``Image``.
    """
    if isinstance(obj, Picture):
        return genai_types.Part.from_bytes(
            data=obj.bytes(), mime_type=obj.media_type or "image/png"
        )

    if isinstance(obj, (bytes, bytearray, memoryview)):
        return genai_types.Part.from_bytes(data=bytes(obj), mime_type="image/png")

    if isinstance(obj, (str, os.PathLike)):
        with open(str(obj), "rb") as f:
            data = f.read()
        if data.startswith(b"\x89PNG"):
            mime = "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            mime = "image/webp"
        else:
            mime = "image/png"
        return genai_types.Part.from_bytes(data=data, mime_type=mime)

    # Duck-type PIL Image
    if hasattr(obj, "save") and hasattr(obj, "mode"):
        buf = BytesIO()
        fmt = getattr(obj, "format", None) or "PNG"
        obj.save(buf, format=fmt)
        return genai_types.Part.from_bytes(
            data=buf.getvalue(), mime_type=f"image/{fmt.lower()}"
        )

    raise TypeError(
        f"Cannot convert {type(obj).__name__!r} to a Gemini Part. "
        "Expected Picture, bytes, file path, or PIL Image."
    )


# ---------------------------------------------------------------------------
# Response-part helpers
# ---------------------------------------------------------------------------


def _collect_image_results(
    response: Any, *, model: str, mode: str, prompt: str
) -> List[Picture]:
    """Extract all image parts from a generate_content response."""
    results: List[Picture] = []
    for part in getattr(response, "parts", None) or []:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        img_bytes: Optional[bytes] = getattr(inline, "data", None)
        mime: Optional[str] = getattr(inline, "mime_type", None)
        if not img_bytes:
            continue
        results.append(
            Picture.from_bytes(
                img_bytes,
                media_type=mime,
                metadata={
                    "provider": "gemini",
                    "model": model,
                    "mode": mode,
                    "prompt": prompt,
                },
            )
        )
    return results


def _is_image_model(model_name: str) -> bool:
    """
    Return True if this model produces image output.

    Google's naming convention is consistent: image-output models contain
    ``-image`` in their base name (e.g. ``gemini-2.5-flash-image``,
    ``gemini-3.1-flash-image-preview``, ``gemini-3-pro-image-preview``).
    """
    base = model_name.split("/")[-1].lower()
    return "-image" in base


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class Gemini(LLMMixin, ImageMixin, FilesystemMixin):
    """
    Google Gemini adapter: text generation + image generation.

    Parameters
    ----------
    model:
        Text / reasoning model string.  ``None`` = no text generation.
    image_model:
        Image generation model string.  ``None`` = no image generation.
    api_key:
        Gemini Developer API key.  Mutually exclusive with
        ``vertex_api_creds`` / ``use_vertexai``.
    vertex_api_creds:
        Path to a service-account JSON file *or* a raw JSON string.
        Enables Vertex AI auth.
    project:
        GCP project ID (Vertex AI only).  Auto-read from SA file if omitted.
    region:
        GCP region (Vertex AI only).  Defaults to ``"us-central1"``.
    use_vertexai:
        Use Application Default Credentials for Vertex AI (no key needed).
    safety_settings:
        Dict mapping harm-category short names to threshold short names.
        Defaults to ``BLOCK_NONE`` for all categories.
    thinking_budget:
        Token budget for extended reasoning.  ``0`` = off, ``None`` = model
        default.  Can be overridden per ``generate``/``stream`` call.
    max_output_tokens:
        Maximum response tokens for text generation.  ``None`` = model default.
    timeout:
        Per-request timeout in seconds.  ``None`` = no timeout.
    """

    def __init__(
        self,
        model: Optional[str] = _DEFAULT_TEXT_MODEL,
        *,
        image_model: Optional[str] = _DEFAULT_IMAGE_MODEL,
        api_key: Optional[str] = None,
        vertex_api_creds: Optional[str] = None,
        project: Optional[str] = None,
        region: Optional[str] = None,
        use_vertexai: bool = False,
        safety_settings: Optional[Dict[str, Any]] = None,
        thinking_budget: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[float] = 300.0,
    ):
        self.model = model
        self.image_model = image_model
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self._safety_settings = _build_safety_settings(
            safety_settings if safety_settings is not None else _DEFAULT_SAFETY
        )

        credentials: Optional[Any] = None

        if vertex_api_creds is not None:
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            if os.path.isfile(vertex_api_creds):
                credentials = SACredentials.from_service_account_file(
                    vertex_api_creds, scopes=scopes
                )
                if project is None:
                    with open(vertex_api_creds) as f:
                        project = json.load(f).get("project_id")
            else:
                sa_dict = json.loads(vertex_api_creds)
                credentials = SACredentials.from_service_account_info(
                    sa_dict, scopes=scopes
                )
                if project is None:
                    project = sa_dict.get("project_id")

            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=region or "us-central1",
                credentials=credentials,
            )

        elif use_vertexai:
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=region or "us-central1",
            )

        else:
            self._client = genai.Client(api_key=api_key)

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def _build_text_contents(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
    ) -> List[genai_types.Content]:
        """Assemble user-turn contents for a text request."""
        user_parts: List[Any] = []
        if images:
            for img in images:
                user_parts.append(_to_part(img))
        user_parts.append(genai_types.Part.from_text(text=prompt))
        return [genai_types.Content(role="user", parts=user_parts)]

    def _make_text_config(
        self, system: str, **overrides: Any
    ) -> genai_types.GenerateContentConfig:
        """Build a GenerateContentConfig for a text request."""
        kwargs: Dict[str, Any] = {}
        if system:
            kwargs["system_instruction"] = system
        if self._safety_settings:
            kwargs["safety_settings"] = self._safety_settings
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens

        # thinking_budget: per-call override > instance default > omit
        budget = overrides.pop("thinking_budget", _UNSET)
        resolved = budget if budget is not _UNSET else self.thinking_budget
        if resolved is not None:
            kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=resolved
            )

        kwargs.update(overrides)
        return genai_types.GenerateContentConfig(**kwargs)

    def generate(
        self,
        system: str,
        prompt: str,
        images: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Synchronous single-shot text generation.

        Thinking tokens arrive in ``LLMResult.scratchpad``; answer tokens in
        ``LLMResult.content``.

        Parameters
        ----------
        images:
            Optional list of input images for img2txt (vision) requests.
        thinking_budget:
            Per-call override of the instance-level thinking budget.
        """
        if self.model is None:
            raise ValueError(
                "model is not set; cannot run text generation. "
                "Pass model= to the constructor."
            )
        config = self._make_text_config(system, **kwargs)
        contents = self._build_text_contents(prompt, images)

        try:
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except Exception as exc:
            return LLMResult(exc, None, None)

        text_parts: List[str] = []
        thought_parts: List[str] = []
        for part in response.parts or []:
            if getattr(part, "thought", False):
                if part.text:
                    thought_parts.append(part.text)
            elif part.text:
                text_parts.append(part.text)

        return LLMResult(
            response,
            "".join(text_parts).strip(),
            "".join(thought_parts).strip() or None,
        )

    async def astream(
        self,
        system: str,
        prompt: str,
        images: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        True async token-level streaming.

        Thought tokens are emitted as ``scratchpad`` on delta events;
        ``LLMMixin.stream`` accumulates both into the final ``end`` event.
        """
        if self.model is None:
            yield {"type": "error", "message": "model is not set."}
            return

        yield {"type": "start", "provider": "gemini", "model": self.model}

        config = self._make_text_config(system, **kwargs)
        contents = self._build_text_contents(prompt, images)

        try:
            async for chunk in await self._client.aio.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            ):
                text_delta = ""
                thought_delta = ""
                for part in chunk.parts or []:
                    if getattr(part, "thought", False):
                        if part.text:
                            thought_delta += part.text
                    elif part.text:
                        text_delta += part.text
                if text_delta or thought_delta:
                    yield {
                        "type": "delta",
                        "text": text_delta,
                        "scratchpad": thought_delta,
                    }
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
            return

        yield {"type": "end", "content": None}

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def _build_image_contents(
        self,
        prompt: str,
        src_img: Optional[Picture],
    ) -> List[Any]:
        """
        Assemble contents for an image generation request.

        For img2img the source image precedes the text so the model interprets
        the prompt as an editing instruction applied to the supplied image.
        """
        if src_img is None:
            return [prompt]
        return [_to_part(src_img), prompt]

    def generate_image(
        self,
        prompt: str,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Picture:
        """
        Generate or edit an image (txt2img or img2img).

        Parameters
        ----------
        prompt:
            Text description or editing instruction.
        image:
            Source image for img2img editing.  ``None`` = txt2img.
        model:
            Override ``self.image_model`` for this call.
        image_index:
            Which result to return when the response contains multiple images
            (default 0).
        aspect_ratio, temperature, seed, image_config, ...:
            Forwarded to ``GenerateContentConfig``.
        """
        effective_model = model or self.image_model
        if effective_model is None:
            raise ValueError(
                "image_model is not set; cannot run image generation. "
                "Pass image_model= to the constructor."
            )

        image_index = int(kwargs.pop("image_index", 0))
        src_img = self._coerce_image_result(image)
        mode = "img2img" if src_img is not None else "txt2img"
        contents = self._build_image_contents(prompt, src_img)

        # Only pass through known GenerateContentConfig keys; drop the rest.
        _config_keys = {
            "temperature",
            "top_p",
            "top_k",
            "candidate_count",
            "stop_sequences",
            "max_output_tokens",
            "seed",
            "aspect_ratio",
            "image_config",
            "thinking_config",
        }
        config_extra = {k: kwargs.pop(k) for k in list(kwargs) if k in _config_keys}
        config = genai_types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=self._safety_settings,
            **config_extra,
        )

        try:
            response = self._client.models.generate_content(
                model=effective_model, contents=contents, config=config
            )
        except Exception as exc:
            raise RuntimeError(
                f"Gemini image generation failed for model {effective_model!r}: {exc}"
            ) from exc

        images = _collect_image_results(
            response, model=effective_model, mode=mode, prompt=prompt
        )

        if not images:
            raise RuntimeError(
                f"Gemini model {effective_model!r} returned no images. "
                "The request may have been filtered or the response contains "
                f"only text: {getattr(response, 'text', '')!r}"
            )

        if not (0 <= image_index < len(images)):
            raise IndexError(
                f"image_index={image_index} is out of range "
                f"({len(images)} image(s) returned)."
            )

        return images[image_index]

    async def astream_image(
        self,
        prompt: str,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Async streaming wrapper for image generation.

        The Gemini generateContent endpoint is single-shot (no server-sent
        progress), so this emits a synthetic progress event before the
        blocking call and an end event when the image resolves::

          {"type": "start",    "provider": "gemini", "model": ..., "mode": ...}
          {"type": "progress", "progress": 0.0, "state": "generating", ...}
          {"type": "end",      "image": Picture, "model": ..., "mode": ...}

        or on failure::

          {"type": "error", "message": "..."}
        """
        effective_model = model or self.image_model
        mode = "img2img" if image is not None else "txt2img"

        yield {
            "type": "start",
            "provider": "gemini",
            "model": effective_model,
            "mode": mode,
        }
        yield {
            "type": "progress",
            "progress": 0.0,
            "eta_relative": None,
            "state": "generating",
            "textinfo": "Waiting for Gemini image API…",
        }

        gen_task: asyncio.Task[Picture] = asyncio.create_task(
            asyncio.to_thread(self.generate_image, prompt, image, model, **kwargs)
        )

        try:
            final_img = await gen_task
        except asyncio.CancelledError:
            if not gen_task.done():
                gen_task.cancel()
                with suppress(Exception):
                    await gen_task
            raise
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
            return

        yield {
            "type": "end",
            "image": final_img,
            "content": final_img,
            "model": effective_model,
            "mode": mode,
        }

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    def models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return a dict classifying available Gemini models by capability.

        ::

            {"text":  [{"name": "models/gemini-3-flash-preview",
                        "display_name": "Gemini 3 Flash Preview",
                        "actions": ["generateContent"]}, ...],
             "image": [{"name": "models/gemini-3.1-flash-image-preview",
                        "display_name": "Nano Banana 2",
                        "actions": ["generateContent"]}, ...]}

        Classification rule: models whose base name (after ``models/``)
        contains ``-image`` are placed in ``"image"``; all other
        ``generateContent``-capable models go into ``"text"``.
        """
        text_models: List[Dict[str, Any]] = []
        image_models: List[Dict[str, Any]] = []

        for m in self._client.models.list():
            actions: List[str] = list(getattr(m, "supported_actions", None) or [])
            if "generateContent" not in actions:
                continue

            entry = {
                "name": getattr(m, "name", ""),
                "display_name": getattr(m, "display_name", ""),
                "actions": actions,
            }

            if _is_image_model(entry["name"]):
                image_models.append(entry)
            else:
                text_models.append(entry)

        return {"text": text_models, "image": image_models}

    def text_model_names(self) -> List[str]:
        """Convenience: return just the name strings for text-output models."""
        return [m["name"] for m in self.models()["text"]]

    def image_model_names(self) -> List[str]:
        """Convenience: return just the name strings for image-output models."""
        return [m["name"] for m in self.models()["image"]]
