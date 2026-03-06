# src/trivialai/bedrock.py
"""
Amazon Bedrock adapter: unified text + image generation.

Text generation  → Converse / ConverseStream APIs  (unchanged from original)
Image generation → InvokeModel API                 (new)
Model discovery  → ListFoundationModels API        (new)

Two distinct boto3 service clients are held:
  self._runtime  – bedrock-runtime  (inference)
  self._control  – bedrock          (model catalogue)

Supported image model families and their invoke_model payload shapes
--------------------------------------------------------------------
Nova Canvas  (amazon.nova-canvas-*)
  txt2img: taskType TEXT_IMAGE       → textToImageParams.text
  img2img: taskType IMAGE_VARIATION  → imageVariationParams.images[0]
  response: {"images": ["<base64>", ...]}

Titan Image  (amazon.titan-image-*)
  Same taskType / param structure as Nova Canvas.
  response: {"images": ["<base64>", ...]}

Stability AI (stability.*)
  txt2img: text_prompts[{text, weight}], cfg_scale, steps, seed, samples
  img2img: same + init_image (base64), init_image_strength
  response: {"artifacts": [{"base64": "<base64>", "finishReason": "SUCCESS"}]}

All other model IDs are routed through the Nova Canvas / Titan shape as a
best-effort fallback (since Amazon's newer models largely share that schema).

Usage
-----
    from trivialai.bedrock import Bedrock

    # Text only (original behaviour preserved)
    b = Bedrock("anthropic.claude-3-5-sonnet-20241022-v2:0")
    result = b.generate("You are helpful.", "What is 2+2?")

    # Image only
    b = Bedrock(image_model_id="amazon.nova-canvas-v1:0")
    img = b.imagen("a red panda in a bamboo forest")

    # Both (model_id for text, image_model_id for images)
    b = Bedrock(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        image_model_id="amazon.nova-canvas-v1:0",
    )
    result = b.generate("system", "prompt")
    img    = b.imagen("a red panda")

    # Model discovery
    info = b.models()
    # {"text": [{"model_id": ..., "name": ..., "provider": ...}, ...],
    #  "image": [...]}
    print(b.text_model_ids())    # ["anthropic.claude...", ...]
    print(b.image_model_ids())   # ["amazon.nova-canvas-v1:0", ...]
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .filesystem import FilesystemMixin
from .image import ImageMixin, Picture
from .llm import LLMMixin, LLMResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model IDs
# ---------------------------------------------------------------------------

_DEFAULT_TEXT_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
_DEFAULT_IMAGE_MODEL = "amazon.nova-canvas-v1:0"


# ---------------------------------------------------------------------------
# Image payload builders
# ---------------------------------------------------------------------------


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _build_nova_titan_txt2img(
    prompt: str,
    *,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    number_of_images: int = 1,
    quality: str = "standard",
    cfg_scale: float = 8.0,
    seed: int = 0,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"text": prompt}
    if negative_prompt:
        params["negativeText"] = negative_prompt
    return {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": params,
        "imageGenerationConfig": {
            "numberOfImages": number_of_images,
            "quality": quality,
            "width": width,
            "height": height,
            "cfgScale": cfg_scale,
            "seed": seed,
        },
    }


def _build_nova_titan_img2img(
    prompt: str,
    src_bytes: bytes,
    *,
    negative_prompt: str = "",
    similarity_strength: float = 0.7,
    number_of_images: int = 1,
    quality: str = "standard",
    cfg_scale: float = 8.0,
    seed: int = 0,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "text": prompt,
        "images": [_b64(src_bytes)],
        "similarityStrength": similarity_strength,
    }
    if negative_prompt:
        params["negativeText"] = negative_prompt
    return {
        "taskType": "IMAGE_VARIATION",
        "imageVariationParams": params,
        "imageGenerationConfig": {
            "numberOfImages": number_of_images,
            "quality": quality,
            "cfgScale": cfg_scale,
            "seed": seed,
        },
    }


def _build_stability_txt2img(
    prompt: str,
    *,
    negative_prompt: str = "",
    cfg_scale: float = 7.0,
    steps: int = 50,
    seed: int = 0,
    samples: int = 1,
) -> Dict[str, Any]:
    text_prompts = [{"text": prompt, "weight": 1.0}]
    if negative_prompt:
        text_prompts.append({"text": negative_prompt, "weight": -1.0})
    return {
        "text_prompts": text_prompts,
        "cfg_scale": cfg_scale,
        "steps": steps,
        "seed": seed,
        "samples": samples,
    }


def _build_stability_img2img(
    prompt: str,
    src_bytes: bytes,
    *,
    negative_prompt: str = "",
    init_image_strength: float = 0.35,
    cfg_scale: float = 7.0,
    steps: int = 50,
    seed: int = 0,
    samples: int = 1,
) -> Dict[str, Any]:
    payload = _build_stability_txt2img(
        prompt,
        negative_prompt=negative_prompt,
        cfg_scale=cfg_scale,
        steps=steps,
        seed=seed,
        samples=samples,
    )
    payload["init_image"] = _b64(src_bytes)
    payload["init_image_strength"] = init_image_strength
    return payload


def _extract_images_nova_titan(body: Dict[str, Any]) -> List[bytes]:
    """Parse Nova Canvas / Titan response body → list of raw PNG bytes."""
    raw_list = body.get("images") or []
    out: List[bytes] = []
    for raw in raw_list:
        if isinstance(raw, str):
            out.append(base64.b64decode(raw))
    return out


def _extract_images_stability(body: Dict[str, Any]) -> List[bytes]:
    """Parse Stability AI response body → list of raw bytes."""
    out: List[bytes] = []
    for artifact in body.get("artifacts") or []:
        if artifact.get("finishReason") in ("SUCCESS", "CONTENT_FILTERED"):
            b64 = artifact.get("base64") or artifact.get("data") or ""
            if b64:
                out.append(base64.b64decode(b64))
    return out


# ---------------------------------------------------------------------------
# Model family routing helpers
# ---------------------------------------------------------------------------


def _is_stability(model_id: str) -> bool:
    return model_id.startswith("stability.")


def _build_image_payload(
    model_id: str,
    prompt: str,
    src_img: Optional[Picture],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Dispatch to the right payload builder based on model family."""
    if _is_stability(model_id):
        if src_img is not None:
            return _build_stability_img2img(prompt, src_img.bytes(), **kwargs)
        return _build_stability_txt2img(prompt, **kwargs)
    else:
        # Nova Canvas, Titan, and unknown Amazon-family models all share
        # the taskType / imageGenerationConfig schema.
        if src_img is not None:
            return _build_nova_titan_img2img(prompt, src_img.bytes(), **kwargs)
        return _build_nova_titan_txt2img(prompt, **kwargs)


def _extract_image_bytes(model_id: str, body: Dict[str, Any]) -> List[bytes]:
    if _is_stability(model_id):
        return _extract_images_stability(body)
    return _extract_images_nova_titan(body)


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class Bedrock(LLMMixin, ImageMixin, FilesystemMixin):
    """
    Amazon Bedrock adapter: text generation (Converse API) + image generation
    (InvokeModel API) + model catalogue (ListFoundationModels API).

    Parameters
    ----------
    model_id:
        Foundation model or inference profile ID for **text** generation.
        Can be ``None`` if you only need image generation.
    image_model_id:
        Foundation model ID for **image** generation.
        Defaults to ``"amazon.nova-canvas-v1:0"`` (Nova Canvas).
        Can be ``None`` if you only need text generation.
    region:
        AWS region (default ``"us-east-1"``).
    max_tokens, temperature, top_p:
        Text generation inference parameters.
    aws_profile, aws_access_key_id, aws_secret_access_key, aws_session_token:
        Authentication (mirrors the original Bedrock class).
    additional_model_fields:
        Extra fields forwarded to ``additionalModelRequestFields`` for text.
    retry_on_throttle, throttle_max_attempts, throttle_base_delay, throttle_max_delay:
        Retry / back-off policy for throttling errors (applied to both APIs).
    """

    def __init__(
        self,
        model_id: Optional[str] = _DEFAULT_TEXT_MODEL,
        *,
        image_model_id: Optional[str] = _DEFAULT_IMAGE_MODEL,
        region: str = "us-east-1",
        max_tokens: Optional[int] = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        aws_profile: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        additional_model_fields: Optional[Dict[str, Any]] = None,
        retry_on_throttle: bool = True,
        throttle_max_attempts: int = 3,
        throttle_base_delay: float = 1.0,
        throttle_max_delay: float = 16.0,
    ):
        self.model_id = model_id
        self.image_model_id = image_model_id
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.additional_model_fields = additional_model_fields or {}

        self.retry_on_throttle = retry_on_throttle
        self.throttle_max_attempts = max(0, throttle_max_attempts)
        self.throttle_base_delay = max(0.0, throttle_base_delay)
        self.throttle_max_delay = max(self.throttle_base_delay, throttle_max_delay)

        # ---- Build a shared boto3 session ----
        session_kwargs: Dict[str, Any] = {}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
            if aws_session_token:
                session_kwargs["aws_session_token"] = aws_session_token
        elif aws_profile:
            session_kwargs["profile_name"] = aws_profile

        session = boto3.Session(**session_kwargs)

        # bedrock-runtime — used for both converse* (text) and invoke_model (image)
        self._runtime = session.client("bedrock-runtime", region_name=region)
        # bedrock (control plane) — used only for list_foundation_models
        self._control = session.client("bedrock", region_name=region)

        # Precompute inference config for text requests.
        inference_config: Dict[str, Any] = {}
        if max_tokens is not None:
            inference_config["maxTokens"] = max_tokens
        if temperature is not None:
            inference_config["temperature"] = temperature
        if top_p is not None:
            inference_config["topP"] = top_p
        self._inference_config = inference_config or None

        self._throttle_error_codes = (
            "ThrottlingException",
            "TooManyRequestsException",
            "TooManyRequests",
            "RequestLimitExceeded",
            "RateLimitExceeded",
        )

    # ------------------------------------------------------------------
    # Throttle detection (shared by text + image paths)
    # ------------------------------------------------------------------

    def _is_throttling_error(self, exc: BaseException) -> bool:
        if isinstance(exc, ClientError):
            err = exc.response.get("Error", {})  # type: ignore[assignment]
            code = (err.get("Code") or "").strip()
            msg = (err.get("Message") or "").strip()
            if any(code == c or code.endswith(c) for c in self._throttle_error_codes):
                return True
            if "too many requests" in msg.lower():
                return True
        text = str(exc).lower()
        return "too many requests" in text or "throttlingexception" in text

    def _backoff_call(self, fn, *args, **kwargs):
        """
        Call ``fn(*args, **kwargs)`` with exponential back-off on throttling.
        Raises on non-throttling errors or when retries are exhausted.
        """
        delay = self.throttle_base_delay
        attempts_left = self.throttle_max_attempts
        while True:
            try:
                return fn(*args, **kwargs)
            except (BotoCoreError, ClientError) as e:
                if not (
                    self.retry_on_throttle
                    and self._is_throttling_error(e)
                    and attempts_left > 0
                ):
                    raise
                logger.warning(
                    "Bedrock throttled (%s); retrying in %.2fs (%d left)",
                    e,
                    delay,
                    attempts_left,
                )
                attempts_left -= 1
                time.sleep(delay)
                delay = min(delay * 2, self.throttle_max_delay)

    # ------------------------------------------------------------------
    # Text generation helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _build_text_kwargs(self, system: str, prompt: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
        }
        if system:
            kwargs["system"] = [{"text": system}]
        if self._inference_config is not None:
            kwargs["inferenceConfig"] = self._inference_config
        if self.additional_model_fields:
            kwargs["additionalModelRequestFields"] = self.additional_model_fields
        return kwargs

    # ------------------------------------------------------------------
    # Text: synchronous generate
    # ------------------------------------------------------------------

    def generate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        if self.model_id is None:
            raise ValueError(
                "model_id is not set; cannot run text generation. "
                "Pass model_id= to the constructor."
            )
        kwargs = self._build_text_kwargs(system, prompt)
        try:
            resp = self._backoff_call(self._runtime.converse, **kwargs)
        except (BotoCoreError, ClientError) as e:
            return LLMResult(raw=e, content=None, scratchpad=None)

        content_blocks = resp.get("output", {}).get("message", {}).get("content", [])
        text = "".join(
            b["text"] for b in content_blocks if isinstance(b.get("text"), str)
        ).strip()
        return LLMResult(raw=resp, content=text or None, scratchpad=None)

    # ------------------------------------------------------------------
    # Text: async streaming (unchanged from original, uses self._runtime)
    # ------------------------------------------------------------------

    async def astream(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        if self.model_id is None:
            yield {
                "type": "error",
                "message": "model_id is not set; cannot run text generation.",
            }
            return

        yield {"type": "start", "provider": "bedrock", "model": self.model_id}

        kwargs = self._build_text_kwargs(system, prompt)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def _worker() -> None:
            delay = self.throttle_base_delay
            attempts_left = self.throttle_max_attempts
            content_buf: List[str] = []
            usage_tokens: Optional[int] = None

            try:
                while True:
                    try:
                        response = self._runtime.converse_stream(**kwargs)
                        break
                    except (BotoCoreError, ClientError) as e:
                        if not (
                            self.retry_on_throttle
                            and self._is_throttling_error(e)
                            and attempts_left > 0
                        ):
                            loop.call_soon_threadsafe(
                                queue.put_nowait,
                                {"type": "error", "message": str(e)},
                            )
                            return
                        logger.warning(
                            "Bedrock.converse_stream throttled (%s); "
                            "retrying in %.2fs (%d left)",
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
                            "message": "converse_stream() returned no stream",
                        },
                    )
                    return

                for event in stream:
                    cbd = event.get("contentBlockDelta")
                    if cbd:
                        txt = (cbd.get("delta") or {}).get("text")
                        if txt:
                            content_buf.append(txt)
                            loop.call_soon_threadsafe(
                                queue.put_nowait, {"type": "delta", "text": txt}
                            )
                    md = event.get("metadata")
                    if md:
                        usage = md.get("usage") or {}
                        tok = usage.get("outputTokens") or usage.get("totalTokens")
                        if tok is not None:
                            usage_tokens = tok

                final = "".join(content_buf)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "type": "end",
                        "content": final,
                        "scratchpad": None,
                        "tokens": (
                            usage_tokens
                            if usage_tokens is not None
                            else (len(final.split()) if final else 0)
                        ),
                    },
                )
            except Exception as e:
                logger.exception("Unexpected error in Bedrock astream worker: %s", e)
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"type": "error", "message": str(e)}
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        loop.run_in_executor(None, _worker)
        while True:
            ev = await queue.get()
            if ev is sentinel:
                break
            yield ev

    # ------------------------------------------------------------------
    # Image: synchronous generate_image  (ImageMixin requirement)
    # ------------------------------------------------------------------

    def generate_image(
        self,
        prompt: str,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Picture:
        """
        Generate or edit an image via InvokeModel.

        Parameters
        ----------
        prompt:
            Text description of the desired image (or edit instruction).
        image:
            Optional source image for img2img.  Accepts anything
            ``ImageMixin._coerce_image_result`` understands.
        model:
            Override ``self.image_model_id`` for this call.
        image_index:
            Which image to return when multiple are generated (default 0).
        negative_prompt, width, height, number_of_images, quality, cfg_scale,
        seed, steps, similarity_strength, init_image_strength:
            Model-specific generation parameters forwarded to the payload
            builder.  Unknown kwargs are silently dropped by the builder.
        """
        effective_model = model or self.image_model_id
        if effective_model is None:
            raise ValueError(
                "image_model_id is not set; cannot run image generation. "
                "Pass image_model_id= to the constructor."
            )

        image_index = int(kwargs.pop("image_index", 0))
        src_img = self._coerce_image_result(image)
        mode = "img2img" if src_img is not None else "txt2img"

        # Separate known image-gen kwargs from any stray extras.
        _img_kwarg_keys = {
            "negative_prompt",
            "width",
            "height",
            "number_of_images",
            "quality",
            "cfg_scale",
            "seed",
            "steps",
            "similarity_strength",
            "init_image_strength",
            "samples",
        }
        img_kwargs = {k: v for k, v in kwargs.items() if k in _img_kwarg_keys}

        payload = _build_image_payload(effective_model, prompt, src_img, **img_kwargs)

        def _invoke() -> Any:
            return self._runtime.invoke_model(
                modelId=effective_model,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )

        try:
            response = self._backoff_call(_invoke)
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"Bedrock invoke_model failed: {e}") from e

        body = json.loads(response["body"].read())
        image_bytes_list = _extract_image_bytes(effective_model, body)

        if not image_bytes_list:
            raise RuntimeError(
                f"Bedrock image model '{effective_model}' returned no images. "
                f"Response keys: {list(body.keys())}"
            )

        if not (0 <= image_index < len(image_bytes_list)):
            raise IndexError(
                f"image_index={image_index} out of range "
                f"({len(image_bytes_list)} image(s) returned)."
            )

        return Picture.from_bytes(
            image_bytes_list[image_index],
            metadata={
                "provider": "bedrock",
                "model": effective_model,
                "mode": mode,
                "prompt": prompt,
                "region": self.region,
            },
        )

    # ------------------------------------------------------------------
    # Image: async streaming  (overrides ImageMixin.astream_image)
    # ------------------------------------------------------------------

    async def astream_image(
        self,
        prompt: str,
        image: Any = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Async streaming wrapper for image generation.

        Bedrock image models are single-shot (no server-sent progress), so
        this emits:
          ``{"type": "start",    "provider": "bedrock", "model": ..., "mode": ...}``
          ``{"type": "progress", "progress": 0.0, "state": "generating"}``
          ``{"type": "end",      "image": Picture, ...}``
          ``{"type": "error",    "message": "..."}``
        """
        effective_model = model or self.image_model_id
        mode = "img2img" if image is not None else "txt2img"

        yield {
            "type": "start",
            "provider": "bedrock",
            "model": effective_model,
            "mode": mode,
        }

        yield {
            "type": "progress",
            "progress": 0.0,
            "eta_relative": None,
            "state": "generating",
            "textinfo": "Waiting for Bedrock InvokeModel…",
        }

        try:
            final_img = await asyncio.to_thread(
                self.generate_image, prompt, image, model, **kwargs
            )
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
        Return a dict with ``"text"`` and ``"image"`` keys, each containing
        a list of model summary dicts available in the current region.

        Each entry has:
          ``model_id``   – the ID to pass to the constructor or per-call
          ``name``       – human-readable model name
          ``provider``   – provider name (e.g. "Amazon", "Anthropic", ...)
          ``input``      – list of input modalities  (e.g. ["TEXT", "IMAGE"])
          ``output``     – list of output modalities (e.g. ["TEXT"])
          ``streaming``  – bool, whether the model supports streaming
          ``status``     – "ACTIVE" | "LEGACY" | "DEPRECATED" (when present)

        Example::

            b = Bedrock()
            info = b.models()
            print(info["image"])   # list of image-capable model dicts
            print(info["text"])    # list of text-capable model dicts

        Note: this calls the Bedrock *control-plane* API (``bedrock``, not
        ``bedrock-runtime``), which requires the ``bedrock:ListFoundationModels``
        IAM permission.
        """

        def _summarise(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out = []
            for s in summaries:
                status_info = s.get("modelLifecycle") or {}
                out.append(
                    {
                        "model_id": s.get("modelId", ""),
                        "name": s.get("modelName", ""),
                        "provider": s.get("providerName", ""),
                        "input": s.get("inputModalities", []),
                        "output": s.get("outputModalities", []),
                        "streaming": s.get("responseStreamingSupported", False),
                        "status": status_info.get("status", "ACTIVE"),
                    }
                )
            return out

        text_resp = self._control.list_foundation_models(byOutputModality="TEXT")
        image_resp = self._control.list_foundation_models(byOutputModality="IMAGE")

        return {
            "text": _summarise(text_resp.get("modelSummaries", [])),
            "image": _summarise(image_resp.get("modelSummaries", [])),
        }

    def text_model_ids(self) -> List[str]:
        """Convenience: return just the model ID strings for text-output models."""
        return [m["model_id"] for m in self.models()["text"]]

    def image_model_ids(self) -> List[str]:
        """Convenience: return just the model ID strings for image-output models."""
        return [m["model_id"] for m in self.models()["image"]]
