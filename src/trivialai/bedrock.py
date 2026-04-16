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

    # Text only — bare model ID; "us." prefix is added automatically for us-east-1
    b = Bedrock("anthropic.claude-3-5-haiku-20241022-v1:0")
    result = b.generate("You are helpful.", "What is 2+2?")

    # Explicit prefixed ID — used as-is; a warning is logged if it contradicts region
    b = Bedrock("eu.anthropic.claude-3-5-haiku-20241022-v1:0", region="eu-west-1")
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

    # Long-term bearer token (AWS_BEARER_TOKEN_BEDROCK from the AWS console).
    # SigV4 signing is skipped; the token is sent as an HTTP
    # "Authorization: Bearer <token>" header on every request.
    b = Bedrock(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        aws_bearer_token="<your-long-term-api-key>",
    )
    result = b.generate("system", "prompt")

    # Model discovery
    info = b.models()
    # {"text": [...], "image": [...], "inference_profiles": [...]}
    #
    # Most newer models (Anthropic Claude 3.5+, etc.) cannot be called by their
    # bare foundation-model ID — use an inference profile ID instead:
    print(b.inference_profile_ids())  # ["us.anthropic.claude-3-5-haiku-20241022-v1:0", ...]
    print(b.text_model_ids())         # foundation model IDs (often not directly callable)
    print(b.image_model_ids())        # ["amazon.nova-canvas-v1:0", ...]
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from .filesystem import FilesystemMixin
from .image import ImageMixin, Picture
from .llm import LLMMixin, LLMResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model IDs
# ---------------------------------------------------------------------------

# Default text model — stored as a bare foundation-model ID (no geo-prefix).
# The correct cross-region prefix ("us.", "eu.", "ap.") is prepended
# automatically at construction time based on the `region` argument.
_DEFAULT_TEXT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"

# ---------------------------------------------------------------------------
# Cross-region inference profile helpers
# ---------------------------------------------------------------------------

# Geo-prefixes used by Bedrock cross-region inference profiles.
_GEO_PREFIXES = ("us.", "eu.", "ap.")


def _region_to_geo_prefix(region: str) -> str:
    """Map an AWS region string to its Bedrock inference-profile geo-prefix."""
    if region.startswith("us-"):
        return "us"
    if region.startswith("eu-"):
        return "eu"
    if region.startswith("ap-"):
        return "ap"
    # Other regions (e.g. ca-*, sa-*, me-*) don't have a dedicated prefix;
    # "us" profiles are generally accessible from those regions.
    logger.debug(
        "No known geo-prefix for region %r; defaulting to 'us' for inference profiles.",
        region,
    )
    return "us"


def _apply_geo_prefix(model_id: str, region: str) -> str:
    """
    Ensure *model_id* carries the geo-prefix that matches *region*.

    * If *model_id* already has a geo-prefix that matches the region → no-op.
    * If *model_id* already has a geo-prefix that **doesn't** match → warn and
      leave it unchanged (caller may have chosen deliberately).
    * If *model_id* has no geo-prefix → prepend the correct one.

    Bare foundation-model IDs (e.g. ``"anthropic.claude-3-5-haiku-20241022-v1:0"``)
    and image / other model IDs that don't use cross-region profiles are returned
    as-is when they carry no recognisable geo-prefix.
    """
    expected = _region_to_geo_prefix(region)
    for prefix in _GEO_PREFIXES:
        if model_id.startswith(prefix):
            actual = prefix.rstrip(".")
            if actual != expected:
                logger.warning(
                    "model_id %r has geo-prefix %r but region is %r (expected prefix %r). "
                    "Leaving model_id unchanged — pass the correct prefix or omit it to "
                    "have it set automatically.",
                    model_id,
                    prefix,
                    region,
                    expected + ".",
                )
            return model_id  # already prefixed — don't touch
    # No geo-prefix present — prepend the one that matches the region.
    return f"{expected}.{model_id}"


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
        Newer models (Claude 3.5+, etc.) require a cross-region inference
        profile ID such as ``"us.anthropic.claude-3-5-haiku-20241022-v1:0"``.
        You may supply the bare foundation-model ID (without a geo-prefix) and
        the correct prefix will be derived automatically from ``region``.  If
        you supply a prefixed ID whose prefix contradicts ``region``, a warning
        is logged and the ID is used as-is.
        Can be ``None`` if you only need image generation.
    image_model_id:
        Foundation model ID for **image** generation.
        Defaults to ``"amazon.nova-canvas-v1:0"`` (Nova Canvas).
        Can be ``None`` if you only need text generation.
    region:
        AWS region (default ``"us-east-1"``).  Also used to derive the
        geo-prefix (``"us."``, ``"eu."``, ``"ap."``) for inference profile IDs.
    max_tokens, temperature, top_p:
        Text generation inference parameters.
    aws_bearer_token:
        Long-term Bedrock API key (``AWS_BEARER_TOKEN_BEDROCK`` in the AWS
        console).  When supplied, SigV4 request signing is disabled and every
        request instead carries an ``Authorization: Bearer <token>`` header.
        This is mutually exclusive with ``aws_profile`` /
        ``aws_access_key_id`` / ``aws_secret_access_key``; if all are
        provided, ``aws_bearer_token`` takes precedence.
    aws_profile, aws_access_key_id, aws_secret_access_key, aws_session_token:
        Standard IAM authentication (mirrors the original Bedrock class).
        Ignored when ``aws_bearer_token`` is set.
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
        aws_bearer_token: Optional[str] = None,
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
        # Prepend the geo-prefix that matches `region` when model_id is a bare
        # foundation-model ID (no existing prefix).  Explicit prefixes that
        # contradict `region` produce a warning but are left unchanged.
        self.model_id = (
            _apply_geo_prefix(model_id, region) if model_id is not None else None
        )
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
        if not aws_bearer_token:
            # Standard IAM auth: explicit keys beat a named profile.
            if aws_access_key_id and aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = aws_access_key_id
                session_kwargs["aws_secret_access_key"] = aws_secret_access_key
                if aws_session_token:
                    session_kwargs["aws_session_token"] = aws_session_token
            elif aws_profile:
                session_kwargs["profile_name"] = aws_profile

        session = boto3.Session(**session_kwargs)

        if aws_bearer_token:
            # Bearer-token mode: disable SigV4 signing entirely and inject
            # the token as an HTTP Authorization header on every request.
            # botocore's UNSIGNED sentinel tells the signer to produce no
            # auth headers, giving us a clean slate for our own header.
            _unsigned = Config(signature_version=UNSIGNED)
            self._runtime = session.client(
                "bedrock-runtime", region_name=region, config=_unsigned
            )
            self._control = session.client(
                "bedrock", region_name=region, config=_unsigned
            )

            _token = aws_bearer_token  # capture for the closure below

            def _inject_bearer(request, **kwargs) -> None:  # type: ignore[type-arg]
                request.headers["Authorization"] = f"Bearer {_token}"

            # before-sign fires after the request is prepared but before any
            # auth header is written — the right place to inject ours.
            self._runtime.meta.events.register(
                "before-sign.bedrock-runtime.*", _inject_bearer
            )
            self._control.meta.events.register("before-sign.bedrock.*", _inject_bearer)
        else:
            # Standard IAM / credential-chain mode.
            self._runtime = session.client("bedrock-runtime", region_name=region)
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

    def models(self, active_only: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return a dict with ``"text"``, ``"image"``, and ``"inference_profiles"``
        keys describing models available in the current region.

        Parameters
        ----------
        active_only:
            When ``True`` (the default) only entries with ``status == "ACTIVE"``
            are returned, filtering out ``"LEGACY"`` and ``"DEPRECATED"`` models
            that AWS will reject at invocation time.  Pass ``False`` to see the
            full catalogue.

        ``"text"`` / ``"image"``
            Foundation model summaries from ``ListFoundationModels``.
            **Note:** most newer Anthropic (and other) models cannot be invoked
            directly by their foundation-model ID — you must use an inference
            profile ID instead.  See ``"inference_profiles"`` below.

        ``"inference_profiles"``
            System-defined cross-region inference profiles from
            ``ListInferenceProfiles``.  These are the IDs you should actually
            pass to the constructor (e.g. ``"us.anthropic.claude-3-5-haiku-20241022-v1:0"``).
            Each entry has:
              ``profile_id``  – the ID to pass to the constructor
              ``name``        – human-readable profile name
              ``models``      – list of underlying foundation-model ARNs
              ``status``      – ``"ACTIVE"`` | ``"INACTIVE"``
              ``type``        – ``"SYSTEM_DEFINED"`` | ``"APPLICATION"``

        Each foundation-model entry has:
          ``model_id``   – the ID (may not be directly invocable; prefer inference profiles)
          ``name``       – human-readable model name
          ``provider``   – provider name (e.g. "Amazon", "Anthropic", ...)
          ``input``      – list of input modalities  (e.g. ["TEXT", "IMAGE"])
          ``output``     – list of output modalities (e.g. ["TEXT"])
          ``streaming``  – bool, whether the model supports streaming
          ``status``     – "ACTIVE" | "LEGACY" | "DEPRECATED"

        Note: requires ``bedrock:ListFoundationModels`` and
        ``bedrock:ListInferenceProfiles`` IAM permissions.
        """

        def _summarise_foundation(
            summaries: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            out = []
            for s in summaries:
                status_info = s.get("modelLifecycle") or {}
                status = status_info.get("status", "ACTIVE")
                if active_only and status != "ACTIVE":
                    continue
                out.append(
                    {
                        "model_id": s.get("modelId", ""),
                        "name": s.get("modelName", ""),
                        "provider": s.get("providerName", ""),
                        "input": s.get("inputModalities", []),
                        "output": s.get("outputModalities", []),
                        "streaming": s.get("responseStreamingSupported", False),
                        "status": status,
                    }
                )
            return out

        def _summarise_profiles(
            summaries: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            out = []
            for s in summaries:
                status = s.get("status", "ACTIVE")
                if active_only and status != "ACTIVE":
                    continue
                out.append(
                    {
                        "profile_id": s.get("inferenceProfileId", ""),
                        "name": s.get("inferenceProfileName", ""),
                        "models": [
                            m.get("modelArn", "") for m in (s.get("models") or [])
                        ],
                        "status": status,
                        "type": s.get("type", "SYSTEM_DEFINED"),
                    }
                )
            return out

        text_resp = self._control.list_foundation_models(byOutputModality="TEXT")
        image_resp = self._control.list_foundation_models(byOutputModality="IMAGE")

        try:
            profile_resp = self._control.list_inference_profiles(
                typeEquals="SYSTEM_DEFINED"
            )
            profiles = _summarise_profiles(
                profile_resp.get("inferenceProfileSummaries", [])
            )
        except (BotoCoreError, ClientError) as e:
            logger.warning("Could not list inference profiles: %s", e)
            profiles = []

        return {
            "text": _summarise_foundation(text_resp.get("modelSummaries", [])),
            "image": _summarise_foundation(image_resp.get("modelSummaries", [])),
            "inference_profiles": profiles,
        }

    def text_model_ids(self, active_only: bool = True) -> List[str]:
        """Convenience: return just the model ID strings for text-output models."""
        return [m["model_id"] for m in self.models(active_only=active_only)["text"]]

    def image_model_ids(self, active_only: bool = True) -> List[str]:
        """Convenience: return just the model ID strings for image-output models."""
        return [m["model_id"] for m in self.models(active_only=active_only)["image"]]

    def inference_profile_ids(self, active_only: bool = True) -> List[str]:
        """
        Convenience: return the profile ID strings for all system-defined
        cross-region inference profiles.

        These are the IDs you should pass to the constructor for text
        generation with newer models, e.g.::

            b = Bedrock()
            print(b.inference_profile_ids())
            # ['us.amazon.nova-lite-v1:0', 'us.anthropic.claude-3-5-haiku-20241022-v1:0', ...]
        """
        return [
            p["profile_id"]
            for p in self.models(active_only=active_only)["inference_profiles"]
        ]
