# src/trivialai/openrouter.py
"""
OpenRouter adapter: one key for text, vision and image models from many
providers.

Text and vision go through OpenRouter's OpenAI-compatible Chat Completions
endpoint, so everything is inherited from ``ChatGPT``.  Image generation
uses OpenRouter's Image API (``POST /images``), which differs from OpenAI's
in two ways handled here: there is a single endpoint for both text-to-image
and reference-guided generation, and references travel as
``input_references``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import httpx

from .chatgpt import ChatGPT
from .image import Picture, encode_image_input

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _reference_part(pic: Picture) -> Dict[str, Any]:
    media_type, data = encode_image_input(
        pic, allowed={"image/png", "image/jpeg", "image/webp"}
    )
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{data}"},
    }


class OpenRouter(ChatGPT):
    """
    OpenRouter client: text + vision (Chat Completions) and images (Image API).

    ``model`` and ``image_model`` are OpenRouter model IDs such as
    ``anthropic/claude-sonnet-4.5`` or ``google/gemini-3.1-flash-image``.

    Text calls behave exactly like ``ChatGPT``'s, including ``images=[...]``
    for vision models.

    ``generate_image(prompt, image=None, **kwargs)`` takes one image or an
    ordered list of reference images.  How many a request may carry depends
    on the model (at most 16); so do the other request fields, which are
    passed through as keywords: ``aspect_ratio``, ``resolution``, ``size``,
    ``quality``, ``background``, ``output_format``, ``seed``, ``n``,
    ``provider``.  When OpenRouter reports the charge, the returned
    Picture has it in ``metadata["cost_usd"]``.
    """

    PROVIDER = "openrouter"
    MAX_REFERENCE_IMAGES = 16

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        image_model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = 300.0,
        base_url: str = DEFAULT_BASE_URL,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
            image_model=image_model,
            base_url=base_url or DEFAULT_BASE_URL,
        )

    # ---- Image generation ----
    def _build_image_request(
        self,
        prompt: str,
        references: List[Picture],
        model: str,
        options: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        if options.get("mask") is not None:
            raise ValueError(
                "OpenRouter's Image API has no mask parameter; send the mask "
                "as a reference image and describe its role in the prompt."
            )
        options = {k: v for k, v in options.items() if k != "mask"}
        body: Dict[str, Any] = {"model": model, "prompt": prompt, **options}
        if references:
            body["input_references"] = [_reference_part(pic) for pic in references]
        return "/images", body

    # ---- Model discovery ----
    def _list(self, client: httpx.Client, path: str) -> List[Dict[str, Any]]:
        res = client.get(
            self._url(path), headers={"Authorization": f"Bearer {self.api_key}"}
        )
        res.raise_for_status()
        body = res.json()
        rows = body.get("data") if isinstance(body, dict) else body
        out = []
        for row in rows or []:
            if isinstance(row, dict) and row.get("id"):
                entry = dict(row)
                entry["label"] = row.get("name") or row["id"]
                out.append(entry)
        out.sort(key=lambda row: str(row["id"]).lower())
        return out

    def models(self) -> Dict[str, list]:
        """
        Return ``{"text": [...], "image": [...]}``: models that produce text
        (from ``/models``) and the Image API's models (from ``/images/models``).
        """
        with httpx.Client(timeout=self.timeout) as client:
            all_models = self._list(client, "/models")
            image_models = self._list(client, "/images/models")

        def _outputs_text(row: Dict[str, Any]) -> bool:
            modalities = (row.get("architecture") or {}).get("output_modalities")
            # Rows that do not say are kept: better listed than hidden.
            return not modalities or "text" in modalities

        return {
            "text": [row for row in all_models if _outputs_text(row)],
            "image": image_models,
        }
