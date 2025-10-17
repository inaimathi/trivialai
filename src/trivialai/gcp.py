from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import google.auth
import vertexai
from vertexai.generative_models import GenerativeModel

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult

# Build safety maps from Vertex AI enums
_SAFETY_MAP: Dict[str, Any] = {
    k.replace("HARM_CATEGORY_", "").lower(): getattr(
        vertexai.generative_models.HarmCategory, k
    )
    for k in dir(vertexai.generative_models.HarmCategory)
    if k.startswith("HARM_CATEGORY_")
}

_SAFETY_LEVEL_MAP: Dict[str, Any] = {
    k.replace("BLOCK_", "").lower(): getattr(
        vertexai.generative_models.HarmBlockThreshold, k
    )
    for k in dir(vertexai.generative_models.HarmBlockThreshold)
    if k.startswith("BLOCK_")
}

assert (
    _SAFETY_MAP and _SAFETY_LEVEL_MAP
), "The GenerativeModel safety interface has changed"


def _dict_to_safety(safety_settings: Dict[str, Any]) -> List[Any]:
    """
    Convert a dict like {"harassment":"none","hate_speech":"low", ...}
    into a list of vertexai.generative_models.SafetySetting.
    Values must be keys in _SAFETY_LEVEL_MAP (stringified, case-insensitive).
    """
    assert isinstance(safety_settings, dict)
    assert set(safety_settings.keys()).issubset(
        _SAFETY_MAP.keys()
    ), f"Valid safety keys are {list(_SAFETY_MAP.keys())}"
    assert {str(v).lower() for v in safety_settings.values()}.issubset(
        _SAFETY_LEVEL_MAP.keys()
    ), f"Valid safety levels are {list(_SAFETY_LEVEL_MAP.keys())}"

    return [
        vertexai.generative_models.SafetySetting(
            category=_SAFETY_MAP[k], threshold=_SAFETY_LEVEL_MAP[str(v).lower()]
        )
        for k, v in safety_settings.items()
    ]


class GCP(LLMMixin, FilesystemMixin):
    """
    Minimal Vertex AI wrapper (sync only).
    This class is primarily for auth/setup and a simple generate() call.

    Streaming is not implemented here; the LLMMixin default `astream` will
    fall back to emitting a single delta containing the full content.
    """

    def __init__(
        self,
        model: str,
        vertex_api_creds: str,
        region: str,
        safety_settings: Optional[Dict[str, Any]] = None,
    ):
        safety = safety_settings or {k: None for k in _SAFETY_MAP.keys()}
        self.safety_settings = _dict_to_safety(safety)
        self.region = region
        self.api_creds = vertex_api_creds
        self.model = model
        self._gcp_creds()
        self.vertex_init()

    def _gcp_creds(self) -> None:
        # Accept either a path to a JSON file or a JSON string
        if os.path.isfile(self.api_creds):
            gcp_creds, gcp_project_id = google.auth.load_credentials_from_file(
                self.api_creds,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            gcp_creds, gcp_project_id = google.auth.load_credentials_from_dict(
                json.loads(self.api_creds),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        self.gcp_creds = gcp_creds
        self.gcp_project_id = gcp_project_id

    def vertex_init(self) -> None:
        vertexai.init(
            project=self.gcp_project_id,
            location=self.region,
            credentials=self.gcp_creds,
        )

    def generate(
        self,
        system: str,
        prompt: str,
        images: Optional[list] = None,  # accepted for LLMMixin parity; unused
    ) -> LLMResult:
        """
        Synchronous content generation via Vertex AI GenerativeModel.
        """
        self.vertex_init()
        model = GenerativeModel(system_instruction=system, model_name=self.model)
        try:
            resp = model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
            )
            text = (resp.text or "").strip()
            return LLMResult(resp, text, None)
        except Exception as e:
            return LLMResult(e, None, None)
