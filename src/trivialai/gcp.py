import json
import os

import google.auth
import vertexai
from vertexai.generative_models import GenerativeModel

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult

_SAFETY_MAP = {
    k.replace("HARM_CATEGORY_", "").lower(): getattr(
        vertexai.generative_models.HarmCategory, k
    )
    for k in dir(vertexai.generative_models.HarmCategory)
    if k.startswith("HARM_CATEGORY_")
}

_SAFETY_LEVEL_MAP = {
    k.replace("BLOCK_", "").lower(): getattr(
        vertexai.generative_models.HarmBlockThreshold, k
    )
    for k in dir(vertexai.generative_models.HarmBlockThreshold)
    if k.startswith("BLOCK_")
}

assert (
    _SAFETY_MAP and _SAFETY_LEVEL_MAP
), "The GenerativeModel safety interface has changed"


def _dict_to_safety(safety_settings):
    assert type(safety_settings) is dict
    assert set(safety_settings.keys()).issubset(
        _SAFETY_MAP.keys()
    ), f"Valid safety settings are {list(_SAFETY_MAP.keys())}"
    assert {str(v).lower() for v in safety_settings.values()}.issubset(
        _SAFETY_LEVEL_MAP.keys()
    ), f"Valid safety levels are  {list(_SAFETY_LEVEL_MAP.keys())}"
    return [
        vertexai.generative_models.SafetySetting(
            category=_SAFETY_MAP[k], threshold=_SAFETY_LEVEL_MAP[str(v).lower()]
        )
        for k, v in safety_settings.items()
    ]


class GCP(LLMMixin, FilesystemMixin):
    def __init__(self, model, vertex_api_creds, region, safety_settings=None):
        safety = safety_settings
        if safety is None:
            safety = {k: None for k in _SAFETY_MAP.keys()}
        self.safety_settings = _dict_to_safety(safety)
        self.region = region
        self.api_creds = vertex_api_creds
        self.model = model
        self._gcp_creds()
        self.vertex_init()

    def _gcp_creds(self):
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

    def vertex_init(self):
        vertexai.init(
            project=self.gcp_project_id,
            location=self.region,
            credentials=self.gcp_creds,
        )

    def generate(self, system, prompt):
        self.vertex_init()
        model = GenerativeModel(system_instruction=system, model_name=self.model)
        try:
            resp = model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
            )
            return LLMResult(resp, resp.text.strip())
        except Exception as e:
            return LLMResult(e, None)
