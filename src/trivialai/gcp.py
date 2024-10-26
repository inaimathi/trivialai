import json
import os

import google.auth
import vertexai
from vertexai.generative_models import GenerativeModel

from .llm import LLMMixin, LLMResult


class GCP(LLMMixin):
    def __init__(self, model, vertex_api_creds, region):
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
            resp = model.generate_content(prompt)
            return LLMResult(resp, resp.text.strip())
        except Exception as e:
            return LLMResult(e, None)
