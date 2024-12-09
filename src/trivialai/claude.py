import requests

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult


class Claude(LLMMixin, FilesystemMixin):
    def __init__(self, model, api_key, anthropic_version=None, max_tokens=None):
        self.max_tokens = max_tokens or 4096
        self.version = anthropic_version or "2023-06-01"
        self.api_key = api_key
        self.model = model

    def generate(self, system, prompt):
        res = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": self.api_key,
                "anthropic-version": self.version,
            },
            json={
                "system": system,
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if res.status_code == 200:
            return LLMResult(res, res.json()["content"][0]["text"])
        return LLMResult(res, None)
