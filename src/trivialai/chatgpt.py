import requests

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult


class ChatGPT(LLMMixin, FilesystemMixin):
    def __init__(self, model, api_key, anthropic_version=None, max_tokens=None):
        self.max_tokens = max_tokens or 4096
        self.version = anthropic_version or "2023-06-01"
        self.api_key = api_key
        self.model = model

    def generate(self, system, prompt):
        res = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        if res.status_code == 200:
            return LLMResult(res, res.json()["choices"][0]["message"]["content"])
        return LLMResult(res, None)
