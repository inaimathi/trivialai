import requests

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult


class Ollama(LLMMixin, FilesystemMixin):
    def __init__(self, model, ollama_server):
        self.model = model
        self.server = ollama_server.rstrip("/")

    def generate(self, system, prompt):
        res = requests.post(
            f"{self.server}/api/generate",
            json={
                "model": self.model,
                "stream": False,
                "prompt": f"SYSTEM PROMPT: {system} PROMPT: {prompt}",
            },
        )
        if res.status_code == 200:
            return LLMResult(res, res.json()["response"].strip())
        return LLMResult(res, None)
