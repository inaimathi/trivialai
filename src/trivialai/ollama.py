import requests

from .util import AiResult


class Ollama:
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
            return AiResult(res, 200, res.json()["response"].strip())
        return AiResult(res, res.status_code, None)
