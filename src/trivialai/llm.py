import json
from collections import namedtuple

LLMResult = namedtuple("LLMResult", ["raw", "status_code", "content"])


def loadch(resp):
    try:
        return json.loads(
            (resp.strip().removeprefix("```json").removesuffix("```").strip())
        )
    except (TypeError, json.decoder.JSONDecodeError):
        pass
    return None


class LLMMixin:
    def generate_checked(self, transformFn, system, prompt, retries=5):
        for i in range(retries):
            res = self.generate(system, prompt)
            transformed = transformFn(res.content)
            if transformed:
                return LLMResult(res.raw, res.status_code, transformed)
                res.content = transformed
                return res
        return None

    def generate_json(self, system, prompt, retries=5):
        return self.generate_checked(loadch, system, prompt, retries=retries)
