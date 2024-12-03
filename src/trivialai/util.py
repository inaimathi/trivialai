import json
import re


class TransformError(Exception):
    def __init__(self, message="Transformation Error", raw=None):
        self.message = message
        self.raw = raw
        super().__init__(self.message)


class GenerationError(Exception):
    def __init__(self, message="Generation Error", raw=None):
        self.message = message
        self.raw = raw
        super().__init__(self.message)


def strip_md_code_marker(block):
    return re.sub("^```\\w+\n", "", block).removesuffix("```").strip()


def loadch(resp):
    if resp is None:
        raise TransformError("no-message-given")
    try:
        return json.loads(strip_md_code_marker(resp.strip()))
    except (TypeError, json.decoder.JSONDecodeError):
        pass
    raise TransformError("parse-failed")
