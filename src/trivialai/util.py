import json
import re


class TransformError(Exception):
    def __init__(self, message="Transformation Error", raw=None):
        self.message = message
        self.raw = raw
        super().__init__(self.message)


def loadch(resp):
    if resp is None:
        raise TransformError("no-message-given")
    try:
        return json.loads(
            re.sub("^```\\w+\n", "", resp.strip()).removesuffix("```").strip()
        )
    except (TypeError, json.decoder.JSONDecodeError):
        pass
    raise TransformError("parse-failed")
