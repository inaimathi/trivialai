import json
import re


def loadch(resp):
    if resp is None:
        return None, False
    try:
        return (
            json.loads(
                re.sub("^```\\w+\n", "", resp.strip()).removesuffix("```").strip()
            ),
            True,
        )
    except (TypeError, json.decoder.JSONDecodeError):
        pass
    return None, False
