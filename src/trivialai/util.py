import json
import re


def loadch(resp):
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
