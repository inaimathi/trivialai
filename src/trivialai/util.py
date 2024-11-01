import json


def loadch(resp):
    try:
        return (
            json.loads(
                (resp.strip().removeprefix("```json").removesuffix("```").strip())
            ),
            True,
        )
    except (TypeError, json.decoder.JSONDecodeError):
        pass
    return None, False
