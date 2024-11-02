from .util import loadch


class Tools:
    def __init__(self):
        self._env = {}

    def _intern(self, name, type, description, fn):
        if name in self._env:
            return False
        self._env[name] = {
            "name": name,
            "type": type,
            "description": description,
            "function": fn,
        }
        return True

    def define(self, fn, name=None, type=None, description=None):
        assert (
            fn.__annotations__ or type
        ), "either annotate the function or pass in a type dictionary for its inputs"
        assert (
            fn.__doc__ or description
        ), "either document the function or pass in a description"
        return self._intern(
            name or fn.__name__,
            type or {k: v for k, v in fn.__annotations__.items() if not k == "return"},
            description or fn.__doc__,
            fn,
        )

    def list(self):
        return [
            {
                "name": k,
                "type": v["type"],
                "description": v["function"].__doc__,
            }
            for k, v in self._env.items()
        ]

    def validate(self, tool_call):
        if (
            "functionName" in tool_call
            and "args" in tool_call
            and tool_call["functionName"] in self._env
        ):
            f = self._env[tool_call["functionName"]]
            if not set(tool_call["args"].keys()).difference(f["type"].keys()):
                return True
        return False

    def transform(self, resp):
        parsed, success = loadch(resp)
        if not success:
            return None, False
        if self.validate(parsed):
            return parsed, True
        return None, False

    def raw_call(self, tool_call):
        return self._env[tool_call["functionName"]]["function"](**tool_call["args"])

    def call(self, tool_call):
        if self.validate(tool_call):
            return self.raw_call(tool_call)
        return None
