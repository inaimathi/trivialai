from . import util


class Tools:
    def __init__(self, extras=None):
        assert (extras is None) or (type(extras) is dict)
        self.extras = None
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

    def define(self, fn=None, *, name=None, type=None, description=None):
        if fn is None:
            # If no function is passed, return a decorator
            def decorator(f):
                self._define_function(f, name, type, description)
                return f

            return decorator
        # If function is passed directly, define it
        return self._define_function(fn, name, type, description)

    def _define_function(self, fn, name=None, type=None, description=None):
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
            if set(tool_call["args"].keys()) == set(f["type"].keys()):
                return True
        return False

    def transform(self, resp):
        parsed = util.loadch(resp)
        if self.validate(parsed):
            return parsed
        raise util.TransformError("invalid-tool-call", raw=resp)

    def transform_multi(self, resp):
        parsed = util.loadch(resp)
        if type(parsed) is not list:
            raise util.TransformError("result-not-list", raw=parsed)
        for call in parsed:
            if not self.validate(call):
                raise util.TransformError("invalid-tool-subcall", raw=call)
        return parsed

    def lookup(self, tool_call):
        return self._env[tool_call["functionName"]]["function"]

    def raw_call(self, tool_call):
        return self.lookup(tool_call)(**tool_call["args"])

    def call_with_extras(self, extras, tool_call):
        if self.validate(tool_call):
            with_extras = {**tool_call, "args": {**tool_call["args"], **extras}}
            return self.raw_call(with_extras)

    def call(self, tool_call):
        if self.validate(tool_call):
            if self.extras is not None:
                return self.call_with_extras(self.extras, tool_call)
            return self.raw_call(tool_call)
        return None
