from .util import LLMResult, TransformError, generate_checked, loadch


class LLMMixin:
    def generate_checked(self, transformFn, system, prompt, retries=5):
        return generate_checked(
            lambda: self.generate(system, prompt), transformFn, retries=retries
        )

    def generate_json(self, system, prompt, retries=5):
        return self.generate_checked(loadch, system, prompt, retries=retries)

    def generate_tool_call(self, tools, system, prompt, retries=5):
        sysprompt = f'You are a computer specialist. Your job is translating client requests into tool calls. Your client has sent a request to use a tool; return the function call corresponding to the request and no other commentary. Return a value of type `{{"functionName" :: string, "args" :: {{arg_name: arg value}} }}`. You have access to the tools: {tools.list()}. {system}'
        return self.generate_checked(
            tools.transform, sysprompt, prompt, retries=retries
        )

    def generate_many_tool_calls(self, tools, prompt, retries=5):
        sysprompt = f'You are a computer specialist. Your job is translating client requests into tool calls. Your client has sent a request to use some number of tools; return a list of function calls corresponding to the request and no other commentary. Return a value of type `[{{"functionName" :: string, "args" :: {{arg_name: arg value}} }}]`. You have access to the tools: {tools.list()}.'
        return self.generate_checked(
            tools.transform_multi, sysprompt, prompt, retries=retries
        )
