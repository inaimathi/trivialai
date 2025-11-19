# TrivialAI

*(A set of **httpx** / **boto3**-based, trivial bindings for AI models — now with optional streaming)*

## Install

```bash
pip install trivialai
# Optional: HTTP/2 for OpenAI/Anthropic
# pip install "trivialai[http2]"
# Optional: AWS Bedrock support (via boto3)
# pip install "trivialai[bedrock]"
```

* Requires **Python ≥ 3.9**.
* Uses **httpx** for HTTP-based providers and **boto3** for Bedrock.

## Quick start

```py
>>> from trivialai import claude, gcp, ollama, chatgpt, bedrock
```

## Synchronous usage (unchanged ergonomics)

### Ollama

```py
>>> client = ollama.Ollama("gemma2:2b", "http://localhost:11434/")
# or ollama.Ollama("deepseek-coder-v2:latest", "http://localhost:11434/")
# or ollama.Ollama("mannix/llama3.1-8b-abliterated:latest", "http://localhost:11434/")
>>> client.generate("sys msg", "Say hi with 'platypus'.").content
"Hi there—platypus!"
>>> client.generate_json("sys msg", "Return {'name': 'Platypus'} as JSON").content
{'name': 'Platypus'}
```

### Claude (Anthropic API)

```py
>>> client = claude.Claude("claude-3-5-sonnet-20240620", os.environ["ANTHROPIC_API_KEY"])
>>> client.generate("sys msg", "Say hi with 'platypus'.").content
"Hello, platypus!"
```

### GCP (Vertex AI)

```py
>>> client = gcp.GCP("gemini-1.5-flash-001", "/path/to/gcp_creds.json", "us-central1")
>>> client.generate("sys msg", "Say hi with 'platypus'.").content
"Hello, platypus!"
```

### ChatGPT (OpenAI API)

```py
>>> client = chatgpt.ChatGPT("gpt-4o-mini", os.environ["OPENAI_API_KEY"])
>>> client.generate("sys msg", "Say hi with 'platypus'.").content
"Hello, platypus!"
```

### AWS Bedrock (Claude / Llama / Nova / etc)

Bedrock support is provided via the `Bedrock` client, which implements the same `LLMMixin` interface as the others.

#### 1) One-time AWS setup

1. **Enable Bedrock and model access**

   * In the AWS console, pick a Bedrock-supported region (e.g. `us-east-1`).
   * Go to **Amazon Bedrock → Model access** and enable access for the models you want (e.g. Claude 3.5 Sonnet, Llama, Nova, etc).

2. **IAM permissions**

   Grant your user/role permission to call Bedrock runtime APIs, for example:

   ```jsonc
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "bedrock:Converse",
           "bedrock:ConverseStream",
           "bedrock:InvokeModel",
           "bedrock:InvokeModelWithResponseStream"
         ],
         "Resource": "*"
       }
     ]
   }
   ```

   You can restrict `Resource` to specific model ARNs later.

3. **Credentials**

   TrivialAI can use either:

   * the normal AWS credential chain (`aws configure`, env vars, instance role), or
   * explicit credentials passed into the `Bedrock` constructor.

#### 2) Choosing the right `model_id`

Bedrock distinguishes between:

* **Foundation model IDs**, like:
  `anthropic.claude-3-5-sonnet-20241022-v2:0`
* **Inference profile IDs**, which are region-prefixed, like:
  `us.anthropic.claude-3-5-sonnet-20241022-v2:0`

Some newer models (like Claude 3.5 Sonnet v2) must be called **via the inference profile ID** from certain regions. If you see a `ValidationException` complaining about “Invocation of model ID ... with on-demand throughput isn’t supported; retry with an inference profile”, swap to the `us.`-prefixed ID.

#### 3) Minimal Bedrock demo

```py
from trivialai import bedrock

# Using an inference profile ID for Claude 3.5 Sonnet v2 from us-east-1:
client = bedrock.Bedrock(
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    region="us-east-1",
    # Either rely on normal AWS creds...
    # aws_profile="my-dev-profile",
    # ...or pass explicit keys (for testing):
    aws_access_key_id="AKIA...",
    aws_secret_access_key="SECRET...",
)

res = client.generate(
    "This is a test message. Make sure your reply contains the word 'margarine'",
    "Hello there! Can you hear me?"
)
print(res.content)
# -> "Yes, I can hear you! ... margarine ..."

# With JSON parsing:
res_json = client.generate_json(
    "You are a JSON-only assistant.",
    "Return {'name':'Platypus'} as JSON."
)
print(res_json.content)
# -> {'name': 'Platypus'}
```

The `Bedrock` client fully participates in the same higher-level helpers:

* `generate_checked(...)`
* `generate_json(...)`
* `generate_tool_call(...)`
* `generate_many_tool_calls(...)`
* `stream_checked(...)` / `stream_json(...)`

No special-casing required in downstream code.

---

## Streaming (NDJSON-style events)

All providers expose a common streaming shape via `stream(...)` (sync iterator) and `astream(...)` (async):

**Event schema**

* `{"type":"start", "provider": "<ollama|openai|anthropic|gcp|bedrock>", "model": "..."}`
* `{"type":"delta", "text":"...", "scratchpad":"..."}`

  * For **Ollama**, `scratchpad` contains model “thinking” extracted from `<think>…</think>`.
  * For **ChatGPT**, **Claude API**, **GCP**, and **Bedrock**, `scratchpad` is `""` (empty) in deltas.
* `{"type":"end", "content":"...", "scratchpad": <str|None>, "tokens": <int>}`
* `{"type":"error", "message":"..."}`

### Example: streaming Ollama (sync)

```py
>>> client = ollama.Ollama("gemma2:2b", "http://localhost:11434/")
>>> for ev in client.stream("sys", "Explain, think step-by-step."):
...     if ev["type"] == "delta":
...         # show model output live
...         print(ev["text"], end="")
...     elif ev["type"] == "end":
...         print("\n-- scratchpad --")
...         print(ev["scratchpad"])
```

### Example: streaming Bedrock (sync)

```py
>>> from trivialai import bedrock
>>> client = bedrock.Bedrock(
...     model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
...     region="us-east-1",
... )
>>> events = list(client.stream(
...     "This is a test message. Make sure your reply contains the word 'margarine'",
...     "Hello there! Can you hear me?"
... ))
>>> events[0]
{'type': 'start', 'provider': 'bedrock', 'model': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'}
>>> events[-1]
{'type': 'end', 'content': 'Yes, I can hear you! ... margarine ...', 'scratchpad': None, 'tokens': 36}
```

### Example: parse-at-end streaming

If you want incremental updates *and* a structured parse at the end:

```py
from trivialai.util import stream_checked, loadch

for ev in client.stream("sys", "Return a JSON object gradually."):
    # pass-through for UI
    if ev["type"] in {"start","delta"}:
        print(ev)
    elif ev["type"] == "end":
        # now emit the final parsed event
        for final_ev in stream_checked(iter([ev]), loadch):
            print(final_ev)  # {"type":"final","ok":True,"parsed":{...}}
```

Shortcut: `stream_json(system, prompt)` yields the same stream and a final parsed event using `loadch`.

### Async flavor

```py
async for ev in client.astream("sys", "Stream something."):
    ...
```

*(For Bedrock, `stream(...)` is the native streaming interface; `astream(...)` currently falls back to the default `LLMMixin` behavior unless you wrap it yourself.)*

---

## Tool Calls

Use `Tools` to register Python functions, describe them to the model, and safely execute the model’s chosen call.

### 1) Define tools

You can register functions directly or with a decorator. Docstring = description. Type hints become the argument schema.

```python
from typing import Optional, List
from trivialai.tools import Tools

tools = Tools()  # or Tools(extras={"api_key": "..."}), see below

@tools.define()
def screenshot(url: str, selectors: Optional[List[str]] = None) -> None:
    """Take a screenshot of a page; optionally highlight CSS selectors."""
    print("shot", url, selectors)

# Or:
def search(query: str, top_k: int = 5) -> List[str]:
    """Search and return top results."""
    return [f"res{i}" for i in range(top_k)]
tools.define(search)
```

### 2) Show tools to the model

`tools.list()` returns LLM-friendly metadata:

```python
>>> tools.list()
[{
  "name": "screenshot",
  "description": "Take a screenshot of a page; optionally highlight CSS selectors.",
  "type": {"url": <class 'str'>, "selectors": typing.Optional[typing.List[str]]},
  "args": {
    "url": {"type": "string"},
    "selectors": {"type": "array", "items": {"type": "string"}, "nullable": True}
  }
},
{
  "name": "search",
  "description": "Search and return top results.",
  "type": {"query": <class 'str'>, "top_k": <class 'int'>},
  "args": {
    "query": {"type": "string"},
    "top_k": {"type": "int"}
  }
}]
```

### 3) Ask the model to choose a tool

All LLM clients (Ollama, Claude API, ChatGPT, GCP, Bedrock) support a helper that prompts for a tool call and validates it:

```python
from trivialai import ollama
client = ollama.Ollama("gemma2:2b", "http://localhost:11434/")

res = client.generate_tool_call(
    tools,
    system="You are a tool-use router.",
    prompt="Take a screenshot of https://example.com and highlight the search box."
)

# Validated, parsed dict:
>>> res.content
{'functionName': 'screenshot', 'args': {'url': 'https://example.com', 'selectors': ['#search']}}
```

Multiple calls? Use `generate_many_tool_calls(...)`:

```python
multi = client.generate_many_tool_calls(
    tools,
    prompt="Search for 'platypus', then screenshot the first result."
)
# -> [{'functionName': 'search', ...}, {'functionName': 'screenshot', ...}]
```

### 4) Validate/execute (with robust errors)

* **Validation rules:** all required params present; optional params may be omitted; unknown params are rejected.
* On invalid input, methods **raise** `TransformError` (no `None` returns).

```python
from trivialai.util import TransformError

tool_call = res.content  # {'functionName': 'screenshot', 'args': {...}}

# Validate explicitly (optional; call() validates too)
assert tools.validate(tool_call)

# Execute
try:
    tools.call(tool_call)
except TransformError as e:
    print("Tool call failed:", e.message, e.raw)
```

If you already have a raw JSON string from a model and want to validate+parse:

```python
parsed = tools.transform('{"functionName":"search","args":{"query":"platypus"}}')
# or for a list of calls:
calls = tools.transform_multi('[{"functionName":"search","args":{"query":"platypus"}}]')
```

### 5) Extras / environment defaults

Attach fixed kwargs (e.g., tokens, org IDs) that merge into every call:

```python
tools = Tools(extras={"api_key": "SECRET"})  # extras override user args by default
tools.call(tool_call)

# Per-call control:
tools.call_with_extras({"api_key": "OTHER"}, tool_call, override=True)   # extras win
tools.call_with_extras({"api_key": "OTHER"}, tool_call, override=False)  # user args win
```

### Notes

* Return values are whatever your function returns—side effects are on you. Keep tools small and deterministic when possible.
* `tools.list()` keeps the original `type` hints for backward compatibility and adds a normalized `args` schema that’s friendlier for prompts.
* Safety: only register functions you actually want the model to invoke.

## Embeddings

The embeddings module uses **httpx** and supports Ollama embeddings:

```py
from trivialai.embedding import OllamaEmbedder
embed = OllamaEmbedder(model="nomic-embed-text", server="http://localhost:11434")
vec = embed("hello world")
```

---

## Notes & compatibility

* **Dependencies**: `httpx` replaces `requests`. Use `httpx[http2]` if you want HTTP/2 for OpenAI/Anthropic. Use `boto3` for AWS Bedrock.
* **Python**: ≥ **3.9** (we use `asyncio.to_thread`).
* **Scratchpad**: only **Ollama** surfaces `<think>` content; others emit `scratchpad` as `""` in deltas and `None` in the final event.
* **GCP/Vertex AI**: primarily for setup/auth. No native provider streaming; `astream` falls back to a single final chunk unless you override.
* **Bedrock**: `stream(...)` uses `converse_stream()`; token counts (when available) are surfaced as `tokens` in the final `end` event.
