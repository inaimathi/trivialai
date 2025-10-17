# TrivialAI

*(A set of **httpx**-based, trivial bindings for AI models — now with optional streaming)*

## Install

```bash
pip install pytrivialai
# Optional: HTTP/2 for OpenAI/Anthropic
# pip install "pytrivialai[http2]"
```

* Requires **Python ≥ 3.9**.
* Uses **httpx** (no more `requests`).

## Quick start

```py
>>> from trivialai import claude, gcp, ollama, chatgpt
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

### Claude

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

### ChatGPT

```py
>>> client = chatgpt.ChatGPT("gpt-4o-mini", os.environ["OPENAI_API_KEY"])
>>> client.generate("sys msg", "Say hi with 'platypus'.").content
"Hello, platypus!"
```

---

## Streaming (NDJSON-style events)

All providers expose a common streaming shape via `stream(...)` (sync iterator) and `astream(...)` (async):

**Event schema**

* `{"type":"start", "provider": "<ollama|openai|anthropic|gcp>", "model": "..."}`
* `{"type":"delta", "text":"...", "scratchpad":"..."}`

  * For **Ollama**, `scratchpad` contains model “thinking” extracted from `<think>…</think>`.
  * For **ChatGPT**/**Claude**, `scratchpad` is `""` (empty).
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

All LLM clients support a helper that prompts for a tool call and validates it:

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

* **Dependencies**: `httpx` replaces `requests`. Use `httpx[http2]` if you want HTTP/2 for OpenAI/Anthropic.
* **Python**: ≥ **3.9** (we use `asyncio.to_thread`).
* **Scratchpad**: only **Ollama** surfaces `<think>` content; others emit `scratchpad` as `""` in deltas and `None` in the final event.
* **GCP/Vertex AI**: primarily for setup/auth. No native provider streaming; `astream` falls back to a single final chunk unless you override.

---

## Changelog (highlights)

* **0.3.0**

  * Switched to **httpx**; removed `requests`.
  * Added **streaming** interface (`stream`, `astream`) with a unified event schema.
  * Exposed **Ollama** `<think>` content live via `scratchpad` in deltas.
  * Added `stream_checked` / `astream_checked` helpers to parse the final output while preserving deltas.
  * Tightened typing across modules; added tests.

