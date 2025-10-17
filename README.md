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

## Tool calls (unchanged)

```py
>>> from trivialai import ollama
>>> from trivialai.tools import Tools
>>> from typing import Optional, List
>>> tls = Tools()

>>> @tls.define()
... def _screenshot(url: str, selectors: Optional[List[str]] = None) -> None:
...     "Takes a url and an optional list of selectors. Takes a screenshot"
...     print("GOT", url, selectors)

>>> client = ollama.Ollama("deepseek-v2:16b", "http://localhost:11434/")
>>> res = client.generate_tool_call(tls, "sys", "Take a screenshot of Google and highlight the search box")
>>> res.content
{'functionName': '_screenshot', 'args': {'url': 'https://www.google.com', 'selectors': ['#search']}}
>>> tls.call(res.content)
GOT https://www.google.com ['#search']
```

---

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

