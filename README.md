# TrivialAI

*(A set of trivial bindings for AI models)*

## Install

```bash
pip install trivialai
# Optional: HTTP/2 for OpenAI/Anthropic
# pip install "trivialai[http2]"
# Optional: AWS Bedrock support (via boto3)
# pip install "trivialai[bedrock]"
````

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
* `stream_checked(...)` / `stream_json(...)`

No special-casing required in downstream code.

---

## Streaming (NDJSON-style events, via `BiStream`)

All providers expose a common streaming shape via `stream(...)`.

**Important:** `stream(...)` (and the higher-level `stream_checked(...)`, `stream_json(...)`) now return a **`BiStream`**, which:

* acts as a **normal iterator** in sync code (`for ev in client.stream(...): ...`), and
* acts as an **async iterator** in async code (`async for ev in client.stream(...): ...`).

You almost never need to touch `astream(...)` directly anymore; `stream(...)` is the unified interface.

**Event schema**

Each streaming LLM yields NDJSON-style events:

* `{"type":"start", "provider": "<ollama|openai|anthropic|gcp|bedrock>", "model": "..."}`
* `{"type":"delta", "text":"...", "scratchpad":"..."}`

  * For **Ollama**, `scratchpad` contains model “thinking” extracted from `<think>…</think>`.
  * For **ChatGPT**, **Claude API**, **GCP**, and **Bedrock**, `scratchpad` is `""` (empty) in deltas.
* `{"type":"end", "content":"...", "scratchpad": <str|None>, "tokens": <int>}`
* `{"type":"error", "message":"..."}`

On top of that, the `stream_checked(...)` / `stream_json(...)` helpers append a **final** event with a parsed payload:

* `{"type":"final", "ok": true|false, "parsed": ..., "error": ..., "raw": ...}`

(See below.)

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

### Parse-at-end streaming (`stream_checked` / `stream_json`)

If you want incremental updates *and* a structured parse at the end, use the LLM-level helpers:

```py
from trivialai.util import loadch

# Parse final output as JSON, with retries, but still get deltas for UI.
for ev in client.stream_checked(loadch, "sys", "Return a JSON object gradually."):
    if ev["type"] in {"start", "delta", "end"}:
        # good for UI updates
        print(ev)
    elif ev["type"] == "final":
        print("Parsed JSON:", ev["parsed"])
```

Shortcut: `stream_json(...)` is just `stream_checked(loadch, ...)`:

```py
for ev in client.stream_json("sys", "Return {'name':'Platypus'} as JSON."):
    if ev["type"] == "final":
        print("Parsed:", ev["parsed"])
```

You can also use the *utility-level* `stream_checked` on arbitrary streams (not just LLMs):

```py
from trivialai.util import stream_checked, loadch

raw_stream = client.stream("sys", "Return JSON gradually.")
for ev in stream_checked(raw_stream, loadch):
    ...
```

In all cases, `stream_checked(...)` returns a **`BiStream`**, so you can use it in sync or async code.

### Async flavor (using `BiStream`)

Because `stream(...)` returns a `BiStream`, you can use it directly in async code:

```py
# Recommended: use .stream(...) everywhere, even in async code
async for ev in client.stream("sys", "Stream something."):
    ...

# If you really want the provider's raw async stream:
async for ev in client.astream("sys", "Stream something."):
    ...
```

*(For Bedrock, `stream(...)` is the native streaming interface; `astream(...)` currently falls back to the default `LLMMixin` behavior unless you wrap it yourself.)*

---

## `BiStream`: sync/async bridge for streams

`BiStream` is a tiny adapter type that underpins all of TrivialAI’s streaming APIs.

```py
from trivialai.bistream import BiStream
```

### What it does

`BiStream[T]` wraps:

* a **synchronous** `Iterable[T]` (e.g. generator, list, `range(...)`),
* an **asynchronous** `AsyncIterable[T]` (e.g. `async def` generator), or
* another `BiStream[T]`,

and presents **both** of these interfaces:

* `Iterator[T]` — usable with `for`, `next()`, `list()`, `itertools`, etc.
* `AsyncIterator[T]` — usable with `async for`.

In other words:

```py
bs = BiStream(async_stream())   # or BiStream(sync_stream())

# Sync code:
for item in bs:
    ...

# Async code:
async for item in bs:
    ...
```

### How it behaves

* **Single-shot:** it’s a stream, not a list. Once you’ve iterated it fully (sync *or* async), it’s exhausted.
* **One underlying source:** if the source is async, there’s a single async iterator and the sync side bridges via a background loop.
  If the source is sync, the async side bridges via a tiny async wrapper.
* **Idempotent construction:** `BiStream.ensure(bs)` returns `bs` unchanged if it’s already a `BiStream`.
* **Sharing progress:** `BiStream` constructed from another `BiStream` reuses its underlying iterators, so partial consumption is shared.

A few practical notes:

* It’s great for **library boundaries**: your code can accept or return “something stream-like” and not care if callers are sync or async.
* It’s not a random-access container. Don’t rely on indexing or resetting; if you need that, buffer into a list yourself.
* Avoid consuming the same `BiStream` concurrently in multiple tasks; it’s single-consumer by design.

TrivialAI uses `BiStream` for:

* LLM methods: `stream(...)`, `stream_checked(...)`, `stream_json(...)`
* Utility helpers: `util.stream_checked(...)`, `util.astream_checked(...)` (the latter normalizes to `BiStream` internally)
* Higher-level orchestration code (e.g., RAG + chat pipelines) that wants a single streaming interface for Tornado handlers and REPLs.

---

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
* **Scratchpad**:

  * **Ollama** surfaces `<think>` content as `scratchpad` deltas and a final scratchpad string.
  * Other providers emit `scratchpad` as `""` in deltas and `None` in the final event.
* **GCP/Vertex AI**: primarily for setup/auth. No native provider streaming; `stream(...)` falls back to a single final chunk unless you override.
* **Bedrock**: `stream(...)` uses `converse_stream()`; token counts (when available) are surfaced as `tokens` in the final `end` event.
* **BiStream**:

  * All `stream*` helpers return `BiStream` so you can write code once and use it in both sync + async contexts.
  * A `BiStream` is single-use; don’t try to consume the same instance from multiple tasks at once.
