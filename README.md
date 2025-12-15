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

**Requirements**

* **Python ≥ 3.10** (the codebase uses `X | Y` type unions).
* Uses **httpx** for HTTP-based providers and **boto3** for Bedrock.

---

## Quick start

```py
>>> from trivialai import claude, gcp, ollama, chatgpt, bedrock
```

---

## Synchronous usage

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

1. Enable Bedrock + model access in a Bedrock-supported region.
2. Ensure your IAM user/role can call Bedrock runtime APIs (`bedrock:Converse*`, `bedrock:InvokeModel*`, etc).
3. Provide credentials via the normal AWS credential chain (`aws configure`, env vars, instance role) or explicit keys.

#### 2) Choosing the right `model_id`

Bedrock distinguishes between:

* **Foundation model IDs**, like: `anthropic.claude-3-5-sonnet-20241022-v2:0`
* **Inference profile IDs**, which are region-prefixed, like: `us.anthropic.claude-3-5-sonnet-20241022-v2:0`

Some models/regions require using the inference profile ID. If you see a validation error about on-demand throughput, switch to the region-prefixed ID.

#### 3) Minimal Bedrock demo

```py
from trivialai import bedrock

client = bedrock.Bedrock(
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    region="us-east-1",
)

res = client.generate(
    "This is a test message. Make sure your reply contains the word 'margarine'",
    "Hello there! Can you hear me?"
)
print(res.content)

res_json = client.generate_json(
    "You are a JSON-only assistant.",
    "Return {'name':'Platypus'} as JSON."
)
print(res_json.content)
```

---

## Streaming (NDJSON-style events) via `BiStream`

All providers expose a common streaming shape via `stream(...)`.

**Important:** `stream(...)` (and helpers like `stream_checked(...)` / `stream_json(...)`) return a **`BiStream`**, which supports both:

* sync iteration (`for ev in ...`)
* async iteration (`async for ev in ...`)

You usually don’t need to call provider-specific `astream(...)` anymore.

### Event schema

A streaming LLM yields NDJSON-style events:

* `{"type":"start", "provider":"<ollama|openai|anthropic|gcp|bedrock>", "model":"..."}`
* `{"type":"delta", "text":"...", "scratchpad":"..."}`

  * For **Ollama**, `scratchpad` may contain model “thinking” extracted from `<think>…</think>`.
  * For other providers, `scratchpad` is typically `""` in deltas.
* `{"type":"end", "content":"...", "scratchpad": <str|None>, "tokens": <int|None>}`
* `{"type":"error", "message":"..."}`

On top of that, `stream_checked(...)` / `stream_json(...)` append a final parse event:

* `{"type":"final", "ok": true|false, "parsed": ..., "error": ..., "raw": ...}`

### Example: streaming (sync)

```py
client = ollama.Ollama("gemma2:2b", "http://localhost:11434/")

for ev in client.stream("sys", "Explain, think step-by-step."):
    if ev["type"] == "delta":
        print(ev["text"], end="")
    elif ev["type"] == "end":
        print("\n-- scratchpad --")
        print(ev["scratchpad"])
```

### Example: streaming + parse-at-end

```py
from trivialai.util import loadch

for ev in client.stream_checked(loadch, "sys", "Return a JSON object gradually."):
    if ev["type"] in {"start", "delta", "end"}:
        # UI updates
        pass
    elif ev["type"] == "final":
        print("Parsed JSON:", ev["parsed"])
```

Shortcut:

```py
for ev in client.stream_json("sys", "Return {'name':'Platypus'} as JSON."):
    if ev["type"] == "final":
        print("Parsed:", ev["parsed"])
```

### Example: streaming (async)

```py
async for ev in client.stream("sys", "Stream something."):
    ...
```

---

## `BiStream`: one stream interface for sync + async

```py
from trivialai.bistream import BiStream
```

### What it wraps

`BiStream[T]` can wrap:

* a **sync** `Iterable[T]` (generator/list/range/…)
* an **async** `AsyncIterable[T]` (async generator/…)
* another `BiStream[T]`

…and exposes **both** iterator interfaces.

### Key behavior (important)

* **Single-consumer:** it’s a stream, not a list. Once consumed, it’s exhausted.
* **Mode-locked:** a given instance may be consumed **either** sync **or** async.
  If you start consuming it sync, you can’t later consume the *same instance* async (and vice versa). This prevents subtle “half-sync / half-async” bugs.
* **Bridging behavior:**

  * async → sync: driven by a dedicated background event loop thread (used only for bridging).
  * sync → async: an async wrapper calls `next()` inside the event loop thread; if a `next()` blocks, the loop is blocked and `BiStream` will log a warning once.

### Construction notes

* `BiStream.ensure(x)` returns `x` unchanged if it’s already a `BiStream`.
* `BiStream(other_bistream)` **shares** the same underlying iterators, so consumption progress is shared.

---

## Chaining streams with `then` / `map` / `mapcat` / `branch`

TrivialAI uses a small set of *mode-preserving* combinators to build pipelines without caring whether you’re in sync or async code.

### `then(...)`: append a follow-up stage after upstream terminates

`then` is **termination-driven** (not event-driven):

* yields all upstream events unchanged
* when upstream ends, it calls your follow-up exactly once
* yields all events from the returned follow-up stream (if any)

**New behavior:** your follow-up can be either:

1. **0-arg**: `then(lambda: stream)`
2. **1-arg**: `then(lambda done: stream)`

`done` is:

* **sync**: `StopIteration.value` if the generator `return`s a value (else `None`)
* **async**: first `StopAsyncIteration` arg if present (else `None`)

#### Pseudocode: append a constant postlude

```py
base = client.stream("sys", "Answer, streaming.")

pipeline = base.then(lambda: [
    {"type": "note", "text": "stream ended"},
    {"type": "done", "ok": True},
])

for ev in pipeline:
    handle(ev)
```

#### Pseudocode: use `done` when you have it

```py
def gen():
    yield {"type": "delta", "text": "hi"}
    return {"tokens": 123}

pipeline = BiStream(gen()).then(lambda done: [{"type": "stats", "done": done}])
# yields: delta, then stats
```

#### Pattern: parse/validate after end

```py
def parse_after_end(_done):
    yield {"type": "final", "ok": True, "parsed": compute_structured_result()}

pipeline = client.stream("sys", "Return JSON gradually.").then(parse_after_end)
```

---

### `map(...)`: transform each event

`map` is the standard per-event transformation:

```py
# prefix all delta text with ">> "
pipeline = client.stream("sys", "Stream.").map(
    lambda ev: (ev | {"text": ">> " + ev["text"]}) if ev.get("type") == "delta" else ev
)
```

This stays mode-preserving: sync in → sync out, async in → async out.

---

### `mapcat(...)`: per-item stream expansion (flatMap), with optional concurrency

`mapcat` lets you turn each event/item into an entire stream and flatten the result.

* `mapcat(fn)` defaults to sequential flattening (like `sequence()`).
* `mapcat(fn, concurrency=N)` flattens by interleaving up to `N` active branches.

#### Pseudocode: expand “files” into per-file agent streams (sequential)

```py
files = BiStream(["a.py", "b.py", "c.py"])

def per_file(path):
    return agent.streamed(f"Analyze {path}")

events = files.mapcat(per_file)  # sequential
for ev in events:
    handle(ev)
```

#### Pseudocode: concurrent interleaving (async-friendly)

```py
files = BiStream(["a.py", "b.py", "c.py"])

def per_file(path):
    return agent.streamed(f"Analyze {path}")  # may be async stream

events = files.mapcat(per_file, concurrency=8)  # interleaved merge
async for ev in events:
    handle(ev)
```

Notes:

* `mapcat(..., concurrency>0)` uses `FanOut.interleave(...)` internally.
* If you consume the result synchronously, it will be bridged via the background loop (same as any async BiStream).

---

### `branch(...)`: fan-out, then fan-in via `.sequence()` / `.interleave()`

There are two entry points:

* **Free function**: `bistream.branch(src_items, mk_stream)` → returns `FanOut`
* **Method**: `BiStream.branch(items, per_item, ...)` → “gated” fan-out (drain prefix first)

A `FanOut` is not an event stream yet — it must be fanned back in:

* `.sequence()` — run branches one-by-one, preserving order
* `.interleave(concurrency=...)` — run branches concurrently and merge events as they arrive

#### Pseudocode: gated fan-out

```py
base = client.stream("sys", "First: describe the plan.")
docs = ["doc1", "doc2", "doc3"]

def per_doc(doc):
    return client.stream("sys", f"Summarize: {doc}")

fan = base.branch(docs, per_doc)     # base is the prefix
merged = fan.interleave(concurrency=8)

for ev in merged:
    handle(ev)
```

---

## Extra helpers you’ll see in pipelines

### `tap(...)`: side effects without changing events

```py
stream = client.stream("sys", "Stream.").tap(lambda ev: log(ev))
```

Optional filters:

* `focus(ev) -> bool`: only tap matching events
* `ignore(ev) -> bool`: tap everything except matching events

---

### `repeat_until(...)`: loop a stream-producing step with an event-based stop

Useful for “agent loops” that keep running steps until a “final”/“conclusion”/etc appears.

```py
from trivialai.bistream import repeat_until, is_type

looped = repeat_until(
    src=client.stream("sys", "First attempt..."),
    step=lambda driver: client.stream("sys", f"Next attempt, based on {driver}..."),
    stop=is_type("final"),
    max_iters=10,
)
```

`repeat_until` best-effort closes underlying iterators on early exit and on exceptions/consumer abort.

---

## Embeddings

```py
from trivialai.embedding import OllamaEmbedder

embed = OllamaEmbedder(model="nomic-embed-text", server="http://localhost:11434")
vec = embed("hello world")
```

---

## Notes & compatibility

* **Dependencies**: `httpx` replaces `requests`. Use `httpx[http2]` if you want HTTP/2 for OpenAI/Anthropic. Use `boto3` for AWS Bedrock.
* **Scratchpad**:

  * **Ollama** may surface `<think>` content as `scratchpad` deltas and a final scratchpad string.
  * Other providers usually emit `scratchpad=""` in deltas and `None` in the final `end`.
* **GCP/Vertex AI**: streaming may fall back to a single final chunk unless a native streaming provider implementation is present.
* **BiStream**: single-use and single-consumer by design — don’t try to consume the same instance concurrently from multiple tasks.
