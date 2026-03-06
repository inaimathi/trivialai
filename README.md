# TrivialAI

*(A set of trivial bindings for AI models)*

## Install

```bash
pip install trivialai
# Optional: HTTP/2 for OpenAI/Anthropic
# pip install "trivialai[http2]"
# Optional: AWS Bedrock support (via boto3)
# pip install "trivialai[bedrock]"
# Optional: Google Gemini support
# pip install "trivialai[gemini]"
```

**Requirements**

* **Python ≥ 3.10** (the codebase uses `X | Y` type unions)
* Uses **httpx** for HTTP-based providers, **boto3** for Bedrock, and **google-genai** for Gemini

---

## Quick start

```py
>>> from trivialai import claude, gemini, ollama, chatgpt, bedrock
>>> from trivialai.stabdiff import StabDiff
```

> **Note:** The legacy `gcp` module (backed by `vertexai.generative_models`) has been removed.
> Use `gemini.Gemini` instead — it supports both the Gemini Developer API and Vertex AI,
> and provides text *and* image generation through a single client.

---

## Credentials

### Anthropic (Claude)

Use an [Anthropic Console](https://console.anthropic.com) API key directly:

```py
claude.Claude("claude-3-5-sonnet-20241022", os.environ["ANTHROPIC_API_KEY"])
```

### OpenAI (ChatGPT)

Use an [OpenAI Platform](https://platform.openai.com) API key:

```py
chatgpt.ChatGPT("gpt-4o-mini", os.environ["OPENAI_API_KEY"])
```

### Google Gemini

Go to **[Google AI Studio](https://aistudio.google.com)**, sign in with a Google account, and click
**"Get API key" → "Create API key"** in the left sidebar. The key starts with `AIza...`.
No billing setup is required for the free tier.

```py
gemini.Gemini(api_key=os.environ["GEMINI_API_KEY"])
```

For Vertex AI (service account or Application Default Credentials), see the
[Vertex AI auth section](#vertex-ai-auth) below.

### AWS Bedrock

1. Enable Bedrock and request model access in a supported region via the AWS Console.
2. Ensure your IAM user/role has `bedrock:Converse*` and `bedrock:InvokeModel*` permissions.
3. Provide credentials via `aws configure`, environment variables, instance role, or explicit keys.

```py
bedrock.Bedrock(
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    region="us-east-1",
)
```

### Stable Diffusion (AUTOMATIC1111 WebUI)

Start your AUTOMATIC1111 WebUI with the `--api` flag, then point `StabDiff` at it. No API key
is required; the client talks directly to the local (or remote) WebUI HTTP server.

```py
from trivialai.stabdiff import StabDiff

sd = StabDiff("http://127.0.0.1:7860")
```

If your WebUI is password-protected, pass `auth=("user", "password")`. The constructor performs
a health-check against `/sdapi/v1/sd-models` on startup; pass `skip_healthcheck=True` to
suppress this.

Key constructor parameters:

| Parameter | Default | Description |
|---|---|---|
| `webui_server` | `"http://127.0.0.1:7860"` | Base URL of the A1111 WebUI |
| `model` | `None` | Default checkpoint name (overrides per-call) |
| `timeout` | `300.0` | Generation request timeout in seconds |
| `progress_poll_interval` | `0.5` | Seconds between progress polls during streaming |
| `auth` | `None` | `(username, password)` tuple for basic auth |
| `use_override_settings` | `True` | Apply model via `override_settings` (non-mutating) |
| `include_previews` | `True` | Include in-progress preview images during streaming |

---

## Synchronous usage

### Ollama

```py
>>> client = ollama.Ollama("gemma2:2b", "http://localhost:11434/")
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

### ChatGPT (OpenAI API)

```py
>>> client = chatgpt.ChatGPT("gpt-4o-mini", os.environ["OPENAI_API_KEY"])
>>> client.generate("sys msg", "Say hi with 'platypus'.").content
"Hello, platypus!"
```

### Gemini (Google) — text + image

`Gemini` is a unified client: one object, one set of credentials, two capabilities.
`model` targets text generation; `image_model` targets image generation.
Both default to sensible values, so you can use either or both.

```py
# Text generation
>>> gem = gemini.Gemini(api_key=os.environ["GEMINI_API_KEY"])
>>> gem.generate(
...     system="Reply concisely.",
...     prompt="What is the capital of France?",
... ).content
"Paris."

# Image generation (txt2img)
>>> img = gem.generate_image("A corgi in a spacesuit floating above the Earth")
>>> img.file()
'/tmp/trivialai-img-ho9ftavj.png'

# Image editing (img2img)
>>> edited = gem.generate_image("Make it sunset colours", image=img)
>>> edited.file()
'/tmp/trivialai-img-x7q2kl1m.png'
```

Image and text models are independent — you can override either per-call or at construction:

```py
gem = gemini.Gemini(
    model="gemini-3-pro-preview",                     # text model
    image_model="gemini-3-pro-image-preview",         # image model (Nano Banana Pro)
    api_key=os.environ["GEMINI_API_KEY"],
)
```

To discover what models are available on your key:

```py
>>> gem.models()
{'text': [{'name': 'models/gemini-3-flash-preview', ...}, ...],
 'image': [{'name': 'models/gemini-3.1-flash-image-preview', ...}, ...]}

>>> gem.text_model_names()
['models/gemini-3-flash-preview', 'models/gemini-3-pro-preview', ...]
>>> gem.image_model_names()
['models/gemini-3.1-flash-image-preview', 'models/gemini-3-pro-image-preview', ...]
```

#### Vertex AI auth

```py
# Service account JSON file (project auto-read from the file)
gem = gemini.Gemini(vertex_api_creds="/path/to/sa.json", region="us-central1")

# Application Default Credentials (gcloud auth application-default login)
gem = gemini.Gemini(project="my-gcp-project", region="us-central1", use_vertexai=True)
```

### Bedrock (AWS) — text + image

`Bedrock` is also a unified client. `model_id` targets text (via the Converse API);
`image_model_id` targets image generation (via InvokeModel). Both are optional and independent.

```py
client = bedrock.Bedrock(
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    image_model_id="amazon.nova-canvas-v1:0",           # default
    region="us-east-1",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)

# Text
res = client.generate(
    system="You are a helpful assistant.",
    prompt="Explain neural networks in one sentence.",
)
print(res.content)

# Image (txt2img)
img = client.generate_image("A watercolour fox reading a book in an autumn forest")
img.file()   # → '/tmp/trivialai-img-4ai11zoz.png'

# Image (img2img)
edited = client.generate_image("Add snow", image=img)
```

Supported image models: Nova Canvas (`amazon.nova-canvas-v1:0`), Titan Image
(`amazon.titan-image-generator-v2:0`), and Stability AI (`stability.*`).

To discover available models in your region:

```py
>>> client.models()
{'text': [{'model_id': 'anthropic.claude-3-5-sonnet-20241022-v2:0', ...}, ...],
 'image': [{'model_id': 'amazon.nova-canvas-v1:0', ...}, ...]}

>>> client.image_model_ids()
['amazon.nova-canvas-v1:0', 'amazon.titan-image-generator-v2:0', ...]
```

#### Choosing the right `model_id`

Bedrock distinguishes between **foundation model IDs** (`anthropic.claude-3-5-sonnet-20241022-v2:0`)
and **inference profile IDs** (`us.anthropic.claude-3-5-sonnet-20241022-v2:0`).
Some models/regions require the region-prefixed profile ID. If you get a validation error
about on-demand throughput, switch to the `us.` / `eu.` prefixed form.

### Stable Diffusion (AUTOMATIC1111 WebUI) — image only

`StabDiff` wraps the AUTOMATIC1111 WebUI REST API, routing to `/sdapi/v1/txt2img` or
`/sdapi/v1/img2img` automatically based on whether an input image is provided.

```py
from trivialai.stabdiff import StabDiff

sd = StabDiff("http://127.0.0.1:7860", model="realisticVisionV60B1_v51VAE.safetensors")

# Text-to-image
img = sd.generate_image("A fox in a moonlit forest, oil painting style")
img.file()   # → '/tmp/trivialai-img-xxxx.png'

# Image-to-image (pass any Picture, bytes, file path, or PIL image)
edited = sd.generate_image("Add falling cherry blossoms", image=img)
edited.file()
```

Any standard A1111 payload field can be passed as a keyword argument:

```py
img = sd.generate_image(
    "Portrait of an astronaut, cinematic lighting",
    steps=30,
    width=768,
    height=1024,
    cfg_scale=7.5,
    sampler_name="DPM++ 2M Karras",
    negative_prompt="blurry, watermark, low quality",
    seed=42,
)
```

By default, the active checkpoint is changed non-destructively via `override_settings` —
the WebUI's globally loaded model is left untouched after the request completes. To switch the
globally loaded model instead, use `set_model`:

```py
sd.set_model("dreamshaper_8.safetensors")          # updates globally
sd.set_model("another.safetensors", persist=False) # updates self.model only
```

#### Model and LoRA discovery

```py
>>> sd.models()
['realisticVisionV60B1_v51VAE.safetensors', 'dreamshaper_8.safetensors', ...]

>>> sd.loras()
['add-detail-xl', 'epi_noiseoffset2', ...]
```

`models_full()` and `loras_full()` return the raw dicts from the WebUI API if you need
additional metadata such as file paths or hashes.

#### WebUI options

```py
# Read current WebUI options
opts = sd.options()

# Write one or more options
sd.set_options(sd_vae="vae-ft-mse-840000-ema-pruned.safetensors")
```

---

## Streaming (NDJSON-style events) via `BiStream`

All providers expose a common streaming shape via `stream(...)`.

**Important:** `stream(...)` (and helpers like `stream_checked(...)` / `stream_json(...)`) return a
**`BiStream`**, which supports both sync and async iteration.

### LLM event schema

* `{"type":"start", "provider":"<ollama|openai|anthropic|gemini|bedrock>", "model":"..."}`
* `{"type":"delta", "text":"...", "scratchpad":"..."}`
  * **Ollama**: `scratchpad` may contain content extracted from `<think>…</think>`.
  * **Gemini**: `scratchpad` carries native thought tokens (no tag parsing needed).
  * Other providers: `scratchpad` is typically `""` in deltas.
* `{"type":"end", "content":"...", "scratchpad": <str|None>, "tokens": <int|None>}`
* `{"type":"error", "message":"..."}`

`stream_checked(...)` / `stream_json(...)` append a final parse event:

* `{"type":"final", "ok": true|false, "parsed": ..., "error": ..., "raw": ...}`

### Image stream event schema

Image generation via `imagestream(...)` yields:

* `{"type":"start", "provider":"...", "model":"...", "mode":"txt2img"|"img2img"}`
* `{"type":"progress", "progress": 0.0–1.0, "state":"...", "textinfo":"..."}` *(where supported)*
* `{"type":"end", "image": ImageResult, "model":"...", "mode":"..."}`
* `{"type":"error", "message":"..."}`

> **Note on Gemini / Bedrock image streaming:** both APIs are single-shot REST calls with no
> server-sent progress. The stream emits a synthetic `progress: 0.0` event immediately
> before the blocking call (so progress-bar consumers see activity), then an `end` event
> when the image resolves. The `end` payload is identical to other providers.

### Example: streaming text (sync)

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
    if ev["type"] == "final":
        print("Parsed JSON:", ev["parsed"])

# Shortcut:
for ev in client.stream_json("sys", "Return {'name':'Platypus'} as JSON."):
    if ev["type"] == "final":
        print("Parsed:", ev["parsed"])
```

### Example: streaming image (Gemini)

```py
gem = gemini.Gemini(api_key=os.environ["GEMINI_API_KEY"])

for ev in gem.imagestream("A rainy Tokyo street at night, neon reflections"):
    if ev["type"] == "progress":
        print(f"  {ev['textinfo']}")
    elif ev["type"] == "end":
        ev["image"].file("tokyo.png")
```

### Example: streaming image (Bedrock)

```py
client = bedrock.Bedrock(image_model_id="amazon.nova-canvas-v1:0", region="us-east-1")

for ev in client.imagestream("A watercolour fox reading a book in an autumn forest"):
    if ev["type"] == "end":
        ev["image"].file("fox.png")
```

### Example: streaming image (Stable Diffusion)

`StabDiff` provides real step-by-step progress from the A1111 progress API, including
optional in-progress preview images.

```py
sd = StabDiff("http://127.0.0.1:7860")

for ev in sd.imagestream("A misty mountain landscape at dawn, Studio Ghibli style"):
    if ev["type"] == "progress":
        pct = (ev["progress"] or 0) * 100
        eta = ev.get("eta_relative") or 0
        print(f"  {pct:.0f}%  ETA {eta:.1f}s  {ev.get('textinfo', '')}")
        if ev.get("image"):          # live preview frame (if include_previews=True)
            ev["image"].file("preview.png")
    elif ev["type"] == "end":
        ev["image"].file("result.png")
    elif ev["type"] == "error":
        print("Error:", ev["message"])
```

The progress event also carries a `"state"` dict with the raw A1111 job state, and a
`"progress-error"` event type is emitted (non-fatally) if a single poll fails mid-generation.

To cancel an in-progress generation, call `sd.interrupt()` (or `await sd.ainterrupt()` in
async contexts). When using `imagestream` via an async generator, cancellation is handled
automatically via `asyncio.CancelledError`.

```py
# Disable preview frames to reduce polling overhead
for ev in sd.imagestream("...", include_previews=False):
    ...
```

### Example: streaming text (async)

```py
async for ev in client.stream("sys", "Stream something."):
    ...
```

---

## `BiStream`: one stream interface for sync + async

```py
from trivialai.bistream import BiStream
```

`BiStream[T]` wraps a sync `Iterable[T]`, an async `AsyncIterable[T]`, or another `BiStream[T]`
and exposes **both** iterator interfaces.

**Key behaviour:**

* **Single-consumer:** once consumed, exhausted.
* **Mode-locked:** a given instance may be consumed *either* sync *or* async.
* **Bridging:** async → sync driven by a background event loop thread; sync → async wraps `next()`.

---

## Chaining streams with `then` / `map` / `mapcat` / `branch`

All combinators are mode-preserving (sync in → sync out, async in → async out).

### `then(...)`: append a follow-up stage after upstream terminates

```py
pipeline = client.stream("sys", "Answer, streaming.").then(lambda: [
    {"type": "note", "text": "stream ended"},
])
```

Your follow-up can be 0-arg or 1-arg (`done` receives `StopIteration.value` if present).

### `map(...)`: transform each event

```py
pipeline = client.stream("sys", "Stream.").map(
    lambda ev: (ev | {"text": ">> " + ev["text"]}) if ev.get("type") == "delta" else ev
)
```

### `mapcat(...)`: per-item stream expansion (flatMap), with optional concurrency

```py
events = BiStream(["a.py", "b.py", "c.py"]).mapcat(
    lambda path: agent.streamed(f"Analyze {path}"),
    concurrency=8,
)
```

### `branch(...)`: fan-out, then fan-in via `.sequence()` / `.interleave()`

```py
base = client.stream("sys", "First: describe the plan.")
fan  = base.branch(["doc1", "doc2", "doc3"],
                   lambda doc: client.stream("sys", f"Summarize: {doc}"))
for ev in fan.interleave(concurrency=8):
    handle(ev)
```

---

## Extra helpers

### `tap(...)`: side effects without changing events

```py
stream = client.stream("sys", "Stream.").tap(lambda ev: log(ev))
```

### `repeat_until(...)`: agent loops

```py
from trivialai.bistream import repeat_until, is_type

looped = repeat_until(
    src=client.stream("sys", "First attempt..."),
    step=lambda driver: client.stream("sys", f"Next attempt, based on {driver}..."),
    stop=is_type("final"),
    max_iters=10,
)
```

---

## Embeddings

```py
from trivialai.embedding import OllamaEmbedder

embed = OllamaEmbedder(model="nomic-embed-text", server="http://localhost:11434")
vec = embed("hello world")
```

---

## Notes & compatibility

* **Dependencies:** `httpx` for HTTP providers; `boto3` for Bedrock; `google-genai` + optionally
  `google-auth` for Gemini. `StabDiff` uses only `httpx` — no extra install step needed.
* **Scratchpad:** Ollama surfaces `<think>` content; Gemini routes native thought tokens;
  other providers emit `scratchpad=""` in deltas and `None` in the final `end`.
* **`gcp` module removed:** the old `gcp.GCP` class (backed by `vertexai.generative_models`,
  deprecated June 2025) has been removed. Migrate to `gemini.Gemini` — it supports all three
  auth modes the old class did, plus image generation.
* **BiStream:** single-use and single-consumer — don't consume the same instance from multiple tasks.
* **StabDiff model selection:** by default the checkpoint is applied via `override_settings` so
  the globally loaded WebUI model is never mutated. Set `use_override_settings=False` or call
  `set_model(..., persist=True)` if you need to control the global state explicitly.
