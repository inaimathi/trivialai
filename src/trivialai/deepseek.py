# src/trivialai/deepseek.py
"""
DeepSeek adapter for trivialai — text generation via the DeepSeek Messages API.

DeepSeek exposes an OpenAI-compatible HTTP interface, so the wire format
(JSON body, SSE streaming, error shapes) closely mirrors the OpenAI Chat
Completions API.  No OpenAI SDK is required; plain ``httpx`` is used, exactly
as in the ``Claude`` adapter.

Two model families are supported
---------------------------------
``deepseek-chat``
    Standard chat model.  Produces only ``content``; ``scratchpad`` is always
    ``None``.

``deepseek-reasoner``
    Extended-reasoning model.  The API returns ``reasoning_content`` alongside
    the final ``content``.  ``trivialai`` maps ``reasoning_content`` to
    ``LLMResult.scratchpad``.

Scratchpad routing in streaming
--------------------------------
``LLMMixin.stream`` routes scratchpad exclusively through ``<think>…</think>``
tag parsing on the ``text`` field of each delta event — it does not forward a
``scratchpad`` key emitted directly by ``astream``.  For ``deepseek-reasoner``,
``astream`` therefore synthesises the tags around ``reasoning_content`` tokens
as they arrive:

  * First ``reasoning_content`` token  → emits ``"<think>" + token``
  * Subsequent reasoning tokens        → emitted verbatim
  * First ``content`` token after any  → emits ``"</think>" + token``
    reasoning tokens
  * Content tokens with no prior       → emitted verbatim (no tags)
    reasoning

``THINK_OPEN = "<think>"`` and ``THINK_CLOSE = "</think>"`` are set at the
class level so ``LLMMixin`` knows to apply the splitter.  For ``deepseek-chat``
no reasoning tokens appear and no tags are ever injected.

Auth
----
    from trivialai.deepseek import DeepSeek

    ds = DeepSeek(model="deepseek-chat", api_key="sk-...")
    result = ds.generate("You are helpful.", "What is 2 + 2?")
    print(result.content)     # "4"
    print(result.scratchpad)  # None

    ds_r = DeepSeek(model="deepseek-reasoner", api_key="sk-...")
    result = ds_r.generate("You are helpful.", "Prove that sqrt(2) is irrational.")
    print(result.content)     # final answer
    print(result.scratchpad)  # chain-of-thought

Streaming
---------
    async for ev in ds.stream("system", "prompt"):
        if ev["type"] == "delta":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "end":
            print()
            print("scratchpad:", ev["scratchpad"])
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "deepseek-chat"
_BASE_URL = "https://api.deepseek.com"
_MESSAGES_PATH = "/chat/completions"

# deepseek-reasoner is the only current model that exposes reasoning_content.
# All other model IDs are treated as plain-chat.
_REASONER_MODELS = {"deepseek-reasoner"}


def _is_reasoner(model: str) -> bool:
    """Return True if this model returns ``reasoning_content`` natively."""
    return model in _REASONER_MODELS


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class DeepSeek(LLMMixin, FilesystemMixin):
    """
    DeepSeek Messages API adapter: sync + async text generation.

    Parameters
    ----------
    model:
        Model ID string.  Defaults to ``"deepseek-chat"``.
        Use ``"deepseek-reasoner"`` for the extended-reasoning model.
    api_key:
        DeepSeek API key (``sk-...``).
    base_url:
        Override the API root (useful for self-hosted or proxy endpoints).
        Defaults to ``"https://api.deepseek.com"``.
    max_tokens:
        Maximum tokens to generate.  Defaults to 4096.
    temperature:
        Sampling temperature.  Ignored by ``deepseek-reasoner`` (which uses
        a fixed temperature of 1).  Defaults to ``1.0``.
    timeout:
        Per-request timeout in seconds.  ``None`` = no timeout.

    Notes
    -----
    * ``THINK_OPEN`` / ``THINK_CLOSE`` are set to the standard ``<think>``
      tags.  ``astream`` synthesises those tags around ``reasoning_content``
      tokens so that ``LLMMixin.stream`` can route them into ``scratchpad``
      through its normal tag-splitting path.
    * ``images`` parameters are accepted in the LLMMixin signature but
      silently ignored — DeepSeek's current public API does not support
      image inputs.
    """

    # LLMMixin uses these to split <think>…</think> out of the text stream.
    # astream injects the tags around reasoning_content tokens so the mixin's
    # existing parser handles scratchpad routing without any custom logic here.
    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        timeout: Optional[float] = 300.0,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _body(self, system: str, prompt: str, *, stream: bool) -> Dict[str, Any]:
        """Build the JSON request body for the Chat Completions endpoint."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        # deepseek-reasoner ignores temperature, but sending it is harmless.
        body["temperature"] = self.temperature

        if stream and _is_reasoner(self.model):
            # Request reasoning_content deltas in the SSE stream.
            body["stream_options"] = {"include_usage": False}

        return body

    def _url(self) -> str:
        return f"{self.base_url}{_MESSAGES_PATH}"

    # ------------------------------------------------------------------
    # Synchronous generate (LLMMixin requirement)
    # ------------------------------------------------------------------

    def generate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        """
        Blocking single-shot call to ``POST /chat/completions`` with
        ``stream=false``.

        ``reasoning_content`` (``deepseek-reasoner`` only) is returned as
        ``LLMResult.scratchpad``.  For all other models ``scratchpad`` is
        ``None``.
        """
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(
                self._url(),
                headers=self._headers(),
                json=self._body(system, prompt, stream=False),
            )

        if res.status_code != 200:
            return LLMResult(raw=res, content=None, scratchpad=None)

        try:
            j = res.json()
            message = j["choices"][0]["message"]
        except Exception:
            return LLMResult(raw=res, content=None, scratchpad=None)

        content: Optional[str] = (message.get("content") or "").strip() or None
        # deepseek-reasoner populates this field; chat model returns None/missing.
        scratchpad: Optional[str] = (
            message.get("reasoning_content") or ""
        ).strip() or None

        return LLMResult(raw=res, content=content, scratchpad=scratchpad)

    # ------------------------------------------------------------------
    # Async streaming (LLMMixin requirement)
    # ------------------------------------------------------------------

    async def astream(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream ``POST /chat/completions`` with ``stream=true`` (SSE).

        Yields base events consumed by ``LLMMixin.stream``:

        ``{"type": "start", "provider": "deepseek", "model": "..."}``
            Emitted once before any data arrives.

        ``{"type": "delta", "text": "..."}``
            Incremental text.  For ``deepseek-reasoner``, reasoning tokens are
            wrapped in ``<think>…</think>`` tags injected at the phase boundary
            so ``LLMMixin.stream`` can route them into ``scratchpad`` via its
            normal tag-splitting path.  For all other models, ``text`` is the
            raw answer token with no extra markup.

        ``{"type": "end", "content": None}``
            Signals stream completion.  ``LLMMixin.stream`` writes the real
            ``content`` and ``scratchpad`` values from its accumulation buffer.

        ``{"type": "error", "message": "..."}``
            Emitted on HTTP / JSON errors; ``astream`` returns immediately.

        Think-tag injection for ``deepseek-reasoner``
        ---------------------------------------------
        DeepSeek's SSE stream sends ``reasoning_content`` tokens first (the
        thinking phase) and ``content`` tokens second (the answer phase); the
        two phases never interleave.  ``astream`` tracks the phase with a
        boolean flag and injects tags at exactly two points:

          * ``"<think>"`` is prepended to the **first** reasoning token.
          * ``"</think>"`` is prepended to the **first** content token that
            follows at least one reasoning token.

        If no reasoning tokens arrive (e.g. the model skips thinking for a
        trivial query), no tags are injected and the stream is plain text.
        """
        yield {"type": "start", "provider": "deepseek", "model": self.model}

        # Phase-tracking state for think-tag injection (reasoner model only).
        # reasoning_open=True means we have emitted "<think>" but not yet "</think>".
        reasoning_open = False

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    self._url(),
                    headers=self._headers(),
                    json=self._body(system, prompt, stream=True),
                ) as resp:
                    if resp.status_code != 200:
                        yield {
                            "type": "error",
                            "message": f"DeepSeek HTTP {resp.status_code}",
                        }
                        return

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue

                        data = line[5:].strip()
                        if data == "[DONE]":
                            break

                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        choices = obj.get("choices")
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        finish_reason = choices[0].get("finish_reason")

                        text_piece: str = delta.get("content") or ""
                        scratch_piece: str = delta.get("reasoning_content") or ""

                        if scratch_piece:
                            # Reasoning phase: open the think block on first token.
                            if not reasoning_open:
                                reasoning_open = True
                                scratch_piece = "<think>" + scratch_piece
                            yield {"type": "delta", "text": scratch_piece}

                        elif text_piece:
                            # Answer phase: close the think block if one is open.
                            if reasoning_open:
                                reasoning_open = False
                                text_piece = "</think>" + text_piece
                            yield {"type": "delta", "text": text_piece}

                        if finish_reason == "stop":
                            break

            except httpx.HTTPError as exc:
                yield {"type": "error", "message": str(exc)}
                return

        # Close an unclosed think block (shouldn't happen in practice, but be safe).
        if reasoning_open:
            yield {"type": "delta", "text": "</think>"}

        # LLMMixin.stream rewrites content + scratchpad from its accumulation buffer.
        yield {"type": "end", "content": None}
