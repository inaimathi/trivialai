# ollama.py
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult


class Ollama(LLMMixin, FilesystemMixin):
    """
    Ollama-backed LLM implementation.

    - Uses /api/generate with stream=False for sync generate().
    - Uses /api/generate with stream=True for true incremental streaming.
    - Separates internal <think>...</think> blocks into a scratchpad stream,
      using the LLMMixin's scratchpad helpers.
    """

    # Configure scratchpad tag boundaries for LLMMixin helpers
    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"

    def __init__(
        self,
        model: str,
        ollama_server: Optional[str] = None,
        timeout: Optional[float] = 300.0,
        skip_healthcheck: bool = False,
    ):
        self.server = (ollama_server or "http://localhost:11434").rstrip("/")
        self.model = model
        self.timeout = timeout
        if not skip_healthcheck:
            self._startup_health_check()

    # ---- Startup health check (sync, one-shot) ------------------------------

    def _startup_health_check(self) -> None:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                tags_resp = client.get(f"{self.server}/api/tags")
        except httpx.RequestError as e:
            raise ValueError(f"Cannot reach Ollama server at {self.server}: {e}") from e

        if tags_resp.status_code != 200:
            raise ValueError(
                f"Ollama server at {self.server} responded with HTTP "
                f"{tags_resp.status_code} for /api/tags"
            )

        try:
            show_resp = httpx.post(
                f"{self.server}/api/show",
                json={"name": self.model},
                timeout=self.timeout,
            )
        except httpx.RequestError as e:
            raise ValueError(
                f"Failed to query model '{self.model}' on {self.server}: {e}"
            ) from e

        if show_resp.status_code != 200:
            raise ValueError(
                f"Model '{self.model}' is not available on Ollama server {self.server} "
                f"(HTTP {show_resp.status_code} from /api/show)."
            )

    # ---- Sync full-generate (compat) ----------------------------------------

    def generate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        """
        Non-streaming generate using Ollama /api/generate with stream=False.

        Uses LLMMixin._split_think_full to separate public content vs scratchpad.
        """
        data: Dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "prompt": f"SYSTEM PROMPT: {system} PROMPT: {prompt}",
        }
        if images is not None:
            data["images"] = images

        url = f"{self.server}/api/generate"
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(url, json=data)

        if res.status_code != 200:
            return LLMResult(res, None, None)

        raw_resp = res.json().get("response", "").strip()
        content, scratch = self._split_think_full(raw_resp)
        return LLMResult(res, content, scratch)

    # ---- Async full-generate built on top of streaming ----------------------

    async def agenerate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        """
        Aggregate over the streaming interface to reconstruct both public text
        and scratchpad, preserving the same semantics as the streaming path.

        This overrides LLMMixin.agenerate (which would otherwise use to_thread).
        """
        content_parts: list[str] = []
        scratch_parts: list[str] = []

        async for ev in self.stream(system, prompt, images):
            t = ev.get("type")
            if t == "delta":
                content_parts.append(ev.get("text", "") or "")
                scratch_parts.append(ev.get("scratchpad", "") or "")
            elif t == "end":
                # Prefer provider-supplied final values if present
                if "content" in ev and ev["content"] is not None:
                    content_parts = [ev["content"]]
                if "scratchpad" in ev and ev["scratchpad"] is not None:
                    scratch_parts = [ev["scratchpad"]]

        content = "".join(content_parts)
        scratch = "".join(scratch_parts) if any(scratch_parts) else None
        return LLMResult(raw=None, content=content, scratchpad=scratch)

    # ---- True async streaming used by LLMMixin.stream -----------------------

    async def _astream_raw(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streams newline-delimited JSON from Ollama /api/generate with stream=True
        and yields NDJSON-style events:
          - {"type":"start", "provider":"ollama", "model": "..."}
          - {"type":"delta", "text": "...", "scratchpad": "..."}  # one side may be ""
          - {"type":"end", "content": "...", "scratchpad": "...", "tokens": int}
          - {"type":"error", "message": "..."} on failure

        The model's internal <think>…</think> blocks are exposed via 'scratchpad';
        tags themselves are not included in either text or scratchpad. Boundaries
        are handled by LLMMixin._separate_think_delta across chunk boundaries.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "prompt": f"SYSTEM PROMPT: {system} PROMPT: {prompt}",
        }
        if images is not None:
            payload["images"] = images

        yield {"type": "start", "provider": "ollama", "model": self.model}

        url = f"{self.server}/api/generate"
        content_buf: list[str] = []
        scratch_buf: list[str] = []
        in_think = False
        carry = ""

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        yield {
                            "type": "error",
                            "message": f"Ollama HTTP {resp.status_code}",
                        }
                        return

                    async for line in resp.aiter_lines():
                        if not line:
                            continue

                        # Ollama returns NDJSON (one JSON object per line)
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            # tolerate non-JSON noise
                            continue

                        if obj.get("done"):
                            break

                        delta = obj.get("response", "") or ""
                        if not delta:
                            continue

                        out, scr, in_think, carry = self._separate_think_delta(
                            delta, in_think, carry
                        )

                        if scr:
                            scratch_buf.append(scr)
                        if out:
                            content_buf.append(out)

                        if out or scr:
                            # "Either text or scratchpad (or both), but tags stripped"
                            yield {
                                "type": "delta",
                                "text": out or "",
                                "scratchpad": scr or "",
                            }

            except httpx.HTTPError as e:
                yield {"type": "error", "message": str(e)}
                return

        # Flush any carry that turned out not to be a tag
        if carry:
            if in_think:
                scratch_buf.append(carry)
                yield {"type": "delta", "text": "", "scratchpad": carry}
            else:
                content_buf.append(carry)
                yield {"type": "delta", "text": carry, "scratchpad": ""}

        final_content = "".join(content_buf)
        final_scratch = "".join(scratch_buf) or None
        yield {
            "type": "end",
            "content": final_content,
            "scratchpad": final_scratch,
            "tokens": len(final_content.split()) if final_content else 0,
        }
