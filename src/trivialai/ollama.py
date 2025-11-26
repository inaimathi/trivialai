# ollama.py
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult


class Ollama(LLMMixin, FilesystemMixin):
    # Class-level defaults; can be overridden per-instance via __init__.
    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"

    def __init__(
        self,
        model: str,
        ollama_server: Optional[str] = None,
        timeout: Optional[float] = 300.0,
        skip_healthcheck: bool = False,
        open_think: Optional[str] = None,
        close_think: Optional[str] = None,
    ):
        self.server = (ollama_server or "http://localhost:11434").rstrip("/")
        self.model = model
        self.timeout = timeout

        # Per-instance override of think tags if provided.
        if open_think is not None:
            self.THINK_OPEN = open_think
        if close_think is not None:
            self.THINK_CLOSE = close_think

        if not skip_healthcheck:
            self._startup_health_check()

    def _startup_health_check(self) -> None:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                # 1) Server reachable and responding
                tags_resp = client.get(f"{self.server}/api/tags")
        except httpx.RequestError as e:
            raise ValueError(f"Cannot reach Ollama server at {self.server}: {e}") from e

        if tags_resp.status_code != 200:
            raise ValueError(
                f"Ollama server at {self.server} responded with HTTP {tags_resp.status_code} for /api/tags"
            )

        # 2) Model exists exactly as specified (no shorthands, no fallback)
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

    # ---- Sync full-generate (compat) ----

    def generate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        """
        Non-streaming Ollama call via /api/generate with stream=False.
        Scratchpad (think-block) parsing delegated to LLMMixin.split_think_full.
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
        content, scratch = self.split_think_full(raw_resp)
        return LLMResult(res, content, scratch)

    # ---- True async streaming (raw text; THINK handling in LLMMixin.stream) ----

    async def astream(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streams newline-delimited JSON from Ollama /api/generate with stream=True
        and yields base NDJSON-style events:

          - {"type":"start", "provider":"ollama", "model": "..."}
          - {"type":"delta", "text": "..."}       # may contain think tags
          - {"type":"end", "content": None}
          - {"type":"error", "message": "..."} on failure

        LLMMixin.stream(...) is responsible for splitting out think blocks
        into 'scratchpad' and cleaning the final 'content'.
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
                        delta = obj.get("response", "")
                        if not delta:
                            continue
                        yield {
                            "type": "delta",
                            "text": delta,
                        }

            except httpx.HTTPError as e:
                yield {"type": "error", "message": str(e)}
                return

        # Base end event; LLMMixin.stream will rewrite 'content' and 'scratchpad'.
        yield {
            "type": "end",
            "content": None,
        }
