from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx

from .filesystem import FilesystemMixin
from .llm import LLMMixin, LLMResult

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
MAX_TAG_LEN = max(len(_THINK_OPEN), len(_THINK_CLOSE))
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _split_think_full(resp: str) -> Tuple[str, Optional[str]]:
    """
    Given a full response string, remove a single <think>...</think> section (if present)
    and return (public_text, scratchpad_text). If none present, scratchpad is None.
    """
    m = _THINK_RE.search(resp)
    if not m:
        return resp, None
    scratch = m.group(0)[len(_THINK_OPEN) : -len(_THINK_CLOSE)].strip()
    content = (resp[: m.start()] + resp[m.end() :]).strip()
    return content, (scratch or None)


def _separate_think_delta(
    delta: str,
    in_think: bool,
    carry: str,
) -> Tuple[str, str, bool, str]:
    """
    Streaming-safe splitter that handles tag fragments across chunk boundaries.

    Args:
      delta: current chunk
      in_think: whether we're inside <think>…</think>
      carry: trailing fragment from previous chunk that *might* be a tag prefix

    Returns:
      (public_out, scratch_out, new_in_think, new_carry)
    """
    text = carry + delta
    i = 0
    out_pub: list[str] = []
    out_scr: list[str] = []

    def _maybe_partial_tag_at(idx: int) -> bool:
        # If remaining text at idx is a prefix of an open/close tag, we need more bytes.
        remaining = text[idx:]
        # only matters if we're near the end (i.e., not enough chars to decide)
        if len(remaining) >= MAX_TAG_LEN:
            return False
        return _THINK_OPEN.startswith(remaining) or _THINK_CLOSE.startswith(remaining)

    while i < len(text):
        # Full tags
        if text.startswith(_THINK_OPEN, i):
            in_think = True
            i += len(_THINK_OPEN)
            continue
        if text.startswith(_THINK_CLOSE, i):
            in_think = False
            i += len(_THINK_CLOSE)
            continue

        # Partial tag at buffer end? keep it in carry for the next call
        if _maybe_partial_tag_at(i):
            carry = text[i:]
            break

        # Normal emission
        ch = text[i]
        if in_think:
            out_scr.append(ch)
        else:
            out_pub.append(ch)
        i += 1
        carry = ""  # we've consumed any previous carry

    # If we consumed all text, ensure carry is empty
    if i >= len(text):
        carry = ""

    return ("".join(out_pub), "".join(out_scr), in_think, carry)


class Ollama(LLMMixin, FilesystemMixin):
    def __init__(
        self,
        model: str,
        ollama_server: Optional[str] = None,
        timeout: Optional[float] = 300.0,
    ):
        self.server = (ollama_server or "http://localhost:11434").rstrip("/")
        self.model = model
        self.timeout = timeout

        # ---- Tight invariant checks: server up AND model present ----
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
        content, scratch = _split_think_full(raw_resp)
        return LLMResult(res, content, scratch)

    # ---- Async full-generate built on top of streaming ----
    async def agenerate(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> LLMResult:
        content_parts: list[str] = []
        scratch_parts: list[str] = []
        async for ev in self.astream(system, prompt, images):
            t = ev.get("type")
            if t == "delta":
                content_parts.append(ev.get("text", "") or "")
                scratch_parts.append(ev.get("scratchpad", "") or "")
            elif t == "end":
                # prefer provider-supplied final values if present
                if "content" in ev and ev["content"] is not None:
                    content_parts = [ev["content"]]
                if "scratchpad" in ev and ev["scratchpad"] is not None:
                    scratch_parts = [ev["scratchpad"]]
        content = "".join(content_parts)
        scratch = "".join(scratch_parts) if any(scratch_parts) else None
        return LLMResult(raw=None, content=content, scratchpad=scratch)

    # ---- True async streaming ----
    async def astream(
        self, system: str, prompt: str, images: Optional[list] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streams newline-delimited JSON from Ollama /api/generate with stream=True
        and yields NDJSON-style events:
          - {"type":"start", "provider":"ollama", "model": "..."}
          - {"type":"delta", "text": "...", "scratchpad": "..."}  # text and/or scratchpad may be ""
          - {"type":"end", "content": "...", "scratchpad": "...", "tokens": int}
          - {"type":"error", "message": "..."} on failure

        The model's internal <think>…</think> blocks are exposed via the 'scratchpad'
        field; tags are not included in either field.
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
        carry = ""  # track partial tag fragments at boundaries

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
                        out, scr, in_think, carry = _separate_think_delta(
                            delta, in_think, carry
                        )
                        # buffer both
                        if scr:
                            scratch_buf.append(scr)
                        if out:
                            content_buf.append(out)
                        # emit a delta if either side has material (uniform shape)
                        if out or scr:
                            yield {
                                "type": "delta",
                                "text": out or "",
                                "scratchpad": scr or "",
                            }

            except httpx.HTTPError as e:
                yield {"type": "error", "message": str(e)}
                return

        # FLUSH any left-over carry (not a real tag after all)
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
