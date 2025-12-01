# util.py
from __future__ import annotations

import base64
import inspect as _inspect
import os
import re
from collections import namedtuple
from typing import Any, AsyncIterator, Callable, Dict, Optional

import httpx
import json5

from .bistream import BiStream
from .log import getLogger

LLMResult = namedtuple("LLMResult", ["raw", "content", "scratchpad"])


class TransformError(Exception):
    def __init__(self, message: str = "Transformation Error", raw: Any = None):
        self.message = message
        self.raw = raw
        super().__init__(self.message)


class GenerationError(Exception):
    def __init__(self, message: str = "Generation Error", raw: Any = None):
        self.message = message
        self.raw = raw
        super().__init__(self.message)


def generate_checked(
    gen: Callable[[], Any], transformFn: Callable[[str], Any]
) -> LLMResult:
    """
    One-shot call to an LLM generator with a transform function applied to its content.
    """
    res = gen()
    return LLMResult(res.raw, transformFn(res.content), res.scratchpad)


def strip_md_code(block: str) -> str:
    block = block.strip()

    # First, try to match a full fenced block with any language identifier
    m = re.match(r"^```[^\n]*\n(.*)\n```$", block, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Fallback: if it’s just ```...``` on one line or similar, strip the fences
    if block.startswith("```") and block.endswith("```"):
        return block[3:-3].strip()

    # No fences detected; return as-is
    return block


def strip_to_first_md_code(block: str) -> str:
    """
    Extract the contents of the *first* fenced code block in a Markdown string.
    Returns "" if none exists.
    """
    pattern = r"^.*?```\w+\n(.*?)\n```.*$"
    match = re.search(pattern, block, re.DOTALL)
    return match.group(1).strip() if match else ""


def invert_md_code(
    md_block: str,
    comment_start: Optional[str] = None,
    comment_end: Optional[str] = None,
) -> str:
    """
    Invert code vs. non-code lines in a Markdown block.

    Lines inside ``` fences are left as-is.
    Lines outside code blocks are prefixed/suffixed with comment markers.
    """
    lines = md_block.splitlines()
    in_code_block = False
    result: list[str] = []
    c_start = comment_start if comment_start is not None else "## "
    c_end = comment_end if comment_end is not None else ""

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
        else:
            result.append(line if in_code_block else f"{c_start}{line}{c_end}")

    return "\n".join(result)


def relative_path(base: str, path: str, must_exist: bool = True) -> str:
    stripped = path.strip("\\/")
    if (not os.path.isfile(os.path.join(base, stripped))) and must_exist:
        raise TransformError("relative-file-doesnt-exist", raw=stripped)
    return stripped


def loadch(resp: Any) -> Any:
    """
    "Load Chat" helper: parse a JSON-ish response into Python.

    - If resp is a string, it may contain a Markdown code block; we strip it and json5-load it.
    - If resp is already a list/dict/tuple, pass it through.
    - Otherwise raise TransformError("parse-failed").
    """
    if resp is None:
        raise TransformError("no-message-given")
    try:
        if type(resp) is str:
            return json5.loads(strip_md_code(resp.strip()))
        elif type(resp) in {list, dict, tuple}:
            return resp
    except (TypeError, ValueError):
        pass
    raise TransformError("parse-failed")


def slurp(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()


def spit(file_path: str, content: str, mode: Optional[str] = None) -> None:
    dr = os.path.dirname(file_path)
    if dr:
        os.makedirs(dr, exist_ok=True)
    with open(file_path, mode or "w") as dest:
        dest.write(content)


def _tree(target_dir: str, ignore: Optional[str] = None, focus: Optional[str] = None):
    def is_excluded(name: str) -> bool:
        ignore_match = re.search(ignore, name) if ignore else False
        focus_match = re.search(focus, name) if focus else True
        return bool(ignore_match or not focus_match)

    def build_tree(dir_path: str, prefix: str = ""):
        entries = sorted(
            [entry for entry in os.listdir(dir_path) if not is_excluded(entry)]
        )

        for i, entry in enumerate(entries):
            entry_path = os.path.join(dir_path, entry)
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            yield f"{prefix}{connector}{entry}"

            if os.path.isdir(entry_path):
                child_prefix = f"{prefix}    " if is_last else f"{prefix}│   "
                for ln in build_tree(entry_path, child_prefix):
                    yield ln

    yield target_dir
    for ln in build_tree(target_dir):
        yield ln


def tree(
    target_dir: str, ignore: Optional[str] = None, focus: Optional[str] = None
) -> str:
    assert os.path.exists(target_dir) and os.path.isdir(target_dir)
    return "\n".join(_tree(target_dir, ignore, focus))


def deep_ls(directory: str, ignore: Optional[str] = None, focus: Optional[str] = None):
    ignore_pattern = re.compile(ignore) if ignore else None
    focus_pattern = re.compile(focus) if focus else None

    for root, dirs, files in os.walk(directory):
        if ignore_pattern:
            dirs[:] = [
                d for d in dirs if not ignore_pattern.search(os.path.join(root, d))
            ]
        if focus_pattern:
            dirs[:] = [d for d in dirs if focus_pattern.search(os.path.join(root, d))]

        for file in files:
            full_path = os.path.join(root, file)

            if ignore_pattern and ignore_pattern.search(full_path):
                continue

            if focus_pattern and not focus_pattern.search(full_path):
                continue

            yield full_path


def mk_local_files(in_dir: str, must_exist: bool = True):
    def _local_files(resp: Any) -> list[str]:
        try:
            rsp = resp if type(resp) is str else strip_to_first_md_code(resp)
            loaded = loadch(rsp)
            if type(loaded) is not list:
                raise TransformError("relative-file-response-not-list", raw=resp)
            return [relative_path(in_dir, f, must_exist=must_exist) for f in loaded]
        except Exception:
            pass
        raise TransformError("relative-file-translation-failed", raw=resp)

    return _local_files


def b64file(pathname: str) -> str:
    with open(pathname, "rb") as f:
        raw = f.read()
        return base64.b64encode(raw).decode("utf-8")


def b64url(url: str) -> str:
    with httpx.Client() as c:
        r = c.get(url)
        r.raise_for_status()
        return base64.b64encode(r.content).decode("utf-8")


async def astream_checked(
    stream_src: Any,
    transformFn: Callable[[str], Any],
) -> AsyncIterator[Dict[str, Any]]:
    """
    Async one-shot streaming: pass-through events and finally emit {"type":"final",...}.
    No retries here.

    `stream_src` can be:
      - a synchronous iterator / iterable of events
      - an async iterator / async iterable of events
      - a BiStream of events

    We normalize with BiStream and then consume it asynchronously.
    """
    stream = BiStream.ensure(stream_src)

    buf: list[str] = []
    last_end: Optional[Dict[str, Any]] = None

    async for ev in stream:
        t = ev.get("type")
        if t == "delta":
            buf.append(ev.get("text", ""))
        elif t == "end":
            last_end = ev
        yield ev

    full = "".join(buf) if buf else ((last_end or {}).get("content") or "")
    try:
        parsed = transformFn(full)
        yield {"type": "final", "ok": True, "parsed": parsed}
    except TransformError as e:
        yield {"type": "final", "ok": False, "error": e.message, "raw": e.raw}


def stream_checked(
    stream_src: Any,
    transformFn: Callable[[str], Any],
) -> BiStream[Dict[str, Any]]:
    """
    Dual-mode wrapper around astream_checked using BiStream.

    - Sync:
        for ev in stream_checked(stream, transformFn): ...
    - Async:
        async for ev in stream_checked(stream, transformFn): ...

    where `stream` can be a plain generator, an async generator, or a BiStream.
    """
    return BiStream(astream_checked(stream_src, transformFn))
