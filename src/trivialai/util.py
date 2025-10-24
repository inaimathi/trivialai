import base64
import inspect as _inspect
import json
import logging
import os
import re
from collections import namedtuple
from typing import (Any, AsyncIterator, Awaitable, Callable, Dict, Iterator,
                    Optional)

import httpx

LLMResult = namedtuple("LLMResult", ["raw", "content", "scratchpad"])


def getLogger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


class TransformError(Exception):
    def __init__(self, message="Transformation Error", raw=None):
        self.message = message
        self.raw = raw
        super().__init__(self.message)


class GenerationError(Exception):
    def __init__(self, message="Generation Error", raw=None):
        self.message = message
        self.raw = raw
        super().__init__(self.message)


def generate_checked(gen, transformFn, retries=5):
    for i in range(retries):
        res = gen()
        try:
            return LLMResult(res.raw, transformFn(res.content), res.scratchpad)
        except TransformError:
            pass
    raise GenerationError(f"failed-on-{retries}-retries", raw=res)


def strip_md_code(block):
    return re.sub("^```\\w+\n", "", block).removesuffix("```").strip()


def strip_to_first_md_code(block):
    pattern = r"^.*?```\w+\n(.*?)\n```.*$"
    match = re.search(pattern, block, re.DOTALL)
    return match.group(1).strip() if match else ""


def invert_md_code(md_block, comment_start=None, comment_end=None):
    lines = md_block.splitlines()
    in_code_block = False
    result = []
    c_start = comment_start if comment_start is not None else "## "
    c_end = comment_end if comment_end is not None else ""

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
        else:
            result.append(line if in_code_block else f"{c_start}{line}{c_end}")

    return "\n".join(result)


def relative_path(base, path, must_exist=True):
    stripped = path.strip("\\/")
    if (not os.path.isfile(os.path.join(base, stripped))) and must_exist:
        raise TransformError("relative-file-doesnt-exist", raw=stripped)
    return stripped


def loadch(resp):
    if resp is None:
        raise TransformError("no-message-given")
    try:
        if type(resp) is str:
            return json.loads(strip_md_code(resp.strip()))
        elif type(resp) in {list, dict, tuple}:
            return resp
    except (TypeError, json.decoder.JSONDecodeError):
        pass
    raise TransformError("parse-failed")


def slurp(pathname):
    with open(pathname, "r") as f:
        return f.read()


def spit(file_path, content, mode=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode or "w") as dest:
        dest.write(content)


def _tree(target_dir, ignore=None, focus=None):
    def is_excluded(name):
        ignore_match = re.search(ignore, name) if ignore else False
        focus_match = re.search(focus, name) if focus else True
        return ignore_match or not focus_match

    def build_tree(dir_path, prefix=""):
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


def tree(target_dir, ignore=None, focus=None):
    assert os.path.exists(target_dir) and os.path.isdir(target_dir)
    return "\n".join(_tree(target_dir, ignore, focus))


def deep_ls(directory, ignore=None, focus=None):
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


def mk_local_files(in_dir, must_exist=True):
    def _local_files(resp):
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


def b64file(pathname):
    with open(pathname, "rb") as f:
        raw = f.read()
        return base64.b64encode(raw).decode("utf-8")


def b64url(url):
    with httpx.Client() as c:
        r = c.get(url)
        r.raise_for_status()
        return base64.b64encode(r.content).decode("utf-8")


# --- Streaming “check at the end” wrappers ---


def stream_checked(
    stream_iter: Iterator[Dict[str, Any]], transformFn: Callable[[str], Any]
) -> Iterator[Dict[str, Any]]:
    """
    Pass-through NDJSON events *and* emit a final parsed event once complete.
    Yields events. The last event includes {"type":"final","ok":true|false,"parsed":..., "error":...}
    """
    buf: list[str] = []
    last_end: Optional[Dict[str, Any]] = None

    for ev in stream_iter:
        t = ev.get("type")
        if t == "delta":
            # accumulate visible text only
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


async def astream_checked(
    stream_agen: AsyncIterator[Dict[str, Any]], transformFn: Callable[[str], Any]
) -> AsyncIterator[Dict[str, Any]]:
    """
    Async variant of stream_checked for async generators.
    """
    buf: list[str] = []
    last_end: Optional[Dict[str, Any]] = None

    async for ev in stream_agen:
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


# --- NEW: Retry-capable streaming wrappers (attempt logic centralized here) ---


def stream_checked_retries(
    stream_factory: Callable[[], Iterator[Dict[str, Any]]],
    transformFn: Callable[[str], Any],
    *,
    retries: int = 5,
) -> Iterator[Dict[str, Any]]:
    """
    Like stream_checked, but will retry up to `retries` times by re-invoking `stream_factory()`.
    Emits passthrough events for each attempt; on parse failure emits:
      {"type":"attempt-failed","attempt":N,"error": "...","raw": ...}
    On success:
      {"type":"final","ok":true,"parsed":...,"attempt":N}
    After exhausting attempts:
      {"type":"final","ok":false,"error":"failed-on-<retries>-retries","attempts":retries,"last_error": "..."}
    """
    last_error: Optional[TransformError] = None
    for attempt in range(1, max(1, retries) + 1):
        buf: list[str] = []
        last_end: Optional[Dict[str, Any]] = None

        for ev in stream_factory():
            t = ev.get("type")
            if t == "delta":
                buf.append(ev.get("text", ""))
            elif t == "end":
                last_end = ev
            yield ev

        full = "".join(buf) if buf else ((last_end or {}).get("content") or "")
        try:
            parsed = transformFn(full)
            yield {"type": "final", "ok": True, "parsed": parsed, "attempt": attempt}
            return
        except TransformError as e:
            last_error = e
            yield {
                "type": "attempt-failed",
                "attempt": attempt,
                "error": e.message,
                "raw": e.raw,
            }

    yield {
        "type": "final",
        "ok": False,
        "error": f"failed-on-{retries}-retries",
        "attempts": retries,
        "last_error": getattr(last_error, "message", None),
    }


async def astream_checked_retries(
    stream_factory: Callable[
        [], AsyncIterator[Dict[str, Any]] | Awaitable[AsyncIterator[Dict[str, Any]]]
    ],
    transformFn: Callable[[str], Any],
    *,
    retries: int = 5,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Async variant of stream_checked_retries.

    Accepts a factory that returns either:
      - an AsyncIterator[Dict[str, Any]], or
      - an Awaitable that resolves to an AsyncIterator[Dict[str, Any]].
    """
    last_error: Optional[TransformError] = None
    for attempt in range(1, max(1, retries) + 1):
        buf: list[str] = []
        last_end: Optional[Dict[str, Any]] = None

        stream_obj = stream_factory()
        if _inspect.isawaitable(stream_obj):
            stream_obj = await stream_obj  # support async factories

        async for ev in stream_obj:
            t = ev.get("type")
            if t == "delta":
                buf.append(ev.get("text", ""))
            elif t == "end":
                last_end = ev
            yield ev

        full = "".join(buf) if buf else ((last_end or {}).get("content") or "")
        try:
            parsed = transformFn(full)
            yield {"type": "final", "ok": True, "parsed": parsed, "attempt": attempt}
            return
        except TransformError as e:
            last_error = e
            yield {
                "type": "attempt-failed",
                "attempt": attempt,
                "error": e.message,
                "raw": e.raw,
            }

    yield {
        "type": "final",
        "ok": False,
        "error": f"failed-on-{retries}-retries",
        "attempts": retries,
        "last_error": getattr(last_error, "message", None),
    }
