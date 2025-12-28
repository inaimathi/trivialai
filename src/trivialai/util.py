# util.py
from __future__ import annotations

import base64
import os
import re
from collections import namedtuple
from typing import (Any, AsyncIterator, Callable, Dict, Generator, List,
                    Optional)

import httpx

from . import jsonesque
from .bistream import BiStream
from .json_shaped import (Exact, JsonInput,  # re-exported symbols
                          JsonPrimitive, JsonValue, ShapeSpec, TransformError,
                          is_json_shaped, json_shape)
from .log import getLogger

logger = getLogger("trivialai.util")

LLMResult = namedtuple("LLMResult", ["raw", "content", "scratchpad"])


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


def lenient_load(candidate: str) -> JsonValue:
    """
    Parse a JSON-esque candidate string with a bit of leniency:

      1. Try jsonesque.loads(candidate) as-is.
      2. If that fails and there are literal newline characters, try again
         after replacing '\n' chars with '\\n' so multiline string values
         in *non*-triple-quoted contexts get a second chance.

    On success, return the parsed value.
    On failure, raise TransformError("parse-failed").
    """
    candidate = candidate.strip()
    if not candidate:
        raise TransformError("parse-failed", raw=candidate)

    # First attempt: as-is, using the JSON-esque parser
    try:
        return jsonesque.loads(candidate)
    except ValueError:
        pass

    # Second attempt: escape bare newlines if any
    if "\n" in candidate:
        try:
            escaped = candidate.replace("\n", "\\n")
            return jsonesque.loads(escaped)
        except ValueError:
            pass

    raise TransformError("parse-failed", raw=candidate)


def loadch(resp: Any) -> JsonValue:
    """
    Parse a JSON-ish response into Python.

    - If resp is a string, it may contain a Markdown code block; we strip it and
      parse it via `lenient_load` (jsonesque-based).
    - If resp is already a list/dict/tuple, pass it through.
    - Otherwise raise TransformError("parse-failed").

    This uses a JSON-esque parser which is more permissive than the Python
    built-in json library, accepting constructs like

      {foo: 1, 'bar': 2, "baz": 3}

    and some Python-ish extensions.
    """
    if resp is None:
        raise TransformError("no-message-given")

    if isinstance(resp, (list, dict, tuple, float, int, bool)):
        return resp

    if not isinstance(resp, str):
        raise TransformError("parse-failed")

    return lenient_load(strip_md_code(resp.strip()))


def loadchmulti(resp: Any) -> list[JsonValue]:
    """
    Like loadch, but return a list of JSON-ish values instead of a single one.

    - If resp is a list/tuple: return list(resp).
    - If resp is a dict: return [resp].
    - If resp is a string:
        * Try to parse all fenced code blocks as JSON-esque (leniently).
        * If none, scan for top-level {...} blocks in the text and parse each.
    - Otherwise raise TransformError("parse-failed").

    This is more permissive than loadch in that it can recover multiple
    JSON-ish objects from a single string, but it only accepts substrings that
    our lenient loader can actually parse.
    """
    if resp is None:
        raise TransformError("no-message-given")

    # Already-structured values
    if isinstance(resp, (list, tuple)):
        return list(resp)
    if isinstance(resp, dict):
        return [resp]

    if not isinstance(resp, str):
        raise TransformError("parse-failed")

    text = resp.strip()
    results: list[JsonValue] = []

    # ---- 1) Try all fenced markdown code blocks first ---------------------
    # Matches ```lang\n<content>\n``` or ```\n<content>\n```
    code_blocks = re.findall(r"```[^\n]*\n(.*?)\n```", text, re.DOTALL)
    for block in code_blocks:
        candidate = block.strip()
        if not candidate:
            continue

        try:
            value = lenient_load(candidate)
        except TransformError:
            continue

        if isinstance(value, list):
            results.extend(value)
        else:
            results.append(value)

    if results:
        return results

    # ---- 2) Scan for top-level {...} blocks in the raw text --------------
    results.extend(_extract_jsonish_objects(text))

    if results:
        return results

    # No parseable JSON-ish substrings found
    raise TransformError("parse-failed")


def _extract_jsonish_objects(text: str) -> list[JsonValue]:
    """
    Best-effort extraction of JSON-ish objects from a string by scanning
    for balanced {...} blocks outside of markdown fences.

    We only accept substrings that `lenient_load` can parse.
    """
    objs: list[JsonValue] = []

    in_fence = False
    fence_pattern = re.compile(r"^```")
    lines = text.splitlines()
    rebuilt: list[str] = []

    # Strip out fenced code blocks (handled earlier) so brace scanning
    # isn't confused by arbitrary code inside them.
    for line in lines:
        if fence_pattern.match(line.strip()):
            in_fence = not in_fence
            continue
        if not in_fence:
            rebuilt.append(line)

    s = "\n".join(rebuilt)

    start = None
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = s[start : i + 1].strip()
                    if not candidate:
                        start = None
                        continue

                    try:
                        value = lenient_load(candidate)
                    except TransformError:
                        # Not actually JSON-ish (or too malformed); ignore.
                        pass
                    else:
                        if isinstance(value, list):
                            objs.extend(value)
                        else:
                            objs.append(value)

                    start = None

    return objs


def read_ndjson(
    filepath: str,
    encoding: str = "utf-8",
) -> Generator[Dict[str, Any], None, None]:
    """
    Read an ndjson (newline-delimited JSON) file line by line using the
    jsonesque/lenient parser.

    Args:
        filepath: Path to the ndjson file
        encoding: File encoding (default: utf-8)

    Yields:
        Dict: Each parsed JSON(-esque) object from the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If a line cannot be parsed as JSON-esque
    """
    with open(filepath, "r", encoding=encoding) as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    # We could call jsonesque.loads directly, but lenient_load
                    # gives us the same "newline escape" fallback as elsewhere.
                    value = lenient_load(line)
                except TransformError as e:
                    error_msg = e.message or str(e)
                    # Maintain a similar ValueError shape to json/json5 errors:
                    # (msg, doc, pos)
                    raise ValueError(
                        f"Invalid JSON on line {line_num}: {error_msg}",
                        line,
                        0,
                    )
                else:
                    # This generator is historically annotated as yielding Dicts.
                    # If you want stricter behavior, enforce dict here.
                    yield value  # type: ignore[misc]


def slurp_ndjson(filepath: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    return list(read_ndjson(filepath, encoding=encoding))


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
