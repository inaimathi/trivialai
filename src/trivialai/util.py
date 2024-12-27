import json
import os
import re
from collections import namedtuple

LLMResult = namedtuple("LLMResult", ["raw", "content"])


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
            return LLMResult(res.raw, transformFn(res.content))
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
