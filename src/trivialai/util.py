import json
import os
import re


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


def strip_md_code(block):
    return re.sub("^```\\w+\n", "", block).removesuffix("```").strip()


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


def relative_path(base, path):
    stripped = path.strip("\\/")
    if not os.path.isfile(os.path.join(base, stripped)):
        raise TransformError("relative-file-doesnt-exist", raw=stripped)
    return stripped


def loadch(resp):
    if resp is None:
        raise TransformError("no-message-given")
    try:
        return json.loads(strip_md_code(resp.strip()))
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


def tree(target_dir, ignore=None, focus=None):
    assert os.path.exists(target_dir) and os.path.isdir(target_dir)

    if ignore is None:
        ignore = r"(^__|^env|^\.|~$|pyc$)"  # by default, ignore a bunch of python/emacs development intermediate files

    def is_excluded(name):
        ignore_match = re.search(ignore, name) if ignore else False
        focus_match = re.search(focus, name) if focus else True
        return ignore_match or not focus_match

    def build_tree(dir_path, prefix=""):
        entries = sorted(
            [entry for entry in os.listdir(dir_path) if not is_excluded(entry)]
        )
        lines = []

        for i, entry in enumerate(entries):
            entry_path = os.path.join(dir_path, entry)
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry}")

            if os.path.isdir(entry_path):
                child_prefix = f"{prefix}    " if is_last else f"{prefix}│   "
                lines.extend(build_tree(entry_path, child_prefix))

        return lines

    tree_lines = [target_dir]
    tree_lines.extend(build_tree(target_dir))
    return "\n".join(tree_lines)
