import textwrap
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Union)

from .toolkit import ToolKit

DEFAULT_CONTEXT_SIZE_CHARS = 3000


def build_prompt(
    base_system_prompt: str,
    user_prompt: str,
    tools: Optional[ToolKit] = None,
    *,
    context_size: int = DEFAULT_CONTEXT_SIZE_CHARS,
    memory: Optional[Any] = None,  # VectorStore | Collection | None (duck-typed)
    context_summary: Optional[str] = None,
) -> str:
    """
    Build a rich system prompt string from:
    - base_system_prompt: core instructions / persona / high-level goals
    - user_prompt: current task/request (used for memory lookup and guidance)
    - tools: functions or tool snippets the model can call
    - context_size: max size (characters) of the returned prompt
    - memory: VectorStore or Collection holding high-level indexed context
              (e.g. the current codebase index)
    - context_summary: short model-written summary of what's happened so far

    Returns:
        A single string intended as the `system` message for the LLM.
    """
    # 1) Base system instructions (or a default if blank)
    base_section = _build_base_section(base_system_prompt)

    # 2) Tools / capabilities section (optional)
    tools_section = "" if tools is None else tools.to_tool_prompt()

    # 3) High-priority context summary (if given)
    summary_section = _build_summary_section(context_summary)

    # 4) Memory-based retrieved context (lowest priority)
    memory_section = _build_memory_section(memory, user_prompt)

    # Combine sections with priority-aware trimming
    sections: List[Tuple[str, str]] = [
        ("base", base_section),
        ("summary", summary_section),
        ("tools", tools_section),
        ("memory", memory_section),
    ]

    prompt = _combine_sections_with_limit(sections, context_size)
    return prompt


def _build_base_section(base_system_prompt: str) -> str:
    text = (base_system_prompt or "").strip()
    if not text:
        text = textwrap.dedent(
            """
            You are an AI agent that can use tools to reason about complex tasks.

            - Follow the user's instructions carefully.
            - When tools are available, prefer using them to gather information
              or take actions before answering.
            - Explain your reasoning clearly and concisely in your final answer.
            """
        ).strip()
    return text


def _build_tools_section(tools: Sequence[Any]) -> str:
    if not tools:
        return ""

    lines: List[str] = []
    lines.append("## Tools")
    lines.append("")
    lines.append(
        "You have access to the following tools. "
        "Call them when they can help you make progress:"
    )
    lines.append("")

    for tool in tools:
        name, sig, desc = _describe_tool(tool)
        if sig:
            lines.append(f"- **{name}({sig})** – {desc}")
        else:
            lines.append(f"- **{name}** – {desc}")

    return "\n".join(lines).strip()


def _build_summary_section(context_summary: Optional[str]) -> str:
    if not context_summary:
        return ""
    summary = context_summary.strip()
    if not summary:
        return ""
    return textwrap.dedent(
        f"""
        ## Task / conversation summary

        {summary}
        """
    ).strip()


def _build_memory_section(memory: Optional[Any], user_prompt: str) -> str:
    """
    Use the memory (VectorStore or Collection) to retrieve relevant snippets.

    We duck-type:
    - Collection: has .queryBy(...)
    - VectorStore: has .listCollections() and .getCollection(name)
    """
    collection = _resolve_collection(memory)
    if collection is None:
        return ""

    try:
        results = collection.queryBy(
            content=user_prompt,
            maxResults=6,
            maxTokens=None,  # let the backend decide; we’ll truncate text anyway
            filter=None,
        )
    except Exception:
        # Fail-soft: don't break the prompt builder if memory query fails
        return ""

    if not results:
        return ""

    lines: List[str] = []
    lines.append("## Retrieved context")
    lines.append("")
    lines.append(
        "The following snippets were retrieved based on the user's current request. "
        "Use them as background knowledge, but always verify details when needed."
    )
    lines.append("")

    # A simple per-snippet char cap; global trimming happens later.
    max_chars_per_snippet = 400

    for i, res in enumerate(results, start=1):
        snippet = _search_result_to_snippet(res, max_chars=max_chars_per_snippet)
        if not snippet:
            continue
        lines.append(f"### Snippet {i}")
        lines.append(snippet)
        lines.append("")

    return "\n".join(lines).strip()


def _describe_tool(tool: Any) -> tuple[str, str, str]:
    """
    Turn a tool (callable or snippet dict) into (name, signature, description).

    - If it's a dict (e.g. from to_llm_snippet), we use its fields.
    - If it's a callable, we introspect __name__, __doc__, and __annotations__.
    """
    # Tool snippet dict case (e.g. from to_llm_snippet)
    if isinstance(tool, dict) and "name" in tool:
        name = str(tool["name"])
        desc = str(tool.get("description") or "").strip()
        args_schema = tool.get("args") or tool.get("parameters") or {}
        sig_parts: List[str] = []
        for arg_name, schema in args_schema.items():
            t = schema.get("type") if isinstance(schema, dict) else None
            if t:
                sig_parts.append(f"{arg_name}: {t}")
            else:
                sig_parts.append(arg_name)
        sig = ", ".join(sig_parts)
        return name, sig, desc or "(no description provided)"

    # Callable case
    fn = tool
    name = getattr(fn, "__name__", "<tool>")
    desc = (getattr(fn, "__doc__", "") or "").strip()

    ann = getattr(fn, "__annotations__", {}) or {}
    arg_parts: List[str] = []
    for arg, t in ann.items():
        if arg == "return":
            continue
        type_name = getattr(t, "__name__", str(t))
        arg_parts.append(f"{arg}: {type_name}")

    sig = ", ".join(arg_parts)
    if not desc:
        desc = "(no description provided)"
    return name, sig, desc


def _resolve_collection(memory: Optional[Any]) -> Optional[Any]:
    """
    Try to resolve `memory` into something with a .queryBy(...) method.

    - If memory has queryBy: assume it's a Collection.
    - Else if it has listCollections/getCollection: assume it's a VectorStore
      and return the first collection, if any.
    """
    if memory is None:
        return None

    # Collection-like
    if hasattr(memory, "queryBy"):
        return memory

    # VectorStore-like
    has_list = hasattr(memory, "listCollections")
    has_get = hasattr(memory, "getCollection")
    if has_list and has_get:
        try:
            names = memory.listCollections() or []
        except Exception:
            return None
        if not names:
            return None
        # Heuristic: pick the first collection. You can refine this later
        # (e.g. by name, or by a special "current task" collection).
        try:
            return memory.getCollection(names[0])
        except Exception:
            return None

    return None


def _search_result_to_snippet(res: Dict[str, Any], max_chars: int) -> str:
    """
    Convert a SearchResult into a human-readable snippet.

    We try a few common fields, then fall back to str(res).
    """
    meta = res.get("metadata") or {}
    source = (
        meta.get("source")
        or meta.get("path")
        or meta.get("file")
        or meta.get("id")
        or ""
    )

    # Try several possible content keys
    text = (
        res.get("content")
        or res.get("text")
        or meta.get("preview")
        or meta.get("content")
        or ""
    )
    if not text:
        text = str(res)

    text = str(text)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."

    if source:
        return f"Source: {source}\n{text}"
    else:
        return text


def _combine_sections_with_limit(
    sections: List[Tuple[str, str]],
    limit: int,
) -> str:
    """
    Combine named sections into a single string, trimming lower-priority
    sections if necessary to respect the character limit.

    Priority order (hard-coded):
      1. base (highest)
      2. summary
      3. tools
      4. memory (lowest, dropped first)

    If dropping memory/tools/summary is not enough, we hard-truncate.
    """

    # Keep sections in original order when joining
    def join(sec_list: List[Tuple[str, str]]) -> str:
        return "\n\n".join(text for _, text in sec_list if text).strip()

    current_sections = list(sections)
    text = join(current_sections)

    if len(text) <= limit:
        return text

    # Drop sections in order of increasing priority
    drop_order = ["memory", "tools", "summary"]  # base is last-resort

    for name in drop_order:
        modified = False
        for i, (sec_name, sec_text) in enumerate(current_sections):
            if sec_name == name and sec_text:
                current_sections[i] = (sec_name, "")  # drop this section
                modified = True
                break
        if modified:
            text = join(current_sections)
            if len(text) <= limit:
                return text

    # Still too long: last resort, hard truncate everything
    if len(text) > limit:
        return text[:limit]
    return text
