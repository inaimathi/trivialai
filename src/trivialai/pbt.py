# src/trivialai/pbt.py
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent import Agent, AgentScope, AgentStep
from .llm import LLMMixin, LLMResult
from .tools import Tools


def _run_cmd(
    args: List[str],
    cwd: Path,
    timeout_sec: int = 300,
    phase: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Small helper to run a subprocess and normalize output.
    """
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        result: Dict[str, Any] = {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "cmd": args,
        }
    except subprocess.TimeoutExpired as e:
        result = {
            "ok": False,
            "timeout": True,
            "returncode": None,
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
            "cmd": args,
        }
    if phase is not None:
        result["phase"] = phase
    return result


def _prepare_agent_venv(agent: Agent, repo_root: Path) -> Dict[str, Any]:
    """
    Create / reuse a venv under repo_root and install:
      - project dependencies (requirements.txt or pyproject)
      - the project itself (editable if possible)
      - hypothesis

    Returns a dict:
      {
        "ok": bool,
        "env_dir": Path,
        "python_cmd": Path,
        "steps": [...],   # list of per-phase results from _run_cmd
      }
    """
    repo_root = repo_root.resolve()
    env_dir = repo_root / f"env-{agent.name}"

    steps: List[Dict[str, Any]] = []

    # Use the current interpreter as the base for venv creation
    base_python = Path(sys.executable)

    # 1) Create venv if needed
    if not env_dir.exists():
        res_env = _run_cmd(
            [str(base_python), "-m", "venv", env_dir.name],
            cwd=repo_root,
            timeout_sec=900,
            phase="create_venv",
        )
        steps.append(res_env)
        if not res_env["ok"]:
            # Bail early if we can't even create a venv
            return {
                "ok": False,
                "env_dir": env_dir,
                "python_cmd": base_python,
                "steps": steps,
            }
    else:
        steps.append(
            {
                "phase": "reused_venv",
                "ok": True,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "cmd": [],
            }
        )

    # 2) Compute venv python path
    if os.name == "nt":
        python_cmd = env_dir / "Scripts" / "python.exe"
    else:
        python_cmd = env_dir / "bin" / "python3"

    # 3) Upgrade pip
    steps.append(
        _run_cmd(
            [str(python_cmd), "-m", "pip", "install", "--upgrade", "pip"],
            cwd=repo_root,
            timeout_sec=900,
            phase="upgrade_pip",
        )
    )

    # 4) Install dependencies (requirements.txt or pyproject)
    req = repo_root / "requirements.txt"
    pyproj = repo_root / "pyproject.toml"

    if req.exists():
        steps.append(
            _run_cmd(
                [str(python_cmd), "-m", "pip", "install", "-r", req.name],
                cwd=repo_root,
                timeout_sec=900,
                phase="install_requirements",
            )
        )
    elif pyproj.exists():
        # Simple case: project is installable via pyproject
        steps.append(
            _run_cmd(
                [str(python_cmd), "-m", "pip", "install", "."],
                cwd=repo_root,
                timeout_sec=900,
                phase="install_project_from_pyproject",
            )
        )

    # 5) Best-effort install of the project itself into the venv
    #    This is what makes `import pycronado` work in the venv.
    setup_py = repo_root / "setup.py"
    setup_cfg = repo_root / "setup.cfg"
    if pyproj.exists() or setup_py.exists() or setup_cfg.exists():
        res_self_editable = _run_cmd(
            [str(python_cmd), "-m", "pip", "install", "-e", "."],
            cwd=repo_root,
            timeout_sec=900,
            phase="install_self_editable",
        )
        steps.append(res_self_editable)
        if not res_self_editable["ok"]:
            steps.append(
                _run_cmd(
                    [str(python_cmd), "-m", "pip", "install", "."],
                    cwd=repo_root,
                    timeout_sec=900,
                    phase="install_self",
                )
            )

    # 6) Ensure hypothesis is available
    steps.append(
        _run_cmd(
            [str(python_cmd), "-c", "import hypothesis"],
            cwd=repo_root,
            timeout_sec=60,
            phase="check_hypothesis",
        )
    )
    if not steps[-1]["ok"]:
        steps.append(
            _run_cmd(
                [str(python_cmd), "-m", "pip", "install", "hypothesis"],
                cwd=repo_root,
                timeout_sec=900,
                phase="install_hypothesis",
            )
        )

    # 7) Aggregate global ok (ignore some non-critical checks)
    ok = True
    for s in steps:
        phase = s.get("phase")
        # These can fail without necessarily making the whole env unusable
        if phase in {"check_hypothesis", "install_self_editable"}:
            continue
        if not s.get("ok", True):
            ok = False
            break

    return {
        "ok": ok,
        "env_dir": env_dir,
        "python_cmd": python_cmd,
        "steps": steps,
    }


def _collect_import_examples(repo_root: Path, max_examples: int = 5) -> List[str]:
    """
    Scan existing tests for import statements to infer how the project is imported.

    Example output:
      ["from src.pycronado.scheduler import CronScheduler", "import pycronado"]
    """
    examples: List[str] = []
    tests_root = repo_root / "tests"
    if not tests_root.exists():
        return examples

    for p in tests_root.rglob("test*.py"):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("from ") or stripped.startswith("import "):
                examples.append(stripped)
                if len(examples) >= max_examples:
                    return examples
    return examples


def _make_pbt_tools(
    agent: Agent, repo_root: Path, python_cmd: Optional[Path] = None
) -> Tools:
    """
    Build a Tools instance wired to the agent's scope + a single repo root.
    """
    tools = Tools()
    repo_root = repo_root.resolve()
    scope: AgentScope = agent.scope  # just to make intent obvious

    @tools.define()
    def list_python_files(max_files: int = 200) -> List[str]:
        """
        List up to `max_files` Python source files in the target repo.
        Paths are returned relative to the repo root.
        """
        paths: List[str] = []
        for p in repo_root.rglob("*.py"):
            # Skip common junk
            if ".venv" in p.parts or ".git" in p.parts or "site-packages" in p.parts:
                continue
            rel = p.relative_to(repo_root)
            paths.append(str(rel))
            if len(paths) >= max_files:
                break
        return paths

    @tools.define()
    def read_file(rel_path: str, max_chars: int = 20000) -> str:
        """
        Read a file under the repo root and return its contents (truncated).
        `rel_path` is relative to the repo root.
        """
        p = (repo_root / rel_path).resolve()
        if repo_root not in p.parents and p != repo_root:
            raise ValueError(f"read_file: path {p} escapes repo root")
        text = p.read_text(encoding="utf-8")
        if len(text) > max_chars:
            return text[:max_chars] + "\n\n# [truncated by tool]\n"
        return text

    @tools.define()
    def write_file(
        rel_path: str, content: str, overwrite: bool = True
    ) -> Dict[str, Any]:
        """
        Write `content` to a file under the repo root, respecting the agent scope.
        If overwrite=False and file exists, append instead.
        """
        p = (repo_root / rel_path).resolve()
        if not scope.can_write(p):
            return {
                "ok": False,
                "error": f"WRITE_NOT_ALLOWED: {p}",
            }
        p.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not p.exists():
            p.write_text(content, encoding="utf-8")
        else:
            with p.open("a", encoding="utf-8") as fh:
                fh.write(content)
        return {"ok": True, "path": str(p)}

    @tools.define()
    def run_unittest(
        extra_args: Optional[List[str]] = None, timeout_sec: int = 120
    ) -> Dict[str, Any]:
        """
        Run the stdlib unittest test runner in the repo root and return exit code and output.

        Equivalent to:
            <env_python> -m unittest [extra_args...]

        We also ensure that PYTHONPATH includes:
          - the repo root
          - repo_root/src  (for src-layout projects)
        """
        python_exe = str(python_cmd) if python_cmd is not None else "python3"
        args = [python_exe, "-m", "unittest"]
        if extra_args:
            args.extend(extra_args)

        env = os.environ.copy()

        extra_paths: List[str] = [str(repo_root)]
        src_dir = repo_root / "src"
        if src_dir.exists():
            extra_paths.append(str(src_dir))

        existing_pp = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            extra_paths + ([existing_pp] if existing_pp else [])
        )

        try:
            proc = subprocess.run(
                args,
                cwd=str(repo_root),
                text=True,
                capture_output=True,
                timeout=timeout_sec,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            return {
                "ok": False,
                "timeout": True,
                "error": f"unittest timeout after {timeout_sec}s",
                "stdout": e.stdout or "",
                "stderr": e.stderr or "",
            }

        def _clip(s: str, limit: int = 15000) -> str:
            if len(s) <= limit:
                return s
            return s[-limit:]

        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": _clip(proc.stdout or ""),
            "stderr": _clip(proc.stderr or ""),
        }

    return tools


def _log_agent_event(agent: Agent, ev: Dict[str, Any]) -> None:
    """
    Append an event to the agent's history and NDJSON log.
    This mirrors Agent._record_event but stays outside the class.
    """
    agent.history_events.append(ev)
    agent.log_path.parent.mkdir(parents=True, exist_ok=True)
    with agent.log_path.open("a", encoding="utf-8") as f:
        print(json.dumps(ev, ensure_ascii=False), file=f)


def _build_tool_system_prompt(
    agent: Agent, step_index: int, step: AgentStep, tools: Tools
) -> str:
    """
    Use the Agent's context builder as the base system prompt,
    then append tool-specific instructions.
    """
    base = agent._build_system_prompt(step_index, step)  # type: ignore[attr-defined]

    tool_list = tools.list()
    tool_lines = [
        "",
        "TOOLING CONTEXT:",
        "You have access to the following tools. You may call them by returning",
        "a JSON structure describing tool calls.",
        "",
        json.dumps(tool_list, indent=2, ensure_ascii=False),
    ]

    return base + "\n\n" + "\n".join(tool_lines)


def run_pbt_on_repo(
    llm: LLMMixin,
    repo_root: str | Path,
    *,
    max_files: int = 50,
) -> Dict[str, Any]:
    """
    Agent-driven PBT pipeline for a Python repo.

    High-level flow:
      0) Create an Agent rooted at `repo_root`.
      1) Prepare a per-agent virtualenv (env-<agent-name>) and install:
         - project deps (requirements.txt or pyproject)
         - the project itself (editable if possible)
         - Hypothesis
      2) Equip the Agent with repo-aware tools (list/read/write, run_unittest).
      3) PLAN:
         - propose+write Hypothesis property tests,
         - ensure `python -m unittest` is run inside the venv,
         - if failures occur, write a markdown bug report.
      4) Execute the plan imperatively here.

    Returns:
      {
        "tests_written": [...],
        "pytest": { ... unittest result ... },
        "report_path": str|None,
        "report_markdown": str|None,
        "env_setup": { ... details from _prepare_agent_venv ... },
      }
    """
    repo_root = Path(repo_root).expanduser().resolve()

    # ---- 0) Agent + memory + venv ----
    agent = Agent.from_llm(llm, name="pbt-agent", root=repo_root)
    agent.consider_with_write_access(repo_root)

    env_setup = _prepare_agent_venv(agent, repo_root)
    python_cmd: Optional[Path] = env_setup.get("python_cmd")

    # Build PBT tools and equip the agent
    tools = _make_pbt_tools(agent, repo_root, python_cmd=python_cmd)
    agent.equip(tools)

    # Collect import examples from existing tests to guide the LLM
    import_examples = _collect_import_examples(repo_root)
    if import_examples:
        imports_hint = "Existing tests use imports like:\n" + "\n".join(
            f"  - {ln}" for ln in import_examples
        )
    else:
        imports_hint = (
            "No existing tests were found to infer import style.\n"
            "If the project uses a `src/` layout (e.g. `src/<pkg_name>/...`), "
            "imports may need to reference the installed package name after "
            "the project is installed into the virtual environment."
        )

    # PLAN the high-level steps (for introspection/logging)
    agent.plan(
        "Propose and write Hypothesis-based property tests for the repo.", kind="do"
    )
    agent.plan("Ensure tests are run via `python3 -m unittest`.", kind="do")
    agent.plan(
        "If failures exist, write a markdown bug report with repro steps.", kind="do"
    )

    tests_written: List[str] = []
    test_run_result: Dict[str, Any] | None = None
    report_markdown: Optional[str] = None
    report_path: Optional[Path] = None

    # Convenience: refer to plan steps explicitly
    step_propose = agent.steps[0]
    step_run = agent.steps[1]
    step_report = agent.steps[2]

    # ---- 1) Step 0: propose tests + tool-call script ----

    step_index = 0
    _log_agent_event(
        agent,
        {
            "type": "agent-step-start",
            "agent_step": step_index,
            "kind": step_propose.kind,
            "description": step_propose.description,
        },
    )

    system_tools = _build_tool_system_prompt(agent, step_index, step_propose, tools)

    tool_prompt = textwrap.dedent(
        f"""
        You are an expert Python property-based testing engineer.

        TARGET REPOSITORY ROOT:
        - {repo_root}

        VIRTUALENV:
        - A dedicated virtualenv has been created at: env-{agent.name}
        - Tests are run via that environment's python, equivalent to:
          <env_python> -m unittest [...]

        IMPORT STYLE HINTS:
        {imports_hint}

        GOAL:
        - Propose and implement a small set of HIGH-VALUE property-based tests
          for this repository using Hypothesis and the stdlib unittest runner.

        CONSTRAINTS:
        - Prefer testing public, well-documented functions (e.g. library code).
        - Use Hypothesis strategies from `hypothesis.strategies as st`.
        - Create or append to test files under the `tests/` directory.
        - Write complete, runnable tests including imports.
        - Tests MUST be discoverable by `python3 -m unittest` (for example,
          via `tests/test_*.py` files using unittest.TestCase subclasses
          or module-level test functions).
        - Do NOT change existing tests; only ADD new test files or append new tests
          in clearly marked sections.
        - Before writing new tests, inspect one or more existing test files using
          `list_python_files` and `read_file` to see how they import project code,
          and mimic that import style.
        - After writing tests, call `run_unittest()` ONCE at the end.

        AVAILABLE TOOLS (for JSON tool calls):
        - `list_python_files(max_files: int = {max_files})`
        - `read_file(rel_path: str, max_chars: int = 20000)`
        - `write_file(rel_path: str, content: str, overwrite: bool = True)`
        - `run_unittest(extra_args: Optional[List[str]] = None, timeout_sec: int = 120)`

        OUTPUT:
        - Return a JSON LIST of tool calls, where each element has the shape:
          {{"functionName": string, "args": {{...}} }}
        - Calls must be valid given the tool signatures.
        - Ensure that one of the calls is a single `run_unittest(...)` invocation
          that happens AFTER any file writes.
        """
    ).strip()

    # Use generate_checked + tools.transform_multi so we can control the system prompt.
    try:
        tc_result: LLMResult = llm.generate_checked(
            tools.transform_multi,
            system_tools,
            tool_prompt,
            retries=3,
        )
        tool_calls = tc_result.content or []
        llm_ok = True
        llm_error = None
    except Exception as e:
        tool_calls = []
        llm_ok = False
        llm_error = str(e)

    _log_agent_event(
        agent,
        {
            "type": "agent-llm-tool-plan",
            "agent_step": step_index,
            "ok": llm_ok,
            "error": llm_error,
            "num_calls": len(tool_calls),
        },
    )

    # Execute the tool calls in order
    for call in tool_calls:
        name = call.get("functionName")
        try:
            result = tools.call(call)
        except Exception as e:
            result = {"ok": False, "error": f"tool-exception: {e!r}"}

        # Track interesting side effects
        if name == "write_file" and isinstance(result, dict) and result.get("ok"):
            path = result.get("path")
            if path:
                tests_written.append(path)
        if name == "run_unittest":
            test_run_result = result

        _log_agent_event(
            agent,
            {
                "type": "agent-tool-call",
                "agent_step": step_index,
                "tool": name,
                "args": call.get("args", {}),
                "result_ok": (
                    bool(result.get("ok", True)) if isinstance(result, dict) else None
                ),
                "result_returncode": (
                    result.get("returncode", None) if isinstance(result, dict) else None
                ),
            },
        )

    _log_agent_event(
        agent,
        {
            "type": "agent-step-end",
            "agent_step": step_index,
            "kind": step_propose.kind,
        },
    )

    # ---- 2) Step 1: ensure unittest has been run ----

    step_index = 1
    _log_agent_event(
        agent,
        {
            "type": "agent-step-start",
            "agent_step": step_index,
            "kind": step_run.kind,
            "description": step_run.description,
        },
    )

    if test_run_result is None:
        manual_call = {"functionName": "run_unittest", "args": {}}
        try:
            test_run_result = tools.call(manual_call)
        except Exception as e:
            test_run_result = {"ok": False, "error": f"run_unittest-exception: {e!r}"}

        _log_agent_event(
            agent,
            {
                "type": "agent-tool-call",
                "agent_step": step_index,
                "tool": "run_unittest",
                "args": {},
                "result_ok": (
                    bool(test_run_result.get("ok", False))
                    if isinstance(test_run_result, dict)
                    else None
                ),
                "result_returncode": (
                    test_run_result.get("returncode", None)
                    if isinstance(test_run_result, dict)
                    else None
                ),
            },
        )
    else:
        _log_agent_event(
            agent,
            {
                "type": "agent-step-skipped",
                "agent_step": step_index,
                "reason": "unittest already run in previous step",
            },
        )

    _log_agent_event(
        agent,
        {
            "type": "agent-step-end",
            "agent_step": step_index,
            "kind": step_run.kind,
        },
    )

    if not isinstance(test_run_result, dict):
        test_run_result = {"ok": False, "error": "run_unittest tool returned non-dict"}

    # ---- 3) Step 2: bug report if tests failed ----

    step_index = 2
    _log_agent_event(
        agent,
        {
            "type": "agent-step-start",
            "agent_step": step_index,
            "kind": step_report.kind,
            "description": step_report.description,
        },
    )

    if not test_run_result.get("ok", False):
        stdout = test_run_result.get("stdout", "") or ""
        stderr = test_run_result.get("stderr", "") or ""
        returncode = test_run_result.get("returncode", None)

        report_system_base = agent._build_system_prompt(  # type: ignore[attr-defined]
            step_index, step_report
        )
        report_system = (
            report_system_base
            + "\n\n"
            + (
                "You are a careful software testing expert. "
                "Given a Python repository, property-based tests, and failing "
                "`python3 -m unittest` output, you will write a concise but detailed bug report."
            )
        )

        report_prompt = textwrap.dedent(
            f"""
            REPOSITORY ROOT:
            - {repo_root}

            TEST RETURN CODE (python3 -m unittest): {returncode}

            STDOUT (may be truncated):
            -------------------------------
            {stdout}

            STDERR (may be truncated):
            -------------------------------
            {stderr}

            TASK:
            - Infer the most likely bug(s) exposed by the failing property-based tests.
            - If there are multiple apparent failures, focus on the most severe
              and clearly reproducible one.

            WRITE A MARKDOWN BUG REPORT WITH SECTIONS:

            # Summary
            - One or two sentences summarizing the bug and its impact.

            # Reproduction Steps
            - A numbered list of concrete steps to reproduce the bug at a Python REPL
              or via a short script, including imports and example values.

            # Failing Property-Based Test
            - A single Hypothesis test (or minimal subset) that demonstrates the failure.
              If the existing test is written for unittest, you may show it exactly as it appears.
            - Wrap it in a Python fenced code block.

            # Root Cause Hypothesis
            - A short explanation of what is likely going wrong and in which module/function.

            # Proposed Fix
            - A short, high-level description of how you would fix the bug, optionally with
              a small code snippet if helpful (Python fenced code block).

            # Issue Text
            - A short GitHub-style issue body the user could paste into an issue tracker.
            """
        ).strip()

        report_res = llm.generate(report_system, report_prompt)
        report_markdown = report_res.content or ""

        report_path = repo_root / "pbt_report.md"
        if agent.scope.can_write(report_path):
            report_path.write_text(report_markdown, encoding="utf-8")
            _log_agent_event(
                agent,
                {
                    "type": "agent-report-written",
                    "agent_step": step_index,
                    "path": str(report_path),
                },
            )
        else:
            _log_agent_event(
                agent,
                {
                    "type": "agent-report-skipped-write",
                    "agent_step": step_index,
                    "reason": f"WRITE_NOT_ALLOWED for {report_path}",
                },
            )
    else:
        _log_agent_event(
            agent,
            {
                "type": "agent-no-failures",
                "agent_step": step_index,
                "message": "python3 -m unittest passed; no bug report generated",
            },
        )

    _log_agent_event(
        agent,
        {
            "type": "agent-step-end",
            "agent_step": step_index,
            "kind": step_report.kind,
        },
    )

    # ---- Final result ----

    return {
        "tests_written": tests_written,
        "pytest": test_run_result,  # unittest result, kept under old key name
        "report_path": str(report_path) if report_path else None,
        "report_markdown": report_markdown,
        "env_setup": env_setup,
    }
