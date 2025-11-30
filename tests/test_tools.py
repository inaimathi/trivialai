# tests/test_tools.py
from __future__ import annotations

import unittest
from typing import Any, Dict, List, Literal, Optional

from src.trivialai.tools import ToolKit, TransformError, to_summary

# ---------- helper functions used as tools ----------


def add(x: int, y: int) -> int:
    """Add two integers."""
    return x + y


def spit(file_path: str, content: str, mode: Optional[str] = None) -> None:
    """Write content to a file path (dummy impl)."""
    # no-op in tests; just exercise signature / summary
    return None


def kwtool(a: int, *, b: str = "default") -> str:
    """Keyword-only parameter example."""
    return f"{a}:{b}"


def varkw_tool(level: str, **kwargs: Any) -> Dict[str, Any]:
    """Tool that accepts arbitrary keyword arguments."""
    out = {"level": level}
    out.update(kwargs)
    return out


def bad_type_tool(x: int) -> None:
    """Raises TypeError inside the tool."""
    raise TypeError("boom")


class TestToSummary(unittest.TestCase):
    def test_to_summary_basic_signature_and_args(self) -> None:
        summary = to_summary(spit)

        self.assertEqual(summary["name"], "spit")
        self.assertIn("signature", summary)
        sig = summary["signature"]
        # Ensure the compact signature reflects parameters and default
        self.assertIn("spit(", sig)
        self.assertIn("file_path: str", sig)
        self.assertIn("content: str", sig)
        self.assertIn("mode: Optional[str] = None", sig)
        self.assertIn("->", sig)

        # Description comes from docstring
        self.assertTrue(summary["description"].startswith("Write content"))

        args_schema = summary["args"]
        self.assertIsInstance(args_schema, dict)
        self.assertIn("file_path", args_schema)
        self.assertIn("content", args_schema)
        self.assertIn("mode", args_schema)

    def test_to_summary_unannotated_function(self) -> None:
        def no_ann(x, y=5):
            return x + y

        summary = to_summary(no_ann)

        self.assertEqual(summary["name"], "no_ann")
        sig = summary["signature"]
        # We expect Any for unannotated params
        self.assertIn("x: Any", sig)
        self.assertIn("y: Any = 5", sig)
        self.assertIn("-> Any", sig)

        # args schema should have both parameters
        self.assertIn("x", summary["args"])
        self.assertIn("y", summary["args"])


class TestToolKitShapeAndPrompt(unittest.TestCase):
    def setUp(self) -> None:
        self.tk = ToolKit(add, spit, kwtool)

    def test_to_tool_shape_contains_all_tool_names(self) -> None:
        shape = self.tk.to_tool_shape()
        self.assertEqual(shape["type"], "tool-call")
        self.assertIn("<param>", shape["args"])
        self.assertIn("...", shape["args"])

        # The placeholder for "tool" should mention all tool names
        tool_placeholder = shape["tool"]
        self.assertIn("add", tool_placeholder)
        self.assertIn("spit", tool_placeholder)
        self.assertIn("kwtool", tool_placeholder)

    def test_to_tool_prompt_contains_signatures_and_shape(self) -> None:
        prompt = self.tk.to_tool_prompt()
        # Basic header text
        self.assertIn("You have access to the following tools.", prompt)
        self.assertIn('"type": "tool-call"', prompt)

        # Compact signatures for each tool
        self.assertIn("add(", prompt)
        self.assertIn("spit(", prompt)
        self.assertIn("kwtool(", prompt)

        # Simple check that description line is present
        self.assertIn("Add two integers.", prompt)
        self.assertIn("Write content to a file path", prompt)


class TestToolKitCheckTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tk = ToolKit(add, spit, kwtool, varkw_tool, bad_type_tool)

    # ---- happy path ----

    def test_check_tool_valid_call(self) -> None:
        call = {
            "type": "tool-call",
            "tool": "add",
            "args": {"x": 1, "y": 2},
        }
        checked = self.tk.check_tool(call)
        self.assertEqual(checked, call)

    def test_call_tool_executes_function(self) -> None:
        call = {
            "type": "tool-call",
            "tool": "add",
            "args": {"x": 2, "y": 3},
        }
        result = self.tk.call_tool(call)
        self.assertEqual(result, 5)

    def test_check_tool_allows_varkw_extra_args(self) -> None:
        call = {
            "type": "tool-call",
            "tool": "varkw_tool",
            "args": {"level": "info", "message": "hello", "code": 200},
        }
        checked = self.tk.check_tool(call)
        self.assertEqual(checked, call)

        result = self.tk.call_tool(call)
        self.assertEqual(result["level"], "info")
        self.assertEqual(result["message"], "hello")
        self.assertEqual(result["code"], 200)

    # ---- structural error cases ----

    def test_check_tool_rejects_non_dict(self) -> None:
        with self.assertRaises(TransformError) as cm:
            self.tk.check_tool("not-a-dict")  # type: ignore[arg-type]
        self.assertIn("invalid-object-structure", str(cm.exception))

    def test_check_tool_missing_type(self) -> None:
        call = {"tool": "add", "args": {"x": 1, "y": 2}}
        with self.assertRaises(TransformError) as cm:
            self.tk.check_tool(call)
        self.assertIn("invalid-tool-call-type", str(cm.exception))

    def test_check_tool_wrong_type_value(self) -> None:
        call = {"type": "not-tool-call", "tool": "add", "args": {"x": 1, "y": 2}}
        with self.assertRaises(TransformError) as cm:
            self.tk.check_tool(call)
        self.assertIn("invalid-tool-call-type", str(cm.exception))

    def test_check_tool_invalid_tool_name_type(self) -> None:
        call = {"type": "tool-call", "tool": 123, "args": {}}  # type: ignore[dict-item]
        with self.assertRaises(TransformError) as cm:
            self.tk.check_tool(call)
        self.assertIn("invalid-tool-name", str(cm.exception))

    def test_check_tool_unknown_tool(self) -> None:
        call = {"type": "tool-call", "tool": "does_not_exist", "args": {}}
        with self.assertRaises(TransformError) as cm:
            self.tk.check_tool(call)
        self.assertIn("no-such-tool", str(cm.exception))

    def test_check_tool_args_not_dict(self) -> None:
        call = {"type": "tool-call", "tool": "add", "args": 42}
        with self.assertRaises(TransformError) as cm:
            self.tk.check_tool(call)
        self.assertIn("invalid-tool-args", str(cm.exception))

    # ---- arg shape error cases ----

    def test_check_tool_unexpected_arg(self) -> None:
        call = {
            "type": "tool-call",
            "tool": "add",
            "args": {"x": 1, "y": 2, "z": 3},
        }
        with self.assertRaises(TransformError) as cm:
            self.tk.check_tool(call)
        self.assertIn("unexpected-tool-arg", str(cm.exception))

    def test_check_tool_missing_required_arg(self) -> None:
        call = {
            "type": "tool-call",
            "tool": "add",
            "args": {"x": 1},  # missing 'y'
        }
        with self.assertRaises(TransformError) as cm:
            self.tk.check_tool(call)
        self.assertIn("missing-tool-arg", str(cm.exception))

    # ---- type error cases ----

    def test_check_tool_type_mismatch(self) -> None:
        call = {
            "type": "tool-call",
            "tool": "add",
            "args": {"x": "not-an-int", "y": 2},
        }
        with self.assertRaises(TransformError) as cm:
            self.tk.check_tool(call)
        self.assertIn("invalid-tool-arg-type", str(cm.exception))

    def test_call_tool_wraps_internal_typeerror(self) -> None:
        call = {
            "type": "tool-call",
            "tool": "bad_type_tool",
            "args": {"x": 10},
        }
        with self.assertRaises(TransformError) as cm:
            self.tk.call_tool(call)
        self.assertIn("tool-call-failed", str(cm.exception))


class TestTypeOk(unittest.TestCase):
    def test_type_ok_simple_builtin(self) -> None:
        self.assertTrue(ToolKit._type_ok(1, int))
        self.assertFalse(ToolKit._type_ok("1", int))

    def test_type_ok_optional(self) -> None:
        opt_str = Optional[str]
        self.assertTrue(ToolKit._type_ok("hello", opt_str))
        self.assertTrue(ToolKit._type_ok(None, opt_str))
        self.assertFalse(ToolKit._type_ok(123, opt_str))

    def test_type_ok_literal(self) -> None:
        lit = Literal["a", "b"]
        self.assertTrue(ToolKit._type_ok("a", lit))
        self.assertTrue(ToolKit._type_ok("b", lit))
        self.assertFalse(ToolKit._type_ok("c", lit))

    def test_type_ok_list_and_dict(self) -> None:
        list_int = List[int]
        dict_str_any = Dict[str, Any]

        self.assertTrue(ToolKit._type_ok([1, 2, 3], list_int))
        self.assertFalse(ToolKit._type_ok("not-a-list", list_int))

        self.assertTrue(ToolKit._type_ok({"k": 1}, dict_str_any))
        self.assertFalse(ToolKit._type_ok(["k", 1], dict_str_any))


if __name__ == "__main__":
    unittest.main()
