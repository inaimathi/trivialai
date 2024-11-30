import unittest
from typing import List, Optional

from src.trivialai.tools import Tools
from src.trivialai.util import TransformError


class TestTools(unittest.TestCase):
    def setUp(self):
        """Set up a Tools instance for each test."""
        self.tools = Tools()

        # Example function to define
        def _screenshot(url: str, selectors: Optional[List[str]] = None) -> None:
            """Takes a url and an optional list of selectors. Takes a screenshot."""
            return f"Screenshot taken for {url} with selectors {selectors}"

        self._screenshot = _screenshot
        self.tools.define(self._screenshot)

    def test_define(self):
        """Test defining a function in Tools."""

        def new_tool(a: int) -> int:
            """Example tool."""
            return a + 1

        result = self.tools.define(new_tool)
        self.assertTrue(result)
        tool_list = self.tools.list()
        self.assertIn("new_tool", [t["name"] for t in tool_list])

    def test_define_duplicate(self):
        """Test defining a duplicate function."""

        def new_tool(a: int) -> int:
            """Example tool."""
            return a + 1

        # Define the function once
        result = self.tools.define(new_tool)
        self.assertTrue(result)  # First definition should succeed

        # Attempt to define the same function again
        result = self.tools.define(new_tool)
        self.assertFalse(result)  # Duplicate definitions should return False

        # Use decorator-style definition
        @self.tools.define()
        def _duplicate_tool(arg: int) -> int:
            """A tool that already exists."""
            return arg + 1

        # Attempt to define the same function again using the decorator
        result = self.tools.define(_duplicate_tool)
        self.assertFalse(result)  # Duplicate definitions should still return False

    def test_list(self):
        """Test listing defined tools."""
        tools_list = self.tools.list()
        self.assertEqual(len(tools_list), 1)
        self.assertEqual(tools_list[0]["name"], "_screenshot")
        self.assertEqual(
            tools_list[0]["description"],
            "Takes a url and an optional list of selectors. Takes a screenshot.",
        )

    def test_validate(self):
        """Test validation of a tool call."""
        tool_call = {
            "functionName": "_screenshot",
            "args": {"url": "https://www.google.com", "selectors": ["#search"]},
        }
        self.assertTrue(self.tools.validate(tool_call))

    def test_validate_invalid(self):
        """Test validation of an invalid tool call."""
        tool_call = {"functionName": "nonexistent_tool", "args": {"param": "value"}}
        self.assertFalse(self.tools.validate(tool_call))

    def test_transform_valid(self):
        """Test transforming a valid response."""
        response = '{"functionName": "_screenshot", "args": {"url": "https://www.google.com", "selectors": ["#search"]}}'
        result = self.tools.transform(response)
        self.assertEqual(result["functionName"], "_screenshot")
        self.assertEqual(result["args"]["url"], "https://www.google.com")

    def test_transform_invalid(self):
        """Test transforming an invalid response."""
        response = '{"invalid": "data"}'
        with self.assertRaises(TransformError) as context:
            self.tools.transform(response)
        self.assertEqual(str(context.exception), "invalid-tool-call")

    def test_transform_multi_valid(self):
        """Test transforming a valid multi-tool response."""
        response = '[{"functionName": "_screenshot", "args": {"url": "https://example.com", "selectors": ["#header"]}}]'
        result = self.tools.transform_multi(response)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["functionName"], "_screenshot")

    def test_transform_multi_invalid(self):
        """Test transforming an invalid multi-tool response."""
        response = '[{"invalid": "data"}]'
        with self.assertRaises(TransformError) as context:
            self.tools.transform_multi(response)
        self.assertEqual(str(context.exception), "invalid-tool-subcall")

    def test_lookup(self):
        """Test looking up a function."""
        tool_call = {
            "functionName": "_screenshot",
            "args": {"url": "https://www.google.com", "selectors": None},
        }
        function = self.tools.lookup(tool_call)
        self.assertEqual(function, self._screenshot)

    def test_raw_call(self):
        """Test raw_call on a valid tool call."""
        tool_call = {
            "functionName": "_screenshot",
            "args": {"url": "https://example.com", "selectors": None},
        }
        result = self.tools.raw_call(tool_call)
        self.assertEqual(
            result,
            "Screenshot taken for https://example.com with selectors None",
        )

    def test_call_valid(self):
        """Test call on a valid tool call."""
        tool_call = {
            "functionName": "_screenshot",
            "args": {"url": "https://example.com", "selectors": None},
        }
        result = self.tools.call(tool_call)
        self.assertEqual(
            result,
            "Screenshot taken for https://example.com with selectors None",
        )

    def test_call_invalid(self):
        """Test call on an invalid tool call."""
        tool_call = {"functionName": "nonexistent_tool", "args": {"param": "value"}}
        result = self.tools.call(tool_call)
        self.assertIsNone(result)
