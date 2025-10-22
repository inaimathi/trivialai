import asyncio
import inspect
import unittest
from typing import List, Optional

from src.trivialai.tools import Tools
from src.trivialai.util import TransformError


class TestTools(unittest.TestCase):
    def setUp(self):
        """Set up a Tools instance for each test."""
        self.tools = Tools()

        # Example function to define (SYNC)
        def _screenshot(url: str, selectors: Optional[List[str]] = None) -> str:
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
        item = tools_list[0]
        self.assertEqual(item["name"], "_screenshot")
        self.assertEqual(
            item["description"],
            "Takes a url and an optional list of selectors. Takes a screenshot.",
        )
        # sanity check: new 'args' schema exists
        self.assertIn("args", item)
        self.assertIn("type", item)  # raw annotations kept for back-compat
        self.assertIn("async", item)
        self.assertFalse(item["async"])  # _screenshot is sync

        # Check normalized schema for Optional[List[str]]
        sel = item["args"]["selectors"]
        self.assertEqual(sel["type"], "array")
        self.assertIn("items", sel)
        self.assertEqual(sel["items"]["type"], "string")
        # Optional[...] should mark nullable
        self.assertTrue(sel.get("nullable", False))

    def test_validate(self):
        """Test validation of a tool call."""
        tool_call = {
            "functionName": "_screenshot",
            "args": {"url": "https://www.google.com", "selectors": ["#search"]},
        }
        self.assertTrue(self.tools.validate(tool_call))

    def test_validate_missing_optional_ok(self):
        """Optional/defaulted params should be optional during validation."""
        tool_call = {
            "functionName": "_screenshot",
            "args": {"url": "https://www.google.com"},
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
        """Test raw_call on a valid tool call (sync tool)."""
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
        """Test call on a valid tool call (sync tool)."""
        tool_call = {
            "functionName": "_screenshot",
            "args": {"url": "https://example.com", "selectors": None},
        }
        result = self.tools.call(tool_call)
        self.assertEqual(
            result,
            "Screenshot taken for https://example.com with selectors None",
        )

    def test_call_invalid_raises(self):
        """Invalid calls should raise TransformError instead of returning None."""
        tool_call = {"functionName": "nonexistent_tool", "args": {"param": "value"}}
        with self.assertRaises(TransformError) as ctx:
            _ = self.tools.call(tool_call)
        self.assertEqual(str(ctx.exception), "invalid-tool-call")


# -------- New async coverage below -------------------------------------------


class TestToolsAsyncDefinition(unittest.TestCase):
    def setUp(self):
        self.tools = Tools()

        # Define an async tool via the same decorator
        @self.tools.define(description="Add two numbers asynchronously")
        async def async_add(a: int, b: int) -> int:
            await asyncio.sleep(0)  # yield to event loop
            return a + b

        self.async_add = async_add

        # Also define a sync tool to test acall with sync functions
        @self.tools.define(description="Multiply two numbers")
        def mul(a: int, b: int) -> int:
            return a * b

        self.mul = mul

    def test_list_contains_async_flag(self):
        items = {t["name"]: t for t in self.tools.list()}
        self.assertIn("async_add", items)
        self.assertTrue(items["async_add"]["async"])
        self.assertIn("mul", items)
        self.assertFalse(items["mul"]["async"])
        # args schemas exist
        self.assertIn("args", items["async_add"])
        self.assertIn("args", items["mul"])

    def test_lookup_returns_original_fn(self):
        call = {"functionName": "async_add", "args": {"a": 1, "b": 2}}
        fn = self.tools.lookup(call)
        self.assertTrue(inspect.iscoroutinefunction(fn))
        self.assertIs(fn, self.async_add)


class TestToolsAsyncExecution(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tools = Tools()

        @self.tools.define(description="Async add")
        async def async_add(a: int, b: int) -> int:
            await asyncio.sleep(0)
            return a + b

        @self.tools.define(description="Sync mul")
        def mul(a: int, b: int) -> int:
            return a * b

    async def test_acall_async_tool(self):
        res = await self.tools.acall(
            {"functionName": "async_add", "args": {"a": 2, "b": 5}}
        )
        self.assertEqual(res, 7)

    async def test_araw_call_async_tool(self):
        res = await self.tools.araw_call(
            {"functionName": "async_add", "args": {"a": 3, "b": 4}}
        )
        self.assertEqual(res, 7)

    async def test_acall_sync_tool(self):
        """acall should work on sync tools without awaiting anything."""
        res = await self.tools.acall({"functionName": "mul", "args": {"a": 3, "b": 5}})
        self.assertEqual(res, 15)

    async def test_call_async_tool_inside_running_loop(self):
        """
        call() is sync but should still be usable even when an event loop is running.
        It must execute the coroutine in a worker thread and return the result.
        """
        res = self.tools.call({"functionName": "async_add", "args": {"a": 10, "b": 7}})
        self.assertEqual(res, 17)

    async def test_raw_call_async_tool_inside_running_loop(self):
        res = self.tools.raw_call(
            {"functionName": "async_add", "args": {"a": 1, "b": 2}}
        )
        self.assertEqual(res, 3)

    async def test_extras_merge_sync_call(self):
        """
        call() should merge extras; with override=True (default) extras win.
        """
        tools = Tools(extras={"b": 99})

        @tools.define(description="echo add")
        def add(a: int, b: int = 0) -> int:
            return a + b

        # User provides b=5, extras b=99, extras override => result 1+99
        res = tools.call({"functionName": "add", "args": {"a": 1, "b": 5}})
        self.assertEqual(res, 100)

    async def test_extras_merge_override_false(self):
        """
        call_with_extras(..., override=False) should allow user args to override extras.
        """
        tools = Tools(extras={"b": 10})

        @tools.define(description="echo add")
        def add(a: int, b: int = 0) -> int:
            return a + b

        call = {"functionName": "add", "args": {"a": 1, "b": 5}}
        # Explicitly use call_with_extras with override=False
        res = tools.call_with_extras({"b": 10}, call, override=False)
        self.assertEqual(res, 6)  # user b=5 wins over extras b=10

    async def test_async_extras_merge_both_paths(self):
        """
        acall_with_extras works with both async and sync tools.
        """
        # Async tool
        toolsA = Tools(extras={"b": 5})

        @toolsA.define(description="async add")
        async def add_async(a: int, b: int = 0) -> int:
            await asyncio.sleep(0)
            return a + b

        resA = await toolsA.acall({"functionName": "add_async", "args": {"a": 2}})
        self.assertEqual(resA, 7)

        # Sync tool via async API
        toolsB = Tools(extras={"b": 3})

        @toolsB.define(description="sync add")
        def add_sync(a: int, b: int = 0) -> int:
            return a + b

        resB = await toolsB.acall({"functionName": "add_sync", "args": {"a": 10}})
        self.assertEqual(resB, 13)

    async def test_validate_and_transform_with_async_tool(self):
        tools = Tools()

        @tools.define(description="async add")
        async def add(a: int, b: int) -> int:
            await asyncio.sleep(0)
            return a + b

        payload = {"functionName": "add", "args": {"a": 3, "b": 4}}
        self.assertTrue(tools.validate(payload))

        s = '{"functionName":"add","args":{"a":3,"b":4}}'
        parsed = tools.transform(s)
        self.assertEqual(parsed, payload)

        multi = '[{"functionName":"add","args":{"a":1,"b":2}}]'
        parsed_multi = tools.transform_multi(multi)
        self.assertEqual(len(parsed_multi), 1)
        self.assertEqual(parsed_multi[0]["functionName"], "add")
