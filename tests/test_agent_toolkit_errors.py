import asyncio
import unittest

from src.trivialai.agent.toolkit import ToolCallError, ToolKit
from src.trivialai.util import TransformError


class ToolKitErrorTests(unittest.TestCase):
    def setUp(self):
        def edit(path: str, old: str, new: str, repository_id: str = None):
            return {
                "path": path,
                "old": old,
                "new": new,
                "repository_id": repository_id,
            }

        self.tools = ToolKit(edit)

    def test_missing_argument_error_is_actionable_and_compatible(self):
        with self.assertRaises(TransformError) as caught:
            self.tools.check_tool(
                {
                    "type": "tool-call",
                    "tool": "edit",
                    "args": {
                        "path": "x.py",
                        "old": "a",
                    },
                }
            )

        error = caught.exception
        self.assertIsInstance(error, ToolCallError)
        self.assertEqual(error.message, "missing-tool-arg")

        public = error.public_dict()
        self.assertEqual(public["code"], "missing-tool-arg")
        self.assertEqual(public["missing"], ["new"])
        self.assertEqual(
            public["expected"],
            ["new", "old", "path", "repository_id"],
        )
        self.assertEqual(public["received"], ["old", "path"])
        self.assertIn("new", public["message"])

    def test_unexpected_argument_error_lists_expected_and_received(self):
        with self.assertRaises(ToolCallError) as caught:
            self.tools.check_tool(
                {
                    "type": "tool-call",
                    "tool": "edit",
                    "args": {
                        "path": "x.py",
                        "old": "a",
                        "new": "b",
                        "content": "wrong-name",
                    },
                }
            )

        public = caught.exception.public_dict()
        self.assertEqual(public["code"], "unexpected-tool-arg")
        self.assertEqual(public["unexpected"], ["content"])
        self.assertIn("content", public["received"])
        self.assertIn("new", public["expected"])

    def test_invalid_type_error_names_argument(self):
        def typed(value: int):
            return value

        tools = ToolKit(typed)
        with self.assertRaises(ToolCallError) as caught:
            tools.check_tool(
                {
                    "type": "tool-call",
                    "tool": "typed",
                    "args": {"value": "wrong"},
                }
            )

        public = caught.exception.public_dict()
        self.assertEqual(public["code"], "invalid-tool-arg-type")
        self.assertEqual(public["argument"], "value")
        self.assertEqual(public["expected_type"], "int")
        self.assertEqual(public["received_type"], "str")

    def test_sync_call_bridges_async_tool(self):
        async def add(a: int, b: int):
            await asyncio.sleep(0)
            return a + b

        tools = ToolKit(add)
        result = tools.call_tool(
            {
                "type": "tool-call",
                "tool": "add",
                "args": {"a": 2, "b": 3},
            }
        )
        self.assertEqual(result, 5)

    def test_async_call_accepts_sync_and_async_tools(self):
        async def add(a: int, b: int):
            await asyncio.sleep(0)
            return a + b

        def mul(a: int, b: int):
            return a * b

        async_tools = ToolKit(add)
        sync_tools = ToolKit(mul)

        async def run():
            added = await async_tools.acall_tool(
                {
                    "type": "tool-call",
                    "tool": "add",
                    "args": {"a": 2, "b": 3},
                }
            )
            multiplied = await sync_tools.acall_tool(
                {
                    "type": "tool-call",
                    "tool": "mul",
                    "args": {"a": 2, "b": 3},
                }
            )
            return added, multiplied

        self.assertEqual(asyncio.run(run()), (5, 6))


if __name__ == "__main__":
    unittest.main()
