import unittest

from src.trivialai.agent import runtime


class AgentRuntimeDecisionTests(unittest.TestCase):
    def test_tool_call_decision_discards_runtime_metadata(self):
        parsed = runtime._parse_decision(
            """
            {
              "type": "tool-call",
              "tool": "read_file",
              "args": {"path": "x.py"},
              "tool_call_id": "call-9000",
              "step": 99,
              "attempt": 7
            }
            """
        )

        self.assertEqual(
            parsed,
            {
                "type": "tool-call",
                "tool": "read_file",
                "args": {"path": "x.py"},
            },
        )

    def test_final_decision_discards_runtime_metadata(self):
        parsed = runtime._parse_decision(
            '{"type":"final","content":"done","step":99}'
        )
        self.assertEqual(parsed, {"type": "final", "content": "done"})

    def test_public_tool_error_payload_is_preserved(self):
        class ExampleError(RuntimeError):
            def public_dict(self):
                return {
                    "code": "example_error",
                    "message": "Something actionable happened.",
                    "details": {"path": "x.py"},
                }

        payload = runtime._tool_error_payload(
            ExampleError("internal wording")
        )

        self.assertEqual(payload["type"], "tool-error")
        self.assertEqual(payload["code"], "example_error")
        self.assertEqual(
            payload["message"],
            "Something actionable happened.",
        )
        self.assertEqual(payload["details"], {"path": "x.py"})


if __name__ == "__main__":
    unittest.main()
