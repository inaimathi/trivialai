# tests/test_jsonesque.py

import json
import unittest

from src.trivialai import jsonesque


class TestJsonesque(unittest.TestCase):
    # --- helpers ---------------------------------------------------------

    def _loads(self, text):
        """Convenience wrapper for jsonesque.loads."""
        return jsonesque.loads(text)

    def _dumps_roundtrip_json(self, obj):
        """
        For JSON-compatible objects, ensure that:
          - jsonesque.dumps(obj) produces valid JSON
          - json.loads of that JSON is equal to obj
        """
        s = jsonesque.dumps(obj)
        parsed = json.loads(s)
        self.assertEqual(
            parsed,
            obj,
            f"dumps/loads roundtrip mismatch for {obj!r}: got {parsed!r}",
        )

    # --- strict JSON -----------------------------------------------------

    def test_basic_json_object_round_trip(self):
        text = '{"a": 1, "b": [2, 3, 4], "c": true, "d": null}'
        val = self._loads(text)
        # Should agree with standard JSON
        self.assertEqual(val, json.loads(text))
        self._dumps_roundtrip_json(val)

    def test_json_array_round_trip(self):
        text = ' [ {"x": 1}, {"y": 2}, {"z": 3} ] '
        val = self._loads(text)
        self.assertEqual(val, json.loads(text))
        self._dumps_roundtrip_json(val)

    def test_nested_json_round_trip(self):
        text = '{"outer": {"inner": [1, 2, {"k": "v"}]}, "flag": false}'
        val = self._loads(text)
        self.assertEqual(val, json.loads(text))
        self._dumps_roundtrip_json(val)

    # --- Python-ish primitives ------------------------------------------

    def test_python_bools_and_none(self):
        text = "[true, false, null, True, False, None]"
        val = self._loads(text)
        expected = [True, False, None, True, False, None]
        self.assertEqual(val, expected)

    # --- strings & triple-quoted strings --------------------------------

    def test_simple_double_quoted_string(self):
        text = '"hello world"'
        val = self._loads(text)
        self.assertEqual(val, "hello world")

    def test_simple_single_quoted_string(self):
        text = "'hello world'"
        val = self._loads(text)
        self.assertEqual(val, "hello world")

    def test_triple_single_quoted_string(self):
        text = "'''hello\nworld'''"
        val = self._loads(text)
        self.assertEqual(val, "hello\nworld")

    def test_triple_double_quoted_string(self):
        text = '"""hello\nworld"""'
        val = self._loads(text)
        self.assertEqual(val, "hello\nworld")

    def test_string_with_escapes(self):
        text = r'"line1\nline2\t\"quoted\""'
        val = self._loads(text)
        self.assertEqual(val, 'line1\nline2\t"quoted"')

    # --- tuples, sets, bytes --------------------------------------------

    def test_tuple_literal(self):
        text = "(1, 2, 3)"
        val = self._loads(text)
        self.assertEqual(val, (1, 2, 3))

    def test_singleton_tuple_literal(self):
        text = "(1,)"
        val = self._loads(text)
        self.assertEqual(val, (1,))

    def test_set_literal(self):
        text = "{1, 2, 3}"
        val = self._loads(text)
        self.assertIsInstance(val, set)
        self.assertEqual(val, {1, 2, 3})

    def test_bytes_literal_simple(self):
        text = 'b"foo"'
        val = self._loads(text)
        self.assertIsInstance(val, (bytes, bytearray))
        self.assertEqual(bytes(val), b"foo")

    def test_bytes_literal_triple(self):
        text = "b'''hello\nworld'''"
        val = self._loads(text)
        self.assertIsInstance(val, (bytes, bytearray))
        self.assertEqual(bytes(val), b"hello\nworld")

    # --- object vs set disambiguation -----------------------------------

    def test_object_literal_with_string_keys(self):
        text = '{"a": 1, "b": 2}'
        val = self._loads(text)
        self.assertEqual(val, {"a": 1, "b": 2})

    def test_object_literal_with_single_quoted_keys(self):
        text = "{'a': 1, 'b': 2}"
        val = self._loads(text)
        self.assertEqual(val, {"a": 1, "b": 2})

    def test_set_literal_not_object(self):
        text = "{1, 2, 3, 4}"
        val = self._loads(text)
        self.assertIsInstance(val, set)
        self.assertEqual(val, {1, 2, 3, 4})

    # --- real-world-ish example -----------------------------------------

    def test_tool_call_with_triple_quoted_content(self):
        text = """{
            'type': 'tool-call',
            'tool': 'spit',
            'args': {
                'file_path': '/tmp/example.py',
                'content': '''line1
line2
line3'''
            }
        }"""
        val = self._loads(text)

        self.assertIsInstance(val, dict)
        self.assertEqual(val["type"], "tool-call")
        self.assertEqual(val["tool"], "spit")
        self.assertIn("args", val)
        self.assertEqual(val["args"]["file_path"], "/tmp/example.py")
        self.assertEqual(val["args"]["content"], "line1\nline2\nline3")

    # --- error cases -----------------------------------------------------

    def test_invalid_number_raises(self):
        text = "00"  # invalid under JSON & our stricter number rule
        with self.assertRaises(ValueError):
            self._loads(text)

    def test_mixed_set_and_object_raises(self):
        # Something like {'a': 1, 2} should not be accepted
        text = "{'a': 1, 2}"
        with self.assertRaises(ValueError):
            self._loads(text)

    def test_unterminated_string_raises(self):
        text = "'unterminated"
        with self.assertRaises(ValueError):
            self._loads(text)

    def test_garbage_input_raises(self):
        text = "@@@"
        with self.assertRaises(ValueError):
            self._loads(text)


if __name__ == "__main__":
    unittest.main()
