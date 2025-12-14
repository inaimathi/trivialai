# tests/test_json_shape.py
import unittest

from src.trivialai.util import TransformError, json_shape


class TestJsonShape(unittest.TestCase):
    # -------- basic types --------

    def test_json_shape_int_accepts_int(self):
        v = json_shape(int)
        self.assertEqual(v("1"), 1)
        self.assertEqual(v(2), 2)  # passthrough loadch(dict) equivalent for scalars

    def test_json_shape_int_rejects_bool(self):
        v = json_shape(int)
        with self.assertRaises(TransformError):
            v("true")

    def test_json_shape_float_accepts_int_and_float(self):
        v = json_shape(float)
        self.assertEqual(v("1"), 1)
        self.assertEqual(v("1.5"), 1.5)

    def test_json_shape_float_rejects_bool(self):
        v = json_shape(float)
        with self.assertRaises(TransformError):
            v("false")

    def test_json_shape_str_accepts_str(self):
        v = json_shape(str)
        self.assertEqual(v('"hello"'), "hello")

    def test_json_shape_str_rejects_non_str(self):
        v = json_shape(str)
        with self.assertRaises(TransformError):
            v("123")

    def test_json_shape_bool_accepts_bool(self):
        v = json_shape(bool)
        self.assertTrue(v("true"))
        self.assertFalse(v("false"))

    def test_json_shape_bool_rejects_int(self):
        v = json_shape(bool)
        with self.assertRaises(TransformError):
            v("0")

    # -------- list shapes --------

    def test_json_shape_list_of_ints_success(self):
        v = json_shape([int])
        self.assertEqual(v("[1,2,3]"), [1, 2, 3])

    def test_json_shape_list_wrong_container_rejected(self):
        v = json_shape([int])
        with self.assertRaises(TransformError):
            v('{"a": 1}')

    def test_json_shape_list_element_type_mismatch(self):
        v = json_shape([int])
        with self.assertRaises(TransformError):
            v('[1, "nope", 3]')

    # -------- dict shapes (specific keys) --------

    def test_json_shape_specific_keys_success(self):
        v = json_shape({"a": int, "b": str})
        out = v('{"a": 1, "b": "x", "extra": 999}')
        # extra keys are allowed by current validator behavior
        self.assertEqual(out["a"], 1)
        self.assertEqual(out["b"], "x")
        self.assertEqual(out["extra"], 999)

    def test_json_shape_specific_keys_missing_key(self):
        v = json_shape({"a": int, "b": str})
        with self.assertRaises(TransformError):
            v('{"a": 1}')

    def test_json_shape_specific_keys_nested(self):
        v = json_shape({"outer": {"inner": [str]}})
        out = v('{"outer": {"inner": ["x", "y"]}}')
        self.assertEqual(out, {"outer": {"inner": ["x", "y"]}})

    # -------- dict shapes (general typed dicts) --------

    def test_json_shape_general_dict_str_to_int_success(self):
        v = json_shape({str: int})
        out = v('{"a": 1, "b": 2}')
        self.assertEqual(out, {"a": 1, "b": 2})

    def test_json_shape_general_dict_value_type_mismatch(self):
        v = json_shape({str: int})
        with self.assertRaises(TransformError):
            v('{"a": 1, "b": "nope"}')

    def test_json_shape_general_dict_key_type_not_supported_for_json_objects(self):
        # JSON object keys are always strings (after parsing)
        # Your validator currently tries to validate key types; this should fail.
        v = json_shape({int: str})
        with self.assertRaises(TransformError):
            v('{"1": "x"}')  # key parses as "1" (str), not int

    # -------- literal values (NEW FEATURE) --------

    def test_json_shape_literal_in_dict_value_success(self):
        v = json_shape({"type": "conclusion", "summary": str})
        out = v('{"type":"conclusion","summary":"ok"}')
        self.assertEqual(out["type"], "conclusion")
        self.assertEqual(out["summary"], "ok")

    def test_json_shape_literal_in_dict_value_mismatch(self):
        v = json_shape({"type": "conclusion", "summary": str})
        with self.assertRaises(TransformError):
            v('{"type":"summary","summary":"ok"}')

    def test_json_shape_literal_none(self):
        v = json_shape(None)
        self.assertIsNone(v("null"))
        with self.assertRaises(TransformError):
            v("0")

    def test_json_shape_literal_number(self):
        v = json_shape(5)
        self.assertEqual(v("5"), 5)
        with self.assertRaises(TransformError):
            v("6")

    def test_json_shape_literal_string(self):
        v = json_shape("ok")
        self.assertEqual(v('"ok"'), "ok")
        with self.assertRaises(TransformError):
            v('"nope"')

    def test_json_shape_literal_bool(self):
        v = json_shape(True)
        self.assertTrue(v("true"))
        with self.assertRaises(TransformError):
            v("false")

    # -------- parsing failures --------

    def test_json_shape_parse_failed(self):
        v = json_shape({"a": int})
        with self.assertRaises(TransformError) as ctx:
            v("{a: 1}")  # invalid even for your lenient loader per existing tests
        self.assertEqual(str(ctx.exception), "parse-failed")

    # -------- passthrough structured input --------

    def test_json_shape_accepts_already_parsed_dict(self):
        v = json_shape({"type": "conclusion", "summary": str})
        obj = {"type": "conclusion", "summary": "yo"}
        out = v(obj)
        self.assertIs(out, obj)  # loadch passthrough for dicts
