# jsonesque.py
"""
A tiny parser for a JSON-ish language with a strict superset of JSON:

- Standard JSON values/structures
- Python-style triple quoted strings with newlines
- None, True, False as valid values (in addition to null/true/false)
- Set literals: {1, 2, 3} when there are no key:value pairs
- Tuple literals: (1, 2, 3) and (1,) (Python semantics)
- Byte literals: b"...", b'...', and triple-quoted ones too

Interface:
    jsonesque.loads(text: str) -> Any
    jsonesque.dumps(obj, *args, **kwargs) -> str

Parse errors raise ValueError.
Encoding is delegated directly to json.dumps (so json.dumps' errors are preserved).
"""

from __future__ import annotations

import ast
import json
from typing import Any


class _JsonesqueParser:
    def __init__(self, text: str):
        self.text = text
        self.len = len(text)
        self.pos = 0

    # ----- basic utilities -----

    def _peek(self) -> str:
        """Return current character or '' at EOF."""
        if self.pos >= self.len:
            return ""
        return self.text[self.pos]

    def _advance(self, n: int = 1) -> None:
        self.pos += n

    def _expect(self, ch: str) -> None:
        if self._peek() != ch:
            raise ValueError(f"Expected {ch!r} at position {self.pos}")
        self._advance(1)

    def _skip_ws(self) -> None:
        while self.pos < self.len and self.text[self.pos].isspace():
            self.pos += 1

    # ----- top level -----

    def parse(self) -> Any:
        self._skip_ws()
        value = self._parse_value()
        self._skip_ws()
        if self.pos != self.len:
            raise ValueError(f"Extra data after valid value at position {self.pos}")
        return value

    # ----- value dispatcher -----

    def _parse_value(self) -> Any:
        self._skip_ws()
        if self.pos >= self.len:
            raise ValueError("Unexpected end of input")

        ch = self._peek()

        if ch == "{":
            return self._parse_braces()
        if ch == "[":
            return self._parse_array()
        if ch == "(":
            return self._parse_parens()
        # bytes literals: b'...' or b"...", including triple-quoted versions
        if (
            ch in ("b", "B")
            and self.pos + 1 < self.len
            and self.text[self.pos + 1] in ("'", '"')
        ):
            return self._parse_bytes()
        if ch in ("'", '"'):
            return self._parse_string()
        if ch == "-" or ch.isdigit():
            return self._parse_number()
        if ch.isalpha() or ch == "_":
            return self._parse_name()

        raise ValueError(f"Unexpected character {ch!r} at position {self.pos}")

    # ----- numbers -----

    def _parse_number(self) -> Any:
        start = self.pos

        # optional sign
        if self._peek() == "-":
            self._advance()

        # integer part
        if not self._peek().isdigit():
            raise ValueError(f"Expected digit at position {self.pos}")

        while self._peek().isdigit():
            self._advance()

        # fractional part
        if self._peek() == ".":
            self._advance()
            if not self._peek().isdigit():
                raise ValueError(f"Expected digit after '.' at position {self.pos}")
            while self._peek().isdigit():
                self._advance()

        # exponent part
        if self._peek() in ("e", "E"):
            self._advance()
            if self._peek() in ("+", "-"):
                self._advance()
            if not self._peek().isdigit():
                raise ValueError(f"Expected digit in exponent at position {self.pos}")
            while self._peek().isdigit():
                self._advance()

        num_str = self.text[start : self.pos]
        try:
            if any(c in num_str for c in ".eE"):
                return float(num_str)
            else:
                return int(num_str)
        except ValueError as e:
            raise ValueError(f"Invalid number {num_str!r} at position {start}") from e

    # ----- names: null/true/false/None/True/False -----

    def _parse_name(self) -> Any:
        if not (self._peek().isalpha() or self._peek() == "_"):
            raise ValueError(f"Expected identifier at position {self.pos}")

        start = self.pos
        while self.pos < self.len and (
            self.text[self.pos].isalnum() or self.text[self.pos] == "_"
        ):
            self.pos += 1

        token = self.text[start : self.pos]

        if token in ("null",):
            return None
        if token in ("true", "True"):
            return True
        if token in ("false", "False"):
            return False
        if token in ("None",):
            return None

        raise ValueError(f"Unknown identifier {token!r} at position {start}")

    # ----- string & bytes literals -----

    def _parse_string_literal_text(self) -> str:
        """
        Return the exact source slice for a Python-like string literal,
        including the starting and ending quotes. Uses a simple scanner
        that understands single vs triple quotes.
        """
        if self.pos >= self.len:
            raise ValueError("Unexpected end of input while parsing string literal")

        quote = self._peek()
        if quote not in ("'", '"'):
            raise ValueError(f"Expected string quote at position {self.pos}")

        # Triple-quoted?
        if self.pos + 2 < self.len and self.text[self.pos : self.pos + 3] == quote * 3:
            delim = quote * 3
            start = self.pos
            i = self.pos + 3
            while True:
                j = self.text.find(delim, i)
                if j == -1:
                    raise ValueError(
                        f"Unterminated triple-quoted string starting at {start}"
                    )
                literal = self.text[start : j + 3]
                self.pos = j + 3
                return literal
        else:
            # Single-quoted string
            start = self.pos
            i = self.pos + 1
            while i < self.len:
                ch = self.text[i]
                if ch == "\\":
                    # skip escaped char
                    i += 2
                    continue
                if ch == quote:
                    literal = self.text[start : i + 1]
                    self.pos = i + 1
                    return literal
                i += 1
            raise ValueError(f"Unterminated string starting at {start}")

    def _parse_string(self) -> str:
        literal = self._parse_string_literal_text()
        try:
            # ast.literal_eval handles escapes, triple quotes, etc.
            value = ast.literal_eval(literal)
        except (SyntaxError, ValueError) as e:
            raise ValueError(
                f"Invalid string literal {literal!r} at position {self.pos}"
            ) from e
        if not isinstance(value, str):
            # Extremely unlikely, but guard against e.g. weird inputs
            raise ValueError(
                f"String literal did not evaluate to str at position {self.pos}"
            )
        return value

    def _parse_bytes(self) -> bytes:
        # We've already confirmed text[pos] is 'b' or 'B' and next is a quote
        prefix_pos = self.pos
        prefix = self._peek()
        self._advance(1)  # consume 'b' / 'B'

        literal = self._parse_string_literal_text()
        full = prefix + literal  # e.g. b"foo"

        try:
            value = ast.literal_eval(full)
        except (SyntaxError, ValueError) as e:
            raise ValueError(
                f"Invalid bytes literal {full!r} at position {prefix_pos}"
            ) from e
        if not isinstance(value, (bytes, bytearray)):
            raise ValueError(
                f"Bytes literal did not evaluate to bytes at position {prefix_pos}"
            )
        # Normalize to bytes
        return bytes(value)

    # ----- arrays -----

    def _parse_array(self) -> list[Any]:
        self._expect("[")
        self._skip_ws()
        if self._peek() == "]":
            self._advance(1)
            return []

        items = [self._parse_value()]
        self._skip_ws()
        while self._peek() == ",":
            self._advance(1)
            self._skip_ws()
            if self._peek() == "]":
                # allow trailing comma
                break
            items.append(self._parse_value())
            self._skip_ws()

        if self._peek() != "]":
            raise ValueError(f"Expected ']' at position {self.pos}")
        self._advance(1)
        return items

    # ----- parens: tuples / grouping -----

    def _parse_parens(self) -> Any:
        self._expect("(")
        self._skip_ws()
        if self._peek() == ")":
            self._advance(1)
            return ()

        first = self._parse_value()
        self._skip_ws()

        ch = self._peek()
        if ch == ",":
            # Tuple: (a, b, c)
            values = [first]
            while self._peek() == ",":
                self._advance(1)
                self._skip_ws()
                if self._peek() == ")":
                    # allow trailing comma
                    break
                values.append(self._parse_value())
                self._skip_ws()
            if self._peek() != ")":
                raise ValueError(f"Expected ')' to close tuple at position {self.pos}")
            self._advance(1)
            return tuple(values)
        elif ch == ")":
            # Grouping: (value) -> just value (Python semantics)
            self._advance(1)
            return first
        else:
            raise ValueError(
                f"Expected ',' or ')' after value in parentheses at position {self.pos}"
            )

    # ----- braces: objects or sets -----

    def _parse_braces(self) -> Any:
        """
        Parse either:
        - dict/object: {'key': value, ...}  (JSON-style or Python single/triple quotes)
        - set: {value1, value2, ...} when there are no ':' separators at top level
        """
        self._expect("{")
        self._skip_ws()
        if self._peek() == "}":
            self._advance(1)
            return {}

        # Parse first element, then decide whether this is an object or a set
        first_value = self._parse_value()
        self._skip_ws()

        if self._peek() == ":":
            # Object / dict
            self._advance(1)  # consume ':'
            value = self._parse_value()
            obj = {}
            try:
                obj[first_value] = value
            except TypeError as e:
                raise ValueError(f"Unhashable object key: {first_value!r}") from e

            self._skip_ws()
            while self._peek() == ",":
                self._advance(1)
                self._skip_ws()
                if self._peek() == "}":
                    # allow trailing comma
                    break
                key = self._parse_value()
                self._skip_ws()
                if self._peek() != ":":
                    raise ValueError(
                        f"Expected ':' after object key at position {self.pos}"
                    )
                self._advance(1)
                val = self._parse_value()
                try:
                    obj[key] = val
                except TypeError as e:
                    raise ValueError(f"Unhashable object key: {key!r}") from e
                self._skip_ws()

            if self._peek() != "}":
                raise ValueError(f"Expected '}}' at position {self.pos}")
            self._advance(1)
            return obj
        else:
            # Set literal: {v1, v2, ...}
            values = [first_value]
            self._skip_ws()
            while self._peek() == ",":
                self._advance(1)
                self._skip_ws()
                if self._peek() == "}":
                    # allow trailing comma
                    break
                values.append(self._parse_value())
                self._skip_ws()

            if self._peek() != "}":
                raise ValueError(f"Expected '}}' at position {self.pos}")
            self._advance(1)
            try:
                return set(values)
            except TypeError as e:
                raise ValueError("Unhashable value in set literal") from e


# ----- public API -----


def loads(text: str) -> Any:
    """
    Parse a JSON-esque string and return the corresponding Python object.

    Superset of JSON with:
      - triple-quoted strings
      - None / True / False
      - sets, tuples, bytes literals

    Raises ValueError on parse errors.
    """
    parser = _JsonesqueParser(text)
    return parser.parse()


def dumps(obj: Any, *args, **kwargs) -> str:
    """
    Serialize obj using the standard json.dumps.

    Any encoding errors (unsupported types, etc.) are left to json.dumps.
    """
    return json.dumps(obj, *args, **kwargs)
