# json_shaped.py
from __future__ import annotations

import types
from dataclasses import dataclass
from datetime import date, datetime
from typing import (Any, Callable, Dict, List, Literal, TypeAlias, Union,
                    get_args, get_origin)

# ----------------------------- types ------------------------------------

JsonPrimitive: TypeAlias = Union[None, bool, int, float, str]

JsonValue: TypeAlias = Union[
    JsonPrimitive,
    List["JsonValue"],
    Dict[str, "JsonValue"],
]

JsonInput: TypeAlias = JsonValue | str

ShapeSpec: TypeAlias = Union[
    type[int],
    type[float],
    type[str],
    type[bool],
    type[dict],
    type[list],
    type[tuple],
    type[date],
    type[datetime],
    None | bool | int | float | str,
    list["ShapeSpec"],
    tuple["ShapeSpec", ...],
    dict[str, "ShapeSpec"],
    dict[type[Any], "ShapeSpec"],
    Any,  # allows Literal[...] / Union[...] specs + wrappers like Exact(...)
]


# --------------------------- helpers ------------------------------------


class TransformError(Exception):
    def __init__(self, message: str = "Transformation Error", raw: Any = None):
        self.message = message
        self.raw = raw
        super().__init__(self.message)


def _type_sensitive_equal(data: Any, lit: Any) -> bool:
    """
    JSON-literal equality, but type-sensitive:
      - False must not match 0
      - True must not match 1
      - int literals must not match bools
      - float literals may match int/float numerically
    """
    if lit is None:
        return data is None

    if isinstance(lit, bool):
        return isinstance(data, bool) and data == lit

    if isinstance(lit, str):
        return isinstance(data, str) and data == lit

    if isinstance(lit, int) and not isinstance(lit, bool):
        return isinstance(data, int) and not isinstance(data, bool) and data == lit

    if isinstance(lit, float):
        return (
            isinstance(data, (int, float))
            and not isinstance(data, bool)
            and data == lit
        )

    return data == lit


def _parse_iso_datetime(s: str) -> datetime:
    # Accept "Z" suffix
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _parse_iso_date(s: str) -> date:
    return date.fromisoformat(s)


# ----------------------------- Exact ------------------------------------


@dataclass(frozen=True)
class _Exact:
    shape: Any


def Exact(shape: Any) -> _Exact:
    """
    Wrap a specific-keys dict shape to reject extra keys at that subtree.

    Example:
        shape = {"outer": Exact({"a": int})}
        # outer must have exactly {"a": ...} (no extras)
    """
    return _Exact(shape)


# -------------------------- validation ----------------------------------


def _validate_shape(
    data: Any,
    shape: Any,
    path: str,
    *,
    coerce: bool,
) -> Any:
    """
    Validate (and optionally coerce) data against a shape.
    Returns possibly-transformed data if coerce=True, otherwise returns the original.
    """

    # --- Exact wrapper ---------------------------------------------------
    exact = False
    if isinstance(shape, _Exact):
        exact = True
        shape = shape.shape

    origin = get_origin(shape)

    # --- typing.Literal[...] --------------------------------------------
    if origin is Literal:
        allowed = get_args(shape)
        if not any(_type_sensitive_equal(data, a) for a in allowed):
            raise TransformError(
                f"At {path}: Expected one of {allowed!r}, got {data!r}"
            )
        return data

    # --- Union / | -------------------------------------------------------
    if origin in (Union, types.UnionType):
        last_err: Exception | None = None
        for opt in get_args(shape):
            try:
                return _validate_shape(data, opt, path, coerce=coerce)
            except TransformError as e:
                last_err = e
        raise TransformError(
            f"At {path}: Value did not match any allowed union option; last error: {last_err}"
        )

    # --- date ------------------------------------------------------------
    if shape is date:
        if isinstance(data, date) and not isinstance(data, datetime):
            return data
        if not isinstance(data, str):
            raise TransformError(
                f"At {path}: Expected ISO date string, got {type(data).__name__}"
            )
        # Must NOT look like a datetime
        if "T" in data or " " in data:
            raise TransformError(
                f"At {path}: Expected date (YYYY-MM-DD), got datetime-like {data!r}"
            )
        try:
            parsed = _parse_iso_date(data)
        except Exception:
            raise TransformError(f"At {path}: Expected date (YYYY-MM-DD), got {data!r}")
        return parsed if coerce else data

    # --- datetime (reject date-only) ------------------------------------
    if shape is datetime:
        if isinstance(data, datetime):
            return data
        # IMPORTANT: datetime is a subclass of date; reject bare date objects
        if isinstance(data, date) and not isinstance(data, datetime):
            raise TransformError(
                f"At {path}: Expected datetime (with time), got date {data!r}"
            )
        if not isinstance(data, str):
            raise TransformError(
                f"At {path}: Expected ISO datetime string, got {type(data).__name__}"
            )
        # Must have a time component
        if ("T" not in data) and (" " not in data):
            raise TransformError(
                f"At {path}: Expected datetime (with time), got date-only {data!r}"
            )
        try:
            parsed = _parse_iso_datetime(data)
        except Exception:
            raise TransformError(
                f"At {path}: Expected ISO datetime string, got {data!r}"
            )
        return parsed if coerce else data

    # --- Raw literal values ---------------------------------------------
    if shape is None:
        if data is not None:
            raise TransformError(f"At {path}: Expected None, got {data!r}")
        return data

    if isinstance(shape, bool):
        if not isinstance(data, bool) or data != shape:
            raise TransformError(f"At {path}: Expected literal {shape!r}, got {data!r}")
        return data

    if isinstance(shape, str):
        if not isinstance(data, str) or data != shape:
            raise TransformError(f"At {path}: Expected literal {shape!r}, got {data!r}")
        return data

    if isinstance(shape, int) and not isinstance(shape, bool):
        if not isinstance(data, int) or isinstance(data, bool) or data != shape:
            raise TransformError(f"At {path}: Expected literal {shape!r}, got {data!r}")
        return data

    if isinstance(shape, float):
        if (
            not isinstance(data, (int, float))
            or isinstance(data, bool)
            or data != shape
        ):
            raise TransformError(f"At {path}: Expected literal {shape!r}, got {data!r}")
        return data

    # --- Basic types -----------------------------------------------------
    if shape == int:
        if not isinstance(data, int) or isinstance(data, bool):
            raise TransformError(f"At {path}: Expected int, got {type(data).__name__}")
        return data

    if shape == float:
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            raise TransformError(
                f"At {path}: Expected float, got {type(data).__name__}"
            )
        return data

    if shape == str:
        if not isinstance(data, str):
            raise TransformError(f"At {path}: Expected str, got {type(data).__name__}")
        return data

    if shape == bool:
        if not isinstance(data, bool):
            raise TransformError(f"At {path}: Expected bool, got {type(data).__name__}")
        return data

    # --- Container type checks ------------------------------------------
    if shape == dict:
        if not isinstance(data, dict):
            raise TransformError(f"At {path}: Expected dict, got {type(data).__name__}")
        for k in data.keys():
            if not isinstance(k, str):
                raise TransformError(
                    f"At {path}: Expected dict with str keys, got key {k!r} of type {type(k).__name__}"
                )
        return data

    if shape == list:
        if not isinstance(data, list):
            raise TransformError(f"At {path}: Expected list, got {type(data).__name__}")
        return data

    if shape == tuple:
        if not isinstance(data, tuple):
            raise TransformError(
                f"At {path}: Expected tuple, got {type(data).__name__}"
            )
        return data

    # --- List shapes -----------------------------------------------------
    if isinstance(shape, list):
        if not isinstance(data, (list, tuple)):
            raise TransformError(
                f"At {path}: Expected list/tuple, got {type(data).__name__}"
            )
        if len(shape) != 1:
            raise TransformError(
                f"Shape specification error: List must have exactly one element type, got {len(shape)}"
            )

        element_shape = shape[0]
        if not coerce:
            for i, item in enumerate(data):
                _validate_shape(item, element_shape, f"{path}[{i}]", coerce=False)
            return data

        out_items: list[Any] = []
        changed = False
        for i, item in enumerate(data):
            coerced_item = _validate_shape(
                item, element_shape, f"{path}[{i}]", coerce=True
            )
            out_items.append(coerced_item)
            if coerced_item is not item:
                changed = True

        if not changed:
            return data
        return tuple(out_items) if isinstance(data, tuple) else out_items

    # --- Tuple shapes ----------------------------------------------------
    if isinstance(shape, tuple):
        if not isinstance(data, (list, tuple)):
            raise TransformError(
                f"At {path}: Expected list/tuple, got {type(data).__name__}"
            )
        if len(data) != len(shape):
            raise TransformError(
                f"At {path}: Expected tuple/list of length {len(shape)}, got {len(data)}"
            )

        if not coerce:
            for i, (item, item_shape) in enumerate(zip(data, shape)):
                _validate_shape(item, item_shape, f"{path}[{i}]", coerce=False)
            return data

        out_items: list[Any] = []
        changed = False
        for i, (item, item_shape) in enumerate(zip(data, shape)):
            coerced_item = _validate_shape(
                item, item_shape, f"{path}[{i}]", coerce=True
            )
            out_items.append(coerced_item)
            if coerced_item is not item:
                changed = True

        if not changed:
            return data
        return tuple(out_items) if isinstance(data, tuple) else out_items

    # --- Dict shapes -----------------------------------------------------
    if isinstance(shape, dict):
        if not isinstance(data, dict):
            raise TransformError(f"At {path}: Expected dict, got {type(data).__name__}")

        basic_types = {int, float, str, bool}
        type_keys = [k for k in shape.keys() if k in basic_types]

        if type_keys:
            # General type dict like {str: int}
            if exact:
                raise TransformError(
                    f"Shape specification error: Exact() only applies to dicts with specific keys, at {path}"
                )
            if len(shape) != 1:
                raise TransformError(
                    f"Shape specification error: Type-based dict must have exactly one key-value pair, got {len(shape)}"
                )

            key_type, value_type = next(iter(shape.items()))

            if not coerce:
                for key, value in data.items():
                    # Validate key type
                    if key_type == int and not (
                        isinstance(key, int) and not isinstance(key, bool)
                    ):
                        raise TransformError(
                            f"At {path}: Key {key!r} should be int, got {type(key).__name__}"
                        )
                    elif key_type == float and not (
                        isinstance(key, (int, float)) and not isinstance(key, bool)
                    ):
                        raise TransformError(
                            f"At {path}: Key {key!r} should be float, got {type(key).__name__}"
                        )
                    elif key_type == str and not isinstance(key, str):
                        raise TransformError(
                            f"At {path}: Key {key!r} should be str, got {type(key).__name__}"
                        )
                    elif key_type == bool and not isinstance(key, bool):
                        raise TransformError(
                            f"At {path}: Key {key!r} should be bool, got {type(key).__name__}"
                        )

                    _validate_shape(value, value_type, f"{path}[{key!r}]", coerce=False)
                return data

            out: dict[Any, Any] = {}
            changed = False
            for key, value in data.items():
                # Validate key type
                if key_type == int and not (
                    isinstance(key, int) and not isinstance(key, bool)
                ):
                    raise TransformError(
                        f"At {path}: Key {key!r} should be int, got {type(key).__name__}"
                    )
                elif key_type == float and not (
                    isinstance(key, (int, float)) and not isinstance(key, bool)
                ):
                    raise TransformError(
                        f"At {path}: Key {key!r} should be float, got {type(key).__name__}"
                    )
                elif key_type == str and not isinstance(key, str):
                    raise TransformError(
                        f"At {path}: Key {key!r} should be str, got {type(key).__name__}"
                    )
                elif key_type == bool and not isinstance(key, bool):
                    raise TransformError(
                        f"At {path}: Key {key!r} should be bool, got {type(key).__name__}"
                    )

                coerced_val = _validate_shape(
                    value, value_type, f"{path}[{key!r}]", coerce=True
                )
                out[key] = coerced_val
                if coerced_val is not value:
                    changed = True

            if not changed:
                return data
            return out

        # Specific keys dict like {"a": int, "b": str}
        required_keys = set(shape.keys())
        data_keys = set(data.keys())

        missing_keys = required_keys - data_keys
        if missing_keys:
            raise TransformError(f"At {path}: Missing required keys: {missing_keys}")

        if exact:
            extra_keys = data_keys - required_keys
            if extra_keys:
                raise TransformError(f"At {path}: Unexpected extra keys: {extra_keys}")

        if not coerce:
            for key, value_shape in shape.items():
                _validate_shape(data[key], value_shape, f"{path}.{key}", coerce=False)
            return data

        # coerce=True: only coerce the keys we actually validate; leave extra keys as-is
        out = dict(data)  # shallow copy
        changed = False
        for key, value_shape in shape.items():
            oldv = data[key]
            newv = _validate_shape(oldv, value_shape, f"{path}.{key}", coerce=True)
            out[key] = newv
            if newv is not oldv:
                changed = True

        if not changed:
            return data
        return out

    if exact:
        raise TransformError(
            f"Shape specification error: Exact() only applies to dict shapes, at {path}"
        )

    raise TransformError(f"Invalid shape specification: {shape!r}")


# ---------------------------- public API --------------------------------


def json_shape(shape: ShapeSpec, *, coerce: bool = False) -> Callable[[Any], Any]:
    """
    Parse JSON-ish input using util.loadch, then validate against 'shape'.

    If coerce=True:
      - date strings become datetime.date
      - datetime strings become datetime.datetime
      - containers are rebuilt only if a child is coerced (otherwise passthrough)
    """

    def validator(json_input: Any) -> Any:
        # local import avoids import cycle (util imports json_shape)
        from .util import loadch

        parsed = loadch(json_input)
        return _validate_shape(parsed, shape, "root", coerce=coerce)

    return validator


def is_json_shaped(shape: ShapeSpec) -> Callable[[JsonValue], bool]:
    cb = json_shape(shape, coerce=False)

    def _validate(thing: JsonValue) -> bool:
        try:
            cb(thing)
            return True
        except TransformError:
            return False

    return _validate
