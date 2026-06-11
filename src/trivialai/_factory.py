# src/trivialai/_factory.py
"""
Configuration-driven model construction for trivialai.

Public surface (re-exported from the package root)
---------------------------------------------------
    text(config)               -> text-capable model instance (LLMMixin surface)
    image(config)              -> image-capable model instance (ImageMixin surface)
    from_env(prefix, ...)      -> config dict collected from environment variables
    providers()                -> registry listing (names, classes, capabilities)
    env_schema(provider, ...)  -> env-var schema derived live from a provider signature
    ConfigError                -> raised for every configuration problem

Design summary
--------------
* **Capability, not provider.**  Callers say "I need a text model"
  (``text(cfg)``) or "I need an image model" (``image(cfg)``); the
  ``provider`` key in the config selects the implementation.

* **The constructor signature *is* the schema.**  There is no mapping
  table.  Config keys are the literal ``__init__`` parameter names of the
  provider class; env vars are the same names, capitalized, under a role
  prefix::

      comp_PROVIDER=ollama
      comp_MODEL=qwen-coder:latest
      comp_OLLAMA_SERVER=http://some.host:12345
      comp_SKIP_HEALTHCHECK=false

      mcompiler = trivialai.text(trivialai.from_env("comp_"))

  Validation (unknown keys, missing required keys) and type coercion
  (bool / int / float / JSON-for-structured) are driven by
  ``inspect.signature`` + type annotations on the provider constructor.

* **No defaults, loud errors.**  Neither ``from_env`` nor ``text`` /
  ``image`` invents values.  Every failure raises ``ConfigError`` with a
  specific, actionable message.  ``from_env(prefix, default=...)`` is the
  one sanctioned default mechanism, and it is bundle-level: the default
  dict is returned verbatim when the prefix matches *nothing*, and is
  ignored entirely otherwise.  Cross-provider key merging never happens.

* **Construction-time side effects are intentional.**  Providers that
  health-check or build clients in ``__init__`` (Ollama, Bedrock) do so
  when ``text()`` / ``image()`` runs — i.e. at service startup.  This is
  the desired fail-fast behavior; use e.g. ``comp_SKIP_HEALTHCHECK=true``
  if a dependency is allowed to come up later.

* **Lazy provider imports.**  The registry maps provider names to module
  paths; nothing heavier than ``importlib`` is touched until a provider
  is actually requested.  A missing optional dependency (boto3,
  google-genai, ...) surfaces as a ``ConfigError`` naming the package.

Environment variable rules (``from_env``)
------------------------------------------
* ``from_env("comp_")`` scans variables beginning with ``comp_`` exactly.
  ``from_env("comp")`` (no trailing underscore) accepts both ``comp_X``
  and ``compX``; if the same logical key is reachable both ways, that is
  ambiguous and raises ``ConfigError``.
* The prefix is stripped and the remainder lowercased to form the config
  key: ``comp_OLLAMA_SERVER`` -> ``ollama_server``.
* Values are kept as strings; coercion happens later, in ``text`` /
  ``image``, driven by the chosen provider's annotations.
* **Empty values are unset values.**  ``comp_MODEL=`` is identical to
  ``comp_MODEL`` not existing.  Booleans get no exemption: to turn a
  flag off explicitly, say ``false``.
* ``{prefix}CONFIG_FILE`` may point to a JSON file containing an object
  whose keys are constructor parameter names (plus ``provider``).  The
  file is loaded as the base config; individual env vars under the same
  prefix override its keys.  This is the escape hatch for structured
  parameters that are awkward as flat env vars, though those can also be
  supplied inline as JSON strings (e.g.
  ``comp_SAFETY_SETTINGS={"hate_speech": "none"}``).
* A prefix that matches nothing (and no ``default=``) raises
  ``ConfigError`` — silence is never success.

Coercion rules (``text`` / ``image``)
--------------------------------------
String values are coerced according to the annotated parameter type;
non-string values (from ``default=`` dicts, ``CONFIG_FILE``, or configs
built in code) pass through untouched.

    bool                  "true" / "false", case-insensitive; anything
                          else is rejected.
    int / float           parsed strictly; rejected on failure.
    str                   passed through.
    dict / list / tuple   parsed as JSON; the decoded type must match.
    Optional[X]           coerced as X.
    Any / unannotated     passed through as the raw string.
"""

from __future__ import annotations

import difflib
import importlib
import inspect
import json
import os
import types
import typing
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "ConfigError",
    "REGISTRY",
    "register",
    "resolve_module",
    "from_env",
    "text",
    "image",
    "providers",
    "env_schema",
]


class ConfigError(Exception):
    """
    Raised for every model-configuration problem: ambiguous or missing
    environment variables, unknown providers, capability mismatches,
    unknown or missing constructor parameters, and values that fail type
    coercion.

    Services are expected to let this propagate (or catch it, log, and
    exit nonzero) at startup; it always indicates that the deployed
    configuration — not the code — needs fixing.
    """


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
#
# name -> (module path, class name, capabilities)
#
# Capabilities are declared here rather than discovered by importing the
# module, so that `providers()` and capability mismatches stay zero-import.
#
# This is the single source of truth: `text` / `image` resolve providers
# through it, and the package root walks it to lazily export the provider
# classes (`trivialai.Bedrock`, ...) — adding an adapter here is the only
# edit needed.  It is deliberately public: out-of-tree adapters can add
# themselves (preferably via `register()`, which validates the entry) and
# immediately work with `text` / `image` / `from_env` / `providers`.
#
# Module paths starting with "." are resolved relative to this package's
# *runtime* name, so the registry works whether the package is imported as
# `trivialai` (installed), `src.trivialai` (tests run from the repo root),
# or anything else.  Absolute paths are for out-of-tree adapters.

_PACKAGE = __package__ or "trivialai"

REGISTRY: Dict[str, Tuple[str, str, frozenset]] = {
    "ollama": (".ollama", "Ollama", frozenset({"text"})),
    "claude": (".claude", "Claude", frozenset({"text"})),
    "chatgpt": (".chatgpt", "ChatGPT", frozenset({"text"})),
    "openai": (".chatgpt", "ChatGPT", frozenset({"text"})),  # alias
    "deepseek": (".deepseek", "DeepSeek", frozenset({"text"})),
    "bedrock": (".bedrock", "Bedrock", frozenset({"text", "image"})),
    "gemini": (".gemini", "Gemini", frozenset({"text", "image"})),
    "stabdiff": (".stabdiff", "StabDiff", frozenset({"image"})),
}


def resolve_module(module_path: str):
    """
    Import a registry module path: relative (".ollama") against this
    package's runtime name, absolute ("mypkg.mything") as-is.
    """
    if module_path.startswith("."):
        return importlib.import_module(module_path, package=_PACKAGE)
    return importlib.import_module(module_path)


_CAPABILITIES = frozenset({"text", "image"})


def register(
    name: str,
    module_path: str,
    cls_name: str,
    capabilities: Iterable[str],
) -> None:
    """
    Add (or replace) a provider in the registry.

    Equivalent to a direct ``REGISTRY[name] = ...`` assignment, with
    validation.  The module is *not* imported here — laziness is preserved;
    a bad ``module_path`` / ``cls_name`` surfaces as a ``ConfigError`` on
    first use, like any other provider.  Use an absolute ``module_path``
    for out-of-tree adapters; paths starting with "." resolve relative to
    this package.

        trivialai.register("mything", "mypkg.mything", "MyThing", {"text"})
        m = trivialai.text({"provider": "mything", ...})
    """
    caps = frozenset(capabilities)
    bad = caps - _CAPABILITIES
    if bad:
        raise ConfigError(
            f"register({name!r}): unknown capabilities {sorted(bad)}. "
            f"Valid capabilities: {sorted(_CAPABILITIES)}."
        )
    if not caps:
        raise ConfigError(f"register({name!r}): at least one capability is required.")
    REGISTRY[str(name).strip().lower()] = (module_path, cls_name, caps)


# Hints appended to the ConfigError when importing a provider module fails,
# which in practice means an optional dependency is not installed.
_OPTIONAL_DEPS: Dict[str, str] = {
    "bedrock": "boto3",
    "gemini": "google-genai pillow",
    "stabdiff": "pillow",
}


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


def from_env(
    prefix: str,
    *,
    default: Optional[Mapping] = None,
    env: Optional[Mapping] = None,
) -> Dict[str, Any]:
    """
    Collect environment variables under ``prefix`` into a config dict.

    Generic by design: this function knows nothing about providers.  It
    strips the prefix, lowercases the remainder, skips empty values, and
    returns string values verbatim.  Type coercion and validation happen
    later in ``text`` / ``image`` against the chosen provider's signature.

    Parameters
    ----------
    prefix:
        Role prefix, e.g. ``"comp_"``.  A trailing underscore makes the
        match exact; without one, both ``comp_X`` and ``compX`` are
        accepted (ambiguity between the two raises ``ConfigError``).
    default:
        Bundle-level fallback.  Returned **verbatim** (shallow-copied)
        when the prefix matches nothing at all; ignored entirely
        otherwise.  Never merged.
    env:
        Mapping to scan instead of ``os.environ`` (for tests).

    Returns
    -------
    dict
        e.g. ``{"provider": "ollama", "model": "qwq:latest", ...}``

    Raises
    ------
    ConfigError
        If the prefix is empty, the same logical key is reachable through
        multiple variables, ``{prefix}CONFIG_FILE`` is unreadable or not
        a JSON object, or nothing matches and no ``default`` was given.
    """
    if not prefix:
        raise ConfigError(
            "from_env() requires a non-empty prefix; scanning the entire "
            "environment would pick up unrelated variables like PATH."
        )

    environ: Mapping = os.environ if env is None else env

    # Longest prefix first so each variable maps to exactly one key.
    prefixes = [prefix] if prefix.endswith("_") else [prefix + "_", prefix]

    # logical key -> list of (source variable name, value)
    candidates: Dict[str, List[Tuple[str, str]]] = {}
    for name in sorted(environ):
        value = environ[name]
        for p in prefixes:
            if name.startswith(p) and len(name) > len(p):
                if value == "":
                    # Empty values are unset values; they neither set keys
                    # nor participate in ambiguity.
                    break
                key = name[len(p) :].lower()
                candidates.setdefault(key, []).append((name, value))
                break  # first (longest) matching prefix wins for this var

    collisions = {k: v for k, v in candidates.items() if len(v) > 1}
    if collisions:
        details = "; ".join(
            f"key '{key}' is set by {', '.join(name for name, _ in sources)}"
            for key, sources in sorted(collisions.items())
        )
        raise ConfigError(
            f"Ambiguous environment variables for prefix '{prefix}': {details}. "
            f"Each setting must have exactly one source — please rename or "
            f"remove the duplicates."
        )

    config: Dict[str, Any] = {key: srcs[0][1] for key, srcs in candidates.items()}

    # ---- {prefix}CONFIG_FILE escape hatch ----
    config_file = config.pop("config_file", None)
    if config_file is not None:
        try:
            with open(config_file) as f:
                file_config = json.load(f)
        except OSError as e:
            raise ConfigError(
                f"Could not read config file {config_file!r} "
                f"(from {prefixes[0]}CONFIG_FILE): {e}"
            ) from e
        except json.JSONDecodeError as e:
            raise ConfigError(
                f"Config file {config_file!r} (from {prefixes[0]}CONFIG_FILE) "
                f"is not valid JSON: {e}"
            ) from e
        if not isinstance(file_config, dict):
            raise ConfigError(
                f"Config file {config_file!r} must contain a JSON object, "
                f"got {type(file_config).__name__}."
            )
        # File is the base; individual env vars override its keys.
        merged = {str(k).lower(): v for k, v in file_config.items()}
        merged.update(config)
        config = merged

    if not config:
        if default is not None:
            return dict(default)
        raise ConfigError(
            f"No environment variables found with prefix '{prefix}' "
            f"(empty values count as unset), and no default= was provided. "
            f"Set at least {prefixes[0]}PROVIDER, or pass a default config."
        )

    return config


# ---------------------------------------------------------------------------
# Signature-driven validation and coercion
# ---------------------------------------------------------------------------

_NONE_TYPE = type(None)
_UNION_ORIGINS: Tuple[Any, ...] = (
    typing.Union,
    getattr(types, "UnionType", typing.Union),
)


def _unwrap_optional(annotation: Any) -> Any:
    """Optional[X] / X | None -> X.  Ambiguous unions -> None (no coercion)."""
    if typing.get_origin(annotation) in _UNION_ORIGINS:
        args = [a for a in typing.get_args(annotation) if a is not _NONE_TYPE]
        return args[0] if len(args) == 1 else None
    return annotation


def _coerce(value: Any, annotation: Any, *, cls_name: str, param: str) -> Any:
    """
    Coerce a string config value according to the parameter's annotation.

    Non-string values (from default= dicts, CONFIG_FILE JSON, or configs
    built in code) pass through untouched — they already carry real types.
    """
    if not isinstance(value, str):
        return value

    target = _unwrap_optional(annotation)
    if target is None or target is Any or target is inspect.Parameter.empty:
        return value

    if target is bool:
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ConfigError(
            f"{cls_name}.{param} expects a boolean; got {value!r}. "
            f"Use 'true' or 'false' (case-insensitive)."
        )

    if target is int:
        try:
            return int(value.strip())
        except ValueError:
            raise ConfigError(
                f"{cls_name}.{param} expects an integer; got {value!r}."
            ) from None

    if target is float:
        try:
            return float(value.strip())
        except ValueError:
            raise ConfigError(
                f"{cls_name}.{param} expects a number; got {value!r}."
            ) from None

    if target is str:
        return value

    origin = typing.get_origin(target) or target
    if origin in (dict, list, tuple):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as e:
            raise ConfigError(
                f"{cls_name}.{param} expects a JSON-encoded "
                f"{origin.__name__}; got {value!r} ({e})."
            ) from None
        if origin is tuple:
            if not isinstance(parsed, list):
                raise ConfigError(
                    f"{cls_name}.{param} expects a JSON array (for a tuple); "
                    f"got {type(parsed).__name__}."
                )
            return tuple(parsed)
        if not isinstance(parsed, origin):
            raise ConfigError(
                f"{cls_name}.{param} expects a JSON {origin.__name__}; "
                f"decoded a {type(parsed).__name__} instead."
            )
        return parsed

    # Exotic annotation we don't understand: pass the string through and
    # let the constructor decide.
    return value


def _signature_and_hints(cls: type) -> Tuple[inspect.Signature, Dict[str, Any]]:
    sig = inspect.signature(cls.__init__)
    try:
        hints = typing.get_type_hints(cls.__init__)
    except Exception:
        # Unresolvable forward references etc.: fall back to no coercion.
        hints = {}
    return sig, hints


# ---------------------------------------------------------------------------
# Provider loading and construction
# ---------------------------------------------------------------------------


def _load_provider(provider: str) -> type:
    module_path, cls_name, _caps = REGISTRY[provider]
    try:
        module = resolve_module(module_path)
    except ImportError as e:
        hint = _OPTIONAL_DEPS.get(provider)
        extra = f" (try: pip install {hint})" if hint else ""
        raise ConfigError(
            f"Provider '{provider}' could not be loaded: {e}. "
            f"It may require optional dependencies that are not "
            f"installed{extra}."
        ) from e
    return getattr(module, cls_name)


def _build(config: Mapping, capability: str) -> Any:
    if not isinstance(config, Mapping):
        raise ConfigError(
            f"{capability}() expects a config dict, got "
            f"{type(config).__name__}. (Did you mean "
            f"{capability}(from_env('...'))?)"
        )

    cfg = dict(config)

    provider_raw = cfg.pop("provider", None)
    if provider_raw is None:
        raise ConfigError(
            f"Config has no 'provider' key; {capability}() cannot pick an "
            f"implementation. Known providers: {', '.join(sorted(REGISTRY))}."
        )
    provider = str(provider_raw).strip().lower()

    entry = REGISTRY.get(provider)
    if entry is None:
        close = difflib.get_close_matches(provider, REGISTRY, n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ConfigError(
            f"Unknown provider '{provider}'.{hint} "
            f"Known providers: {', '.join(sorted(REGISTRY))}."
        )

    _module_path, cls_name, caps = entry
    if capability not in caps:
        capable = sorted(n for n, (_, _, c) in REGISTRY.items() if capability in c)
        raise ConfigError(
            f"Provider '{provider}' has no {capability} capability "
            f"(it provides: {', '.join(sorted(caps))}). "
            f"Providers with {capability}: {', '.join(capable)}."
        )

    cls = _load_provider(provider)
    sig, hints = _signature_and_hints(cls)
    params = {n: p for n, p in sig.parameters.items() if n != "self"}
    has_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    kwargs: Dict[str, Any] = {}
    for key, value in cfg.items():
        if key not in params:
            if has_var_keyword:
                kwargs[key] = value
                continue
            close = difflib.get_close_matches(key, params, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise ConfigError(
                f"{cls_name}() has no parameter '{key}'.{hint} "
                f"Accepted parameters: {', '.join(sorted(params))}."
            )
        kwargs[key] = _coerce(
            value,
            hints.get(key, inspect.Parameter.empty),
            cls_name=cls_name,
            param=key,
        )

    missing = sorted(
        name
        for name, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and name not in kwargs
    )
    if missing:
        as_env = ", ".join(m.upper() for m in missing)
        raise ConfigError(
            f"{cls_name}() is missing required parameter(s): "
            f"{', '.join(missing)}. Set them in the config "
            f"(env vars: {{prefix}}{as_env})."
        )

    try:
        return cls(**kwargs)
    except TypeError as e:
        # Shouldn't happen after validation above; keep it loud and tagged.
        raise ConfigError(f"Failed to construct {cls_name}(): {e}") from e
    # Anything else (ValueError from health checks, botocore/auth errors,
    # ...) propagates as-is: construction-time side effects failing fast at
    # startup is intended behavior, and those errors are not config-shape
    # problems this layer can describe better than the provider already does.


def text(config: Mapping) -> Any:
    """
    Construct a **text-capable** model from a config dict.

    The returned object implements the ``LLMMixin`` surface: ``generate``,
    ``stream``, ``generate_json``, ``stream_checked``, ...

    ``config`` must contain ``provider`` plus the literal constructor
    parameters of that provider — typically produced by ``from_env``::

        mcompiler = trivialai.text(trivialai.from_env("comp_"))

    Raises ``ConfigError`` for any configuration problem.  Provider
    constructor errors (failed health checks, auth failures) propagate
    unchanged: configuration loading is intentionally fail-fast.
    """
    return _build(config, "text")


def image(config: Mapping) -> Any:
    """
    Construct an **image-capable** model from a config dict.

    The returned object implements the ``ImageMixin`` surface:
    ``generate_image`` / ``imagen``, ``imagestream``, ...

    See ``text`` for config semantics; everything is identical except the
    required capability.
    """
    return _build(config, "image")


# ---------------------------------------------------------------------------
# Discoverability
# ---------------------------------------------------------------------------


def providers() -> List[Dict[str, Any]]:
    """
    List known providers without importing any of them.

    Returns entries like::

        {"provider": "bedrock", "class": "Bedrock",
         "capabilities": ["image", "text"]}
    """
    return [
        {"provider": name, "class": cls_name, "capabilities": sorted(caps)}
        for name, (_mod, cls_name, caps) in sorted(REGISTRY.items())
    ]


def env_schema(provider: str, prefix: str = "") -> List[Dict[str, Any]]:
    """
    Describe the environment variables a provider accepts, derived live
    from its constructor signature.  (This imports the provider module,
    so its dependencies must be installed.)

    Returns entries like::

        {"env_var": "comp_MODEL", "param": "model",
         "type": "Optional[str]", "required": False, "default": None}

    Handy at the REPL::

        >>> for row in trivialai.env_schema("ollama", prefix="comp_"):
        ...     print(row["env_var"], "-", row["type"])
    """
    name = str(provider).strip().lower()
    if name not in REGISTRY:
        raise ConfigError(
            f"Unknown provider '{provider}'. "
            f"Known providers: {', '.join(sorted(REGISTRY))}."
        )
    cls = _load_provider(name)
    sig, hints = _signature_and_hints(cls)

    rows: List[Dict[str, Any]] = [
        {
            "env_var": f"{prefix}PROVIDER",
            "param": "provider",
            "type": "str",
            "required": True,
            "default": name,
        }
    ]
    for pname, p in sig.parameters.items():
        if pname == "self" or p.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        hint = hints.get(pname)
        rows.append(
            {
                "env_var": f"{prefix}{pname.upper()}",
                "param": pname,
                "type": getattr(hint, "__name__", str(hint) if hint else "str"),
                "required": p.default is inspect.Parameter.empty,
                "default": None if p.default is inspect.Parameter.empty else p.default,
            }
        )
    return rows
