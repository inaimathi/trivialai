# src/trivialai/__init__.py
"""
trivialai — small adapters over LLM and image-generation providers.

Two usage styles, both first-class:

Direct (classic)::

    import trivialai
    m = trivialai.Ollama("qwq:latest", "http://localhost:11434")
    m.generate("system", "prompt")

    # or, to avoid importing optional dependencies of other providers:
    from trivialai.ollama import Ollama

Configuration-driven::

    import trivialai
    mcompiler = trivialai.text(trivialai.from_env("comp_"))
    mpic      = trivialai.image(trivialai.from_env("diagram_"))

Provider classes and heavier helpers are exported lazily (PEP 562):
``import trivialai`` never imports boto3 / google-genai / pillow; the
cost is paid only when ``trivialai.Bedrock`` (etc.) is first touched.

Provider class exports are derived by walking ``trivialai.REGISTRY``, so
adding an adapter to the registry is the only edit needed — it becomes
reachable through ``text``/``image``, ``providers()``, *and* as a lazy
``trivialai.<ClassName>`` attribute, with no changes to this file.
"""

from __future__ import annotations

import sys
import types
from typing import Any

from . import _factory as _factory_mod
# Eager: light, factory-layer surface. _factory imports only stdlib.
from ._factory import REGISTRY, ConfigError, env_schema, from_env  # noqa: F401
from ._factory import image as _image_factory
from ._factory import providers, register, text  # noqa: F401

# ---------------------------------------------------------------------------
# `trivialai.image` collision guard
# ---------------------------------------------------------------------------
# The package contains an `image` *submodule* (Picture / ImageMixin).  Any
# `from .image import ...` — which the bedrock, gemini, and stabdiff modules
# all do — makes the import machinery run `setattr(trivialai, "image",
# <module>)`, which would silently clobber the `image()` factory function.
#
# A property on the module's *type* is a data descriptor, so it takes
# precedence over the instance __dict__ entry the import machinery writes,
# keeping `trivialai.image` bound to the factory while `import
# trivialai.image` / `from trivialai.image import Picture` continue to work
# normally (those resolve through sys.modules, not package attributes).
#
# Known wart: the aliasing form `import trivialai.image as x` binds via the
# package attribute (PEP 328 semantics since 3.7) and therefore yields the
# factory function, not the submodule.  Use `from trivialai.image import
# Picture` or `from trivialai import image as ...` instead; nothing in this
# codebase uses the aliasing form.


class _TrivialaiModule(types.ModuleType):
    @property
    def image(self):
        """Construct an image-capable model from a config dict (see _factory.image)."""
        return _image_factory

    @image.setter
    def image(self, value):
        # The import machinery setattr()s the `image` submodule onto the
        # package after importing it.  Swallow that write: the submodule
        # remains importable via its dotted path, and the factory keeps
        # the attribute slot.
        pass


sys.modules[__name__].__class__ = _TrivialaiModule


# ---------------------------------------------------------------------------
# Lazy exports (PEP 562)
# ---------------------------------------------------------------------------
# Non-provider helpers that shouldn't be imported until used.  Provider
# classes are NOT listed here — they are resolved by walking REGISTRY.

_LAZY_EXTRAS = {
    "BiStream": (".bistream", "BiStream"),
    "force": (".bistream", "force"),
    "LLMResult": (".util", "LLMResult"),
    "GenerationError": (".util", "GenerationError"),
    "TransformError": (".util", "TransformError"),
    "Picture": (".image", "Picture"),
}


def _registry_class_location(name: str):
    """Map a class name ('Bedrock') to its (module_path, attr) via REGISTRY."""
    for module_path, cls_name, _caps in REGISTRY.values():
        if cls_name == name:
            return module_path, cls_name
    return None


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute loading for provider classes and helpers."""
    location = _LAZY_EXTRAS.get(name) or _registry_class_location(name)
    if location is None:
        raise AttributeError(f"module 'trivialai' has no attribute {name!r}") from None
    module_path, attr = location
    obj = getattr(_factory_mod.resolve_module(module_path), attr)
    globals()[name] = obj  # cache so __getattr__ only fires once per name
    return obj


def __dir__() -> list[str]:
    provider_classes = {cls_name for _mod, cls_name, _caps in REGISTRY.values()}
    return sorted(set(globals()) | set(_LAZY_EXTRAS) | provider_classes | {"image"})
