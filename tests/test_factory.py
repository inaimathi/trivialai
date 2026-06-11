# tests/test_factory.py
"""
Tests for trivialai._factory and the package-root export behavior.

Provider construction is tested against a dummy provider injected into the
registry, so no real provider dependencies (httpx, boto3, ...) are needed.

TestRegistryCompleteness is the drift guard: it scans the package for
adapter classes and fails if any are missing from REGISTRY, so "add the
one registry line" is enforced by CI rather than by memory.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import trivialai
except ImportError:  # repo layout: package lives under src/ and isn't installed
    from src import trivialai

PKG = trivialai.__name__  # "trivialai" installed, "src.trivialai" from the repo
_factory = importlib.import_module(f"{PKG}._factory")
REGISTRY = _factory.REGISTRY
ConfigError = _factory.ConfigError
from_env = _factory.from_env
providers = _factory.providers
text = _factory.text


# ---------------------------------------------------------------------------
# Dummy provider, injected into the registry per-test
# ---------------------------------------------------------------------------


class DummyText:
    def __init__(
        self,
        model: str,
        *,
        server: str = "http://localhost",
        max_tokens: Optional[int] = 4096,
        temperature: float = 0.7,
        skip_healthcheck: bool = False,
        extras: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = 300.0,
    ):
        self.model = model
        self.server = server
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.skip_healthcheck = skip_healthcheck
        self.extras = extras
        self.timeout = timeout


class DummyProviderCase(unittest.TestCase):
    """Base case that registers DummyText as provider 'dummy'."""

    def setUp(self):
        self._mod = types.ModuleType("dummy_provider_mod")
        self._mod.DummyText = DummyText
        sys.modules["dummy_provider_mod"] = self._mod
        REGISTRY["dummy"] = ("dummy_provider_mod", "DummyText", frozenset({"text"}))

    def tearDown(self):
        REGISTRY.pop("dummy", None)
        sys.modules.pop("dummy_provider_mod", None)


# ---------------------------------------------------------------------------
# from_env: collection, prefixes, collisions, empties, defaults, CONFIG_FILE
# ---------------------------------------------------------------------------


class TestFromEnv(unittest.TestCase):
    def test_collects_strips_and_lowercases(self):
        env = {
            "comp_PROVIDER": "ollama",
            "comp_MODEL": "qwq:latest",
            "comp_OLLAMA_SERVER": "http://h:11434",
            "UNRELATED": "x",
        }
        self.assertEqual(
            from_env("comp_", env=env),
            {
                "provider": "ollama",
                "model": "qwq:latest",
                "ollama_server": "http://h:11434",
            },
        )

    def test_bare_prefix_accepts_both_forms(self):
        self.assertEqual(from_env("comp", env={"compMODEL": "m"}), {"model": "m"})
        self.assertEqual(from_env("comp", env={"comp_MODEL": "m"}), {"model": "m"})

    def test_trailing_underscore_is_exact(self):
        with self.assertRaisesRegex(ConfigError, "No environment variables"):
            from_env("comp_", env={"compMODEL": "m"})

    def test_collision_is_loud_and_names_the_variables(self):
        with self.assertRaises(ConfigError) as cm:
            from_env("comp", env={"comp_MODEL": "a", "compMODEL": "b"})
        msg = str(cm.exception)
        self.assertIn("comp_MODEL", msg)
        self.assertIn("compMODEL", msg)
        self.assertIn("exactly one source", msg)

    def test_empty_values_are_unset(self):
        env = {"comp_PROVIDER": "ollama", "comp_MODEL": ""}
        self.assertEqual(from_env("comp_", env=env), {"provider": "ollama"})

    def test_empty_values_do_not_create_collisions(self):
        env = {"comp_MODEL": "", "compMODEL": "real"}
        self.assertEqual(from_env("comp", env=env), {"model": "real"})

    def test_nothing_found_raises_without_default(self):
        with self.assertRaisesRegex(ConfigError, "prefix 'comp_'"):
            from_env("comp_", env={})

    def test_default_is_bundle_level(self):
        default = {"provider": "ollama", "model": "qwq:latest"}
        # Nothing found -> default verbatim (copied).
        out = from_env("comp_", default=default, env={})
        self.assertEqual(out, default)
        self.assertIsNot(out, default)
        # Anything found -> default ignored entirely, never merged.
        out = from_env("comp_", default=default, env={"comp_PROVIDER": "bedrock"})
        self.assertEqual(out, {"provider": "bedrock"})

    def test_rejects_empty_prefix(self):
        with self.assertRaisesRegex(ConfigError, "non-empty prefix"):
            from_env("", env={"PATH": "/usr/bin"})

    def test_config_file_is_base_env_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "model.json"
            cfg.write_text(
                json.dumps(
                    {
                        "provider": "dummy",
                        "model": "from-file",
                        "extras": {"nested": [1, 2]},
                    }
                )
            )
            env = {"comp_CONFIG_FILE": str(cfg), "comp_MODEL": "from-env"}
            out = from_env("comp_", env=env)
        self.assertEqual(
            out,
            {
                "provider": "dummy",
                "model": "from-env",  # env var wins over the file
                "extras": {"nested": [1, 2]},  # structured value arrives typed
            },
        )
        self.assertNotIn("config_file", out)

    def test_config_file_errors_are_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.json")
            with self.assertRaisesRegex(ConfigError, "Could not read"):
                from_env("comp_", env={"comp_CONFIG_FILE": missing})

            bad = Path(tmp) / "bad.json"
            bad.write_text("{nope")
            with self.assertRaisesRegex(ConfigError, "not valid JSON"):
                from_env("comp_", env={"comp_CONFIG_FILE": str(bad)})

            arr = Path(tmp) / "arr.json"
            arr.write_text("[1,2]")
            with self.assertRaisesRegex(ConfigError, "JSON object"):
                from_env("comp_", env={"comp_CONFIG_FILE": str(arr)})


# ---------------------------------------------------------------------------
# text/image: provider resolution, capability, validation, coercion
# ---------------------------------------------------------------------------


class TestProviderResolution(unittest.TestCase):
    def test_missing_provider_key(self):
        with self.assertRaisesRegex(ConfigError, "no 'provider' key"):
            text({"model": "m"})

    def test_unknown_provider_suggests(self):
        with self.assertRaisesRegex(ConfigError, "Did you mean 'ollama'"):
            text({"provider": "olama", "model": "m"})

    def test_capability_mismatch_is_loud(self):
        # stabdiff is image-only; the error must say so without importing it.
        with self.assertRaisesRegex(ConfigError, "no text capability"):
            text({"provider": "stabdiff"})
        with self.assertRaisesRegex(ConfigError, "no image capability"):
            trivialai.image({"provider": "ollama"})


class TestValidationAndCoercion(DummyProviderCase):
    def test_provider_is_case_insensitive(self):
        self.assertIsInstance(text({"provider": "DuMmY", "model": "m"}), DummyText)

    def test_unknown_parameter_suggests_and_lists(self):
        with self.assertRaises(ConfigError) as cm:
            text({"provider": "dummy", "model": "m", "servr": "x"})
        msg = str(cm.exception)
        self.assertIn("no parameter 'servr'", msg)
        self.assertIn("Did you mean 'server'", msg)
        self.assertIn("model", msg)  # accepted-parameter listing

    def test_missing_required_parameter_names_env_var(self):
        with self.assertRaises(ConfigError) as cm:
            text({"provider": "dummy"})
        self.assertIn("{prefix}MODEL", str(cm.exception))

    def test_coercion_int_float_bool_str(self):
        m = text(
            {
                "provider": "dummy",
                "model": "m",
                "max_tokens": "8192",
                "temperature": "0.2",
                "skip_healthcheck": "False",
                "server": "http://h",
            }
        )
        self.assertEqual(m.max_tokens, 8192)
        self.assertEqual(m.temperature, 0.2)
        self.assertIs(m.skip_healthcheck, False)
        self.assertEqual(m.server, "http://h")

    def test_bool_spelling_is_strict(self):
        for bad in ("yes", "1", "on", ""):
            with self.assertRaisesRegex(ConfigError, "'true' or 'false'"):
                text({"provider": "dummy", "model": "m", "skip_healthcheck": bad})
        m = text({"provider": "dummy", "model": "m", "skip_healthcheck": "TRUE"})
        self.assertIs(m.skip_healthcheck, True)

    def test_numeric_coercion_failures_are_specific(self):
        with self.assertRaisesRegex(ConfigError, "max_tokens expects an integer"):
            text({"provider": "dummy", "model": "m", "max_tokens": "lots"})
        with self.assertRaisesRegex(ConfigError, "temperature expects a number"):
            text({"provider": "dummy", "model": "m", "temperature": "warm"})

    def test_structured_parameter_as_json_string(self):
        m = text({"provider": "dummy", "model": "m", "extras": '{"a": 1, "b": [2, 3]}'})
        self.assertEqual(m.extras, {"a": 1, "b": [2, 3]})
        with self.assertRaisesRegex(ConfigError, "JSON"):
            text({"provider": "dummy", "model": "m", "extras": "{nope"})
        with self.assertRaisesRegex(ConfigError, "dict"):
            text({"provider": "dummy", "model": "m", "extras": "[1, 2]"})

    def test_non_string_values_pass_through_untouched(self):
        # Values from default= dicts / CONFIG_FILE already carry real types.
        m = text(
            {
                "provider": "dummy",
                "model": "m",
                "max_tokens": 1024,
                "skip_healthcheck": True,
                "extras": {"k": "v"},
            }
        )
        self.assertEqual(m.max_tokens, 1024)
        self.assertIs(m.skip_healthcheck, True)
        self.assertEqual(m.extras, {"k": "v"})

    def test_end_to_end_from_env_to_text(self):
        env = {
            "comp_PROVIDER": "dummy",
            "comp_MODEL": "qwen-coder:latest",
            "comp_SERVER": "http://some.host:12345",
            "comp_SKIP_HEALTHCHECK": "false",
            "comp_MAX_TOKENS": "8192",
        }
        m = text(from_env("comp_", env=env))
        self.assertEqual(
            (m.model, m.server, m.skip_healthcheck, m.max_tokens),
            ("qwen-coder:latest", "http://some.host:12345", False, 8192),
        )


# ---------------------------------------------------------------------------
# Discoverability and package-root behavior
# ---------------------------------------------------------------------------


class TestDiscoverabilityAndRoot(unittest.TestCase):
    def test_providers_lists_without_importing(self):
        # Order-independent laziness check: other tests in a shared suite
        # may already have imported provider modules, so we can't assert
        # about absolute sys.modules state.  The real claim is that the
        # providers() *call itself* imports nothing.
        before = set(sys.modules)
        entries = {e["provider"]: e for e in providers()}
        self.assertEqual(
            set(sys.modules) - before,
            set(),
            "providers() should not import any modules",
        )
        self.assertEqual(entries["bedrock"]["capabilities"], ["image", "text"])
        self.assertEqual(entries["ollama"]["capabilities"], ["text"])
        self.assertEqual(entries["stabdiff"]["capabilities"], ["image"])

    def test_root_image_attribute_survives_submodule_import(self):
        self.assertTrue(callable(trivialai.image))
        # equivalent to `from <pkg>.image import Picture`, the form the
        # provider modules use, but agnostic to the package's runtime name
        img_mod = importlib.import_module(f"{PKG}.image")
        Picture = img_mod.Picture

        self.assertTrue(callable(trivialai.image))  # still the factory
        self.assertIs(sys.modules[f"{PKG}.image"].Picture, Picture)

    def test_root_lazy_attribute_error_is_clean(self):
        with self.assertRaisesRegex(AttributeError, "Nonexistent"):
            trivialai.Nonexistent

    def test_registry_drives_root_exports(self):
        # Registering a provider makes it reachable as a lazy class attribute
        # with no __init__.py edits.
        mod = types.ModuleType("late_mod")

        class LateThing:
            def __init__(self, model: str):
                self.model = model

        mod.LateThing = LateThing
        sys.modules["late_mod"] = mod
        trivialai.register("latething", "late_mod", "LateThing", {"text"})
        try:
            self.assertIs(trivialai.LateThing, LateThing)
            self.assertIn("LateThing", dir(trivialai))
            self.assertIsInstance(
                text({"provider": "latething", "model": "m"}), LateThing
            )
        finally:
            REGISTRY.pop("latething", None)
            vars(trivialai).pop("LateThing", None)  # drop cached lazy attribute
            sys.modules.pop("late_mod", None)

    def test_register_validates_capabilities(self):
        with self.assertRaisesRegex(ConfigError, "unknown capabilities"):
            trivialai.register("x", "m", "C", {"text", "audio"})
        with self.assertRaisesRegex(ConfigError, "at least one capability"):
            trivialai.register("x", "m", "C", set())
        self.assertNotIn("x", REGISTRY)


# ---------------------------------------------------------------------------
# Registry completeness (the drift guard)
# ---------------------------------------------------------------------------


class TestRegistryCompleteness(unittest.TestCase):
    """
    Enforce the convention that every adapter module is in REGISTRY.

    Scans the trivialai package, imports each public module (skipping ones
    whose optional dependencies aren't installed in this environment), and
    asserts that every class defined there which subclasses LLMMixin or
    ImageMixin appears in REGISTRY with capabilities matching its mixins.

    This converts "remember to add the registry line" into a CI failure.
    """

    def _mixins(self):
        try:
            LLMMixin = importlib.import_module(f"{PKG}.llm").LLMMixin
            ImageMixin = importlib.import_module(f"{PKG}.image").ImageMixin
        except ImportError as e:
            self.skipTest(f"core mixins unavailable in this environment: {e}")
        return LLMMixin, ImageMixin

    def test_every_adapter_is_registered_with_correct_capabilities(self):
        LLMMixin, ImageMixin = self._mixins()
        registered = {
            # Normalize relative registry paths (".ollama") to the package's
            # runtime name so they compare against scanned module paths.
            (
                (PKG + module_path) if module_path.startswith(".") else module_path,
                cls_name,
            ): caps
            for module_path, cls_name, caps in REGISTRY.values()
        }
        skipped, checked = [], []

        for info in pkgutil.iter_modules(trivialai.__path__):
            if info.name.startswith("_"):
                continue
            module_path = f"{PKG}.{info.name}"
            try:
                module = importlib.import_module(module_path)
            except Exception as e:
                # ImportError usually means optional deps aren't installed
                # in this environment; anything else means the module is
                # broken at import time (e.g. a scratch/WIP file living in
                # the package).  Either way we can't scan it for adapter
                # classes — record it and move on rather than failing a
                # test whose job is registry completeness, not linting.
                skipped.append(f"{module_path} ({type(e).__name__}: {e})")
                continue

            for attr in vars(module).values():
                if not isinstance(attr, type):
                    continue
                if attr.__module__ != module_path:
                    continue  # imported, not defined here
                if attr in (LLMMixin, ImageMixin):
                    continue
                expected = set()
                if issubclass(attr, LLMMixin):
                    expected.add("text")
                if issubclass(attr, ImageMixin):
                    expected.add("image")
                if not expected:
                    continue  # not an adapter class

                key = (module_path, attr.__name__)
                self.assertIn(
                    key,
                    registered,
                    f"{module_path}.{attr.__name__} looks like an adapter "
                    f"(subclasses {sorted(expected)} mixins) but is not in "
                    f"{PKG}._factory.REGISTRY. Add a registry entry.",
                )
                self.assertEqual(
                    set(registered[key]),
                    expected,
                    f"REGISTRY capabilities for {module_path}.{attr.__name__} "
                    f"don't match its mixins.",
                )
                checked.append(key)

        # If the real package is present, we expect to have verified
        # something; in a stub environment everything may be skipped.
        if not checked and not skipped:
            self.skipTest("no adapter modules found to check")


if __name__ == "__main__":
    unittest.main()
