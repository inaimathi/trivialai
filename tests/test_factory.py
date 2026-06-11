# tests/test_factory.py
"""
Tests for trivialai._factory and the package-root export behavior.

Provider construction is tested against a dummy provider injected into the
registry, so no real provider dependencies (httpx, boto3, ...) are needed.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any, Dict, Optional

import pytest
from src import trivialai
from src.trivialai._factory import (REGISTRY, ConfigError, from_env, providers,
                                    text)

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


@pytest.fixture
def dummy_provider(monkeypatch):
    mod = types.ModuleType("dummy_provider_mod")
    mod.DummyText = DummyText
    monkeypatch.setitem(sys.modules, "dummy_provider_mod", mod)
    monkeypatch.setitem(
        REGISTRY, "dummy", ("dummy_provider_mod", "DummyText", frozenset({"text"}))
    )
    return "dummy"


# ---------------------------------------------------------------------------
# from_env: collection, prefixes, collisions, empties, defaults, CONFIG_FILE
# ---------------------------------------------------------------------------


def test_from_env_collects_strips_and_lowercases():
    env = {
        "comp_PROVIDER": "ollama",
        "comp_MODEL": "qwq:latest",
        "comp_OLLAMA_SERVER": "http://h:11434",
        "UNRELATED": "x",
    }
    assert from_env("comp_", env=env) == {
        "provider": "ollama",
        "model": "qwq:latest",
        "ollama_server": "http://h:11434",
    }


def test_from_env_bare_prefix_accepts_both_forms():
    assert from_env("comp", env={"compMODEL": "m"}) == {"model": "m"}
    assert from_env("comp", env={"comp_MODEL": "m"}) == {"model": "m"}


def test_from_env_trailing_underscore_is_exact():
    with pytest.raises(ConfigError, match="No environment variables"):
        from_env("comp_", env={"compMODEL": "m"})


def test_from_env_collision_is_loud_and_names_the_variables():
    env = {"comp_MODEL": "a", "compMODEL": "b"}
    with pytest.raises(ConfigError) as e:
        from_env("comp", env=env)
    msg = str(e.value)
    assert "comp_MODEL" in msg and "compMODEL" in msg and "exactly one source" in msg


def test_from_env_empty_values_are_unset():
    env = {"comp_PROVIDER": "ollama", "comp_MODEL": ""}
    assert from_env("comp_", env=env) == {"provider": "ollama"}


def test_from_env_empty_values_do_not_create_collisions():
    env = {"comp_MODEL": "", "compMODEL": "real"}
    assert from_env("comp", env=env) == {"model": "real"}


def test_from_env_nothing_found_raises_without_default():
    with pytest.raises(ConfigError, match="prefix 'comp_'"):
        from_env("comp_", env={})


def test_from_env_default_is_bundle_level():
    default = {"provider": "ollama", "model": "qwq:latest"}
    # Nothing found -> default verbatim (copied).
    out = from_env("comp_", default=default, env={})
    assert out == default and out is not default
    # Anything found -> default ignored entirely, never merged.
    out = from_env("comp_", default=default, env={"comp_PROVIDER": "bedrock"})
    assert out == {"provider": "bedrock"}


def test_from_env_rejects_empty_prefix():
    with pytest.raises(ConfigError, match="non-empty prefix"):
        from_env("", env={"PATH": "/usr/bin"})


def test_from_env_config_file_is_base_env_overrides(tmp_path):
    cfg = tmp_path / "model.json"
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
    assert out == {
        "provider": "dummy",
        "model": "from-env",  # env var wins over the file
        "extras": {"nested": [1, 2]},  # structured value arrives typed
    }
    assert "config_file" not in out


def test_from_env_config_file_errors_are_specific(tmp_path):
    with pytest.raises(ConfigError, match="Could not read"):
        from_env("comp_", env={"comp_CONFIG_FILE": str(tmp_path / "missing.json")})
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(ConfigError, match="not valid JSON"):
        from_env("comp_", env={"comp_CONFIG_FILE": str(bad)})
    arr = tmp_path / "arr.json"
    arr.write_text("[1,2]")
    with pytest.raises(ConfigError, match="JSON object"):
        from_env("comp_", env={"comp_CONFIG_FILE": str(arr)})


# ---------------------------------------------------------------------------
# text/image: provider resolution, capability, validation, coercion
# ---------------------------------------------------------------------------


def test_missing_provider_key():
    with pytest.raises(ConfigError, match="no 'provider' key"):
        text({"model": "m"})


def test_unknown_provider_suggests():
    with pytest.raises(ConfigError, match="Did you mean 'ollama'"):
        text({"provider": "olama", "model": "m"})


def test_provider_is_case_insensitive(dummy_provider):
    m = text({"provider": "DuMmY", "model": "m"})
    assert isinstance(m, DummyText)


def test_capability_mismatch_is_loud():
    # stabdiff is image-only; the error must say so without importing it.
    with pytest.raises(ConfigError, match="no text capability"):
        text({"provider": "stabdiff"})
    with pytest.raises(ConfigError, match="no image capability"):
        trivialai.image({"provider": "ollama"})


def test_unknown_parameter_suggests_and_lists(dummy_provider):
    with pytest.raises(ConfigError) as e:
        text({"provider": "dummy", "model": "m", "servr": "x"})
    msg = str(e.value)
    assert "no parameter 'servr'" in msg
    assert "Did you mean 'server'" in msg
    assert "model" in msg  # accepted-parameter listing


def test_missing_required_parameter_names_env_var(dummy_provider):
    with pytest.raises(ConfigError, match=r"\{prefix\}MODEL"):
        text({"provider": "dummy"})


def test_coercion_int_float_bool_str(dummy_provider):
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
    assert m.max_tokens == 8192
    assert m.temperature == 0.2
    assert m.skip_healthcheck is False
    assert m.server == "http://h"


def test_bool_spelling_is_strict(dummy_provider):
    for bad in ("yes", "1", "on", ""):
        with pytest.raises(ConfigError, match="'true' or 'false'"):
            text({"provider": "dummy", "model": "m", "skip_healthcheck": bad})
    assert (
        text(
            {"provider": "dummy", "model": "m", "skip_healthcheck": "TRUE"}
        ).skip_healthcheck
        is True
    )


def test_numeric_coercion_failures_are_specific(dummy_provider):
    with pytest.raises(ConfigError, match="max_tokens expects an integer"):
        text({"provider": "dummy", "model": "m", "max_tokens": "lots"})
    with pytest.raises(ConfigError, match="temperature expects a number"):
        text({"provider": "dummy", "model": "m", "temperature": "warm"})


def test_structured_parameter_as_json_string(dummy_provider):
    m = text(
        {
            "provider": "dummy",
            "model": "m",
            "extras": '{"a": 1, "b": [2, 3]}',
        }
    )
    assert m.extras == {"a": 1, "b": [2, 3]}
    with pytest.raises(ConfigError, match="JSON"):
        text({"provider": "dummy", "model": "m", "extras": "{nope"})
    with pytest.raises(ConfigError, match="dict"):
        text({"provider": "dummy", "model": "m", "extras": "[1, 2]"})


def test_non_string_values_pass_through_untouched(dummy_provider):
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
    assert m.max_tokens == 1024
    assert m.skip_healthcheck is True
    assert m.extras == {"k": "v"}


def test_end_to_end_from_env_to_text(dummy_provider):
    env = {
        "comp_PROVIDER": "dummy",
        "comp_MODEL": "qwen-coder:latest",
        "comp_SERVER": "http://some.host:12345",
        "comp_SKIP_HEALTHCHECK": "false",
        "comp_MAX_TOKENS": "8192",
    }
    m = text(from_env("comp_", env=env))
    assert (m.model, m.server, m.skip_healthcheck, m.max_tokens) == (
        "qwen-coder:latest",
        "http://some.host:12345",
        False,
        8192,
    )


# ---------------------------------------------------------------------------
# Discoverability and package-root behavior
# ---------------------------------------------------------------------------


def test_providers_lists_without_importing():
    entries = {e["provider"]: e for e in providers()}
    assert entries["bedrock"]["capabilities"] == ["image", "text"]
    assert entries["ollama"]["capabilities"] == ["text"]
    assert entries["stabdiff"]["capabilities"] == ["image"]
    # Nothing heavy was imported just by listing.
    assert "trivialai.bedrock" not in sys.modules


def test_root_image_attribute_survives_submodule_import():
    assert callable(trivialai.image)
    from trivialai.image import Picture  # the form provider modules use

    assert callable(trivialai.image)  # still the factory, not the module
    assert sys.modules["trivialai.image"].Picture is Picture


def test_root_lazy_attribute_error_is_clean():
    with pytest.raises(AttributeError, match="Nonexistent"):
        trivialai.Nonexistent


def test_registry_drives_root_exports(monkeypatch):
    # Registering a provider makes it reachable as a lazy class attribute
    # with no __init__.py edits.
    mod = types.ModuleType("late_mod")

    class LateThing:
        def __init__(self, model: str):
            self.model = model

    mod.LateThing = LateThing
    monkeypatch.setitem(sys.modules, "late_mod", mod)
    trivialai.register("latething", "late_mod", "LateThing", {"text"})
    try:
        assert trivialai.LateThing is LateThing
        assert "LateThing" in dir(trivialai)
        assert isinstance(text({"provider": "latething", "model": "m"}), LateThing)
    finally:
        REGISTRY.pop("latething", None)
        # drop the cached lazy attribute so other tests see a clean module
        vars(trivialai).pop("LateThing", None)


def test_register_validates_capabilities():
    with pytest.raises(ConfigError, match="unknown capabilities"):
        trivialai.register("x", "m", "C", {"text", "audio"})
    with pytest.raises(ConfigError, match="at least one capability"):
        trivialai.register("x", "m", "C", set())
    assert "x" not in REGISTRY
