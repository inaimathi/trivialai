import unittest

from src.trivialai.embedding.core import Embedder, OllamaEmbedder


class TestEmbeddingRegistry(unittest.TestCase):
    def test_register_decorator_and_from_config_with_dummy(self):
        # Define a dummy embedder to exercise the registry without network calls
        @Embedder.register("dummy-test")
        class DummyEmbedder(Embedder):
            def __init__(self, foo: str = "bar"):
                self.foo = foo

            def __call__(self, thing, metadata=None):
                # Return a deterministic "vector" without external deps
                return [1.0, 2.0, 3.0]

            def to_config(self):
                return {"type": "dummy-test", "foo": self.foo}

        # Build via from_config
        inst = Embedder.from_config({"type": "dummy-test", "foo": "baz"})
        self.assertIsInstance(inst, DummyEmbedder)
        self.assertEqual(inst.foo, "baz")
        self.assertEqual(inst.to_config(), {"type": "dummy-test", "foo": "baz"})
        self.assertEqual(inst("anything"), [1.0, 2.0, 3.0])

    def test_from_config_missing_type(self):
        with self.assertRaises(ValueError) as ctx:
            Embedder.from_config({})
        self.assertIn("missing 'type'", str(ctx.exception))

    def test_from_config_unknown_type(self):
        with self.assertRaises(ValueError) as ctx:
            Embedder.from_config({"type": "nope"})
        self.assertIn("Unknown embedder type", str(ctx.exception))


class TestOllamaEmbedderConfig(unittest.TestCase):
    def test_to_config_roundtrip_values(self):
        cfg = {
            "server": "http://example:11434",
            "model": "nomic-embed-text",
            "retries": 5,
            "timeout": 10.0,
        }
        emb = OllamaEmbedder(**cfg)
        out = emb.to_config()
        # Should include type and reflect our values
        self.assertEqual(out["type"], "ollama")
        self.assertEqual(out["server"], cfg["server"])
        self.assertEqual(out["model"], cfg["model"])
        self.assertEqual(out["retries"], cfg["retries"])
        self.assertEqual(out["timeout"], cfg["timeout"])

        # from_config should reconstruct an equivalent instance
        emb2 = Embedder.from_config(out)
        self.assertIsInstance(emb2, OllamaEmbedder)
        self.assertEqual(emb2.server, cfg["server"])
        self.assertEqual(emb2.model, cfg["model"])
        self.assertEqual(emb2.retries, cfg["retries"])
        self.assertEqual(emb2.timeout, cfg["timeout"])

    def test_defaults_are_sane(self):
        emb = OllamaEmbedder()
        self.assertEqual(emb.server, "http://localhost:11434")
        self.assertEqual(emb.model, "nomic-embed-text")
        self.assertEqual(emb.retries, 3)
        self.assertIsInstance(emb.timeout, (float, type(None)))


if __name__ == "__main__":
    unittest.main()
