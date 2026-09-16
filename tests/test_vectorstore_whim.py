import unittest

from src.trivialai.embedding.core import Embedder
from src.trivialai.vectorstore.chromadb import ChromaCollection
from src.trivialai.vectorstore.core import Whim


@Embedder.register("dummy-whim-test")
class DummyWhimEmbedder(Embedder):
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, thing, metadata=None):
        text = str(thing)
        bias = float((metadata or {}).get("bias", 0.0))
        return [
            len(text) * self.scale,
            (sum(ord(c) for c in text) % 997) * self.scale + bias,
            1.0,
        ]

    def to_config(self):
        return {
            "type": "dummy-whim-test",
            "scale": self.scale,
        }


class TestWhim(unittest.TestCase):
    def setUp(self):
        self.embedder = DummyWhimEmbedder()
        self.collection = Whim(self.embedder)

    def assertVectorEqual(self, actual, expected):
        self.assertEqual(list(actual), list(expected))

    def test_uses_vectorstore_adapter(self):
        self.assertIsInstance(
            self.collection._collection,
            ChromaCollection,
        )

    def test_extra_metadata_is_retained(self):
        collection = Whim(
            self.embedder,
            extra_metadata={
                "purpose": "test",
                "source": "temporary",
            },
        )

        self.assertEqual(
            collection.extra_metadata,
            {
                "purpose": "test",
                "source": "temporary",
            },
        )

    def test_insert_and_lookup_by_id(self):
        vector = self.collection.insert(
            "hello",
            metadata={"source": "test"},
        )

        self.assertEqual(len(self.collection), 1)

        row = self.collection.lookup(
            id=self.collection.doc_id("hello"),
        )

        self.assertEqual(row["value"], "hello")
        self.assertVectorEqual(row["vector"], vector)
        self.assertEqual(row["meta"], {"source": "test"})
        self.assertEqual(
            row["id"],
            self.collection.doc_id("hello"),
        )

    def test_insert_vector(self):
        vector = [101.0, 202.0, 303.0]

        self.collection.insert_vector(
            "hello",
            vector,
            metadata={"source": "supplied"},
        )

        row = self.collection.lookup(
            id=self.collection.doc_id("hello"),
        )

        self.assertEqual(row["value"], "hello")
        self.assertVectorEqual(row["vector"], vector)
        self.assertEqual(
            row["meta"],
            {"source": "supplied"},
        )

    def test_lookup_missing_id_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.collection.lookup(
                id="does-not-exist",
            )

    def test_lookup_requires_exactly_one_selector(self):
        with self.assertRaises(ValueError):
            self.collection.lookup()

        with self.assertRaises(ValueError):
            self.collection.lookup(
                id="abc",
                vector=[1.0, 2.0, 3.0],
            )

    def test_lookup_by_vector(self):
        self.collection.insert("alpha")
        self.collection.insert("bravo")

        vector = self.collection.embedding("alpha")
        row = self.collection.lookup(vector=vector)

        self.assertEqual(row["value"], "alpha")
        self.assertEqual(
            row["id"],
            self.collection.doc_id("alpha"),
        )

    def test_query_by_content(self):
        self.collection.insert(
            "alpha",
            {"kind": "letter"},
        )
        self.collection.insert(
            "bravo",
            {"kind": "word"},
        )
        self.collection.insert(
            "charlie",
            {"kind": "word"},
        )

        results = self.collection.queryBy(
            content="alpha",
            maxResults=3,
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(
            results[0]["value"],
            "alpha",
        )
        self.assertEqual(
            results[0]["meta"],
            {"kind": "letter"},
        )

    def test_query_by_vector(self):
        self.collection.insert("alpha")
        self.collection.insert("bravo")

        results = self.collection.queryBy(
            vector=self.collection.embedding("bravo"),
            maxResults=2,
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(
            results[0]["value"],
            "bravo",
        )

    def test_query_filter(self):
        self.collection.insert(
            "alpha",
            {"include": True},
        )
        self.collection.insert(
            "bravo",
            {"include": False},
        )
        self.collection.insert(
            "charlie",
            {"include": True},
        )

        results = self.collection.queryBy(
            content="alpha",
            maxResults=3,
            filter=lambda meta: meta.get("include") is True,
        )

        self.assertTrue(results)
        self.assertTrue(
            all(result["meta"].get("include") is True for result in results)
        )

    def test_upsert_inserts_new_document(self):
        vector = self.collection.upsert(
            "hello",
            metadata={"version": 1},
        )

        self.assertEqual(len(self.collection), 1)

        row = self.collection.lookup(
            id=self.collection.doc_id("hello"),
        )

        self.assertEqual(row["value"], "hello")
        self.assertEqual(
            row["meta"],
            {"version": 1},
        )
        self.assertVectorEqual(
            row["vector"],
            vector,
        )

    def test_upsert_replaces_existing_metadata(self):
        self.collection.insert(
            "hello",
            metadata={"refs": '["a"]'},
        )

        self.collection.upsert(
            "hello",
            metadata={"refs": '["a", "b"]'},
        )

        self.assertEqual(len(self.collection), 1)

        row = self.collection.lookup(
            id=self.collection.doc_id("hello"),
        )

        self.assertEqual(
            row["meta"],
            {"refs": '["a", "b"]'},
        )

    def test_upsert_recomputes_embedding(self):
        first = self.collection.upsert(
            "hello",
            metadata={"bias": 1.0},
        )

        second = self.collection.upsert(
            "hello",
            metadata={"bias": 100.0},
        )

        self.assertNotEqual(first, second)
        self.assertEqual(len(self.collection), 1)

        row = self.collection.lookup(
            id=self.collection.doc_id("hello"),
        )

        self.assertVectorEqual(
            row["vector"],
            second,
        )
        self.assertEqual(
            row["meta"],
            {"bias": 100.0},
        )

    def test_delete_by_content(self):
        self.collection.insert("hello")

        deleted = self.collection.deleteBy(
            content="hello",
        )

        self.assertTrue(deleted)
        self.assertEqual(
            len(self.collection),
            0,
        )

    def test_delete_by_id(self):
        self.collection.insert("hello")
        doc_id = self.collection.doc_id("hello")

        deleted = self.collection.deleteBy(
            id=doc_id,
        )

        self.assertTrue(deleted)
        self.assertEqual(
            len(self.collection),
            0,
        )

    def test_delete_by_vector(self):
        self.collection.insert("hello")

        deleted = self.collection.deleteBy(
            vector=self.collection.embedding("hello"),
        )

        self.assertTrue(deleted)
        self.assertEqual(
            len(self.collection),
            0,
        )

    def test_delete_by_vector_on_empty_collection_returns_false(self):
        deleted = self.collection.deleteBy(
            vector=[1.0, 2.0, 3.0],
        )

        self.assertFalse(deleted)

    def test_iter_rows(self):
        self.collection.insert(
            "alpha",
            {"index": 1},
        )
        self.collection.insert(
            "bravo",
            {"index": 2},
        )
        self.collection.insert(
            "charlie",
            {"index": 3},
        )

        rows = list(
            self.collection.iter_rows(
                batch_size=2,
            )
        )

        self.assertEqual(
            len(rows),
            3,
        )

        by_value = {row["value"]: row for row in rows}

        self.assertEqual(
            by_value["alpha"]["meta"],
            {"index": 1},
        )
        self.assertEqual(
            by_value["bravo"]["meta"],
            {"index": 2},
        )
        self.assertEqual(
            by_value["charlie"]["meta"],
            {"index": 3},
        )

        self.assertEqual(
            by_value["alpha"]["id"],
            self.collection.doc_id("alpha"),
        )

    def test_insert_many(self):
        vectors = self.collection.insert_many(
            ["alpha", "bravo"],
            metadatas=[
                {"index": 1},
                {"index": 2},
            ],
        )

        self.assertEqual(
            len(vectors),
            2,
        )
        self.assertEqual(
            len(self.collection),
            2,
        )

        alpha = self.collection.lookup(
            id=self.collection.doc_id("alpha"),
        )
        bravo = self.collection.lookup(
            id=self.collection.doc_id("bravo"),
        )

        self.assertEqual(
            alpha["meta"],
            {"index": 1},
        )
        self.assertEqual(
            bravo["meta"],
            {"index": 2},
        )

    def test_insert_many_requires_matching_metadata_length(self):
        with self.assertRaises(ValueError):
            self.collection.insert_many(
                ["alpha", "bravo"],
                metadatas=[
                    {"index": 1},
                ],
            )

    def test_reset(self):
        self.collection.insert("alpha")
        self.collection.insert("bravo")

        self.assertEqual(
            len(self.collection),
            2,
        )

        self.collection.reset()

        self.assertEqual(
            len(self.collection),
            0,
        )

    def test_consume(self):
        source = Whim(self.embedder)
        source.insert(
            "alpha",
            {"source": "original"},
        )
        source.insert(
            "bravo",
            {"source": "original"},
        )

        target = Whim(self.embedder)
        target.consume(
            source,
            extra_meta={"copied": True},
        )

        self.assertEqual(
            len(target),
            2,
        )

        alpha = target.lookup(
            id=target.doc_id("alpha"),
        )

        self.assertEqual(
            alpha["meta"],
            {
                "source": "original",
                "copied": True,
            },
        )

    def test_consume_reuses_existing_embedding(self):
        source = Whim(
            DummyWhimEmbedder(scale=1.0),
        )
        source.insert("alpha")

        source_row = source.lookup(
            id=source.doc_id("alpha"),
        )

        target = Whim(
            DummyWhimEmbedder(scale=2.0),
        )
        target.consume(
            source,
            strict_embedder=False,
        )

        target_row = target.lookup(
            id=target.doc_id("alpha"),
        )

        self.assertVectorEqual(
            target_row["vector"],
            source_row["vector"],
        )

        self.assertNotEqual(
            list(target_row["vector"]),
            target.embedding("alpha"),
        )

    def test_consume_meta_filter(self):
        source = Whim(self.embedder)
        source.insert(
            "alpha",
            {"keep": True},
        )
        source.insert(
            "bravo",
            {"keep": False},
        )

        target = Whim(self.embedder)
        target.consume(
            source,
            meta_filter=lambda meta: meta.get("keep") is True,
        )

        self.assertEqual(
            len(target),
            1,
        )

        row = target.lookup(
            id=target.doc_id("alpha"),
        )

        self.assertEqual(
            row["value"],
            "alpha",
        )

        with self.assertRaises(KeyError):
            target.lookup(
                id=target.doc_id("bravo"),
            )

    def test_consume_rejects_different_embedder_config(self):
        source = Whim(
            DummyWhimEmbedder(scale=1.0),
        )
        source.insert("alpha")

        target = Whim(
            DummyWhimEmbedder(scale=2.0),
        )

        with self.assertRaises(ValueError):
            target.consume(source)

    def test_consume_can_ignore_embedder_mismatch(self):
        source = Whim(
            DummyWhimEmbedder(scale=1.0),
        )
        source.insert("alpha")

        target = Whim(
            DummyWhimEmbedder(scale=2.0),
        )

        target.consume(
            source,
            strict_embedder=False,
        )

        self.assertEqual(
            len(target),
            1,
        )


if __name__ == "__main__":
    unittest.main()
