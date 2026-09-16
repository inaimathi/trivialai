import tempfile
import unittest
import uuid

from src.trivialai.embedding.core import Embedder
from src.trivialai.vectorstore.chromadb import ChromaDB


@Embedder.register("dummy-chromadb-test")
class DummyChromaEmbedder(Embedder):
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
            "type": "dummy-chromadb-test",
            "scale": self.scale,
        }


class TestChromaDB(unittest.TestCase):
    def setUp(self):
        self.embedder = DummyChromaEmbedder()
        self.db = ChromaDB()

        self.collection_name = f"test-{uuid.uuid4()}"

        self.collection = self.db.createCollection(
            self.collection_name,
            self.embedder,
        )

    def assertVectorEqual(self, actual, expected):
        self.assertEqual(
            list(actual),
            list(expected),
        )

    def test_create_and_list_collection(self):
        self.assertIn(
            self.collection_name,
            self.db.listCollections(),
        )

    def test_get_collection_restores_embedder(self):
        collection = self.db.getCollection(
            self.collection_name,
        )

        self.assertEqual(
            collection.name,
            self.collection_name,
        )
        self.assertIsInstance(
            collection._embedder,
            DummyChromaEmbedder,
        )
        self.assertEqual(
            collection._embedder.scale,
            1.0,
        )

    def test_insert_and_lookup_by_id(self):
        vector = self.collection.insert(
            "hello",
            metadata={"source": "test"},
        )

        self.assertEqual(
            len(self.collection),
            1,
        )

        row = self.collection.lookup(
            id=self.collection.doc_id("hello"),
        )

        self.assertEqual(
            row["value"],
            "hello",
        )
        self.assertVectorEqual(
            row["vector"],
            vector,
        )
        self.assertEqual(
            row["meta"],
            {"source": "test"},
        )
        self.assertEqual(
            row["id"],
            self.collection.doc_id("hello"),
        )

    def test_insert_vector_uses_supplied_embedding(self):
        vector = [
            101.0,
            202.0,
            303.0,
        ]

        self.collection.insert_vector(
            "hello",
            vector,
            metadata={"source": "supplied"},
        )

        self.assertEqual(
            len(self.collection),
            1,
        )

        row = self.collection.lookup(
            id=self.collection.doc_id("hello"),
        )

        self.assertEqual(
            row["value"],
            "hello",
        )
        self.assertVectorEqual(
            row["vector"],
            vector,
        )
        self.assertEqual(
            row["meta"],
            {"source": "supplied"},
        )

        self.assertNotEqual(
            list(row["vector"]),
            self.collection.embedding("hello"),
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

        row = self.collection.lookup(
            vector=self.collection.embedding("alpha"),
        )

        self.assertEqual(
            row["value"],
            "alpha",
        )
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

        self.assertGreaterEqual(
            len(results),
            1,
        )
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

        self.assertGreaterEqual(
            len(results),
            1,
        )
        self.assertEqual(
            results[0]["value"],
            "bravo",
        )

    def test_query_with_no_content_or_vector_returns_empty(self):
        self.collection.insert("alpha")

        self.assertEqual(
            self.collection.queryBy(),
            [],
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

        self.assertEqual(
            len(self.collection),
            1,
        )

        row = self.collection.lookup(
            id=self.collection.doc_id("hello"),
        )

        self.assertEqual(
            row["value"],
            "hello",
        )
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

        self.assertEqual(
            len(self.collection),
            1,
        )

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

        self.assertNotEqual(
            first,
            second,
        )
        self.assertEqual(
            len(self.collection),
            1,
        )

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

    def test_delete_with_no_selector_returns_false(self):
        self.collection.insert("hello")

        deleted = self.collection.deleteBy()

        self.assertFalse(deleted)
        self.assertEqual(
            len(self.collection),
            1,
        )

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

        self.assertVectorEqual(
            by_value["alpha"]["vector"],
            self.collection.embedding(
                "alpha",
                metadata={"index": 1},
            ),
        )

    def test_doc_id_is_stable_and_content_based(self):
        first = self.collection.doc_id("hello")
        second = self.collection.doc_id("hello")
        different = self.collection.doc_id("goodbye")

        self.assertEqual(
            first,
            second,
        )
        self.assertNotEqual(
            first,
            different,
        )

    def test_persistent_collection_can_be_reopened(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collection_name = f"persistent-{uuid.uuid4()}"

            db1 = ChromaDB(
                config=tmpdir,
            )
            col1 = db1.createCollection(
                collection_name,
                DummyChromaEmbedder(scale=2.0),
            )

            col1.insert(
                "hello",
                metadata={"source": "persistent"},
            )

            db2 = ChromaDB(
                config=tmpdir,
            )
            col2 = db2.getCollection(
                collection_name,
            )

            self.assertEqual(
                len(col2),
                1,
            )
            self.assertIsInstance(
                col2._embedder,
                DummyChromaEmbedder,
            )
            self.assertEqual(
                col2._embedder.scale,
                2.0,
            )

            row = col2.lookup(
                id=col2.doc_id("hello"),
            )

            self.assertEqual(
                row["value"],
                "hello",
            )
            self.assertEqual(
                row["meta"],
                {"source": "persistent"},
            )


if __name__ == "__main__":
    unittest.main()
