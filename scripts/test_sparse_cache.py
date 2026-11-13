"""Offline BM25 snapshot contracts. Fake versions test control flow, not DB consistency/performance."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

from sqlalchemy import create_engine, text
from app.core.config import Settings
from app.search import sparse, stages
from app.search.sparse_cache import CACHE, CacheLimits, SnapshotCache, SparseSnapshot
from app.search.sparse_version import CorpusStamp, Uncacheable
from app.corpus.scope import Scope


class CacheConcurrency(unittest.TestCase):
    def setUp(self):
        self.cache = SnapshotCache()
        self.limits = CacheLimits(2, 1000, 2)
        self.stamp = CorpusStamp(("test",), "epoch", 1)
        self.item = SparseSnapshot((), None, 100)

    def get(self, base="owner", build=None, validate=None):
        return self.cache.get(
            base, self.stamp, build or (lambda: self.item), validate or (lambda: self.stamp), self.limits
        )

    def test_concurrent_cold_requests_build_once(self):
        entered, release = threading.Event(), threading.Event()
        builds = []

        def build():
            builds.append(1)
            entered.set()
            self.assertTrue(release.wait(2))
            return self.item

        with ThreadPoolExecutor(4) as pool:
            first = pool.submit(self.get, build=build)
            self.assertTrue(entered.wait(2))
            others = [pool.submit(self.get, build=build) for _ in range(3)]
            release.set()
            self.assertTrue(all(x.result(3) is self.item for x in [first, *others]))
        self.assertEqual(len(builds), 1)
        self.assertEqual(self.cache.stats()["publish"], 1)

    def test_mutation_during_build_is_not_published(self):
        with self.assertRaisesRegex(Uncacheable, "changed_during_build"):
            self.get(validate=lambda: CorpusStamp(("test",), "epoch", 2))
        self.assertEqual(self.cache.stats()["entries"], 0)
        self.assertEqual(self.cache.stats()["builders"], 0)

    def test_exception_and_cancellation_release_builder(self):
        for error in (ValueError("broken"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):

                def fail():
                    raise error

                with self.assertRaises(type(error)):
                    self.get(build=fail)
                self.assertEqual(self.cache.stats()["builders"], 0)
                self.assertEqual(self.cache.stats()["entries"], 0)
        self.assertIs(self.get(), self.item)

    def test_lru_budget_invalidation_and_rebuild(self):
        self.get("a")
        self.get("b")
        self.get("a")
        self.get("c")
        self.assertEqual(self.cache.stats()["eviction"], 1)
        self.get("b")
        self.assertEqual(self.cache.stats()["cold_build"], 4)
        self.stamp = CorpusStamp(("test",), "epoch", 2)
        self.get("b")
        self.assertEqual(self.cache.stats()["invalidation"], 1)
        self.assertLessEqual(self.cache.stats()["estimated_bytes"], 1000)

    def test_oversized_snapshot_is_not_published(self):
        with self.assertRaisesRegex(Uncacheable, "snapshot_over_budget"):
            self.get(build=lambda: SparseSnapshot((), None, 1001))
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_wait_timeout_and_other_key_fall_back(self):
        entered, release = threading.Event(), threading.Event()
        self.limits = CacheLimits(2, 1000, 0.01)

        def build():
            entered.set()
            release.wait(2)
            return self.item

        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(self.get, build=build)
            self.assertTrue(entered.wait(2))
            try:
                with self.assertRaisesRegex(Uncacheable, "wait_timeout"):
                    self.get()
                with self.assertRaisesRegex(Uncacheable, "other_key_building"):
                    self.get("other")
            finally:
                release.set()
            future.result(3)


class SearchEquivalence(unittest.TestCase):
    def setUp(self):
        CACHE.clear()
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        self.db = self.engine.connect()
        self.db.execute(
            text(
                "CREATE TABLE chunks (id TEXT, doc_id TEXT, page INT, start_offset INT, end_offset INT, text TEXT, created_at INT)"
            )
        )
        for i in range(30):
            self.db.execute(
                text("INSERT INTO chunks VALUES (:id,'d',1,0,500,:text,:created)"),
                {
                    "id": str(i),
                    "text": (_TEST_MATERIALS["text_015"] if i < 8 else _TEST_MATERIALS["text_016"]) * 40,
                    "created": i,
                },
            )
        self.settings = Settings(_env_file=None, sparse_index_cache_enabled=True)
        self.stamp = CorpusStamp((self.engine,), "test", 1)
        self.version = patch("app.search.sparse_version.read_stamp", side_effect=lambda db: self.stamp)
        self.version.start()
        self.addCleanup(self.version.stop)
        self.addCleanup(self.engine.dispose)
        self.addCleanup(self.db.close)
        stages.reset_stages(capture_candidates=True)

    def query(self, query=_TEST_MATERIALS["text_001"], cache=True):
        self.settings.sparse_index_cache_enabled = cache
        return sparse.sparse_search(self.db, query, top_k=10, settings=self.settings)

    def test_full_protocol_equivalence_cold_hot_different_query_ties(self):
        for query in (
            _TEST_MATERIALS["text_002"],
            _TEST_MATERIALS["text_003"],
            _TEST_MATERIALS["text_004"],
            _TEST_MATERIALS["text_005"],
        ):
            expected = self.query(query, cache=False)
            for _ in range(2):
                self.assertEqual(self.query(query), expected)
        self.assertEqual(CACHE.stats()["publish"], 1)
        self.assertEqual(CACHE.stats()["hit"], 7)
        self.assertEqual([x.chunk_id for x in self.query()], [str(i) for i in range(8)])

    def test_hot_has_no_corpus_read_tokenize_or_build(self):
        self.query()
        stages.reset_stages()
        with (
            patch.object(sparse, "BM25Okapi", side_effect=AssertionError("hot rebuild")),
            patch.object(self.db, "execute", side_effect=AssertionError(_TEST_MATERIALS["text_014"])),
            patch.object(sparse, "tokenize", wraps=sparse.tokenize) as tokenize,
        ):
            self.query(_TEST_MATERIALS["text_008"])
        self.assertEqual(tokenize.call_count, 1)
        self.assertEqual(stages.snapshot_meta()["sparse_index_reused"], 1)
        for key in ("sparse_corpus_load_ms", "sparse_tokenize_ms", "sparse_index_build_ms"):
            self.assertNotIn(key, stages.snapshot_stages())

    def test_mutations_with_new_versions_and_empty_corpus(self):
        self.query()
        operations = [
            _TEST_MATERIALS["text_006"],
            "DELETE FROM chunks WHERE id='1'",
            _TEST_MATERIALS["text_007"],
            "DELETE FROM chunks",
        ]
        for number, sql in enumerate(operations, 2):
            self.db.execute(text(sql))
            self.stamp = CorpusStamp((self.engine,), "test", number)
            self.assertEqual(self.query(), self.query(cache=False))

    def test_empty_query_and_invalid_tokens_do_not_probe(self):
        with patch("app.search.sparse_version.read_stamp", side_effect=AssertionError("unneeded stamp")):
            self.assertEqual(self.query("  "), [])
            self.assertEqual(self.query("!!!"), [])

    def test_unavailable_version_and_source_budget_fallback(self):
        expected = self.query(cache=False)
        with patch("app.search.sparse_version.read_stamp", side_effect=Uncacheable("version_unavailable")):
            self.assertEqual(self.query(), expected)
        self.assertEqual(CACHE.stats()["entries"], 0)
        self.settings.sparse_index_cache_max_chunks = 1
        self.assertEqual(self.query(), expected)
        self.assertEqual(stages.snapshot_meta()["sparse_index_cache_reason"], "source_over_budget")

    def test_same_created_at_is_cached_with_id_order(self):
        self.db.execute(text("UPDATE chunks SET created_at=0"))
        expected = self.query(cache=False)
        self.assertEqual([x.chunk_id for x in expected], [str(i) for i in range(8)])
        self.assertEqual(self.query(), expected)
        self.assertEqual(stages.snapshot_meta()["sparse_index_cache_status"], "cold_build")
        stages.reset_stages()
        self.assertEqual(self.query(), expected)
        self.assertEqual(stages.snapshot_meta()["sparse_index_cache_status"], "hit")
        self.assertEqual(CACHE.stats()["entries"], 1)

    def test_null_created_at_is_not_cached(self):
        self.db.execute(text("UPDATE chunks SET created_at=NULL WHERE id='0'"))
        expected = self.query(cache=False)
        self.assertEqual(self.query(), expected)
        self.assertEqual(stages.snapshot_meta()["sparse_index_cache_reason"], "ambiguous_created_at")
        self.assertEqual(CACHE.stats()["entries"], 0)

    def test_historical_created_at_only_order_can_differ(self):
        self.db.execute(text("DELETE FROM chunks"))
        inserted = []
        for index in range(9, -1, -1):
            body = (_TEST_MATERIALS["text_012"] if index < 3 else _TEST_MATERIALS["text_013"]) * 20
            self.db.execute(
                text("INSERT INTO chunks VALUES (:id,'d',1,0,500,:text,0)"), {"id": str(index), "text": body}
            )
            inserted.append({"id": str(index), "text": body})
        new_ids = [str(hit.chunk_id) for hit in self.query(cache=False)]
        tokens = [sparse.tokenize(row["text"]) for row in inserted]
        scores = sparse.BM25Okapi(tokens).get_scores(sparse.tokenize(_TEST_MATERIALS["text_009"]))
        old_ids = [
            row["id"]
            for row, score in sorted(zip(inserted, scores), key=lambda item: item[1], reverse=True)[:10]
            if score > 0
        ]
        self.assertEqual(old_ids, ["2", "1", "0"])
        self.assertEqual(new_ids, ["0", "1", "2"])
        self.assertNotEqual(old_ids, new_ids)

    def test_diagnostics_toggle_and_broken_sink_do_not_change_hits(self):
        expected = self.query(cache=False)
        for enabled in (False, True):
            stages.reset_stages(capture_candidates=enabled)
            self.assertEqual(self.query(), expected)
        token = stages._meta.set(object())
        try:
            self.assertEqual(self.query(), expected)
        finally:
            stages._meta.reset(token)

    def test_scope_key_matches_effective_sql(self):
        self.assertEqual(sparse._effective_scope(None), sparse._effective_scope(Scope(mode="hard")))
        self.assertEqual(
            sparse._effective_scope(Scope(mode="soft", collection_ids=("b", "a", "a"))),
            sparse._effective_scope(Scope(mode="soft", collection_ids=("a", "b"), widened=True)),
        )
        self.assertNotEqual(
            sparse._effective_scope(Scope(mode="hard", collection_ids=("a",))),
            sparse._effective_scope(Scope(mode="soft", collection_ids=("a",))),
        )

    def test_hybrid_and_rerank_inputs_unchanged(self):
        # Existing offline harness supplies inert model imports, never real inference.
        import test_retrieval_diagnostics as harness

        fixture = harness.FakeDB()
        self.settings.hybrid_dense_candidates = 3
        self.settings.rerank_candidates = 2

        def cold(db, query, **kw):
            return sparse.sparse_search(self.db, query, **kw)

        with (
            patch.object(harness.dense, "encode_query", return_value=[0.1] * 384),
            patch.object(harness.hybrid, "sparse_search", side_effect=cold),
            patch.object(harness.rerank, "rerank_scores", return_value=[0.1, 0.9]) as score,
        ):
            self.settings.sparse_index_cache_enabled = False
            old = harness.rerank.rerank_search(fixture, _TEST_MATERIALS["text_010"], top_k=2, settings=self.settings)
            before = score.call_args
            self.settings.sparse_index_cache_enabled = True
            new = harness.rerank.rerank_search(fixture, _TEST_MATERIALS["text_011"], top_k=2, settings=self.settings)
            self.assertEqual(new, old)
            self.assertEqual(score.call_args, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
