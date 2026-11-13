"""Real local PostgreSQL generation/concurrency tests in a disposable private schema."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import text
from sqlalchemy.orm import Session
from app.core.config import Settings
from app.corpus.scope import Scope
from app.search import sparse, stages
from app.search.sparse_cache import CACHE
from app.search.sparse_version import read_stamp, Uncacheable
from lib.sparse_lab import SparseLab


class PostgresCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab = SparseLab()
        cls.lab.__enter__()

    @classmethod
    def tearDownClass(cls):
        CACHE.clear()
        cls.lab.__exit__(None, None, None)

    def setUp(self):
        self.lab.seed()
        CACHE.clear()
        self.settings = Settings(_env_file=None, sparse_index_cache_enabled=True)

    def stamp(self):
        with self.lab.engine.connect() as db:
            return read_stamp(db)

    def search(self, cache=True, owner=None, scope=None, query=_TEST_MATERIALS["text_001"]):
        settings = self.settings.model_copy(update={"sparse_index_cache_enabled": cache})
        stages.reset_stages(capture_candidates=True)
        with self.lab.engine.connect() as db:
            hits = sparse.sparse_search(db, query, owner_id=owner, scope=scope, settings=settings)
        return hits, stages.snapshot_meta(), stages.snapshot_stages()

    def equal(self, owner=None, scope=None):
        old = self.search(False, owner, scope)[0]
        new, meta, _ = self.search(True, owner, scope)
        self.assertEqual(new, old)
        return meta

    def test_committed_cold_hot_different_query_and_no_fulltext_hot(self):
        self.assertEqual(self.equal()["sparse_index_cache_status"], "cold_build")
        with patch.object(sparse, "_build_snapshot", side_effect=AssertionError("hot rebuild")):
            self.assertEqual(self.equal()["sparse_index_cache_status"], "hit")
        old = self.search(False, query=_TEST_MATERIALS["text_009"])[0]
        hits, meta, times = self.search(True, query=_TEST_MATERIALS["text_004"])
        self.assertEqual(hits, old)
        self.assertEqual(meta["sparse_index_reused"], 1)
        for name in ("sparse_corpus_load_ms", "sparse_tokenize_ms", "sparse_index_build_ms"):
            self.assertNotIn(name, times)

    def test_owner_hard_soft_all_and_normalized_scope(self):
        for owner in (None, UUID(int=1), UUID(int=2)):
            for scope in (
                None,
                Scope(mode="hard", collection_ids=(str(UUID(int=11)),)),
                Scope(mode="soft", collection_ids=(str(UUID(int=12)),)),
            ):
                self.equal(owner, scope)
                hits = self.search(True, owner, scope)[0]
                if owner:
                    for hit in hits:
                        self.assertEqual(1 + (hit.doc_id.int - 1000) % 2, owner.int)
        a = Scope(mode="soft", collection_ids=(str(UUID(int=11)), str(UUID(int=12))))
        b = Scope(mode="soft", collection_ids=tuple(reversed(a.collection_ids)), widened=True)
        self.equal(UUID(int=1), a)
        self.assertEqual(self.search(True, UUID(int=1), b)[1]["sparse_index_cache_status"], "hit")

    def test_writes_commit_and_rollback_are_atomic(self):
        self.equal()
        before = self.stamp()
        with self.lab.engine.connect() as writer:
            transaction = writer.begin()
            writer.execute(text(_TEST_MATERIALS["text_010"]), {"id": UUID(int=10001)})
            self.assertEqual(self.stamp(), before)  # uncommitted version AND data invisible
            self.assertEqual(self.search()[1]["sparse_index_cache_status"], "hit")
            transaction.rollback()
        self.assertEqual(self.stamp(), before)
        self.lab.execute(_TEST_MATERIALS["text_002"], {"id": UUID(int=10001)})
        self.assertGreater(self.stamp().generation, before.generation)
        self.assertEqual(self.equal()["sparse_index_cache_status"], "cold_build")

    def test_mutation_matrix_metadata_rechunk_delete_membership_owner_cascade(self):
        operations = [
            ("INSERT INTO documents VALUES (:id,:owner,NULL)", {"id": UUID(int=500), "owner": UUID(int=1)}),
            (
                _TEST_MATERIALS["text_005"],
                {"id": UUID(int=500), "doc": UUID(int=500)},
            ),
            ("UPDATE chunks SET page=2,start_offset=9,end_offset=20 WHERE id=:id", {"id": UUID(int=500)}),
            ("UPDATE documents SET collection_id=:c WHERE id=:id", {"c": UUID(int=11), "id": UUID(int=500)}),
            ("UPDATE documents SET collection_id=NULL WHERE id=:id", {"id": UUID(int=500)}),
            ("UPDATE documents SET owner_id=:o WHERE id=:id", {"o": UUID(int=2), "id": UUID(int=500)}),
            ("DELETE FROM chunks WHERE id=:id", {"id": UUID(int=500)}),
            (
                _TEST_MATERIALS["text_006"],
                {"id": UUID(int=501), "doc": UUID(int=500)},
            ),
            ("DELETE FROM documents WHERE id=:id", {"id": UUID(int=500)}),
            ("DELETE FROM collections WHERE id=:id", {"id": UUID(int=11)}),
            ("DELETE FROM users WHERE id=:id", {"id": UUID(int=2)}),
            ("TRUNCATE chunks", {}),
        ]
        for sql, params in operations:
            with self.subTest(sql=sql):
                self.equal()
                before = self.stamp()
                self.lab.execute(sql, params)
                self.assertGreater(self.stamp().generation, before.generation)
                self.assertEqual(self.equal()["sparse_index_cache_status"], "cold_build")
                self.equal(UUID(int=1), Scope(mode="soft", collection_ids=(str(UUID(int=11)),)))

    def test_embedding_updates_do_not_advance_generation_or_change_ties(self):
        expected = self.search(False)[0]
        self.search()
        before = self.stamp()
        self.lab.execute("UPDATE chunks SET embedding='numeric vector placeholder', embedded_at=now()")
        self.assertEqual(self.stamp(), before)
        hits, meta, _ = self.search()
        self.assertEqual(hits, expected)
        self.assertEqual(meta["sparse_index_cache_status"], "hit")
        self.equal()

    def test_same_timestamp_is_cached_and_stable_after_embedding_write(self):
        self.lab.execute("UPDATE chunks SET created_at=now()")
        first, meta, _ = self.search()
        self.assertEqual(meta["sparse_index_cache_status"], "cold_build")
        self.lab.execute("UPDATE chunks SET embedding='another embedding' WHERE id=:id", {"id": UUID(int=10000)})
        second, meta, _ = self.search()
        self.assertEqual(second, first)
        self.assertEqual(meta["sparse_index_cache_status"], "hit")
        self.assertEqual(self.equal()["sparse_index_cache_status"], "hit")

    def test_null_created_at_still_falls_back(self):
        self.lab.execute("UPDATE chunks SET created_at=NULL WHERE id=:id", {"id": UUID(int=10000)})
        self.assertEqual(self.equal()["sparse_index_cache_reason"], "ambiguous_created_at")
        self.assertEqual(CACHE.stats()["entries"], 0)

    def test_writer_transaction_never_publishes_rollback_version(self):
        with self.lab.engine.connect() as writer:
            writer.execute(text(_TEST_MATERIALS["text_011"]), {"id": UUID(int=10001)})
            stages.reset_stages()
            actual = sparse.sparse_search(writer, _TEST_MATERIALS["text_007"], settings=self.settings)
            self.assertEqual(stages.snapshot_meta()["sparse_index_cache_reason"], "transaction_not_committed_reader")
            self.assertEqual(CACHE.stats()["entries"], 0)
            legacy = self.settings.model_copy(update={"sparse_index_cache_enabled": False})
            self.assertEqual(actual, sparse.sparse_search(writer, _TEST_MATERIALS["text_012"], settings=legacy))
            writer.rollback()
        self.assertEqual(self.equal()["sparse_index_cache_status"], "cold_build")

    def test_repeatable_read_falls_back(self):
        with self.lab.engine.connect().execution_options(isolation_level="REPEATABLE READ") as db:
            with self.assertRaisesRegex(Uncacheable, "transaction_not_committed_reader"):
                read_stamp(db)

    def test_update_during_build_rejected_and_legacy_reloads(self):
        original = sparse._build_snapshot

        def changing(*args):
            snapshot = original(*args)
            self.lab.execute(_TEST_MATERIALS["text_008"], {"id": UUID(int=10001)})
            return snapshot

        with patch.object(sparse, "_build_snapshot", side_effect=changing):
            hits, meta, _ = self.search()
        self.assertEqual(meta["sparse_index_cache_reason"], "changed_during_build")
        self.assertEqual(CACHE.stats()["entries"], 0)
        self.assertEqual(hits, self.search(False)[0])

    def test_concurrent_real_cold_queries_only_build_once(self):
        gate = threading.Barrier(4)

        def run():
            gate.wait(timeout=5)
            return self.search()[0]

        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(run) for _ in range(4)]
            results = [f.result(10) for f in futures]
        self.assertTrue(all(item == results[0] for item in results))
        self.assertEqual(CACHE.stats()["cold_build"], 1)
        self.assertEqual(CACHE.stats()["publish"], 1)

    def test_build_sql_error_does_not_abort_legacy_fallback(self):
        def fail(db, *args):
            db.execute(text("SELECT 1/0"))

        with patch.object(sparse, "_construct_snapshot", side_effect=fail):
            hits, meta, _ = self.search()
        self.assertEqual(meta["sparse_index_cache_reason"], "cache_error")
        self.assertEqual(hits, self.search(False)[0])
        self.assertEqual(CACHE.stats()["entries"], 0)
        self.assertEqual(CACHE.stats()["builders"], 0)

    def test_missing_disabled_guards_and_migration_down_up(self):
        self.search()
        self.lab.execute("ALTER TABLE chunks DISABLE TRIGGER sparse_chunks_update")
        try:
            self.assertEqual(self.equal()["sparse_index_cache_reason"], "version_guards_missing")
        finally:
            self.lab.execute("ALTER TABLE chunks ENABLE TRIGGER sparse_chunks_update")
        before = self.stamp()
        with self.lab.engine.begin() as db:
            self.lab.migrate(db, "downgrade")
        self.assertEqual(self.equal()["sparse_index_cache_reason"], "version_unavailable")
        with self.lab.engine.begin() as db:
            self.lab.migrate(db, "upgrade")
        self.assertNotEqual(self.stamp().epoch, before.epoch)
        self.assertEqual(self.equal()["sparse_index_cache_status"], "cold_build")

    def test_actual_session_pending_write_falls_back_without_flushing(self):
        # A pending unrelated ORM object must not be flushed by version probing.
        from app.models import Document

        with Session(self.lab.engine) as db:
            db.add(Document(filename=_TEST_MATERIALS["text_013"], byte_size=1, file_hash="pending"))
            with self.assertRaisesRegex(Uncacheable, "pending_orm_writes"):
                read_stamp(db)
            self.assertEqual(len(db.new), 1)


if __name__ == "__main__":
    if "--run-local" not in sys.argv:
        raise SystemExit(_TEST_MATERIALS["text_003"])
    sys.argv.remove("--run-local")
    unittest.main(verbosity=2)
