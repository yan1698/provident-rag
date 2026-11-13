"""Offline retrieval regression: real BM25/SQLite, fake vectors/CE; no model downloads."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import json
import os
import sys
import unittest
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Import the actual adapters against inert external model classes, never import torch.
_model_stubs = {
    "torch": SimpleNamespace(
        nn=SimpleNamespace(Sigmoid=lambda: None), cuda=SimpleNamespace(is_available=lambda: False)
    ),
    "sentence_transformers": SimpleNamespace(SentenceTransformer=object, CrossEncoder=object),
}
_previous_modules = {name: sys.modules.get(name) for name in _model_stubs}
sys.modules.update(_model_stubs)
try:
    from app.embed import model as embed_model
    from app.rerank import model as ce_model
    from app.search import dense, hybrid, rerank, sparse, stages
    from app.api import search as api
finally:
    for _name, _previous in _previous_modules.items():
        if _previous is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _previous

from sqlalchemy import create_engine, text
from app.core.config import Settings
from app.corpus.scope import Scope
from app.eval.benchmarks.retrieval_diagnostics import mapped_ids, passage_diagnostics


class Clock:
    now = 100.0

    def tick(self, seconds):
        self.now += seconds


def fixtures():
    return [
        dict(
            id=UUID(int=i),
            doc_id=UUID(int=10 + i),
            page=1,
            start_offset=0,
            end_offset=20,
            text=value,
            score=1.0 - i / 10,
        )
        for i, value in enumerate(
            (
                _TEST_MATERIALS["text_014"],
                _TEST_MATERIALS["text_015"],
                _TEST_MATERIALS["text_016"],
                _TEST_MATERIALS["text_017"],
                _TEST_MATERIALS["text_018"],
            ),
            1,
        )
    ]


class FakeDB:
    def __init__(self, clock=None):
        self.clock = clock
        self.calls = []

    def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params))
        if self.clock:
            self.clock.tick(0.004)
        rows = fixtures()
        sql = str(stmt)
        if "embedding <=>" in sql:
            rows = rows[: params["top_k"]]
        elif "chunks.id" in sql:
            rows = [SimpleNamespace(id=x["id"], text=x["text"]) for x in rows]
        return SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: rows), all=lambda: rows)


class RetrievalDiagnostics(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            _env_file=None,
            search_cache_enabled=False,
            hybrid_dense_candidates=3,
            hybrid_sparse_candidates=3,
            rerank_candidates=2,
            rerank_max_inflight=1,
        )
        self.db = FakeDB()
        stages.reset_stages(capture_candidates=True)
        self.embedding = patch.object(dense, "encode_query", return_value=[0.1] * 384)
        self.scores = patch.object(rerank, "rerank_scores", return_value=[0.1, 0.9])
        self.embedding.start()
        self.scores.start()
        self.addCleanup(self.embedding.stop)
        self.addCleanup(self.scores.stop)

    def test_all_routes_contract_and_diagnostics_toggle(self):
        common = {"chunk_id", "doc_id", "page", "snippet", "score", "start", "end"}
        extra = {"dense_score", "sparse_score", "dense_rank", "sparse_rank"}
        routes = [
            (dense.dense_search, common),
            (sparse.sparse_search, common),
            (hybrid.hybrid_search, common | extra),
            (rerank.rerank_search, common | extra | {"rrf_score"}),
        ]
        for function, fields in routes:
            with self.subTest(route=function.__name__):
                stages.reset_stages()
                before = function(self.db, _TEST_MATERIALS["text_019"], top_k=2, settings=self.settings)
                self.assertEqual(stages.snapshot_candidates(), [])
                stages.reset_stages(capture_candidates=True)
                after = function(self.db, _TEST_MATERIALS["text_020"], top_k=2, settings=self.settings)
                self.assertEqual(before, after)
                self.assertEqual(set(asdict(after[0])), fields)
                self.assertEqual(stages.snapshot_meta()["returned_chunk_count"], len(after))
                self.assertEqual(stages.snapshot_candidates()[-1]["candidate_count"], len(after))

    def test_api_response_schema_unchanged(self):
        for route in ("dense", "sparse", "hybrid", "rerank"):
            request = getattr(api, route.capitalize() + "SearchRequest")(query=_TEST_MATERIALS["text_021"], top_k=2)
            response = getattr(api, "search_" + route)(request, db=self.db, settings=self.settings, user=None)
            fields = {"query", "top_k", "count", "results", "cached"}
            if route == "hybrid":
                fields |= {"dense_candidates", "sparse_candidates"}
            if route == "rerank":
                fields.add("candidates")
            payload = response.model_dump(mode="json")
            self.assertEqual(set(payload), fields)
            self.assertFalse(payload["cached"])
            self.assertEqual(payload["count"], len(payload["results"]))

    def test_complete_chain_and_pruning(self):
        hits = rerank.rerank_search(self.db, _TEST_MATERIALS["text_001"], top_k=1, settings=self.settings)
        events = stages.snapshot_candidates()
        self.assertEqual(
            [e["stage"] for e in events], ["dense", "sparse", "rrf", "rerank_input", "rerank_output", "final"]
        )
        self.assertEqual([e["candidate_count"] for e in events], [3, 2, 3, 2, 2, 1])
        self.assertEqual(events[2]["candidates"][2]["chunk_id"], str(UUID(int=3)))
        self.assertEqual([x["chunk_id"] for x in events[3]["candidates"]], [str(UUID(int=1)), str(UUID(int=2))])
        self.assertEqual([x["chunk_id"] for x in events[4]["candidates"]], [str(UUID(int=2)), str(UUID(int=1))])
        self.assertEqual(hits[0].chunk_id, UUID(int=2))
        self.assertAlmostEqual(events[2]["candidates"][0]["score"], 2 / 61)
        for event in events:
            self.assertEqual([r["rank"] for r in event["candidates"]], list(range(1, event["candidate_count"] + 1)))
            for row in event["candidates"]:
                self.assertEqual(set(row), {"chunk_id", "doc_id", "score", "rank"})
        self.assertNotIn(_TEST_MATERIALS["text_002"], json.dumps(events, ensure_ascii=False))

    def test_sparse_rebuilds_each_call_real_bm25_and_sqlite(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        with engine.begin() as db:
            db.execute(
                text(
                    "CREATE TABLE chunks (id TEXT, doc_id TEXT, page INT, start_offset INT, end_offset INT, text TEXT, created_at INT)"
                )
            )
            for i, row in enumerate(fixtures()):
                db.execute(
                    text("INSERT INTO chunks VALUES (:id,:doc_id,:page,:start_offset,:end_offset,:text,:created_at)"),
                    {**row, "id": str(row["id"]), "doc_id": str(row["doc_id"]), "created_at": i},
                )
            with (
                patch.object(sparse, "BM25Okapi", wraps=sparse.BM25Okapi) as build,
                patch.object(sparse, "tokenize", wraps=sparse.tokenize) as tokenize,
            ):
                first = sparse.sparse_search(db, _TEST_MATERIALS["text_022"], top_k=3, settings=self.settings)
                second = sparse.sparse_search(db, _TEST_MATERIALS["text_023"], top_k=3, settings=self.settings)
                self.assertEqual(first, second)
                self.assertEqual(build.call_count, 2)
                self.assertEqual(tokenize.call_count, 12)  # query + all five chunks, twice
            self.assertEqual([h.chunk_id for h in first], [str(UUID(int=1)), str(UUID(int=2))])
            self.assertEqual(first[0].score, 1.009417)  # Scenario-specific notes are maintained with local materials.
            db.execute(text("DELETE FROM chunks WHERE id = :id"), {"id": str(UUID(int=1))})
            third = sparse.sparse_search(db, _TEST_MATERIALS["text_009"], top_k=3, settings=self.settings)
            self.assertEqual([h.chunk_id for h in third], [str(UUID(int=2))])
        engine.dispose()
        self.assertEqual(stages.snapshot_meta()["sparse_index_reused"], 0)
        for key in (
            "sparse_corpus_load_ms",
            "sparse_tokenize_ms",
            "sparse_index_build_ms",
            "sparse_search_ms",
            "bm25_ms",
        ):
            self.assertIn(key, stages.snapshot_stages())

    def test_scope_and_owner_sql_preserved(self):
        owner = UUID(int=99)
        scope = Scope(mode="hard", collection_ids=(str(UUID(int=88)),))
        sparse.sparse_search(self.db, _TEST_MATERIALS["text_003"], owner_id=owner, scope=scope, settings=self.settings)
        sql, params = self.db.calls[-1]
        self.assertIn(_TEST_MATERIALS["text_004"], sql)
        self.assertIn("owner_id = CAST(:owner_id AS uuid)", sql)
        self.assertIn("collection_id = ANY", sql)
        self.assertEqual(params["owner_id"], str(owner))
        self.assertEqual(params["scope_ids"], "{" + str(UUID(int=88)) + "}")

    def test_mapping_preserves_ranks_and_handles_duplicates_missing_many(self):
        hits = dense.dense_search(self.db, _TEST_MATERIALS["text_005"], top_k=3, settings=self.settings)
        mapping = {str(hits[0].chunk_id): ["p1"], str(hits[1].chunk_id): ["p1", "p2"], str(hits[2].chunk_id): ["p3"]}
        original = stages.snapshot_candidates()
        annotated = passage_diagnostics(original, mapping)
        self.assertEqual(mapped_ids(hits, mapping), ["p1", "p2", "p3"])
        self.assertEqual(original, stages.snapshot_candidates())
        for raw, item in zip(original, annotated):
            self.assertEqual(item["returned_chunk_count"], 3)
            self.assertEqual(item["unique_passage_count"], 3)
            self.assertEqual(item["duplicated_passage_count"], 1)
            self.assertEqual(item["repeated_passage_count"], 1)
            self.assertEqual(
                [{k: row[k] for k in ("chunk_id", "doc_id", "score", "rank")} for row in item["candidates"]],
                raw["candidates"],
            )
        missing = passage_diagnostics(original, {})[0]
        self.assertEqual(missing["unmapped_chunk_count"], 3)
        self.assertNotIn("passage_ids", missing["candidates"][0])
        self.assertEqual(passage_diagnostics(original, object()), [])

    def test_repeated_passes_and_request_reset(self):
        sparse.sparse_search(self.db, _TEST_MATERIALS["text_006"], settings=self.settings)
        sparse.sparse_search(self.db, _TEST_MATERIALS["text_007"], settings=self.settings)
        self.assertEqual([e["candidate_count"] for e in stages.snapshot_candidates()], [0, 0, 2, 2])
        stages.reset_stages()
        self.assertEqual(stages.snapshot_candidates(), [])
        self.assertEqual(stages.snapshot_stages(), {})

    def test_diagnostic_sink_failure_does_not_affect_results_or_exceptions(self):
        expected = rerank.rerank_search(self.db, _TEST_MATERIALS["text_008"], top_k=1, settings=self.settings)
        tokens = [var.set(object()) for var in (stages._stages, stages._meta, stages._candidates)]
        try:
            actual = rerank.rerank_search(self.db, _TEST_MATERIALS["text_010"], top_k=1, settings=self.settings)
            self.assertEqual(actual, expected)
            self.assertEqual(stages.snapshot_stages(), {})
            self.assertGreater(stages.snapshot_meta()["diagnostics_error_count"], 0)
            self.assertEqual(stages.snapshot_candidates(), [])
            with patch.object(self.db, "execute", side_effect=ValueError("database failed")):
                with self.assertRaisesRegex(ValueError, "database failed"):
                    sparse.sparse_search(self.db, _TEST_MATERIALS["text_025"], settings=self.settings)
        finally:
            for var, token in zip((stages._stages, stages._meta, stages._candidates), tokens):
                var.reset(token)
        self.assertEqual(stages._depth.get(), 0)

    def test_timers_wait_load_predict_and_nested_totals(self):
        clock = Clock()

        @contextmanager
        def slot(limit):
            clock.tick(0.020)
            yield

        def load():
            clock.tick(0.030)
            return SimpleNamespace(predict=predict)

        def predict(*args, **kwargs):
            clock.tick(0.050)
            return [0.1, 0.9]

        def encode(*args, **kwargs):
            clock.tick(0.010)
            return [0.1] * 384

        with (
            patch("time.perf_counter", side_effect=lambda: clock.now),
            patch("app.rerank.limit.rerank_slot", slot),
            patch.object(ce_model, "get_reranker", side_effect=load),
            patch.object(rerank, "rerank_scores", ce_model.rerank_scores),
            patch.object(dense, "encode_query", side_effect=encode),
        ):
            rerank.rerank_search(FakeDB(clock), _TEST_MATERIALS["text_011"], top_k=1, settings=self.settings)
        times = stages.snapshot_stages()
        for key, expected in {
            "embed_ms": 10,
            "ann_ms": 4,
            "sparse_corpus_load_ms": 4,
            "rerank_wait_ms": 20,
            "rerank_model_load_ms": 30,
            "rerank_ms": 50,
            "rerank_call_ms": 100,
            "retrieval_total_ms": 122,
        }.items():
            self.assertAlmostEqual(times[key], expected, delta=1)
        self.assertEqual(stages.leaf_total_ms(times), times["retrieval_total_ms"])
        self.assertEqual(stages.leaf_total_ms({"embed_ms": 10, "ann_ms": 4}), 14)

    def test_cache_hit_is_explicit_and_does_not_execute_search(self):
        from app.cache import service as cache

        self.settings.search_cache_enabled = True
        with (
            patch.object(
                cache, "cache_get", return_value={"query": _TEST_MATERIALS["text_026"], "results": [], "cached": False}
            ),
            patch.object(cache, "_incr"),
            patch.object(sparse, "sparse_search") as search,
        ):
            payload, hit = cache.cached_payload("sparse", {}, search, settings=self.settings)
        self.assertTrue(hit)
        search.assert_not_called()
        self.assertEqual(stages.snapshot_meta()["search_cache_status"], "hit")
        self.assertEqual(stages.snapshot_stages(), {})
        self.assertNotIn("cached", payload)

    def test_query_persists_candidates_in_existing_diagnostics_only(self):
        from app.api import query as query_api
        from app.generate.service import AnswerResult

        self.settings.qa_verify = False
        self.settings.search_trace_candidates = True

        def answer(*args, **kwargs):
            rerank.rerank_search(self.db, _TEST_MATERIALS["text_012"], top_k=1, settings=self.settings)
            return AnswerResult(
                query=_TEST_MATERIALS["text_024"], answer="", provider="stub", citations=[], evidence=[]
            )

        with (
            patch("app.corpus.router.resolve_scope", return_value=Scope()),
            patch.object(query_api, "answer_query", side_effect=answer),
            patch.object(query_api, "save_trace", return_value=SimpleNamespace(id=UUID(int=9))) as save,
        ):
            response = query_api.query(
                query_api.QueryRequest(query=_TEST_MATERIALS["text_027"]),
                db=self.db,
                settings=self.settings,
                user=None,
                _=None,
            )
        diag = save.call_args.kwargs["diagnostics"]
        self.assertEqual(len(diag["candidate_stages"]), 6)
        json.dumps(diag, allow_nan=False)
        self.assertNotIn("candidate_stages", response.model_dump())
        self.assertEqual(response.trace_id, str(UUID(int=9)))

    def test_embedding_load_and_inference_boundaries(self):
        clock = Clock()

        class Vector:
            def tolist(self):
                return [0.1] * 384

        def encode(*args, **kwargs):
            clock.tick(0.030)
            return [Vector()]

        def load():
            clock.tick(0.020)
            return SimpleNamespace(encode=encode)

        with (
            patch("time.perf_counter", side_effect=lambda: clock.now),
            patch.object(embed_model, "get_embedder", side_effect=load),
            patch.object(dense, "encode_query", embed_model.encode_query),
        ):
            dense.dense_search(FakeDB(clock), _TEST_MATERIALS["text_013"], settings=self.settings)
        times = stages.snapshot_stages()
        self.assertAlmostEqual(times["embed_model_load_ms"], 20, delta=1)
        self.assertAlmostEqual(times["embed_inference_ms"], 30, delta=1)
        self.assertAlmostEqual(times["embed_ms"], 50, delta=1)
        self.assertEqual(stages.leaf_total_ms(times), times["retrieval_total_ms"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
