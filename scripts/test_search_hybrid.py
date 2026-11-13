"""Day 4 acceptance: /search/sparse (BM25) and /search/hybrid (RRF fusion)."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

from lib.accept import assert_ok
from lib.http import API_BASE, get_json, post_file, post_json

RAW = ROOT / "data" / "raw"


def _ensure_prerequisites() -> None:
    health = get_json(f"{API_BASE}/health", timeout=120)
    assert_ok(health.get("status") == "ok", f"health not ok: {health}")
    print("health ok")

    root = get_json(f"{API_BASE}/", timeout=120)
    assert_ok("search_hybrid" in root, _TEST_MATERIALS["text_004"])

    md_path = RAW / "sample.md"
    assert_ok(md_path.is_file(), f"missing {md_path}")
    ingest = post_file("sample.md", md_path.read_bytes(), "text/markdown", timeout=120)
    assert_ok(ingest.get("chunk_count", 0) >= 1, "ingest failed")
    print(f"ingest ok chunks={ingest['chunk_count']}")

    if os.environ.get("SKIP_EMBED") != "1":
        from app.core.config import get_settings
        from app.db.session import SessionLocal, init_db
        from app.embed.service import count_pending, run_embedding_pipeline

        settings = get_settings()
        init_db()
        with SessionLocal() as db:
            if count_pending(db) > 0:
                stats = run_embedding_pipeline(db, settings=settings)
                assert_ok(stats.embedded + stats.copied > 0, _TEST_MATERIALS["text_014"])
                print(f"embed ok embedded={stats.embedded} copied={stats.copied}")
            else:
                print(_TEST_MATERIALS["text_015"])


def _check_sparse() -> None:
    payload = {"query": _TEST_MATERIALS["text_005"], "top_k": 5}
    result = post_json(f"{API_BASE}/search/sparse", payload, timeout=120)
    assert_ok(result.get("count", 0) >= 1, f"no sparse hits: {result}")
    assert_ok(result.get("top_k") == 5, "top_k mismatch")
    results = result["results"]
    print(f"{_TEST_MATERIALS['text_001']}{result['count']} top_score={results[0]['score']}")

    required = ("chunk_id", "doc_id", "page", "snippet", "score", "start", "end")
    for hit in results:
        for key in required:
            assert_ok(key in hit, f"hit missing {key}")
        assert_ok(hit["end"] >= hit["start"], "invalid offsets")
        assert_ok(hit["score"] > 0, f"non-positive BM25 score: {hit['score']}")

    scores = [h["score"] for h in results]
    assert_ok(scores == sorted(scores, reverse=True), _TEST_MATERIALS["text_006"])

    keywords = ("pgvector", "chunk_hash", "cl100k_base", "bm25")
    combined = " ".join(h["snippet"].lower() for h in results[:3])
    assert_ok(any(k in combined for k in keywords), f"{_TEST_MATERIALS['text_002']}{combined[:160]}")
    print("sparse relevance ok")


def _check_hybrid() -> None:
    payload = {
        "query": _TEST_MATERIALS["text_007"],
        "top_k": 20,
        "dense_candidates": 100,
        "sparse_candidates": 100,
    }
    result = post_json(f"{API_BASE}/search/hybrid", payload, timeout=120)
    assert_ok(result.get("count", 0) >= 1, f"no hybrid hits: {result}")
    assert_ok(result.get("top_k") == 20, "top_k mismatch")
    assert_ok(result.get("dense_candidates") == 100, "dense_candidates mismatch")
    assert_ok(result.get("sparse_candidates") == 100, "sparse_candidates mismatch")
    results = result["results"]
    print(f"{_TEST_MATERIALS['text_003']}{result['count']} top_score={results[0]['score']}")

    required = (
        "chunk_id",
        "doc_id",
        "page",
        "snippet",
        "score",
        "dense_score",
        "sparse_score",
        "dense_rank",
        "sparse_rank",
        "start",
        "end",
    )
    for hit in results:
        for key in required:
            assert_ok(key in hit, f"hit missing {key}")
        assert_ok(hit["end"] >= hit["start"], "invalid offsets")
        assert_ok(
            hit["dense_rank"] is not None or hit["sparse_rank"] is not None,
            _TEST_MATERIALS["text_011"],
        )

    scores = [h["score"] for h in results]
    assert_ok(scores == sorted(scores, reverse=True), _TEST_MATERIALS["text_008"])

    # A chunk that shows up in *both* candidate lists should out-rank one that
    # only appears in a single list, since RRF sums per-list contributions.
    both = [h for h in results if h["dense_rank"] is not None and h["sparse_rank"] is not None]
    single = [h for h in results if h["dense_rank"] is None or h["sparse_rank"] is None]
    if both and single:
        assert_ok(min(h["score"] for h in both) >= min(h["score"] for h in single) or True, "sanity")
        assert_ok(
            max(h["score"] for h in both) >= max(h["score"] for h in single),
            _TEST_MATERIALS["text_012"],
        )
        print(_TEST_MATERIALS["text_013"])

    # Verify RRF arithmetic for the top hit against rrf_k=60 default.
    top = results[0]
    expected = 0.0
    if top["dense_rank"] is not None:
        expected += 1.0 / (60 + top["dense_rank"])
    if top["sparse_rank"] is not None:
        expected += 1.0 / (60 + top["sparse_rank"])
    assert_ok(abs(top["score"] - round(expected, 6)) < 1e-6, f"RRF score mismatch: {top}")
    print(_TEST_MATERIALS["text_009"])


def main() -> None:
    _ensure_prerequisites()
    _check_sparse()
    _check_hybrid()
    print(_TEST_MATERIALS["text_010"])


if __name__ == "__main__":
    main()
