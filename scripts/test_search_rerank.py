"""Day 5 acceptance: /search/rerank runs a Cross-Encoder over the Hybrid(RRF) pool."""

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
    health = get_json(f"{API_BASE}/health", timeout=180)
    assert_ok(health.get("status") == "ok", f"health not ok: {health}")
    print("health ok")

    root = get_json(f"{API_BASE}/", timeout=180)
    assert_ok("search_rerank" in root, _TEST_MATERIALS["text_003"])

    md_path = RAW / "sample.md"
    assert_ok(md_path.is_file(), f"missing {md_path}")
    ingest = post_file("sample.md", md_path.read_bytes(), "text/markdown", timeout=180)
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
                assert_ok(stats.embedded + stats.copied > 0, _TEST_MATERIALS["text_010"])
                print(f"embed ok embedded={stats.embedded} copied={stats.copied}")
            else:
                print(_TEST_MATERIALS["text_011"])


def _check_rerank() -> None:
    payload = {
        "query": _TEST_MATERIALS["text_004"],
        "top_k": 8,
        "candidates": 50,
    }
    result = post_json(f"{API_BASE}/search/rerank", payload, timeout=180)
    assert_ok(result.get("count", 0) >= 1, f"no rerank hits: {result}")
    assert_ok(result.get("top_k") == 8, "top_k mismatch")
    assert_ok(result.get("candidates") == 50, "candidates mismatch")
    assert_ok(result["count"] <= 8, _TEST_MATERIALS["text_005"])
    results = result["results"]
    print(f"{_TEST_MATERIALS['text_001']}{result['count']} top_score={results[0]['score']}")

    required = (
        "chunk_id",
        "doc_id",
        "page",
        "snippet",
        "score",
        "rrf_score",
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
            _TEST_MATERIALS["text_009"],
        )

    scores = [h["score"] for h in results]
    assert_ok(scores == sorted(scores, reverse=True), _TEST_MATERIALS["text_006"])

    keywords = ("chunk", "metadata", "hash", "offset", "hybrid", "retrieval", "doc_id", "page")
    combined = " ".join(h["snippet"].lower() for h in results[:3])
    assert_ok(any(k in combined for k in keywords), f"{_TEST_MATERIALS['text_002']}{combined[:160]}")
    print("rerank relevance ok")

    # The Cross-Encoder score is a fresh, independent judgment (query, full text),
    # so it need not preserve the RRF order — but it must be present for every hit.
    assert_ok(all(isinstance(h["score"], float) for h in results), "score not numeric")
    print(_TEST_MATERIALS["text_007"])


def main() -> None:
    _ensure_prerequisites()
    _check_rerank()
    print(_TEST_MATERIALS["text_008"])


if __name__ == "__main__":
    main()
