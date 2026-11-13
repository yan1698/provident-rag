"""Day 3 dense search acceptance: /search/dense returns ranked hits with metadata."""

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
    assert_ok("search_dense" in root or root.get("version", "") >= "0.3.0", _TEST_MATERIALS["text_004"])

    md_path = RAW / "sample.md"
    assert_ok(md_path.is_file(), f"missing {md_path}")
    ingest = post_file("sample.md", md_path.read_bytes(), "text/markdown", timeout=120)
    assert_ok(ingest.get("chunk_count", 0) >= 1, "ingest failed")
    print(f"ingest ok chunks={ingest['chunk_count']}")

    if os.environ.get("SKIP_EMBED") != "1":
        from app.db.session import SessionLocal, init_db
        from app.core.config import get_settings
        from app.embed.service import count_pending, run_embedding_pipeline

        settings = get_settings()
        init_db()
        with SessionLocal() as db:
            if count_pending(db) > 0:
                stats = run_embedding_pipeline(db, settings=settings)
                assert_ok(stats.embedded + stats.copied > 0, _TEST_MATERIALS["text_008"])
                print(f"embed ok embedded={stats.embedded} copied={stats.copied}")
            else:
                print(_TEST_MATERIALS["text_009"])


def main() -> None:
    _ensure_prerequisites()

    payload = {
        "query": _TEST_MATERIALS["text_005"],
        "top_k": 5,
    }
    result = post_json(f"{API_BASE}/search/dense", payload, timeout=120)
    assert_ok(result.get("count", 0) >= 1, f"no search hits: {result}")
    assert_ok(result.get("top_k") == 5, "top_k mismatch")
    results = result["results"]
    print(f"search ok count={result['count']} top_score={results[0]['score']}")

    required = ("chunk_id", "doc_id", "page", "snippet", "score", "start", "end")
    for hit in results:
        for key in required:
            assert_ok(key in hit, f"hit missing {key}")
        assert_ok(hit["end"] >= hit["start"], "invalid offsets")
        assert_ok(0.0 <= hit["score"] <= 1.0, f"{_TEST_MATERIALS['text_003']}{hit['score']}")

    scores = [h["score"] for h in results]
    assert_ok(scores == sorted(scores, reverse=True), _TEST_MATERIALS["text_006"])
    assert_ok(results[0]["score"] >= 0.2, f"{_TEST_MATERIALS['text_001']}{results[0]['score']}")

    keywords = ("chunk", "metadata", "hash", "offset", "hybrid", "retrieval", "doc_id", "page")
    combined = " ".join(h["snippet"].lower() for h in results[:3])
    assert_ok(any(k in combined for k in keywords), f"{_TEST_MATERIALS['text_002']}{combined[:160]}")
    print("relevance ok")
    print(_TEST_MATERIALS["text_007"])


if __name__ == "__main__":
    main()
