"""Day 6 acceptance: /query ties Rerank Top-K evidence to an LLM-generated,
citation-annotated answer ([S<n>] -> doc_id/page)."""

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
    assert_ok("query" in root, _TEST_MATERIALS["text_002"])

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
                assert_ok(stats.embedded + stats.copied > 0, _TEST_MATERIALS["text_014"])
                print(f"embed ok embedded={stats.embedded} copied={stats.copied}")
            else:
                print(_TEST_MATERIALS["text_015"])


def _check_query() -> None:
    payload = {
        "query": _TEST_MATERIALS["text_003"],
        "top_k": 6,
    }
    result = post_json(f"{API_BASE}/query", payload, timeout=180)

    required = ("query", "answer", "provider", "citations", "evidence", "usage")
    for key in required:
        assert_ok(key in result, f"response missing {key}: {result}")

    assert_ok(isinstance(result["answer"], str) and result["answer"].strip(), "empty answer")
    assert_ok(isinstance(result["usage"], list), f"{_TEST_MATERIALS['text_001']}{result.get('usage')}")
    print(f"query ok provider={result['provider']}")
    print(f"answer preview: {result['answer'][:160]!r}")

    evidence = result["evidence"]
    assert_ok(len(evidence) >= 1, "no evidence returned")
    assert_ok(len(evidence) <= 6, _TEST_MATERIALS["text_004"])
    evidence_required = ("index", "chunk_id", "doc_id", "page", "text")
    for item in evidence:
        for key in evidence_required:
            assert_ok(key in item, f"evidence item missing {key}")
    indices = [item["index"] for item in evidence]
    assert_ok(indices == sorted(indices), _TEST_MATERIALS["text_005"])
    assert_ok(indices == list(range(1, len(evidence) + 1)), _TEST_MATERIALS["text_006"])
    print(f"evidence ok count={len(evidence)}")

    citations = result["citations"]
    citation_required = ("index", "chunk_id", "doc_id", "page")
    evidence_by_index = {item["index"]: item for item in evidence}
    for cite in citations:
        for key in citation_required:
            assert_ok(key in cite, f"citation missing {key}")
        assert_ok(cite["index"] in evidence_by_index, f"citation index {cite['index']} not in evidence")
        ev = evidence_by_index[cite["index"]]
        assert_ok(cite["chunk_id"] == ev["chunk_id"], _TEST_MATERIALS["text_008"])
        assert_ok(cite["doc_id"] == ev["doc_id"], _TEST_MATERIALS["text_009"])
        assert_ok(cite["page"] == ev["page"], _TEST_MATERIALS["text_010"])
    print(f"citations ok count={len(citations)}")

    if result["provider"] == "stub":
        assert_ok(len(citations) >= 1, _TEST_MATERIALS["text_011"])
        assert_ok("[S1]" in result["answer"], _TEST_MATERIALS["text_012"])
        print(_TEST_MATERIALS["text_013"])


def main() -> None:
    _ensure_prerequisites()
    _check_query()
    print(_TEST_MATERIALS["text_007"])


if __name__ == "__main__":
    main()
