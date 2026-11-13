"""Day 8 acceptance: /query persists traces; GET /traces replays evidence + verification."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

from lib.accept import assert_ok, version_tuple
from lib.http import API_BASE, get_json, post_file, post_json

RAW = ROOT / "data" / "raw"


def _ensure_prerequisites() -> None:
    health = get_json(f"{API_BASE}/health", timeout=240)
    assert_ok(health.get("status") == "ok", f"health not ok: {health}")
    print("health ok")

    root = get_json(f"{API_BASE}/", timeout=240)
    assert_ok("traces" in root, _TEST_MATERIALS["text_008"])
    assert_ok(
        version_tuple(root.get("version", "")) >= (0, 8),
        f"expected >= 0.8.0, got {root.get('version')}",
    )
    print(f"api version={root.get('version')}")

    md_path = RAW / "sample.md"
    assert_ok(md_path.is_file(), f"missing {md_path}")
    ingest = post_file("sample.md", md_path.read_bytes(), "text/markdown", timeout=240)
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
                assert_ok(stats.embedded + stats.copied > 0, _TEST_MATERIALS["text_016"])
                print(f"embed ok embedded={stats.embedded} copied={stats.copied}")
            else:
                print(_TEST_MATERIALS["text_017"])


def _check_query_persists() -> tuple[str, dict]:
    result = post_json(
        f"{API_BASE}/query",
        {
            "query": _TEST_MATERIALS["text_014"],
            "top_k": 6,
            "verify": True,
            "persist": True,
        },
        timeout=240,
    )
    assert_ok(result.get("trace_id"), f"/query missing trace_id: {result.keys()}")
    trace_id = result["trace_id"]
    uuid.UUID(trace_id)
    assert_ok(result.get("verification") is not None, "/query verification is null")
    print(f"query persisted trace_id={trace_id} provider={result['provider']}")
    return trace_id, result


def _check_get_trace(trace_id: str, query_result: dict) -> dict:
    stored = get_json(f"{API_BASE}/traces/{trace_id}", timeout=240)
    required = (
        "id",
        "query",
        "answer",
        "provider",
        "llm_model",
        "citations",
        "evidence",
        "verification",
        "usage",
        "coverage_rate",
        "hallucination_rate",
        "created_at",
    )
    for key in required:
        assert_ok(key in stored, f"{_TEST_MATERIALS['text_006']}{key}: {stored.keys()}")

    assert_ok(stored["id"] == trace_id, "trace id mismatch")
    assert_ok(stored["query"] == query_result["query"], "stored query mismatch")
    assert_ok(stored["answer"] == query_result["answer"], "stored answer mismatch")
    assert_ok(stored["provider"] == query_result["provider"], "stored provider mismatch")
    assert_ok(stored["citations"] == query_result["citations"], "stored citations mismatch")
    assert_ok(len(stored["evidence"]) == len(query_result["evidence"]), _TEST_MATERIALS["text_009"])
    assert_ok(stored["verification"] == query_result["verification"], "stored verification mismatch")
    assert_ok(stored.get("usage") == query_result.get("usage"), "stored usage mismatch")
    assert_ok(
        isinstance(query_result.get("usage"), list), f"/query usage should be a list: {query_result.get('usage')}"
    )
    print(f"{_TEST_MATERIALS['text_001']}{stored['coverage_rate']}")
    return stored


def _check_list_includes(trace_id: str) -> None:
    listing = get_json(f"{API_BASE}/traces?limit=20", timeout=240)
    assert_ok("results" in listing, f"{_TEST_MATERIALS['text_002']}{listing}")
    ids = [item["id"] for item in listing["results"]]
    assert_ok(trace_id in ids, f"trace {trace_id}{_TEST_MATERIALS['text_003']}{ids[:5]}")
    summary = next(item for item in listing["results"] if item["id"] == trace_id)
    assert_ok("answer" not in summary, _TEST_MATERIALS["text_010"])
    assert_ok("evidence" not in summary, _TEST_MATERIALS["text_011"])
    print(f"{_TEST_MATERIALS['text_004']}{listing['count']}")


def _check_replay_verify(stored: dict) -> None:
    replay = post_json(
        f"{API_BASE}/verify",
        {"answer": stored["answer"], "evidence": stored["evidence"]},
        timeout=240,
    )
    assert_ok("coverage_rate" in replay, f"replay /verify missing coverage_rate: {replay}")
    stored_v = stored["verification"]
    assert_ok(
        abs(replay["coverage_rate"] - stored_v["coverage_rate"]) < 1e-6,
        f"replay coverage_rate {replay['coverage_rate']} != stored {stored_v['coverage_rate']}",
    )
    assert_ok(
        abs(replay["hallucination_rate"] - stored_v["hallucination_rate"]) < 1e-6,
        f"replay hallucination_rate mismatch: {replay} vs {stored_v}",
    )
    print(f"{_TEST_MATERIALS['text_005']}{replay['coverage_rate']} hallucination={replay['hallucination_rate']}")


def _check_missing_trace() -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    try:
        get_json(f"{API_BASE}/traces/{missing}", timeout=240)
    except SystemExit as exc:
        assert_ok("HTTP 404" in str(exc), f"{_TEST_MATERIALS['text_007']}{exc}")
        print(_TEST_MATERIALS["text_015"])
        return
    raise SystemExit(_TEST_MATERIALS["text_012"])


def main() -> None:
    _ensure_prerequisites()
    trace_id, query_result = _check_query_persists()
    stored = _check_get_trace(trace_id, query_result)
    _check_list_includes(trace_id)
    _check_replay_verify(stored)
    _check_missing_trace()
    print(_TEST_MATERIALS["text_013"])


if __name__ == "__main__":
    main()
