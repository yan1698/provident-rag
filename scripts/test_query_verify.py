"""Day 7 acceptance: Verifier checks each [S<n>] claim against cited evidence."""

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

EVIDENCE_TEXT = _TEST_MATERIALS["text_010"]

EVIDENCE = [
    {
        "index": 1,
        "chunk_id": "00000000-0000-0000-0000-000000000001",
        "doc_id": "00000000-0000-0000-0000-000000000002",
        "page": 1,
        "text": EVIDENCE_TEXT,
    }
]


def _check_shape(result: dict, *, context: str) -> None:
    required = (
        "claim_count",
        "supported_count",
        "hallucination_count",
        "coverage_rate",
        "hallucination_rate",
        "claims",
    )
    for key in required:
        assert_ok(key in result, f"{context} missing {key}: {result}")
    assert_ok(0.0 <= result["coverage_rate"] <= 1.0, f"{context} coverage_rate out of range")
    assert_ok(0.0 <= result["hallucination_rate"] <= 1.0, f"{context} hallucination_rate out of range")
    claim_required = (
        "text",
        "citation_indices",
        "lexical_coverage",
        "entailment_score",
        "status",
        "hallucination",
    )
    for item in result["claims"]:
        for key in claim_required:
            assert_ok(key in item, f"{context} claim missing {key}: {item}")


def _verify(answer: str, evidence: list[dict] | None = None) -> dict:
    result = post_json(
        f"{API_BASE}/verify",
        {"answer": answer, "evidence": evidence if evidence is not None else EVIDENCE},
        timeout=180,
    )
    _check_shape(result, context=_TEST_MATERIALS["text_027"])
    return result


def _ensure_api() -> None:
    health = get_json(f"{API_BASE}/health", timeout=180)
    assert_ok(health.get("status") == "ok", f"health not ok: {health}")
    print("health ok")

    root = get_json(f"{API_BASE}/", timeout=180)
    assert_ok("verify" in root, _TEST_MATERIALS["text_012"])
    print(f"api version={root.get('version')}")


def _check_supported() -> None:
    result = _verify(_TEST_MATERIALS["text_013"])
    assert_ok(result["claim_count"] == 1, f"{_TEST_MATERIALS['text_001']}{result}")
    assert_ok(result["claims"][0]["status"] == "supported", f"{_TEST_MATERIALS['text_002']}{result}")
    assert_ok(result["claims"][0]["hallucination"] is False, _TEST_MATERIALS["text_014"])
    assert_ok(result["coverage_rate"] == 1.0, f"supported coverage_rate != 1: {result}")
    assert_ok(result["hallucination_count"] == 0, f"supported hallucination_count != 0: {result}")
    print("verify supported ok")


def _check_unsupported() -> None:
    result = _verify(_TEST_MATERIALS["text_015"])
    assert_ok(result["claim_count"] == 1, f"{_TEST_MATERIALS['text_003']}{result}")
    assert_ok(
        result["claims"][0]["status"] == "unsupported",
        f"{_TEST_MATERIALS['text_004']}{result}",
    )
    assert_ok(result["claims"][0]["hallucination"] is True, _TEST_MATERIALS["text_016"])
    assert_ok(result["coverage_rate"] == 0.0, f"unsupported coverage_rate != 0: {result}")
    print(_TEST_MATERIALS["text_017"])


def _check_uncited() -> None:
    result = _verify(_TEST_MATERIALS["text_018"])
    assert_ok(result["claim_count"] == 1, f"{_TEST_MATERIALS['text_005']}{result}")
    assert_ok(result["claims"][0]["status"] == "uncited", f"{_TEST_MATERIALS['text_006']}{result}")
    assert_ok(result["claims"][0]["hallucination"] is True, _TEST_MATERIALS["text_019"])
    print("verify uncited ok")


def _check_invalid_citation() -> None:
    result = _verify(_TEST_MATERIALS["text_020"])
    assert_ok(result["claims"][0]["status"] == "invalid_citation", f"invalid citation missed: {result}")
    assert_ok(result["claims"][0]["hallucination"] is True, _TEST_MATERIALS["text_021"])
    print("verify invalid_citation ok")


def _check_insufficient() -> None:
    result = _verify(_TEST_MATERIALS["text_022"], evidence=[])
    assert_ok(result["claims"][0]["status"] == "insufficient", f"insufficient missed: {result}")
    assert_ok(result["claims"][0]["hallucination"] is False, _TEST_MATERIALS["text_023"])
    assert_ok(result["coverage_rate"] == 1.0, f"insufficient coverage_rate != 1: {result}")
    assert_ok(result["hallucination_rate"] == 0.0, f"insufficient hallucination_rate != 0: {result}")
    print(_TEST_MATERIALS["text_024"])


def _check_mixed() -> None:
    answer = _TEST_MATERIALS["text_011"]
    result = _verify(answer)
    assert_ok(result["claim_count"] == 3, f"{_TEST_MATERIALS['text_007']}{result}")
    statuses = [item["status"] for item in result["claims"]]
    assert_ok(
        statuses == ["supported", "unsupported", "uncited"],
        f"mixed statuses {statuses} != ['supported', 'unsupported', 'uncited']",
    )
    assert_ok(result["supported_count"] == 1, f"mixed supported_count: {result}")
    assert_ok(result["hallucination_count"] == 2, f"mixed hallucination_count: {result}")
    assert_ok(abs(result["coverage_rate"] - 1 / 3) < 1e-6, f"mixed coverage_rate: {result}")
    assert_ok(abs(result["hallucination_rate"] - 2 / 3) < 1e-6, f"mixed hallucination_rate: {result}")
    print(_TEST_MATERIALS["text_025"])


def _ensure_query_prerequisites() -> None:
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
                assert_ok(stats.embedded + stats.copied > 0, _TEST_MATERIALS["text_031"])
                print(f"embed ok embedded={stats.embedded} copied={stats.copied}")
            else:
                print(_TEST_MATERIALS["text_032"])


def _check_query_includes_verification() -> None:
    result = post_json(
        f"{API_BASE}/query",
        {
            "query": _TEST_MATERIALS["text_028"],
            "top_k": 6,
            "verify": True,
        },
        timeout=240,
    )
    assert_ok("verification" in result, f"/query missing verification: {result.keys()}")
    verification = result["verification"]
    assert_ok(verification is not None, "/query verification is null")
    _check_shape(verification, context=_TEST_MATERIALS["text_029"])
    assert_ok(verification["claim_count"] >= 1, f"/query produced zero claims: {verification}")
    print(
        f"{_TEST_MATERIALS['text_008']}{result['provider']} claims={verification['claim_count']} coverage={verification['coverage_rate']} hallucination={verification['hallucination_rate']}"
    )
    if result["provider"] == "stub":
        supported = [item for item in verification["claims"] if item["status"] == "supported"]
        assert_ok(len(supported) >= 1, f"{_TEST_MATERIALS['text_009']}{verification}")
        print(_TEST_MATERIALS["text_030"])


def main() -> None:
    _ensure_api()
    _check_supported()
    _check_unsupported()
    _check_uncited()
    _check_invalid_citation()
    _check_insufficient()
    _check_mixed()
    _ensure_query_prerequisites()
    _check_query_includes_verification()
    print(_TEST_MATERIALS["text_026"])


if __name__ == "__main__":
    main()
