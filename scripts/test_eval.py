"""Day 9 acceptance: golden query set + /eval/summary aggregation."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

from lib.accept import assert_ok, version_tuple
from lib.http import API_BASE, get_json, post_file, post_json

RAW = ROOT / "data" / "raw"
GOLDEN = ROOT / "data" / "eval" / "golden.json"


def _check_golden_file() -> None:
    from app.eval.golden import GoldenQuery, GoldenSet, load_golden
    from app.eval.metrics import aggregate, score_query

    golden = load_golden(GOLDEN)
    assert_ok(len(golden.queries) >= 5, f"{_TEST_MATERIALS['text_001']}{len(golden.queries)}")
    ids = [item.id for item in golden.queries]
    assert_ok(len(ids) == len(set(ids)), "duplicate golden ids")
    assert_ok(any(item.expect == "grounded" for item in golden.queries), _TEST_MATERIALS["text_006"])
    assert_ok(any(item.expect == "refusal" for item in golden.queries), _TEST_MATERIALS["text_007"])
    print(f"golden ok name={golden.name} queries={len(golden.queries)}")

    grounded = GoldenQuery(
        id="q_test",
        query=_TEST_MATERIALS["text_014"],
        expect="grounded",
        relevant_terms=("file_hash", "doc_id"),
    )
    refusal = GoldenQuery(
        id="q_ref",
        query=_TEST_MATERIALS["text_015"],
        expect="refusal",
        relevant_terms=(),
    )
    tiny = GoldenSet(name="unit", corpus="", queries=(grounded, refusal))
    hit = score_query(
        grounded,
        trace_id="t1",
        evidence=[{"text": _TEST_MATERIALS["text_019"]}],
        citations=[{"index": 1}],
    )
    miss = score_query(grounded, trace_id="t2", evidence=[{"text": _TEST_MATERIALS["text_020"]}], citations=[])
    refuse = score_query(
        refusal,
        trace_id="t3",
        answer=_TEST_MATERIALS["text_016"],
        verification={"coverage_rate": 1.0, "hallucination_rate": 0.0, "claims": [{"status": "insufficient"}]},
    )
    assert_ok(hit.term_hit is True, f"{_TEST_MATERIALS['text_002']}{hit}")
    assert_ok(miss.term_hit is False, f"{_TEST_MATERIALS['text_003']}{miss}")
    assert_ok(refuse.refusal_ok is True, _TEST_MATERIALS["text_008"])
    report = aggregate(tiny, [hit, refuse])
    assert_ok(report.traced_count == 2, f"traced_count {report.traced_count}")
    assert_ok(report.term_hit_rate == 1.0, f"term_hit_rate {report.term_hit_rate}")
    assert_ok(report.refusal_ok_rate == 1.0, f"refusal_ok_rate {report.refusal_ok_rate}")
    print("metrics unit ok")


def _ensure_eval_corpus() -> None:
    from app.db.session import SessionLocal, init_db
    from app.eval.golden import corpus_filenames, load_golden
    from app.models import Document

    golden = load_golden(GOLDEN)
    names = set(corpus_filenames(golden.corpus))
    assert_ok(bool(names), _TEST_MATERIALS["text_009"])
    init_db()
    with SessionLocal() as db:
        have = {row[0] for row in db.query(Document.filename).all()}
    missing = [name for name in sorted(names) if name not in have]
    if not missing:
        print(f"eval corpus present: {sorted(names)}")
        return
    for name in missing:
        path = RAW / name
        assert_ok(path.is_file(), f"eval corpus missing {name}{_TEST_MATERIALS['text_005']}")
        ingest = post_file(name, path.read_bytes(), "text/markdown")
        assert_ok(ingest.get("chunk_count", 0) >= 1, f"ingest failed for {name}")
        print(f"ingest ok {name} chunks={ingest['chunk_count']}")


def _ensure_api() -> None:
    health = get_json(f"{API_BASE}/health")
    assert_ok(health.get("status") == "ok", f"health not ok: {health}")
    root = get_json(f"{API_BASE}/")
    assert_ok("eval_summary" in root, _TEST_MATERIALS["text_010"])
    assert_ok(
        version_tuple(root.get("version", "")) >= (0, 9),
        f"expected >= 0.9.0, got {root.get('version')}",
    )
    print(f"api version={root.get('version')}")

    _ensure_eval_corpus()

    if os.environ.get("SKIP_EMBED") != "1":
        from app.core.config import get_settings
        from app.db.session import SessionLocal, init_db
        from app.embed.service import count_pending, run_embedding_pipeline

        settings = get_settings()
        init_db()
        with SessionLocal() as db:
            if count_pending(db) > 0:
                stats = run_embedding_pipeline(db, settings=settings)
                assert_ok(stats.embedded + stats.copied > 0, _TEST_MATERIALS["text_021"])
                print(f"embed ok embedded={stats.embedded} copied={stats.copied}")
            else:
                print(_TEST_MATERIALS["text_022"])


def _run_two_queries() -> None:
    from app.eval.golden import load_golden

    golden = load_golden(GOLDEN)
    picked = [item for item in golden.queries if item.expect == "grounded"][:1]
    picked += [item for item in golden.queries if item.expect == "refusal"][:1]
    assert_ok(len(picked) == 2, _TEST_MATERIALS["text_011"])

    for item in picked:
        result = post_json(
            f"{API_BASE}/query",
            {
                "query": item.query,
                "top_k": 6,
                "verify": True,
                "persist": True,
                "eval_id": item.id,
            },
        )
        assert_ok(result.get("trace_id"), f"{item.id} missing trace_id")
        assert_ok(result.get("eval_id") == item.id, f"{item.id} eval_id not echoed")
        print(f"ran {item.id} trace_id={result['trace_id']} provider={result['provider']}")


def _check_summary() -> None:
    summary = get_json(f"{API_BASE}/eval/summary")
    required = (
        "eval_set",
        "query_count",
        "traced_count",
        "missing",
        "avg_coverage_rate",
        "avg_hallucination_rate",
        "term_hit_rate",
        "results",
    )
    for key in required:
        assert_ok(key in summary, f"/eval/summary missing {key}: {summary.keys()}")
    assert_ok(summary["query_count"] >= 5, f"summary query_count {summary['query_count']}")
    assert_ok(summary["traced_count"] >= 2, f"{_TEST_MATERIALS['text_004']}{summary}")
    assert_ok(len(summary["results"]) == summary["query_count"], "results length mismatch")

    traced = [item for item in summary["results"] if item.get("trace_id")]
    assert_ok(any(item.get("expect") == "grounded" for item in traced), _TEST_MATERIALS["text_012"])
    grounded = next(item for item in traced if item["expect"] == "grounded")
    assert_ok("term_hit" in grounded, f"grounded result missing term_hit: {grounded}")
    print(
        f"summary ok traced={summary['traced_count']}/{summary['query_count']} "
        f"coverage={summary['avg_coverage_rate']} "
        f"hallucination={summary['avg_hallucination_rate']} "
        f"term_hit={summary['term_hit_rate']}"
    )


def main() -> None:
    _check_golden_file()
    if os.environ.get("SKIP_EVAL_LLM") == "1":
        print(_TEST_MATERIALS["text_017"])
        print(_TEST_MATERIALS["text_018"])
        return
    _ensure_api()
    _run_two_queries()
    _check_summary()
    print(_TEST_MATERIALS["text_013"])


if __name__ == "__main__":
    main()
