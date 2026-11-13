"""Regression helper; scenario text is supplied locally."""

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
_RETRY = {"retries": 2, "backoff": 25.0}


def _check_gold_labels_offline() -> None:
    from app.eval.gold import LabeledChunk, label_sections, resolve_gold
    from app.eval.golden import GoldenQuery, load_golden

    golden = load_golden(GOLDEN)
    grounded = [item for item in golden.queries if item.expect == "grounded"]
    assert_ok(bool(grounded), _TEST_MATERIALS["text_011"])
    unlabeled = [item.id for item in grounded if not item.has_gold]
    assert_ok(not unlabeled, f"{_TEST_MATERIALS['text_001']}{unlabeled}")
    print(f"{_TEST_MATERIALS['text_002']}{len(grounded)}")

    sections = label_sections(
        [
            _TEST_MATERIALS["text_018"],
            _TEST_MATERIALS["text_019"],
            _TEST_MATERIALS["text_020"],
            _TEST_MATERIALS["text_021"],
        ]
    )
    assert_ok(sections[1][0].startswith("1."), f"heading not detected: {sections}")
    assert_ok(sections[2][0].startswith("1."), f"{_TEST_MATERIALS['text_003']}{sections}")
    assert_ok(sections[3][0].startswith("2."), f"section not switched: {sections}")
    print("section labeling ok")

    chunks = [
        LabeledChunk("c1", "aaaa1111" * 8, "d1", 1, (_TEST_MATERIALS["text_030"],), "..."),
        LabeledChunk("c2", "bbbb2222" * 8, "d1", 1, (_TEST_MATERIALS["text_031"], _TEST_MATERIALS["text_032"]), "..."),
        LabeledChunk("c3", "cccc3333" * 8, "d1", 2, (_TEST_MATERIALS["text_033"],), "..."),
    ]
    by_section = resolve_gold(
        GoldenQuery(
            id="q1",
            query=_TEST_MATERIALS["text_034"],
            expect="grounded",
            relevant_terms=(),
            gold_sections=(_TEST_MATERIALS["text_038"],),
        ),
        chunks,
    )
    assert_ok([c.chunk_id for c in by_section.chunks] == ["c2"], f"section match wrong: {by_section}")
    by_hash = resolve_gold(
        GoldenQuery(
            id="q2",
            query=_TEST_MATERIALS["text_035"],
            expect="grounded",
            relevant_terms=(),
            gold_chunk_hashes=("cccc3333",),
        ),
        chunks,
    )
    assert_ok([c.chunk_id for c in by_hash.chunks] == ["c3"], f"{_TEST_MATERIALS['text_004']}{by_hash}")
    typo = resolve_gold(
        GoldenQuery(
            id="q3",
            query=_TEST_MATERIALS["text_036"],
            expect="grounded",
            relevant_terms=(),
            gold_sections=(_TEST_MATERIALS["text_039"],),
        ),
        chunks,
    )
    assert_ok(typo.chunks == () and typo.missing_labels == (_TEST_MATERIALS["text_040"],), f"typo not reported: {typo}")
    print("gold resolution ok")


def _check_metrics_unit() -> None:
    from app.eval.golden import GoldenQuery, GoldenSet
    from app.eval.metrics import aggregate, score_query

    spec = GoldenQuery(
        id="q_metrics",
        query=_TEST_MATERIALS["text_022"],
        expect="grounded",
        relevant_terms=("file_hash",),
        gold_sections=(_TEST_MATERIALS["text_037"],),
    )
    gold = {"h_gold_a", "h_gold_b"}
    evidence = [
        {"index": 1, "chunk_hash": "h_noise", "text": _TEST_MATERIALS["text_023"]},
        {"index": 2, "chunk_hash": "h_gold_a", "text": _TEST_MATERIALS["text_024"]},
        {"index": 3, "chunk_hash": "h_noise2", "text": _TEST_MATERIALS["text_025"]},
    ]
    hit = score_query(spec, trace_id="t1", evidence=evidence, gold_hashes=gold)
    assert_ok(hit.gold_count == 2, f"gold_count {hit.gold_count}")
    assert_ok(hit.gold_hit_count == 1, f"gold_hit_count {hit.gold_hit_count}")
    assert_ok(hit.recall_at_k == 0.5, f"recall_at_k {hit.recall_at_k}")
    assert_ok(hit.first_hit_rank == 2, f"first_hit_rank {hit.first_hit_rank}")
    assert_ok(hit.reciprocal_rank == 0.5, f"reciprocal_rank {hit.reciprocal_rank}")
    assert_ok(hit.retrieved_count == 3, f"retrieved_count {hit.retrieved_count}")

    miss = score_query(
        spec,
        trace_id="t2",
        evidence=[{"index": 1, "chunk_hash": "h_noise", "text": _TEST_MATERIALS["text_041"]}],
        gold_hashes=gold,
    )
    assert_ok(miss.recall_at_k == 0.0 and miss.reciprocal_rank == 0.0, f"miss score {miss}")

    refusal = GoldenQuery(id="q_ref", query=_TEST_MATERIALS["text_026"], expect="refusal", relevant_terms=())
    unscored = score_query(refusal, trace_id="t3", answer=_TEST_MATERIALS["text_027"])
    assert_ok(unscored.recall_at_k is None, _TEST_MATERIALS["text_012"])

    report = aggregate(GoldenSet(name="unit", corpus="", queries=(spec, refusal)), [hit, miss, unscored])
    assert_ok(report.retrieval_scored_count == 2, f"retrieval_scored_count {report.retrieval_scored_count}")
    assert_ok(report.avg_recall_at_k == 0.25, f"avg_recall_at_k {report.avg_recall_at_k}")
    assert_ok(report.mrr == 0.25, f"mrr {report.mrr}")
    assert_ok(report.gold_labeled_count == 2, f"gold_labeled_count {report.gold_labeled_count}")
    print(_TEST_MATERIALS["text_013"])


def _ensure_eval_corpus() -> None:
    from app.db.session import SessionLocal, init_db
    from app.eval.golden import corpus_filenames, load_golden
    from app.models import Document

    golden = load_golden(GOLDEN)
    names = set(corpus_filenames(golden.corpus))
    assert_ok(bool(names), _TEST_MATERIALS["text_014"])
    init_db()
    with SessionLocal() as db:
        have = {row[0] for row in db.query(Document.filename).all()}
    missing = [name for name in sorted(names) if name not in have]
    if not missing:
        print(f"eval corpus present: {sorted(names)}")
        return
    for name in missing:
        path = RAW / name
        assert_ok(path.is_file(), f"eval corpus missing {name}{_TEST_MATERIALS['text_009']}")
        ingest = post_file(name, path.read_bytes(), "text/markdown", **_RETRY)
        assert_ok(ingest.get("chunk_count", 0) >= 1, f"ingest failed for {name}")
        print(f"ingest ok {name} chunks={ingest['chunk_count']}")


def _ensure_api() -> None:
    health = get_json(f"{API_BASE}/health", **_RETRY)
    assert_ok(health.get("status") == "ok", f"health not ok: {health}")
    root = get_json(f"{API_BASE}/", **_RETRY)
    assert_ok("eval_gold" in root, _TEST_MATERIALS["text_015"])
    version = version_tuple(root.get("version", ""))
    assert_ok(version >= (0, 10), f"expected >= 0.10.0, got {root.get('version')}")
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
                assert_ok(stats.embedded + stats.copied > 0, _TEST_MATERIALS["text_042"])
                print(f"embed ok embedded={stats.embedded} copied={stats.copied}")
            else:
                print(_TEST_MATERIALS["text_043"])


def _check_gold_endpoint() -> list[dict]:
    data = get_json(f"{API_BASE}/eval/gold", **_RETRY)
    assert_ok(data.get("corpus_chunk_count", 0) > 0, f"{_TEST_MATERIALS['text_005']}{data}")
    labeled = [item for item in data["results"] if item["gold_sections"] or item["gold_chunk_hashes"]]
    assert_ok(bool(labeled), _TEST_MATERIALS["text_016"])
    for item in labeled:
        assert_ok(not item["missing_labels"], f"{item['eval_id']} has unresolved labels: {item['missing_labels']}")
        assert_ok(item["gold_count"] >= 1, f"{item['eval_id']}{_TEST_MATERIALS['text_010']}")
        for chunk in item["chunks"]:
            assert_ok(len(chunk["chunk_hash"]) == 64, f"bad chunk_hash: {chunk}")
    print(
        f"/eval/gold ok chunks={data['corpus_chunk_count']} labeled={data['labeled_count']} "
        f"sections={len(data['sections'])}"
    )
    return labeled


def _run_labeled_queries(labeled: list[dict]) -> None:
    picked = labeled[:2]
    for item in picked:
        result = post_json(
            f"{API_BASE}/query",
            {
                "query": item["query"],
                "top_k": 6,
                "verify": True,
                "persist": True,
                "eval_id": item["eval_id"],
            },
            **_RETRY,
        )
        assert_ok(result.get("trace_id"), f"{item['eval_id']} missing trace_id")
        evidence = result.get("evidence") or []
        assert_ok(bool(evidence), f"{item['eval_id']} retrieved no evidence")
        assert_ok(
            all(len(e.get("chunk_hash", "")) == 64 for e in evidence),
            f"{item['eval_id']} evidence missing chunk_hash: {evidence[:1]}",
        )
        print(f"ran {item['eval_id']} trace_id={result['trace_id']} evidence={len(evidence)}")


def _check_summary() -> None:
    summary = get_json(f"{API_BASE}/eval/summary", **_RETRY)
    for key in ("gold_labeled_count", "retrieval_scored_count", "avg_recall_at_k", "mrr"):
        assert_ok(key in summary, f"/eval/summary missing {key}: {list(summary)}")
    assert_ok(summary["gold_labeled_count"] >= 1, f"{_TEST_MATERIALS['text_006']}{summary}")
    assert_ok(summary["retrieval_scored_count"] >= 1, f"{_TEST_MATERIALS['text_007']}{summary}")
    assert_ok(summary["avg_recall_at_k"] is not None, "avg_recall_at_k is null")
    assert_ok(summary["mrr"] is not None, "mrr is null")
    assert_ok(summary["avg_recall_at_k"] > 0, f"avg_recall_at_k should be > 0: {summary['avg_recall_at_k']}")

    scored = [item for item in summary["results"] if item["recall_at_k"] is not None]
    for item in scored:
        assert_ok(
            item["gold_hit_count"] <= item["gold_count"],
            f"{item['eval_id']} hit_count > gold_count: {item}",
        )
        if item["first_hit_rank"]:
            assert_ok(
                item["first_hit_rank"] <= item["retrieved_count"],
                f"{item['eval_id']} first_hit_rank out of range: {item}",
            )
    print(
        f"{_TEST_MATERIALS['text_008']}{summary['avg_recall_at_k']} mrr={summary['mrr']} scored={summary['retrieval_scored_count']}/{summary['gold_labeled_count']}"
    )


def main() -> None:
    _check_gold_labels_offline()
    _check_metrics_unit()
    if os.environ.get("SKIP_EVAL_LLM") == "1":
        print(_TEST_MATERIALS["text_028"])
        print(_TEST_MATERIALS["text_029"])
        return
    _ensure_api()
    labeled = _check_gold_endpoint()
    _run_labeled_queries(labeled)
    _check_summary()
    print(_TEST_MATERIALS["text_017"])


if __name__ == "__main__":
    main()
