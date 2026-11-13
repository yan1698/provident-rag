"""Day 11: freeze a one-shot eval report from stored traces. No generator LLM.

python scripts/report_eval.py
python scripts/report_eval.py --reverify
python scripts/report_eval.py --baseline data/eval/baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

REPORT = ROOT / "data" / "eval" / "report.json"
BASELINE = ROOT / "data" / "eval" / "baseline.json"


def _print_report(snapshot: dict) -> None:
    metrics = snapshot.get("metrics") or {}
    corpus = snapshot.get("corpus") or {}
    traces = snapshot.get("traces") or {}
    print(
        f"report passed={snapshot.get('passed')} "
        f"coverage={metrics.get('avg_coverage_rate')} "
        f"hallucination={metrics.get('avg_hallucination_rate')} "
        f"recall@k={metrics.get('avg_recall_at_k')} "
        f"mrr={metrics.get('mrr')} "
        f"term_hit={metrics.get('term_hit_rate')} "
        f"refusal_ok={metrics.get('refusal_ok_rate')} "
        f"off_corpus_evidence={metrics.get('avg_off_corpus_evidence_rate')}"
    )
    print(
        f"corpus path={corpus.get('path')} chunks={corpus.get('chunk_count')} "
        f"off_corpus_docs={corpus.get('off_corpus_count')} "
        f"traces={len(traces.get('trace_ids') or [])} source={traces.get('source')}"
    )
    for item in corpus.get("documents") or []:
        flag = "keep" if item.get("expected") else "extra"
        print(
            f"  {flag} {item.get('filename')} hash={str(item.get('file_hash') or '')[:12]} chunks={item.get('chunk_count')}"
        )
    fixtures = corpus.get("forbidden_documents") or []
    if fixtures:
        print(f"  !! 验收夹具入库: {fixtures}（跑 scripts/reset_corpus.py 清掉）")
    failures = snapshot.get("gate_failures") or []
    if failures:
        print(f"gate failures: {failures}")
        for gate in snapshot.get("gates") or []:
            if not gate.get("passed"):
                print(f"  {gate['name']} {gate['op']} {gate['threshold']} actual={gate['actual']}")
    else:
        print("gates passed")


def _print_compare(payload: dict) -> None:
    comparison = payload["comparison"]
    print(
        f"compare vs {payload['baseline']['name']} "
        f"scored={payload['baseline']['scored']} "
        f"corpus_match={comparison.get('corpus_match')} "
        f"eval_set_match={comparison.get('eval_set_match')}"
    )
    for key, delta in (comparison.get("delta") or {}).items():
        print(
            f"  {key:24} ours={comparison['ours'].get(key)} "
            f"baseline={comparison['baseline'].get(key)} "
            f"delta={delta} winner={comparison['winner'].get(key)}"
        )
    for note in comparison.get("notes") or []:
        print(f"  !! {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze golden traces into data/eval/report.json")
    parser.add_argument("--reverify", action="store_true", help="Re-run local Cross-Encoder Verifier (still no LLM)")
    parser.add_argument(
        "--baseline",
        default="",
        help="Baseline JSON to score/compare (default: data/eval/baseline.json if it has data)",
    )
    parser.add_argument("--out", default=str(REPORT), help="Where to write the frozen report")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--no-fail", action="store_true", help="Do not exit 1 when quality gates fail")
    parser.add_argument("--set", dest="eval_set", default="", help="Eval set name (default: golden-agent-notes-v1)")
    args = parser.parse_args()

    from app.db.session import SessionLocal, init_db
    from app.eval.baseline import (
        compare_metrics,
        load_baseline,
        precomputed_metrics,
        report_metrics,
        score_baseline_from_db,
    )
    from app.eval.golden import DEFAULT_EVAL_SET, load_golden
    from app.eval.replay import replay_snapshot
    from app.eval.snapshot import corpus_snapshot

    init_db()
    golden = load_golden(name=args.eval_set or DEFAULT_EVAL_SET)
    with SessionLocal() as db:
        snapshot = replay_snapshot(db, golden, reverify=args.reverify)
        _print_report(snapshot)

        missing = (snapshot.get("metrics") or {}).get("missing") or []
        if missing:
            print(f"missing traces: {missing}（先跑 python scripts/run_eval.py）")

        out = Path(args.out)
        if not args.no_write:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"wrote {out}")

        baseline_path = Path(args.baseline) if args.baseline else BASELINE
        should_compare = bool(args.baseline)
        if not args.baseline and baseline_path.is_file():
            raw = json.loads(baseline_path.read_text(encoding="utf-8"))
            should_compare = bool(raw.get("results")) or any(
                value is not None for value in (raw.get("metrics") or {}).values()
            )
        if should_compare:
            if not baseline_path.is_file():
                raise SystemExit(f"baseline not found: {baseline_path}")
            baseline = load_baseline(baseline_path)
            if baseline.records:
                theirs = score_baseline_from_db(db, golden, baseline, reverify=args.reverify)
                theirs_metrics = report_metrics(theirs)
                scored = True
            else:
                theirs_metrics = precomputed_metrics(baseline)
                theirs_metrics.setdefault("eval_set", baseline.eval_set)
                scored = False
            comparison = compare_metrics(
                {**snapshot["metrics"], "eval_set": snapshot["eval_set"]},
                theirs_metrics,
                ours_corpus=corpus_snapshot(db, golden),
                baseline=baseline,
            )
            payload = {
                "ours": {
                    "eval_set": snapshot["eval_set"],
                    "metrics": snapshot["metrics"],
                    "passed": snapshot["passed"],
                },
                "baseline": {
                    "system": baseline.system,
                    "name": baseline.name,
                    "eval_set": baseline.eval_set,
                    "metrics": theirs_metrics,
                    "scored": scored,
                },
                "comparison": comparison,
            }
            _print_compare(payload)

    if not snapshot.get("passed") and not args.no_fail:
        raise SystemExit("quality gates failed")


if __name__ == "__main__":
    main()
