"""Batch-run the golden query set through /query and print /eval/report（含 Recall@k / MRR）。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lib.auth import resolve_bearer_token
from lib.golden import DEFAULT_EVAL_SET, load_golden_json
from lib.http import API_BASE, get_json, post_json

LAST_RUN = ROOT / "data" / "eval" / "last_run.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden queries against /query")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N queries (0 = all)")
    parser.add_argument("--ids", default="", help="Comma-separated eval ids to run")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--sleep", type=float, default=21.0, help="Delay between queries (Kimi 免费额度约 3 RPM)")
    parser.add_argument("--retries", type=int, default=3, help="Retries when the provider rate-limits")
    parser.add_argument("--token", default="", help="Bearer token (or set PROVIDENT_TOKEN)")
    parser.add_argument(
        "--username",
        default=os.environ.get("EVAL_GATE_USERNAME", ""),
        help="Developer account; alternatively set EVAL_GATE_USERNAME",
    )
    parser.add_argument("--password", default="", help="Login password (or set PROVIDENT_PASSWORD)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--set", dest="eval_set", default=DEFAULT_EVAL_SET, help="Eval set name (default: golden-agent-notes-v1)"
    )
    args = parser.parse_args()

    status = get_json(f"{API_BASE}/auth/status")
    auth_enabled = bool(status.get("auth_enabled"))
    token = (
        ""
        if args.dry_run
        else resolve_bearer_token(
            token=args.token,
            username=args.username,
            password=args.password,
            auth_enabled=auth_enabled,
            default_username=os.environ.get("EVAL_GATE_USERNAME", ""),
        )
    )

    eval_set, all_queries = load_golden_json(root=ROOT, name=args.eval_set)
    queries = list(all_queries)
    wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
    if wanted:
        queries = [item for item in queries if item["id"] in wanted]
        missing_ids = wanted - {item["id"] for item in queries}
        if missing_ids:
            raise SystemExit(f"unknown eval ids: {sorted(missing_ids)}")
    if args.limit > 0:
        queries = queries[: args.limit]

    print(f"eval_set={eval_set} selected={len(queries)}/{len(all_queries)}")
    if args.dry_run:
        for item in queries:
            print(f"  {item['id']}\t{item.get('expect', '')}\t{item.get('query', '')}")
        return

    runs = []
    for position, item in enumerate(queries):
        if position and args.sleep > 0:
            time.sleep(args.sleep)
        print(f"query {item['id']} ...", flush=True)
        result = post_json(
            f"{API_BASE}/query",
            {
                "query": item["query"],
                "top_k": args.top_k,
                "verify": True,
                "persist": True,
                "eval_id": item["id"],
            },
            token=token,
            retries=args.retries,
        )
        trace_id = result.get("trace_id")
        coverage = (result.get("verification") or {}).get("coverage_rate")
        print(f"  trace_id={trace_id} provider={result.get('provider')} coverage={coverage}")
        runs.append({"eval_id": item["id"], "trace_id": trace_id, "query": item["query"]})

    LAST_RUN.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN.write_text(
        json.dumps({"eval_set": eval_set, "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {LAST_RUN}")

    report = get_json(f"{API_BASE}/eval/report?eval_set={eval_set}", token=token)
    metrics = report.get("metrics") or {}
    results = report.get("results") or []
    print(
        "report "
        f"traced={metrics.get('traced_count')}/{metrics.get('query_count')} "
        f"coverage={metrics.get('avg_coverage_rate')} "
        f"hallucination={metrics.get('avg_hallucination_rate')} "
        f"term_hit={metrics.get('term_hit_rate')} "
        f"refusal_ok={metrics.get('refusal_ok_rate')} "
        f"passed={report.get('passed')}"
    )
    print(
        "retrieval "
        f"scored={metrics.get('retrieval_scored_count')}/{metrics.get('gold_labeled_count')} "
        f"recall@k={metrics.get('avg_recall_at_k')} "
        f"mrr={metrics.get('mrr')}"
    )
    for item in results:
        if item.get("recall_at_k") is None:
            continue
        print(
            f"  {item['eval_id']}\trecall={item['recall_at_k']}"
            f"\thit={item['gold_hit_count']}/{item['gold_count']}"
            f"\trank={item['first_hit_rank']}\tk={item['retrieved_count']}"
        )
    unresolved = [item["eval_id"] for item in results if item.get("gold_missing_labels")]
    if unresolved:
        print(f"gold labels unresolved: {unresolved}（用 scripts/label_gold.py 核对）")
    missing = metrics.get("missing") or []
    if missing:
        print(f"missing traces: {missing}（重跑: python scripts/run_eval.py --ids <eval_id>）")


if __name__ == "__main__":
    main()
