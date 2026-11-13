"""Warmed single-request profile, then 1/2/4-way overlap.

Prints durations and structured metadata only. No query text, answers, or evidence.
"""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import math
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sqlalchemy import text

from app.api.query import QueryRequest, query
from app.core.config import get_settings
from app.db.session import SessionLocal, engine
from app.models import User

QUERY = _TEST_MATERIALS["text_001"]
RUNS = 20
STAGES = (
    "scope_ms",
    "embed_ms",
    "ann_ms",
    "vector_ms",
    "bm25_ms",
    "rrf_ms",
    "rerank_wait_ms",
    "rerank_ms",
    "generate_ms",
    "verify_wait_ms",
    "verify_ms",
)


def _user(db) -> uuid.UUID:
    row = db.execute(
        text(
            """
            SELECT d.owner_id
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            WHERE c.embedding IS NOT NULL AND d.owner_id IS NOT NULL
            GROUP BY d.owner_id
            ORDER BY count(*) DESC
            LIMIT 1
            """
        )
    ).first()
    if row is None:
        raise SystemExit("no embedded chunks")
    return row[0]


def _percentile(values: list[int], pct: float) -> int:
    ordered = sorted(values)
    rank = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def _once(user_id: uuid.UUID) -> dict:
    settings = get_settings()
    started = time.perf_counter()
    try:
        with SessionLocal() as db:
            user = db.get(User, user_id)
            response = query(
                QueryRequest(query=QUERY, top_k=4, verify=True, persist=False),
                db=db,
                settings=settings,
                user=user,
                _=None,
            )
    except Exception as exc:  # noqa: BLE001 — profile records the failure class only
        text = str(exc)
        detail = "rate_limited" if "429" in text or "rate limit" in text.lower() else type(exc).__name__
        return {
            "error": type(exc).__name__,
            "detail": detail,
            "wall_ms": int((time.perf_counter() - started) * 1000),
            "timings": {},
            "meta": {},
        }
    return {
        "error": None,
        "wall_ms": int((time.perf_counter() - started) * 1000),
        "timings": dict(response.timings_ms),
        "meta": dict(response.timing_meta),
    }


def _summarize(rows: list[dict]) -> None:
    ok = [row for row in rows if row["error"] is None]
    failed = [row for row in rows if row["error"]]
    print(f"samples={len(rows)} ok={len(ok)} errors={len(failed)}", flush=True)
    for row in failed:
        print(f"  ERR {row['error']} {row.get('detail', '')}", flush=True)
    if not ok:
        return
    for name in ("wall_ms", *STAGES):
        values = [int(row["wall_ms"] if name == "wall_ms" else row["timings"].get(name, 0)) for row in ok]
        print(
            f"  {name} median={_percentile(values, 50)} p95={_percentile(values, 95)} "
            f"min={min(values)} max={max(values)}"
        )
    meta = ok[-1]["meta"]
    prompt_tokens = [int(row["meta"].get("prompt_tokens") or 0) for row in ok]
    completion_tokens = [int(row["meta"].get("completion_tokens") or 0) for row in ok]
    inflight = [int(row["meta"].get("rerank_inflight") or 0) for row in ok]
    print(
        "  meta "
        f"device={meta.get('rerank_device')} cuda={meta.get('rerank_cuda')} "
        f"model={meta.get('rerank_model')} batch={meta.get('rerank_batch_size')} "
        f"candidates={meta.get('rerank_candidates')} pairs={meta.get('rerank_pairs')} "
        f"inflight_max={max(inflight)} "
        f"provider={meta.get('llm_provider')} llm={meta.get('llm_model')} "
        f"prompt_chars={meta.get('generate_prompt_chars')} "
        f"prompt_tokens={min(prompt_tokens)}-{max(prompt_tokens)} "
        f"completion_tokens={min(completion_tokens)}-{max(completion_tokens)} "
        f"verify_pairs={meta.get('verify_pairs')}"
    )


def _concurrent(user_id: uuid.UUID, width: int) -> None:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=width) as pool:
        rows = list(pool.map(lambda _item: _once(user_id), range(width)))
    wall = int((time.perf_counter() - started) * 1000)
    errors = [row["error"] for row in rows if row["error"]]
    pool_state = engine.pool
    print(
        f"CONCURRENCY {width} batch_ms={wall} errors={errors or 0} "
        f"pool checked_out={pool_state.checkedout()} checked_in={pool_state.checkedin()}"
    )
    _summarize(rows)


def main() -> int:
    with SessionLocal() as db:
        user_id = _user(db)
    warm = _once(user_id)
    if warm["error"]:
        print(f"FAIL warmup {warm['error']}")
        return 1
    for key in ("rerank_device", "rerank_candidates", "rerank_model", "prompt_tokens", "completion_tokens"):
        if key not in warm["meta"]:
            print(f"FAIL warmup missing {key} meta={warm['meta']}")
            return 1
    print("PASS warmup", flush=True)
    rows = [_once(user_id) for _ in range(RUNS)]
    print(f"SINGLE sequential={RUNS}", flush=True)
    _summarize(rows)
    for width in (1, 2, 4):
        _concurrent(user_id, width)
    print("profile checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
