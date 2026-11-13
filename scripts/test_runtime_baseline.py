"""Phase 2D: stage timings, queue metrics, and one owner-filtered vector plan.

Prints durations and planner nodes only. Document text, answers, and secrets stay out.
"""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sqlalchemy import text

from app.agent.lifecycle import claim_queued_run
from app.agent.service import create_agent_run
from app.api.metrics import _agent_runtime
from app.db.session import SessionLocal
from app.models import AgentRun
from app.queue import client as queue
from app.search.hybrid import hybrid_search
from app.search.stages import reset_stages, snapshot_stages

TOKEN = uuid.uuid4().hex[:12]
QUERY = _TEST_MATERIALS["text_002"]


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
        row = db.execute(text("SELECT id FROM users ORDER BY created_at LIMIT 1")).first()
    if row is None:
        raise SystemExit("no user")
    return row[0]


def _cleanup(db, user_id) -> None:
    db.execute(
        text("DELETE FROM agent_runs WHERE user_id = CAST(:u AS uuid) AND goal LIKE :g"),
        {"u": str(user_id), "g": f"%{TOKEN}%"},
    )
    db.commit()


def test_claim_records_queue_wait(user_id) -> None:
    with SessionLocal() as db:
        row = create_agent_run(db, f"{TOKEN} metrics", owner_id=user_id, session_id=None)
        run_id = row.id
        created = row.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        db.execute(
            text("UPDATE agent_runs SET created_at = :ts WHERE id = CAST(:id AS uuid)"),
            {"ts": datetime.now(timezone.utc) - timedelta(seconds=2), "id": str(run_id)},
        )
        db.commit()
        assert claim_queued_run(run_id) is True
    with SessionLocal() as db:
        run = db.get(AgentRun, run_id)
        metrics = (run.outline or {}).get("_metrics") or {}
        assert isinstance(metrics.get("queue_ms"), int) and metrics["queue_ms"] >= 1000, metrics
        db.delete(run)
        db.commit()
    print(f"PASS queue_ms={metrics['queue_ms']}")


def test_retrieval_stages(user_id) -> dict:
    with SessionLocal() as db:
        hybrid_search(db, QUERY, top_k=4, owner_id=user_id)
    reset_stages()
    with SessionLocal() as db:
        hits = hybrid_search(db, QUERY, top_k=4, owner_id=user_id)
    stages = snapshot_stages()
    for name in ("vector_ms", "bm25_ms", "rrf_ms"):
        assert isinstance(stages.get(name), int), stages
    print(f"PASS stages={stages} hits={len(hits)}")
    return stages


def test_vector_plan(user_id) -> None:
    from app.embed.model import encode_query

    vector = encode_query(QUERY)
    literal = "[" + ",".join(f"{value:.8f}" for value in vector) + "]"
    sql = text(_TEST_MATERIALS["text_003"])
    with SessionLocal() as db:
        lines = [row[0] for row in db.execute(sql, {"owner": str(user_id), "vector": literal}).all()]
    joined = "\n".join(lines)
    assert "Planning Time" in joined and "Execution Time" in joined
    interesting = [
        line.strip()
        for line in lines
        if any(
            word in line
            for word in ("Index", "Seq Scan", "Bitmap", "Limit", "Buffers", "Planning Time", "Execution Time")
        )
    ]
    print("PASS vector plan")
    for line in interesting:
        print("  " + line)


def test_runtime_counts(user_id) -> None:
    with SessionLocal() as db:
        payload = _agent_runtime(db, owner_id=user_id)
    for key in ("by_status", "queued", "running", "stale_running", "queued_jobs", "processing_jobs"):
        assert key in payload, payload
    print(
        "PASS runtime "
        f"queued={payload['queued']} running={payload['running']} "
        f"stale={payload['stale_running']} jobs={payload['queued_jobs']}/{payload['processing_jobs']}"
    )


def test_query_timings(user_id) -> None:
    from app.api.query import QueryRequest, query
    from app.core.config import get_settings
    from app.models import User

    settings = get_settings()
    with SessionLocal() as db:
        user = db.get(User, user_id)
        first = query(
            QueryRequest(query=QUERY, top_k=4, verify=True, persist=False),
            db=db,
            settings=settings,
            user=user,
            _=None,
        )
        second = query(
            QueryRequest(query=QUERY, top_k=4, verify=True, persist=False),
            db=db,
            settings=settings,
            user=user,
            _=None,
        )
    timings = dict(second.timings_ms)
    for name in ("vector_ms", "bm25_ms", "rrf_ms", "rerank_ms", "generate_ms", "verify_ms"):
        assert isinstance(timings.get(name), int), timings
    print(f"PASS query cold={dict(first.timings_ms)}")
    print(
        f"{_TEST_MATERIALS['text_001']}{second.provider} evidence={len(second.evidence)} verified={second.verification is not None} timings={timings}"
    )


def main() -> int:
    queue.QUEUE_KEY = f"provident:test:{TOKEN}:jobs"
    queue.PROCESSING_KEY = f"provident:test:{TOKEN}:processing"
    with SessionLocal() as db:
        user_id = _user(db)
        _cleanup(db, user_id)
    try:
        test_claim_records_queue_wait(user_id)
        test_retrieval_stages(user_id)
        test_vector_plan(user_id)
        test_query_timings(user_id)
        test_runtime_counts(user_id)
    finally:
        with SessionLocal() as db:
            _cleanup(db, user_id)
        client = queue._redis()
        if client is not None:
            client.delete(queue.QUEUE_KEY, queue.PROCESSING_KEY)
    print("baseline checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
