from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user, owner_id as current_owner, require_eval_gate_user
from app.cache.service import cache_stats
from app.core.version import API_VERSION
from app.db.session import get_db
from app.graph.service import graph_counts
from app.models import AgentRun, Chunk, Document, Feedback, LlmCall, QueryTrace
from app.models.user import User

router = APIRouter(tags=["metrics"])


def _trace_quality(db: Session, *, owner_id) -> dict[str, Any]:
    """Aggregate Verifier rates from persisted /query traces (daily Q&A, not golden eval)."""
    stmt = select(
        func.count(QueryTrace.id),
        func.count(QueryTrace.coverage_rate),
        func.avg(QueryTrace.coverage_rate),
        func.avg(QueryTrace.hallucination_rate),
    )
    if owner_id is not None:
        stmt = stmt.where(QueryTrace.user_id == owner_id)
    total, verified, avg_coverage, avg_hallucination = db.execute(stmt).one()
    return {
        "total": int(total or 0),
        "verified": int(verified or 0),
        "avg_coverage_rate": float(avg_coverage) if avg_coverage is not None else None,
        "avg_hallucination_rate": float(avg_hallucination) if avg_hallucination is not None else None,
    }


@router.get("/metrics")
def metrics(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> dict[str, Any]:
    uid = current_owner(user)
    doc_stmt = select(func.count(Document.id))
    chunk_stmt = select(func.count(Chunk.id))
    embedded_stmt = select(func.count(Chunk.id)).where(Chunk.embedded_at.is_not(None))
    trace_stmt = select(func.count(QueryTrace.id))
    agent_stmt = select(func.count(AgentRun.id))
    feedback_stmt = select(Feedback.rating, func.count(Feedback.id)).group_by(Feedback.rating)
    if uid is not None:
        doc_stmt = doc_stmt.where(Document.owner_id == uid)
        chunk_stmt = chunk_stmt.join(Document, Chunk.doc_id == Document.id).where(Document.owner_id == uid)
        embedded_stmt = embedded_stmt.join(Document, Chunk.doc_id == Document.id).where(Document.owner_id == uid)
        trace_stmt = trace_stmt.where(QueryTrace.user_id == uid)
        agent_stmt = agent_stmt.where(AgentRun.user_id == uid)
        feedback_stmt = feedback_stmt.where(Feedback.user_id == uid)
    documents = db.execute(doc_stmt).scalar_one()
    chunks = db.execute(chunk_stmt).scalar_one()
    embedded = db.execute(embedded_stmt).scalar_one()
    traces = db.execute(trace_stmt).scalar_one()
    agent_runs = db.execute(agent_stmt).scalar_one()
    feedback_rows = db.execute(feedback_stmt).all()
    feedback = {str(rating): int(count) for rating, count in feedback_rows}
    return {
        "api_version": API_VERSION,
        "documents": int(documents or 0),
        "chunks": int(chunks or 0),
        "embedded_chunks": int(embedded or 0),
        "traces": int(traces or 0),
        "agent_runs": int(agent_runs or 0),
        "feedback": {
            "up": feedback.get("up", 0),
            "down": feedback.get("down", 0),
            "count": sum(feedback.values()),
        },
        "cache": cache_stats(),
        "graph": graph_counts(db, owner_id=uid),
        "trace_quality": _trace_quality(db, owner_id=uid),
        "llm": _llm_rollup(db, owner_id=uid),
        "agent_runtime": _agent_runtime(db, owner_id=uid),
    }


def _agent_runtime(db: Session, *, owner_id) -> dict[str, Any]:
    """Queue and terminal counts. Durations only; no goals, answers, or document text."""
    from datetime import datetime, timedelta, timezone

    from app.core.config import get_settings
    from app.queue.client import PROCESSING_KEY, QUEUE_KEY

    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(int(settings.agent_running_stale_seconds), 1))
    stmt = select(AgentRun.status, func.count(AgentRun.id)).group_by(AgentRun.status)
    stale_stmt = select(func.count(AgentRun.id)).where(
        AgentRun.status == "running",
        (AgentRun.heartbeat_at.is_(None)) | (AgentRun.heartbeat_at < cutoff),
    )
    if owner_id is not None:
        stmt = stmt.where(AgentRun.user_id == owner_id)
        stale_stmt = stale_stmt.where(AgentRun.user_id == owner_id)
    by_status = {str(status): int(count or 0) for status, count in db.execute(stmt).all()}
    depth = {"queued_jobs": None, "processing_jobs": None}
    try:
        from app.queue.client import _redis

        client = _redis()
        if client is not None:
            depth = {
                "queued_jobs": int(client.llen(QUEUE_KEY)),
                "processing_jobs": int(client.llen(PROCESSING_KEY)),
            }
    except Exception:  # noqa: BLE001 — metrics stay numeric when Redis is down
        depth = {"queued_jobs": None, "processing_jobs": None}
    return {
        "by_status": by_status,
        "queued": int(by_status.get("queued") or 0),
        "running": int(by_status.get("running") or 0),
        "stale_running": int(db.execute(stale_stmt).scalar_one() or 0),
        **depth,
    }


def _llm_rollup(db: Session, *, owner_id, purpose: str | None = None) -> dict[str, Any]:
    stmt = select(
        func.count(LlmCall.id),
        func.coalesce(func.sum(LlmCall.prompt_tokens), 0),
        func.coalesce(func.sum(LlmCall.completion_tokens), 0),
        func.coalesce(func.sum(case((LlmCall.status == "error", 1), else_=0)), 0),
    )
    if owner_id is not None:
        stmt = stmt.where(LlmCall.user_id == owner_id)
    if purpose:
        stmt = stmt.where(LlmCall.purpose == purpose)
    calls, prompt, completion, errors = db.execute(stmt).one()
    by_purpose_stmt = select(
        LlmCall.purpose,
        func.count(LlmCall.id),
        func.coalesce(func.sum(LlmCall.prompt_tokens), 0),
        func.coalesce(func.sum(LlmCall.completion_tokens), 0),
        func.coalesce(func.sum(case((LlmCall.status == "error", 1), else_=0)), 0),
    ).group_by(LlmCall.purpose)
    if owner_id is not None:
        by_purpose_stmt = by_purpose_stmt.where(LlmCall.user_id == owner_id)
    by_purpose = {}
    for name, count, ptok, ctok, err in db.execute(by_purpose_stmt):
        by_purpose[str(name or "")] = {
            "calls": int(count or 0),
            "prompt_tokens": int(ptok or 0),
            "completion_tokens": int(ctok or 0),
            "errors": int(err or 0),
        }
    return {
        "calls": int(calls or 0),
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "errors": int(errors or 0),
        "by_purpose": by_purpose,
    }


@router.get("/metrics/llm")
def metrics_llm(
    db: Session = Depends(get_db),
    user: User | None = Depends(require_eval_gate_user),
    purpose: str | None = Query(default=None, max_length=32),
) -> dict[str, Any]:
    """Tenant-wide ledger for the maintainer. Regular /metrics stays per-user."""
    return _llm_rollup(db, owner_id=None, purpose=purpose)
