from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.config import Settings
from app.models.tag import query_trace_tags
from app.models.trace import QueryTrace


def llm_model_name(settings: Settings) -> str:
    from app.llm import answer_model_name

    return answer_model_name(settings)


def save_trace(
    db: Session,
    *,
    query: str,
    answer: str,
    provider: str,
    llm_model: str,
    top_k: int | None,
    citations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    verification: dict[str, Any] | None,
    usage: list[dict[str, Any]] | None = None,
    eval_id: str | None = None,
    user_id: uuid.UUID | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> QueryTrace:
    row = QueryTrace(
        user_id=user_id,
        query=query,
        answer=answer,
        provider=provider,
        llm_model=llm_model,
        eval_id=eval_id,
        top_k=top_k,
        citations=citations,
        evidence=evidence,
        verification=verification,
        usage=list(usage or []),
        diagnostics=diagnostics,
        coverage_rate=None if verification is None else verification.get("coverage_rate"),
        hallucination_rate=None if verification is None else verification.get("hallucination_rate"),
        claim_count=None if verification is None else verification.get("claim_count"),
        supported_count=None if verification is None else verification.get("supported_count"),
        hallucination_count=None if verification is None else verification.get("hallucination_count"),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_trace(db: Session, trace_id: uuid.UUID, *, owner_id: uuid.UUID | None = None) -> QueryTrace | None:
    row = db.get(QueryTrace, trace_id)
    if row is None:
        return None
    if owner_id is not None and row.user_id != owner_id:
        return None
    return row


def delete_trace(db: Session, trace_id: uuid.UUID, *, owner_id: uuid.UUID | None = None) -> QueryTrace | None:
    """Delete a trace and any feedback rows pointing at it. Returns the deleted row or None."""
    from sqlalchemy import delete

    from app.models.feedback import Feedback

    row = get_trace(db, trace_id, owner_id=owner_id)
    if row is None:
        return None
    db.execute(delete(Feedback).where(Feedback.trace_id == trace_id))
    db.delete(row)
    db.commit()
    return row


def list_traces(
    db: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    unique: bool = False,
    tag_id: uuid.UUID | None = None,
    untagged: bool = False,
    owner_id: uuid.UUID | None = None,
) -> list[QueryTrace]:
    """List recent traces. ``unique=True`` keeps only the newest row per trimmed query."""
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    if tag_id is not None and untagged:
        raise ValueError("tag_id and untagged cannot be combined")

    def apply_filters(stmt):
        if owner_id is not None:
            stmt = stmt.where(QueryTrace.user_id == owner_id)
        if tag_id is not None:
            return stmt.join(query_trace_tags, QueryTrace.id == query_trace_tags.c.trace_id).where(
                query_trace_tags.c.tag_id == tag_id
            )
        if untagged:
            tagged = select(query_trace_tags.c.trace_id)
            return stmt.where(QueryTrace.id.notin_(tagged))
        return stmt

    if not unique:
        stmt = select(QueryTrace).options(selectinload(QueryTrace.tags))
        stmt = apply_filters(stmt)
        stmt = stmt.order_by(QueryTrace.created_at.desc()).offset(offset).limit(limit)
        return list(db.execute(stmt).scalars().unique().all())

    ranked = select(
        QueryTrace.id.label("id"),
        func.row_number()
        .over(
            partition_by=(
                func.trim(QueryTrace.query),
                func.coalesce(QueryTrace.diagnostics["answer_source"].as_string(), "document_only"),
            ),
            order_by=QueryTrace.created_at.desc(),
        )
        .label("rn"),
    )
    if owner_id is not None:
        ranked = ranked.where(QueryTrace.user_id == owner_id)
    ranked = ranked.subquery()

    stmt = (
        select(QueryTrace)
        .options(selectinload(QueryTrace.tags))
        .join(ranked, QueryTrace.id == ranked.c.id)
        .where(ranked.c.rn == 1)
    )
    stmt = apply_filters(stmt)
    stmt = stmt.order_by(QueryTrace.created_at.desc()).offset(offset).limit(limit)
    return list(db.execute(stmt).scalars().unique().all())
