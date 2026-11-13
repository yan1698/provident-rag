from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.api.verify import VerificationOut
from app.auth.deps import get_current_user, owner_id as current_owner
from app.db.session import get_db
from app.models.trace import QueryTrace
from app.models.user import User
from app.tag.service import set_trace_tags
from app.trace.service import delete_trace, get_trace, list_traces
from app.generate.general import AnswerSource, source_mode

router = APIRouter(tags=["traces"])


class TagBriefOut(BaseModel):
    id: uuid.UUID
    name: str
    color: str = ""


class TraceSummaryOut(BaseModel):
    answer_source: AnswerSource = "document_only"
    source_trace_id: str = ""
    id: uuid.UUID
    query: str
    provider: str
    llm_model: str
    eval_id: str | None = None
    top_k: int | None
    coverage_rate: float | None
    hallucination_rate: float | None
    claim_count: int | None
    refused: bool = False
    tags: list[TagBriefOut] = Field(default_factory=list)
    created_at: datetime


class TraceOut(TraceSummaryOut):
    answer: str
    citations: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    verification: VerificationOut | None
    usage: list[dict[str, Any]] | None = None
    diagnostics: dict[str, Any] | None = None
    citation_integrity: list[dict[str, Any]] | None = None
    source_checked_at: datetime | None = None
    supported_count: int | None
    hallucination_count: int | None


class TraceListOut(BaseModel):
    count: int
    limit: int
    offset: int
    results: list[TraceSummaryOut]


class TraceDeleteOut(BaseModel):
    id: uuid.UUID
    query: str
    deleted: bool = True


class TraceTagsIn(BaseModel):
    tag_ids: list[uuid.UUID] = Field(default_factory=list)


class TraceTagsOut(BaseModel):
    id: uuid.UUID
    tags: list[TagBriefOut]


def _is_refused(row: QueryTrace) -> bool:
    if "现有证据不足以回答" in (row.answer or ""):
        return True
    claims = (row.verification or {}).get("claims") or []
    return any(isinstance(item, dict) and item.get("status") == "insufficient" for item in claims)


def _tag_briefs(row: QueryTrace) -> list[TagBriefOut]:
    tags = sorted(row.tags or [], key=lambda item: item.name.lower())
    return [TagBriefOut(id=tag.id, name=tag.name, color=tag.color or "") for tag in tags]


def to_summary(row: QueryTrace) -> TraceSummaryOut:
    return TraceSummaryOut(
        answer_source=source_mode(row),
        source_trace_id=str((row.diagnostics or {}).get("source_trace_id") or ""),
        id=row.id,
        query=row.query,
        provider=row.provider,
        llm_model=row.llm_model,
        eval_id=row.eval_id,
        top_k=row.top_k,
        coverage_rate=row.coverage_rate,
        hallucination_rate=row.hallucination_rate,
        claim_count=row.claim_count,
        refused=_is_refused(row),
        tags=_tag_briefs(row),
        created_at=row.created_at,
    )


def to_detail(row: QueryTrace) -> TraceOut:
    verification = None
    if row.verification is not None:
        verification = VerificationOut.model_validate(row.verification)
    return TraceOut(
        **to_summary(row).model_dump(),
        answer=row.answer,
        citations=list(row.citations or []),
        evidence=list(row.evidence or []),
        verification=verification,
        usage=list(row.usage) if isinstance(row.usage, list) else None,
        diagnostics=row.diagnostics if isinstance(row.diagnostics, dict) else None,
        supported_count=row.supported_count,
        hallucination_count=row.hallucination_count,
    )


@router.get("/traces", response_model=TraceListOut)
def traces_list(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    unique: bool = Query(
        default=True,
        description="Keep only the newest trace for each identical question text",
    ),
    tag_id: uuid.UUID | None = Query(default=None, description="Only traces that have this tag"),
    untagged: bool = Query(default=False, description="Only traces with no tags"),
) -> TraceListOut:
    try:
        rows = list_traces(
            db,
            limit=limit,
            offset=offset,
            unique=unique,
            tag_id=tag_id,
            untagged=untagged,
            owner_id=current_owner(user),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TraceListOut(
        count=len(rows),
        limit=limit,
        offset=offset,
        results=[to_summary(row) for row in rows],
    )


@router.get("/traces/{trace_id}", response_model=TraceOut)
def traces_get(
    trace_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> TraceOut:
    row = get_trace(db, trace_id, owner_id=current_owner(user))
    if row is None:
        raise HTTPException(status_code=404, detail=f"trace {trace_id} not found")
    # Authorize and materialize the saved answer before querying current sources.
    # diagnostics.citation_integrity remains the immutable generation-time snapshot.
    payload = to_detail(row)
    if source_mode(row) == "general_explanation":
        return payload
    from app.verify.structure import assess_snapshot

    try:
        payload.citation_integrity = assess_snapshot(db, payload.evidence, owner_id=current_owner(user))
    except SQLAlchemyError:
        db.rollback()
        payload.citation_integrity = [
            {"index": item.get("index"), "chunk_id": item.get("chunk_id", ""), "status": "unavailable"}
            for item in payload.evidence
        ]
    payload.source_checked_at = datetime.now(timezone.utc)
    return payload


@router.put("/traces/{trace_id}/tags", response_model=TraceTagsOut)
def traces_set_tags(
    trace_id: uuid.UUID,
    body: TraceTagsIn,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> TraceTagsOut:
    try:
        tags = set_trace_tags(db, trace_id=trace_id, tag_ids=list(body.tag_ids), owner_id=current_owner(user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TraceTagsOut(
        id=trace_id,
        tags=[
            TagBriefOut(id=tag.id, name=tag.name, color=tag.color or "")
            for tag in sorted(tags, key=lambda item: item.name.lower())
        ],
    )


@router.delete("/traces/{trace_id}", response_model=TraceDeleteOut)
def traces_delete(
    trace_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> TraceDeleteOut:
    row = delete_trace(db, trace_id, owner_id=current_owner(user))
    if row is None:
        raise HTTPException(status_code=404, detail=f"trace {trace_id} not found")
    return TraceDeleteOut(id=trace_id, query=row.query, deleted=True)
