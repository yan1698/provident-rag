from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user, owner_id as current_owner
from app.db.session import get_db
from app.models import AgentRun, Feedback, QueryTrace
from app.models.user import User

router = APIRouter(tags=["feedback"])


class FeedbackIn(BaseModel):
    rating: str = Field(..., pattern=r"^(up|down)$")
    comment: str = Field(default="", max_length=2000)
    trace_id: uuid.UUID | None = None
    agent_run_id: uuid.UUID | None = None


class FeedbackOut(BaseModel):
    id: uuid.UUID
    rating: str
    comment: str
    trace_id: uuid.UUID | None = None
    agent_run_id: uuid.UUID | None = None
    created_at: datetime


class FeedbackListOut(BaseModel):
    count: int
    up: int
    down: int
    results: list[FeedbackOut]


def _to_out(row: Feedback) -> FeedbackOut:
    return FeedbackOut(
        id=row.id,
        rating=row.rating,
        comment=row.comment or "",
        trace_id=row.trace_id,
        agent_run_id=row.agent_run_id,
        created_at=row.created_at,
    )


@router.post("/feedback", response_model=FeedbackOut)
def feedback_create(
    body: FeedbackIn,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> FeedbackOut:
    if body.trace_id is None and body.agent_run_id is None:
        raise HTTPException(status_code=400, detail="trace_id or agent_run_id is required")
    uid = current_owner(user)
    if body.trace_id is not None:
        trace = db.get(QueryTrace, body.trace_id)
        if trace is None or (uid is not None and trace.user_id != uid):
            raise HTTPException(status_code=404, detail=f"trace {body.trace_id} not found")
    if body.agent_run_id is not None:
        run = db.get(AgentRun, body.agent_run_id)
        if run is None or (uid is not None and run.user_id != uid):
            raise HTTPException(status_code=404, detail=f"agent run {body.agent_run_id} not found")

    row = Feedback(
        rating=body.rating,
        comment=(body.comment or "").strip(),
        trace_id=body.trace_id,
        agent_run_id=body.agent_run_id,
        user_id=uid,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _to_out(row)


@router.get("/feedback", response_model=FeedbackListOut)
def feedback_list(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
    trace_id: uuid.UUID | None = None,
    agent_run_id: uuid.UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> FeedbackListOut:
    stmt = select(Feedback).order_by(Feedback.created_at.desc()).limit(limit)
    uid = current_owner(user)
    if uid is not None:
        stmt = stmt.where(Feedback.user_id == uid)
    if trace_id is not None:
        stmt = stmt.where(Feedback.trace_id == trace_id)
    if agent_run_id is not None:
        stmt = stmt.where(Feedback.agent_run_id == agent_run_id)
    rows = list(db.execute(stmt).scalars().all())
    up = sum(1 for row in rows if row.rating == "up")
    down = sum(1 for row in rows if row.rating == "down")
    return FeedbackListOut(
        count=len(rows),
        up=up,
        down=down,
        results=[_to_out(row) for row in rows],
    )


@router.get("/feedback/summary")
def feedback_summary(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> dict[str, int]:
    stmt = select(Feedback.rating, func.count(Feedback.id)).group_by(Feedback.rating)
    uid = current_owner(user)
    if uid is not None:
        stmt = stmt.where(Feedback.user_id == uid)
    rows = db.execute(stmt).all()
    counts = {str(rating): int(count) for rating, count in rows}
    return {
        "up": counts.get("up", 0),
        "down": counts.get("down", 0),
        "count": sum(counts.values()),
    }
