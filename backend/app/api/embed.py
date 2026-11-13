from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user, owner_id as current_owner
from app.core.ratelimit import rate_limit_jobs
from app.db.session import get_db
from app.embed.jobs import embed_job_state, start_embed_job
from app.embed.service import count_pending
from app.models import Chunk, Document
from app.models.user import User

router = APIRouter(tags=["embed"])


class EmbedStatusOut(BaseModel):
    running: bool
    started: bool = False
    pending: int
    chunks: int
    embedded: int
    error: str | None = None


def _status(db: Session, *, started: bool = False, owner_id=None) -> EmbedStatusOut:
    job = embed_job_state(owner_id=owner_id)
    chunk_stmt = select(func.count(Chunk.id))
    if owner_id is not None:
        chunk_stmt = chunk_stmt.join(Document, Chunk.doc_id == Document.id).where(Document.owner_id == owner_id)
    chunks = int(db.execute(chunk_stmt).scalar_one() or 0)
    pending = count_pending(db, owner_id=owner_id)
    return EmbedStatusOut(
        running=bool(job["running"]),
        started=started,
        pending=pending,
        chunks=chunks,
        embedded=max(chunks - pending, 0),
        error=job["error"] if isinstance(job["error"], str) else None,
    )


@router.get("/embed", response_model=EmbedStatusOut)
def embed_status(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> EmbedStatusOut:
    return _status(db, owner_id=current_owner(user))


@router.post("/embed", response_model=EmbedStatusOut)
def embed_start(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
    _: None = Depends(rate_limit_jobs("embed")),
) -> EmbedStatusOut:
    uid = current_owner(user)
    pending = count_pending(db, owner_id=uid)
    started = False
    if pending > 0:
        started = start_embed_job(owner_id=uid)
    return _status(db, started=started, owner_id=uid)
