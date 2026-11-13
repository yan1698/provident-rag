from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user, owner_id as current_owner
from app.db.session import get_db
from app.models.user import User
from app.tag.service import create_tag, delete_tag, get_tag, list_tags, update_tag

router = APIRouter(tags=["tags"])


class TagIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    color: str = Field(default="", max_length=16, description="Optional #RRGGBB")


class TagPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=64)
    color: str | None = Field(default=None, max_length=16)


class TagOut(BaseModel):
    id: uuid.UUID
    name: str
    color: str = ""
    trace_count: int = 0
    created_at: datetime


class TagListOut(BaseModel):
    count: int
    results: list[TagOut]


class TagDeleteOut(BaseModel):
    id: uuid.UUID
    name: str
    deleted: bool = True


def to_tag_out(row, trace_count: int = 0) -> TagOut:
    return TagOut(
        id=row.id,
        name=row.name,
        color=row.color or "",
        trace_count=trace_count,
        created_at=row.created_at,
    )


@router.get("/tags", response_model=TagListOut)
def tags_list(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> TagListOut:
    rows = list_tags(db, owner_id=current_owner(user))
    return TagListOut(
        count=len(rows),
        results=[to_tag_out(tag, count) for tag, count in rows],
    )


@router.post("/tags", response_model=TagOut, status_code=201)
def tags_create(
    body: TagIn,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> TagOut:
    try:
        row = create_tag(db, name=body.name, color=body.color, owner_id=current_owner(user))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return to_tag_out(row, 0)


@router.patch("/tags/{tag_id}", response_model=TagOut)
def tags_patch(
    tag_id: uuid.UUID,
    body: TagPatch,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> TagOut:
    if body.name is None and body.color is None:
        raise HTTPException(status_code=400, detail="name or color is required")
    uid = current_owner(user)
    try:
        row = update_tag(db, tag_id, name=body.name, color=body.color, owner_id=uid)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"tag {tag_id} not found")
    counts = {tag.id: count for tag, count in list_tags(db, owner_id=uid)}
    return to_tag_out(row, counts.get(row.id, 0))


@router.delete("/tags/{tag_id}", response_model=TagDeleteOut)
def tags_delete(
    tag_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> TagDeleteOut:
    row = delete_tag(db, tag_id, owner_id=current_owner(user))
    if row is None:
        raise HTTPException(status_code=404, detail=f"tag {tag_id} not found")
    return TagDeleteOut(id=tag_id, name=row.name, deleted=True)


@router.get("/tags/{tag_id}", response_model=TagOut)
def tags_get(
    tag_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> TagOut:
    uid = current_owner(user)
    row = get_tag(db, tag_id, owner_id=uid)
    if row is None:
        raise HTTPException(status_code=404, detail=f"tag {tag_id} not found")
    counts = {tag.id: count for tag, count in list_tags(db, owner_id=uid)}
    return to_tag_out(row, counts.get(row.id, 0))
