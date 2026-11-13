from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from app.models.tag import QueryTag

_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def normalize_tag_name(name: str) -> str:
    return " ".join((name or "").strip().split())


def normalize_tag_color(color: str | None) -> str:
    value = (color or "").strip()
    if not value:
        return ""
    if not _COLOR_RE.match(value):
        raise ValueError("color must be a #RRGGBB hex value")
    return value.upper()


def list_tags(db: Session, *, owner_id: uuid.UUID | None = None) -> list[tuple[QueryTag, int]]:
    from app.models.tag import QueryTag, query_trace_tags

    count_col = func.count(query_trace_tags.c.trace_id).label("trace_count")
    stmt = (
        select(QueryTag, count_col)
        .outerjoin(query_trace_tags, QueryTag.id == query_trace_tags.c.tag_id)
        .group_by(QueryTag.id)
        .order_by(QueryTag.name.asc())
    )
    if owner_id is not None:
        stmt = stmt.where(QueryTag.user_id == owner_id)
    return [(row[0], int(row[1])) for row in db.execute(stmt).all()]


def get_tag(db: Session, tag_id: uuid.UUID, *, owner_id: uuid.UUID | None = None) -> QueryTag | None:
    from app.models.tag import QueryTag

    row = db.get(QueryTag, tag_id)
    if row is None:
        return None
    if owner_id is not None and row.user_id != owner_id:
        return None
    return row


def create_tag(db: Session, *, name: str, color: str = "", owner_id: uuid.UUID | None = None) -> QueryTag:
    from app.models.tag import QueryTag

    cleaned = normalize_tag_name(name)
    if not cleaned:
        raise ValueError("name must not be empty")
    if len(cleaned) > 64:
        raise ValueError("name must be at most 64 characters")
    color_value = normalize_tag_color(color)
    row = QueryTag(name=cleaned, color=color_value, user_id=owner_id)
    db.add(row)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ValueError(f"tag '{cleaned}' already exists") from exc
    db.refresh(row)
    return row


def update_tag(
    db: Session,
    tag_id: uuid.UUID,
    *,
    name: str | None = None,
    color: str | None = None,
    owner_id: uuid.UUID | None = None,
) -> QueryTag | None:
    row = get_tag(db, tag_id, owner_id=owner_id)
    if row is None:
        return None
    if name is not None:
        cleaned = normalize_tag_name(name)
        if not cleaned:
            raise ValueError("name must not be empty")
        if len(cleaned) > 64:
            raise ValueError("name must be at most 64 characters")
        row.name = cleaned
    if color is not None:
        row.color = normalize_tag_color(color)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ValueError(f"tag '{row.name}' already exists") from exc
    db.refresh(row)
    return row


def delete_tag(db: Session, tag_id: uuid.UUID, *, owner_id: uuid.UUID | None = None) -> QueryTag | None:
    from app.models.tag import query_trace_tags

    row = get_tag(db, tag_id, owner_id=owner_id)
    if row is None:
        return None
    db.execute(delete(query_trace_tags).where(query_trace_tags.c.tag_id == tag_id))
    db.delete(row)
    db.commit()
    return row


def load_tags_by_ids(db: Session, tag_ids: list[uuid.UUID], *, owner_id: uuid.UUID | None = None) -> list[QueryTag]:
    from app.models.tag import QueryTag

    if not tag_ids:
        return []
    unique_ids = list(dict.fromkeys(tag_ids))
    stmt = select(QueryTag).where(QueryTag.id.in_(unique_ids))
    if owner_id is not None:
        stmt = stmt.where(QueryTag.user_id == owner_id)
    rows = list(db.execute(stmt).scalars().all())
    by_id = {row.id: row for row in rows}
    missing = [str(tid) for tid in unique_ids if tid not in by_id]
    if missing:
        raise ValueError(f"tag(s) not found: {', '.join(missing)}")
    return [by_id[tid] for tid in unique_ids]


def set_trace_tags(
    db: Session,
    *,
    trace_id: uuid.UUID,
    tag_ids: list[uuid.UUID],
    owner_id: uuid.UUID | None = None,
) -> list[QueryTag]:
    from app.models.trace import QueryTrace

    trace = db.get(QueryTrace, trace_id)
    if trace is None or (owner_id is not None and trace.user_id != owner_id):
        raise LookupError(f"trace {trace_id} not found")
    tags = load_tags_by_ids(db, tag_ids, owner_id=owner_id)
    trace.tags = tags
    db.commit()
    db.refresh(trace)
    return list(trace.tags)
