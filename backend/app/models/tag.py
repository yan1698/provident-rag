from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, ForeignKey, Index, String, Table, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.trace import QueryTrace


query_trace_tags = Table(
    "query_trace_tags",
    Base.metadata,
    Column(
        "trace_id",
        UUID(as_uuid=True),
        ForeignKey("query_traces.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "tag_id",
        UUID(as_uuid=True),
        ForeignKey("query_tags.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Index("idx_query_trace_tags_tag_id", "tag_id"),
)


class QueryTag(Base):
    """User-defined tag for organizing query history (multi-label)."""

    __tablename__ = "query_tags"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    color: Mapped[str] = mapped_column(String(16), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    traces: Mapped[list[QueryTrace]] = relationship(
        "QueryTrace",
        secondary=query_trace_tags,
        back_populates="tags",
    )
