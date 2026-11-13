from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.tag import query_trace_tags

if TYPE_CHECKING:
    from app.models.tag import QueryTag


class QueryTrace(Base):
    """Persisted /query run for eval replay and provenance (Day 8)."""

    __tablename__ = "query_traces"
    __table_args__ = (
        Index("idx_query_traces_created_at", "created_at"),
        Index("idx_query_traces_eval_id", "eval_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    eval_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    top_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    citations: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    evidence: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    verification: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    usage: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    coverage_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    hallucination_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    claim_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    supported_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hallucination_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diagnostics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    tags: Mapped[list[QueryTag]] = relationship(
        "QueryTag",
        secondary=query_trace_tags,
        back_populates="traces",
        lazy="selectin",
    )
