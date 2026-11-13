from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class AgentSession(Base):
    """One conversation. Runs inside it may follow up with a summary, not raw evidence."""

    __tablename__ = "agent_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    pinned_scope: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    runs: Mapped[list["AgentRun"]] = relationship(back_populates="session")


class AgentRun(Base):
    """Planner → tool executor audit log (Day 13)."""

    __tablename__ = "agent_runs"
    __table_args__ = (Index("idx_agent_runs_created_at", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    planner: Mapped[str] = mapped_column(String(16), nullable=False, default="rules")
    step_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    steps: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    verification: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    task_kind: Mapped[str | None] = mapped_column(String(16), nullable=True)
    outline: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    gaps: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    catalog_digest: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rolled_back: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    scope: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    session: Mapped[AgentSession | None] = relationship(back_populates="runs")


class AgentSessionState(Base):
    """Authoritative short-term state. checkpoint_json is not None ⇔ in-flight turn."""

    __tablename__ = "agent_session_state"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    slots: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    catalog_digest: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    gaps: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    checkpoint_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class AgentSessionEvent(Base):
    __tablename__ = "agent_session_events"
    __table_args__ = (Index("idx_agent_session_events_session", "session_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    revision_after: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    type: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class MemoryItem(Base):
    __tablename__ = "memory_items"
    __table_args__ = (Index("idx_memory_items_owner_kind", "owner_id", "kind"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    key: Mapped[str] = mapped_column(Text, nullable=False, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    source_event_ids: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    source_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    rule_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    stale: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retracted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    stale_ingest_id: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class AgentIdempotency(Base):
    """One accepted run for a client key. Scoped by owner and session, not globally."""

    __tablename__ = "agent_idempotency"
    __table_args__ = (
        UniqueConstraint(
            "owner_key",
            "session_key",
            "idempotency_key",
            name="uq_agent_idempotency",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_key: Mapped[str] = mapped_column(String(64), nullable=False)
    session_key: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class CorpusIndexState(Base):
    """chunk_generation bumps on ingest; committed_index_version on embed complete."""

    __tablename__ = "corpus_index_state"

    owner_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    chunk_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    committed_index_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
