"""Enqueue agent runs. Execution lives in the queue worker."""

from __future__ import annotations

from uuid import UUID

from app.queue.client import enqueue_agent


def start_agent_job(
    run_id: UUID,
    *,
    max_steps: int | None = None,
    planner: str | None = None,
    intent: str | None = None,
    owner_id: UUID | None = None,
) -> bool:
    return enqueue_agent(
        run_id,
        max_steps=max_steps,
        planner=planner,
        intent=intent,
        owner_id=owner_id,
    )
