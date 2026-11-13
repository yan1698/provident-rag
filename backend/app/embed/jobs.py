"""Enqueue embedding jobs. State lives in Redis so API and worker agree."""

from __future__ import annotations

from app.queue.client import enqueue_embed, job_state


def embed_job_state(*, owner_id=None) -> dict[str, str | bool | None]:
    state = job_state("embed", owner_id)
    return {"running": state["running"], "error": state.get("error")}


def start_embed_job(*, owner_id=None) -> bool:
    return enqueue_embed(owner_id=owner_id)
