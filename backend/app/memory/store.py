"""Persist / stale / retract memory items."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.memory.promote import mark_stale_on_ingest, retract_stale_on_hit


def list_items(db: Session, owner_id: UUID | None) -> list[dict]:
    from app.models.agent import MemoryItem

    stmt = select(MemoryItem)
    if owner_id is not None:
        stmt = stmt.where(MemoryItem.owner_id == owner_id)
    rows = list(db.execute(stmt).scalars())
    return [_to_dict(row) for row in rows]


def stale_on_ingest(db: Session, owner_id: UUID | None, *, ingest_id: str) -> None:
    from app.models.agent import MemoryItem

    stmt = select(MemoryItem)
    if owner_id is not None:
        stmt = stmt.where(MemoryItem.owner_id == owner_id)
    rows = list(db.execute(stmt).scalars())
    now = datetime.now(timezone.utc)
    run_times = {str(row.source_run_id or ""): row.created_at for row in rows}
    mapped = mark_stale_on_ingest(
        [_to_dict(row) for row in rows],
        ingest_id=ingest_id,
        ingest_time=now,
        run_time_by_id=run_times,
    )
    by_id = {str(row.id): row for row in rows}
    for item in mapped:
        row = by_id.get(str(item.get("id") or ""))
        if row is None:
            continue
        row.stale = bool(item.get("stale"))
        row.stale_ingest_id = str(item.get("stale_ingest_id") or "")
        db.add(row)


def retract_if_answered(db: Session, owner_id: UUID | None, *, key: str, answered: bool) -> None:
    from app.models.agent import MemoryItem

    stmt = select(MemoryItem).where(MemoryItem.key == key)
    if owner_id is not None:
        stmt = stmt.where(MemoryItem.owner_id == owner_id)
    for row in db.execute(stmt).scalars():
        updated = retract_stale_on_hit(_to_dict(row), answered=answered)
        row.stale = bool(updated.get("stale"))
        row.retracted = bool(updated.get("retracted"))
        row.version = int(updated.get("version") or row.version)
        db.add(row)


def persist_turn_memory(
    db: Session,
    *,
    run,
    steps: list[dict],
    retrieve_obs: dict | None,
    owner_id: UUID | None,
    session_id: UUID | None,
    answered: bool,
) -> None:
    """Project + gate + persist. Cancelled/crashed/unhealthy runs are no-ops."""
    if (getattr(run, "outline", None) or {}).get("answer_source") == "general_explanation":
        return
    from app.memory.extractor import events_from_steps, extractor_allows_run, project
    from app.memory.promote import promote_gap

    run_id = str(getattr(run, "id", "") or "")
    events = events_from_steps(run_id, steps)
    run_dict = {
        "id": run_id,
        "status": str(getattr(run, "status", "") or ""),
        "rolled_back": bool(getattr(run, "rolled_back", False)),
        "steps": list(steps or []),
        "events": events,
    }
    if not extractor_allows_run(run_dict):
        return
    for draft in project(events):
        item = promote_gap(run_dict, retrieve_obs, draft)
        if item is None:
            continue
        save_item(db, item, owner_id=owner_id, session_id=session_id)
    if answered and retrieve_obs:
        key = str(retrieve_obs.get("query") or "")
        if key:
            retract_if_answered(db, owner_id, key=key, answered=True)


def save_item(db: Session, payload: dict, *, owner_id: UUID | None, session_id: UUID | None) -> None:
    from app.models.agent import MemoryItem

    raw_run = payload.get("source_run_id")
    run_id = None
    if raw_run:
        try:
            run_id = raw_run if isinstance(raw_run, UUID) else UUID(str(raw_run))
        except (TypeError, ValueError):
            run_id = None
    row = MemoryItem(
        id=uuid4(),
        owner_id=owner_id,
        session_id=session_id,
        kind=str(payload.get("kind") or ""),
        key=str(payload.get("key") or ""),
        payload=dict(payload.get("payload") or {}),
        source_event_ids=list(payload.get("source_event_ids") or []),
        source_run_id=run_id,
        version=int(payload.get("version") or 1),
        rule_version=int(payload.get("rule_version") or 1),
        stale=bool(payload.get("stale")),
        retracted=bool(payload.get("retracted")),
    )
    db.add(row)


def _to_dict(row) -> dict:
    return {
        "id": str(row.id),
        "kind": row.kind,
        "key": row.key,
        "payload": dict(row.payload or {}),
        "source_run_id": str(row.source_run_id or ""),
        "created_at": row.created_at,
        "stale": bool(row.stale),
        "retracted": bool(row.retracted),
        "version": int(row.version or 1),
        "rule_version": int(row.rule_version or 1),
        "stale_ingest_id": row.stale_ingest_id or "",
    }
