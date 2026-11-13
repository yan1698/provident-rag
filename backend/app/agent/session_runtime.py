"""DB adapter for checkpoint lifecycle. Uses runtime.py; no second orphan detector."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.agent.runtime import begin_turn, finish_turn, idle_state, is_orphan, recover_orphan


def _payload(row) -> dict[str, Any]:
    return {
        "revision": int(row.revision or 0),
        "slots": dict(row.slots or {}),
        "catalog_digest": list(row.catalog_digest or []),
        "gaps": list(row.gaps or []),
        "checkpoint_json": row.checkpoint_json,
    }


def _write(row, payload: dict[str, Any]) -> None:
    row.revision = int(payload.get("revision") or 0)
    row.slots = dict(payload.get("slots") or {})
    row.catalog_digest = list(payload.get("catalog_digest") or [])
    row.gaps = list(payload.get("gaps") or [])
    row.checkpoint_json = payload.get("checkpoint_json")
    row.updated_at = datetime.now(timezone.utc)


def get_or_create_state(db: Session, session_id: UUID):
    from sqlalchemy.exc import IntegrityError

    from app.models.agent import AgentSessionState

    row = db.get(AgentSessionState, session_id)
    if row is not None:
        return row
    idle = idle_state()
    row = AgentSessionState(
        session_id=session_id,
        revision=idle["revision"],
        slots=idle["slots"],
        catalog_digest=idle["catalog_digest"],
        gaps=idle["gaps"],
        checkpoint_json=None,
    )
    try:
        with db.begin_nested():
            db.add(row)
            db.flush()
    except IntegrityError:
        from sqlalchemy import select

        existing = db.execute(
            select(AgentSessionState)
            .where(AgentSessionState.session_id == session_id)
            .execution_options(populate_existing=True)
        ).scalar_one_or_none()
        if existing is None:
            raise
        return existing
    return row


def _lock_sessions(db: Session, session_ids: list[UUID]) -> None:
    from sqlalchemy import select

    from app.models.agent import AgentSession

    for session_id in sorted(set(session_ids), key=str):
        db.execute(select(AgentSession).where(AgentSession.id == session_id).with_for_update())


def _owner_active(db: Session, owner_id: UUID | None):
    from sqlalchemy import select

    from app.agent.lifecycle import ACTIVE_STATUSES
    from app.models import AgentRun

    stmt = select(AgentRun).where(AgentRun.status.in_(tuple(ACTIVE_STATUSES)))
    if owner_id is None:
        stmt = stmt.where(AgentRun.user_id.is_(None))
    else:
        stmt = stmt.where(AgentRun.user_id == owner_id)
    return list(db.execute(stmt).scalars())


def _bump_recovery(run) -> None:
    from sqlalchemy.orm.attributes import flag_modified

    outline = dict(run.outline or {})
    metrics = dict(outline.get("_metrics") or {})
    metrics["recovery_count"] = int(metrics.get("recovery_count") or 0) + 1
    metrics["terminal"] = "crashed"
    outline["_metrics"] = metrics
    run.outline = outline
    flag_modified(run, "outline")


def _blocks_admission(run, *, now: datetime, stale_seconds: int) -> bool:
    from app.agent.lifecycle import QUEUED, running_is_fresh

    if run.status == QUEUED:
        return True
    return running_is_fresh(run, now=now, stale_seconds=stale_seconds)


def acquire_session(db: Session, session_id: UUID, *, owner_id: UUID | None = None) -> dict[str, Any]:
    """Admit one turn. A live queued/running turn is left untouched."""
    from app.core.config import get_settings
    from app.agent.lifecycle import SessionBusy, lock_owner, mark_crashed

    settings = get_settings()
    lock_owner(db, owner_id)
    now = datetime.now(timezone.utc)
    stale_seconds = int(settings.agent_running_stale_seconds)
    active = _owner_active(db, owner_id)
    extra = [
        run.session_id
        for run in active
        if run.session_id is not None and not _blocks_admission(run, now=now, stale_seconds=stale_seconds)
    ]
    _lock_sessions(db, [session_id, *extra])
    db.expire_all()
    active = _owner_active(db, owner_id)
    blocking = [run for run in active if _blocks_admission(run, now=now, stale_seconds=stale_seconds)]
    same = [run for run in blocking if run.session_id == session_id]
    if same:
        raise SessionBusy("已有任务正在排队或执行，请等当前任务结束后再提交")
    limit = int(settings.rate_limit_agent_concurrency) if settings.rate_limit_enabled else 0
    if limit > 0 and len(blocking) >= limit:
        raise SessionBusy("已有任务正在排队或执行，请等当前任务结束后再提交")
    for run in active:
        if run.status != "running" or _blocks_admission(run, now=now, stale_seconds=stale_seconds):
            continue
        mark_crashed(run, now=now)
        if run.session_id is not None:
            complete_session_turn(db, run.session_id, cancelled=True)
    row = get_or_create_state(db, session_id)
    payload = _payload(row)
    if is_orphan(payload.get("checkpoint_json")):
        restored = recover_orphan(payload)
        _write(row, restored)
        payload = _payload(row)
    started = begin_turn(payload)
    _write(row, started)
    db.add(row)
    db.flush()
    return _payload(row)


def cancel_queued_run(db: Session, run) -> bool:
    """Cancel a turn the worker has not claimed. Reuses checkpoint abort."""
    from app.agent.lifecycle import QUEUED, lock_owner

    if run is None:
        return False
    lock_owner(db, run.user_id)
    if run.session_id is not None:
        _lock_sessions(db, [run.session_id])
    db.refresh(run)
    if run.status != QUEUED:
        return False
    now = datetime.now(timezone.utc)
    run.status = "cancelled"
    run.rolled_back = True
    run.finished_at = now
    if not (run.answer or "").strip():
        run.answer = "任务已取消。"
    if run.session_id is not None:
        complete_session_turn(db, run.session_id, cancelled=True)
    db.add(run)
    db.commit()
    return True


def fail_abandoned_run(db: Session, run_id: UUID) -> bool:
    """Mark a claimed run crashed and restore its checkpoint. No model replay."""
    from app.agent.lifecycle import RUNNING, lock_owner, mark_crashed
    from app.models import AgentRun

    run = db.get(AgentRun, run_id)
    if run is None or run.status != RUNNING:
        return False
    lock_owner(db, run.user_id)
    if run.session_id is not None:
        _lock_sessions(db, [run.session_id])
    db.refresh(run)
    if run.status != RUNNING:
        return False
    mark_crashed(run)
    _bump_recovery(run)
    if run.session_id is not None:
        complete_session_turn(db, run.session_id, cancelled=True)
    db.add(run)
    db.commit()
    return True


def complete_session_turn(
    db: Session,
    session_id: UUID,
    *,
    cancelled: bool = False,
    slots: dict[str, Any] | None = None,
    catalog_digest: list[Any] | None = None,
    gaps: list[Any] | None = None,
) -> dict[str, Any]:
    row = get_or_create_state(db, session_id)
    payload = _payload(row)
    if cancelled:
        if is_orphan(payload.get("checkpoint_json")):
            restored = recover_orphan(payload)
            _write(row, restored)
        else:
            payload["checkpoint_json"] = None
            _write(row, payload)
    elif payload.get("checkpoint_json") is None:
        pass
    else:
        finished = finish_turn(payload)
        if slots is not None:
            finished["slots"] = dict(slots)
        if catalog_digest is not None:
            finished["catalog_digest"] = list(catalog_digest)
        if gaps is not None:
            finished["gaps"] = list(gaps)
        _write(row, finished)
    db.add(row)
    db.flush()
    return _payload(row)


def record_turn_events(
    db: Session,
    session_id: UUID,
    run_id: UUID | None,
    events: list[dict[str, Any]],
    *,
    revision: int = 0,
) -> None:
    from uuid import uuid4

    from app.models.agent import AgentSessionEvent

    for event in events:
        db.add(
            AgentSessionEvent(
                id=uuid4(),
                session_id=session_id,
                run_id=run_id,
                revision_after=int(revision),
                type=str(event.get("type") or ""),
                payload=dict(event),
            )
        )


def session_snapshot(db: Session, session_id: UUID) -> dict[str, Any]:
    row = get_or_create_state(db, session_id)
    return _payload(row)
