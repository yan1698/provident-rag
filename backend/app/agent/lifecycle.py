"""Run status, admission lock, and claim. Checkpoint changes stay in session_runtime."""

from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import text, update
from sqlalchemy.orm import Session

QUEUED = "queued"
RUNNING = "running"
FINISHED = "finished"
REFUSED = "refused"
ERROR = "error"
CANCELLED = "cancelled"
CRASHED = "crashed"

# Turns that may be summarized later. A crashed partial answer is not one of them.
MEMORY_STATUSES = frozenset({FINISHED, REFUSED, ERROR, CANCELLED})
CLOSED_STATUSES = MEMORY_STATUSES | {CRASHED}
ACTIVE_STATUSES = frozenset({QUEUED, RUNNING})

_INTERRUPTED_ANSWER = "任务已中断。"
# One process, one id. A restart is a different executor even on the same host.
WORKER_ID = str(uuid.uuid4())


class SessionBusy(RuntimeError):
    """Another turn still owns this session or the owner's concurrency slot."""


class QueueUnavailable(RuntimeError):
    """Redis did not accept the flag. The database status was left unchanged."""


def owner_lock_key(owner_id: UUID | None) -> int:
    """Stable signed bigint. Anonymous callers share one namespaced key."""
    if owner_id is None:
        return 0x50726F7601  # "Prov" + generation, not the zero lock
    key = int.from_bytes(owner_id.bytes[:8], "big", signed=True)
    return key or 1


def lock_owner(db: Session, owner_id: UUID | None) -> None:
    """Hold a transaction advisory lock until the caller commits or rolls back."""
    db.execute(text("SELECT pg_advisory_xact_lock(CAST(:key AS bigint))"), {"key": owner_lock_key(owner_id)})


def running_is_fresh(run, *, now: datetime, stale_seconds: int) -> bool:
    if str(getattr(run, "status", "") or "") != RUNNING:
        return False
    beat = getattr(run, "heartbeat_at", None)
    if beat is None:
        return False
    if beat.tzinfo is None:
        beat = beat.replace(tzinfo=timezone.utc)
    return (now - beat).total_seconds() <= max(int(stale_seconds), 1)


def _owner_ttl(settings) -> int:
    return max(int(settings.agent_running_stale_seconds) * 3, 60)


def publish_executor(run_id: UUID, worker_id: str | None = None) -> None:
    """Refresh the worker lease and the run's owner pointer."""
    from app.core.config import get_settings
    from app.queue.client import bind_run_executor, register_worker

    settings = get_settings()
    worker_id = worker_id or WORKER_ID
    register_worker(worker_id, int(settings.agent_worker_lease_seconds))
    bind_run_executor(run_id, worker_id, _owner_ttl(settings))


def execution_is_live(run, *, now: datetime, stale_seconds: int) -> bool:
    """Fresh heartbeat plus a living owner. A recorded dead owner is not fresh enough."""
    from app.queue.client import executor_liveness

    if not running_is_fresh(run, now=now, stale_seconds=stale_seconds):
        return False
    return executor_liveness(run.id) != "dead"


def claim_queued_run(run_id: UUID) -> bool:
    """Move queued → running. Only one caller can win."""
    from app.db.session import SessionLocal
    from app.models import AgentRun

    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        result = db.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id, AgentRun.status == QUEUED)
            .values(status=RUNNING, heartbeat_at=now)
        )
        won = int(result.rowcount or 0) == 1
        db.commit()
    if won:
        publish_executor(run_id)
        _note_queue_wait(run_id, now)
    return won


def _note_queue_wait(run_id: UUID, claimed_at: datetime) -> None:
    """Record how long the run sat in queued. Later heartbeats must not overwrite it."""
    from sqlalchemy.orm.attributes import flag_modified

    from app.db.session import SessionLocal
    from app.models import AgentRun

    with SessionLocal() as db:
        run = db.get(AgentRun, run_id)
        if run is None:
            return
        outline = dict(run.outline or {})
        metrics = dict(outline.get("_metrics") or {})
        if "queue_ms" not in metrics and run.created_at is not None:
            created = run.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            metrics["queue_ms"] = max(0, int((claimed_at - created).total_seconds() * 1000))
            metrics["claimed_at"] = claimed_at.isoformat()
        outline["_metrics"] = metrics
        run.outline = outline
        flag_modified(run, "outline")
        db.add(run)
        db.commit()


def touch_heartbeat(run_id: UUID) -> None:
    from app.db.session import SessionLocal
    from app.models import AgentRun
    from app.queue.client import writer_owns

    if not writer_owns(run_id, WORKER_ID):
        return
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        result = db.execute(
            update(AgentRun).where(AgentRun.id == run_id, AgentRun.status == RUNNING).values(heartbeat_at=now)
        )
        still_running = int(result.rowcount or 0) == 1
        db.commit()
    if still_running:
        publish_executor(run_id)


@contextmanager
def heartbeat_scope(run_id: UUID, interval_seconds: int):
    """Refresh heartbeat on its own session while a model call blocks the worker."""
    interval = max(int(interval_seconds), 1)
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(interval):
            try:
                touch_heartbeat(run_id)
            except Exception:  # noqa: BLE001 — a missed beat must not kill the run
                from app.core.log import get_logger

                get_logger("agent").exception("heartbeat update failed")

    thread = threading.Thread(target=_loop, name=f"heartbeat-{run_id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def mark_crashed(run, *, now: datetime | None = None) -> None:
    """Set crash fields. Caller restores the checkpoint and commits."""
    now = now or datetime.now(timezone.utc)
    if str(run.status or "") != RUNNING:
        return
    run.status = CRASHED
    run.rolled_back = True
    run.finished_at = now
    if not (run.answer or "").strip():
        run.answer = _INTERRUPTED_ANSWER
