"""Long-running Redis consumer for agent / embed / graph jobs."""

from __future__ import annotations

import sys
from uuid import UUID

from app.core.log import configure_logging, get_logger
from app.queue.client import (
    clear_cancel,
    is_cancelled,
    pop_job,
    release_lock,
    write_job_state,
)

_log = get_logger("queue.worker")


def _uuid(value: str | None) -> UUID | None:
    if not value:
        return None
    return UUID(str(value))


def dispatch(job: dict) -> None:
    kind = str(job.get("kind") or "")
    if kind == "agent":
        _run_agent(job)
        return
    if kind == "embed":
        _run_embed(job)
        return
    if kind == "graph":
        _run_graph(job)
        return
    _log.info("unknown job kind %s", kind)


def _run_agent(job: dict) -> None:
    from datetime import datetime, timezone

    from sqlalchemy.orm.attributes import flag_modified

    from app.agent.lifecycle import CLOSED_STATUSES, QUEUED, claim_queued_run
    from app.agent.service import execute_agent_run
    from app.core.config import get_settings
    from app.db.session import SessionLocal
    from app.models import AgentRun

    run_id = _uuid(job.get("run_id"))
    if run_id is None:
        return
    owner_id = _uuid(job.get("owner_id"))
    try:
        with SessionLocal() as db:
            row = db.get(AgentRun, run_id)
            if row is None or row.status in CLOSED_STATUSES:
                return
            if row.status == QUEUED and is_cancelled(run_id):
                from app.agent.session_runtime import cancel_queued_run

                cancel_queued_run(db, row)
                return
            if row.status != QUEUED:
                return
        if not claim_queued_run(run_id):
            return
        with SessionLocal() as db:
            execute_agent_run(
                db,
                run_id,
                max_steps=job.get("max_steps"),
                planner=job.get("planner"),
                intent=job.get("intent"),
                owner_id=owner_id,
                settings=get_settings(),
            )
    except Exception as exc:  # noqa: BLE001
        _log.exception("agent job failed")
        with SessionLocal() as db:
            row = db.get(AgentRun, run_id)
            if row is None or row.status != "running":
                return
            steps = list(row.steps or [])
            steps.append(
                {
                    "thought": "queue worker failed",
                    "action": "error",
                    "input": "",
                    "source": "system",
                    "phase": "done",
                    "observation": {"error": str(exc)},
                    "elapsed_ms": 0,
                }
            )
            row.steps = steps
            flag_modified(row, "steps")
            row.step_count = len(steps)
            row.status = "error"
            row.answer = row.answer or str(exc)
            row.finished_at = datetime.now(timezone.utc)
            if row.session_id is not None:
                from app.agent.session_runtime import complete_session_turn

                complete_session_turn(db, row.session_id, cancelled=True)
            db.add(row)
            db.commit()
    finally:
        clear_cancel(run_id)


def recover_abandoned_jobs(db, *, startup: bool) -> None:
    """Reconcile the processing list. A live executor is left in place.

    ``startup`` is accepted so existing callers stay source-compatible. It no
    longer means every running row is dead.
    """
    del startup
    from datetime import datetime, timezone

    from app.agent.lifecycle import CLOSED_STATUSES, QUEUED, RUNNING
    from app.agent.session_runtime import fail_abandoned_run
    from app.core.config import get_settings
    from app.models import AgentRun
    from app.queue.client import ack_job, list_processing_jobs, requeue_job

    settings = get_settings()
    now = datetime.now(timezone.utc)
    for job in list_processing_jobs():
        if str(job.get("kind") or "") != "agent":
            requeue_job(job)
            continue
        run_id = _uuid(job.get("run_id"))
        if run_id is None:
            ack_job(job)
            continue
        run = db.get(AgentRun, run_id)
        if run is None or run.status in CLOSED_STATUSES:
            ack_job(job)
            continue
        if run.status == QUEUED:
            requeue_job(job)
            continue
        if run.status == RUNNING:
            from app.agent.lifecycle import execution_is_live

            if execution_is_live(run, now=now, stale_seconds=int(settings.agent_running_stale_seconds)):
                continue
            fail_abandoned_run(db, run_id)
            ack_job(job)
            continue
        if run.status != RUNNING:
            ack_job(job)


def deliver_queued_runs(db, *, only_run_id=None) -> int:
    """Push committed queued rows that never reached Redis. Does not create another run."""
    from sqlalchemy import select

    from app.agent.lifecycle import QUEUED
    from app.models import AgentRun
    from app.queue.client import enqueue_agent, queued_run_ids

    present = queued_run_ids()
    stmt = select(AgentRun).where(AgentRun.status == QUEUED)
    if only_run_id is not None:
        stmt = stmt.where(AgentRun.id == only_run_id)
    pushed = 0
    for run in list(db.execute(stmt).scalars()):
        if str(run.id) in present:
            continue
        dispatch = {}
        if isinstance(run.outline, dict) and isinstance(run.outline.get("_dispatch"), dict):
            dispatch = run.outline["_dispatch"]
        if enqueue_agent(
            run.id,
            max_steps=dispatch.get("max_steps"),
            planner=str(dispatch.get("planner") or run.planner),
            intent=dispatch.get("intent"),
            owner_id=run.user_id,
        ):
            present.add(str(run.id))
            pushed += 1
    return pushed


def abandon_running_on_startup(db) -> None:
    """Crash running rows whose executor lease is gone or whose heartbeat is stale.

    A second worker starting up must leave a live executor's run alone. The old
    single-worker behavior crashed every running row, including those heartbeats.
    """
    from datetime import datetime, timezone

    from sqlalchemy import select

    from app.agent.lifecycle import RUNNING, execution_is_live
    from app.agent.session_runtime import fail_abandoned_run
    from app.core.config import get_settings
    from app.models import AgentRun

    settings = get_settings()
    now = datetime.now(timezone.utc)
    run_ids = list(db.execute(select(AgentRun.id).where(AgentRun.status == RUNNING)).scalars())
    for run_id in run_ids:
        run = db.get(AgentRun, run_id)
        if run is None or run.status != RUNNING:
            continue
        if execution_is_live(run, now=now, stale_seconds=int(settings.agent_running_stale_seconds)):
            continue
        fail_abandoned_run(db, run_id)


def sweep_stale_running(db) -> None:
    """Crash running rows with a dead heartbeat that are not sitting in the processing list."""
    from datetime import datetime, timezone

    from sqlalchemy import select

    from app.agent.lifecycle import RUNNING, running_is_fresh
    from app.agent.session_runtime import fail_abandoned_run
    from app.core.config import get_settings
    from app.models import AgentRun
    from app.queue.client import list_processing_jobs

    settings = get_settings()
    now = datetime.now(timezone.utc)
    held = {
        str(job.get("run_id"))
        for job in list_processing_jobs()
        if str(job.get("kind") or "") == "agent" and job.get("run_id")
    }
    run_ids = list(db.execute(select(AgentRun.id).where(AgentRun.status == RUNNING)).scalars())
    for run_id in run_ids:
        if str(run_id) in held:
            continue
        run = db.get(AgentRun, run_id)
        if run is None or running_is_fresh(run, now=now, stale_seconds=int(settings.agent_running_stale_seconds)):
            continue
        fail_abandoned_run(db, run_id)


def _run_embed(job: dict) -> None:
    from app.core.config import get_settings
    from app.db.session import SessionLocal
    from app.embed.service import run_embedding_pipeline

    owner_id = _uuid(job.get("owner_id"))
    try:
        write_job_state("embed", owner_id, {"running": True, "error": ""})
        with SessionLocal() as db:
            run_embedding_pipeline(db, settings=get_settings(), owner_id=owner_id)
        write_job_state("embed", owner_id, {"running": False, "error": ""})
    except Exception as exc:  # noqa: BLE001
        _log.exception("embed job failed")
        write_job_state("embed", owner_id, {"running": False, "error": str(exc)})
    finally:
        release_lock("embed", owner_id)


def _run_graph(job: dict) -> None:
    from uuid import UUID

    from app.db.session import SessionLocal
    from app.graph.service import run_graph_pipeline

    owner_id = _uuid(job.get("owner_id"))
    doc_ids = []
    for item in job.get("doc_ids") or []:
        try:
            doc_ids.append(UUID(str(item)))
        except ValueError:
            continue
    try:
        from app.llm.ledger import bind_call_context

        write_job_state("graph", owner_id, {"running": True, "error": ""})
        with SessionLocal() as db, bind_call_context(user_id=owner_id):
            stats = run_graph_pipeline(
                db,
                owner_id=owner_id,
                all_docs=bool(job.get("all_docs")),
                extract=bool(job.get("extract")),
                doc_ids=doc_ids or None,
            )
        write_job_state(
            "graph",
            owner_id,
            {
                "running": False,
                "error": "",
                "note": stats.note or "",
                "skipped_chunks": stats.skipped_chunks,
                "extract_calls": stats.extract_calls,
                "extract_failed": stats.extract_failed,
                "extract_empty": stats.extract_empty,
                "extract_dropped": stats.extract_dropped,
                "source": stats.source or "rules",
            },
        )
    except Exception as exc:  # noqa: BLE001
        _log.exception("graph job failed")
        write_job_state("graph", owner_id, {"running": False, "error": str(exc)})
    finally:
        release_lock("graph", owner_id)


def _keep_worker_lease(stop: "object") -> None:
    from app.agent.lifecycle import WORKER_ID
    from app.core.config import get_settings
    from app.queue.client import register_worker

    while not stop.wait(5):  # type: ignore[attr-defined]
        settings = get_settings()
        register_worker(WORKER_ID, int(settings.agent_worker_lease_seconds))


def main() -> int:
    import signal
    import threading

    from app.agent.lifecycle import WORKER_ID
    from app.core.config import get_settings, validate_runtime_timeouts
    from app.db.session import SessionLocal
    from app.queue.client import ack_job, clear_worker, register_worker

    configure_logging()
    settings = get_settings()
    validate_runtime_timeouts(settings)
    register_worker(WORKER_ID, int(settings.agent_worker_lease_seconds))
    stop = threading.Event()
    refresher = threading.Thread(target=_keep_worker_lease, args=(stop,), name="worker-lease", daemon=True)
    refresher.start()

    def _release(*_args: object) -> None:
        clear_worker(WORKER_ID)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _release)
    with SessionLocal() as db:
        recover_abandoned_jobs(db, startup=True)
        abandon_running_on_startup(db)
    _log.info("queue worker listening worker_id=%s", WORKER_ID)
    try:
        while True:
            with SessionLocal() as db:
                deliver_queued_runs(db)
            job = pop_job(timeout=5)
            if job is None:
                with SessionLocal() as db:
                    recover_abandoned_jobs(db, startup=False)
                    sweep_stale_running(db)
                    abandon_running_on_startup(db)
                continue
            try:
                dispatch(job)
            except Exception:  # noqa: BLE001
                _log.exception("dispatch crashed")
            finally:
                ack_job(job)
    finally:
        stop.set()
        clear_worker(WORKER_ID)
    return 0


if __name__ == "__main__":
    sys.exit(main())
