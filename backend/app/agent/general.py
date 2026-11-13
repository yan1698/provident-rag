"""A single explanation call using the existing worker lease and session checkpoint."""

from datetime import datetime, timezone
import time

from sqlalchemy import select

from app.generate.general import explain
from app.llm import public_llm_failure
from app.llm.ledger import bind_call_context
from app.llm.meter import bind_llm_stop, collect_usage


def execute_explanation(db, row, *, settings):
    from app.agent.lifecycle import WORKER_ID
    from app.agent.service import _committed_status, _merge_run_metrics
    from app.agent.session_runtime import complete_session_turn
    from app.models import AgentRun
    from app.queue.client import writer_owns, is_cancelled

    run_id = row.id
    outline = dict(row.outline or {})
    started = time.perf_counter()
    budget = float(settings.agent_run_timeout_seconds or 0)

    def stopped():
        return (
            is_cancelled(run_id)
            or _committed_status(db, run_id) != "running"
            or not writer_owns(run_id, WORKER_ID)
            or (budget > 0 and time.perf_counter() - started >= budget)
        )

    ledger = []
    status, answer = "finished", ""
    try:
        with bind_call_context(user_id=row.user_id, run_id=run_id), collect_usage(ledger), bind_llm_stop(stopped):
            result = explain(
                row.goal,
                topic=outline.get("topic", ""),
                selected_context=outline.get("selected_context", ""),
                settings=settings,
            )
            answer = result.answer
        if budget > 0 and time.perf_counter() - started >= budget:
            status, answer = "error", "模型服务超时，请重试。"
    except Exception as exc:
        status, answer = "error", public_llm_failure(exc)
    if not writer_owns(run_id, WORKER_ID):
        db.rollback()
        return db.get(AgentRun, run_id)
    # Same conditional final write as document runs: cancellation always wins.
    held = db.execute(
        select(AgentRun).where(AgentRun.id == run_id, AgentRun.status == "running").with_for_update()
    ).scalar_one_or_none()
    if held is None:
        db.rollback()
        return db.get(AgentRun, run_id)
    row = held
    if is_cancelled(run_id):
        status, answer = "cancelled", "任务已取消。"
        row.rolled_back = True
    row.status, row.answer = status, answer
    row.verification = None
    row.steps = [
        {
            "action": "general_explanation",
            "source": "model",
            "usage": [u.to_dict() for u in ledger],
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    ]
    row.step_count = 1
    row.finished_at = row.heartbeat_at = datetime.now(timezone.utc)
    _merge_run_metrics(row, status, outline)
    if row.session_id:
        # Restore the document snapshot even on success. Model knowledge is not document memory.
        complete_session_turn(db, row.session_id, cancelled=True)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
