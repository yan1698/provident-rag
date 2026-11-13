"""Phase 1: session admission, claim, queue recovery, cancel, heartbeat.

Uses the real Postgres and Redis. A controlled barrier forces the double-submit
race; recovery tests use the queue the worker actually consumes.
"""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import os
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_backend = ROOT / "backend"
if _backend.is_dir():
    sys.path.insert(0, str(_backend))
elif (ROOT / "app").is_dir():
    sys.path.insert(0, str(ROOT))

from sqlalchemy import delete, select, text

from app.agent.planner import Plan
from app.agent.service import create_agent_run, execute_agent_run
from app.agent.session_runtime import session_snapshot
from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models import AgentRun, AgentSession, AgentSessionEvent, AgentSessionState, LlmCall
from app.queue import client as queue_client
from app.queue.client import enqueue_agent, pop_job

TOKEN = uuid.uuid4().hex[:12]
GOAL = f"{_TEST_MATERIALS['text_001']}{TOKEN}{_TEST_MATERIALS['text_002']}"
FAILURES: list[str] = []


def _fail(name: str, message: str) -> None:
    FAILURES.append(f"{name}: {message}")
    print(f"FAIL {name}: {message}")


def _ok(name: str) -> None:
    print(f"PASS {name}")


def _isolate_queue() -> None:
    queue_client.QUEUE_KEY = f"provident:testlifecycle:{TOKEN}:jobs"
    queue_client.PROCESSING_KEY = f"provident:testlifecycle:{TOKEN}:processing"
    queue_client._force_memory = False


def _redis():
    return queue_client._redis()


def _wipe_queue() -> None:
    client = _redis()
    if client is None:
        return
    client.delete(queue_client.QUEUE_KEY, queue_client.PROCESSING_KEY)


def _cleanup_db() -> None:
    with SessionLocal() as db:
        run_ids = list(
            db.execute(
                select(AgentRun.id).where(AgentRun.goal.contains(TOKEN) | AgentRun.goal.startswith("lifecycle-"))
            ).scalars()
        )
        session_ids = list(
            db.execute(
                select(AgentSession.id).where(
                    AgentSession.title.contains(TOKEN) | AgentSession.title.contains("lifecycle-")
                )
            ).scalars()
        )
        if run_ids:
            db.execute(delete(LlmCall).where(LlmCall.run_id.in_(run_ids)))
            db.execute(delete(AgentSessionEvent).where(AgentSessionEvent.run_id.in_(run_ids)))
            db.execute(delete(AgentRun).where(AgentRun.id.in_(run_ids)))
        if session_ids:
            db.execute(delete(AgentSessionEvent).where(AgentSessionEvent.session_id.in_(session_ids)))
            db.execute(delete(AgentSessionState).where(AgentSessionState.session_id.in_(session_ids)))
            db.execute(delete(AgentSession).where(AgentSession.id.in_(session_ids)))
        db.commit()


def _new_session(db, owner_id: uuid.UUID | None = None) -> AgentSession:
    now = datetime.now(timezone.utc)
    session = AgentSession(
        user_id=owner_id, title=f"{TOKEN}{_TEST_MATERIALS['text_003']}", created_at=now, updated_at=now
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _anchor_outline(run_id: uuid.UUID) -> dict:
    return {"answer_anchor": {"run_id": str(run_id)}, "memory": "prior"}


def _require_redis() -> None:
    client = _redis()
    if client is None:
        raise SystemExit(_TEST_MATERIALS["text_030"])
    info = client.info("server")
    version = str(info.get("redis_version") or "0")
    parts = [int(piece) for piece in version.split(".")[:2]]
    if tuple(parts) < (6, 2):
        raise SystemExit(f"BLOCKED: Redis {version} has no BLMOVE")
    print(f"redis {version} blmove ok")


def _require_postgres() -> None:
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{_TEST_MATERIALS['text_024']}{exc}") from exc
    print("postgres ok")


def test_a_concurrent_submit() -> None:
    """Two overlapping submits. One is admitted. The other is 409. No crash."""
    from fastapi.testclient import TestClient

    from app.auth.deps import get_current_user
    from app.core.ratelimit import rate_limit_agent_run
    from app.main import app

    settings = get_settings()
    with SessionLocal() as db:
        session = _new_session(db)
        prior_id = uuid.uuid4()
        prior = AgentRun(
            id=prior_id,
            user_id=None,
            session_id=session.id,
            goal=f"{TOKEN}{_TEST_MATERIALS['text_025']}",
            status="finished",
            answer=_TEST_MATERIALS["text_041"],
            planner="rules",
            task_kind="factoid",
            steps=[],
            step_count=0,
            outline=_anchor_outline(prior_id),
            finished_at=datetime.now(timezone.utc),
        )
        db.add(prior)
        db.commit()
        session_id = session.id
        anchor_before = dict(prior.outline or {})

    barrier = threading.Barrier(2, timeout=8)
    import app.agent.session_runtime as runtime

    original = runtime.acquire_session

    def _gated(*args, **kwargs):
        barrier.wait()
        return original(*args, **kwargs)

    runtime.acquire_session = _gated
    app.dependency_overrides[get_current_user] = lambda: None
    app.dependency_overrides[rate_limit_agent_run] = lambda: None
    statuses: list[int] = []
    bodies: list[dict] = []
    errors: list[str] = []
    payload = {
        "goal": GOAL,
        "task_kind": "factoid",
        "planner": "rules",
        "session_id": str(session_id),
    }

    def _post(client: TestClient) -> None:
        try:
            response = client.post("/agent/run", json=payload)
            statuses.append(response.status_code)
            bodies.append(response.json() if response.content else {})
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    try:
        with TestClient(app) as client:
            threads = [threading.Thread(target=_post, args=(client,), name=f"submit-{i}") for i in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=20)
    finally:
        runtime.acquire_session = original
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(rate_limit_agent_run, None)

    if errors:
        _fail("A", f"request errors {errors} statuses={statuses}")
        return
    if any(thread.is_alive() for thread in threads):
        _fail("A", _TEST_MATERIALS["text_031"])
        return
    with SessionLocal() as db:
        runs = list(
            db.execute(select(AgentRun).where(AgentRun.session_id == session_id, AgentRun.goal == GOAL)).scalars()
        )
        prior = db.get(AgentRun, prior_id)
        snap = session_snapshot(db, session_id)
        anchor_after = dict(prior.outline or {}) if prior is not None else {}
    ok_bodies = [body for code, body in zip(statuses, bodies) if code == 200]
    conflicts = [code for code in statuses if code == 409]
    crashed = [run for run in runs if run.status == "crashed"]
    if statuses.count(200) != 1 or len(conflicts) != 1:
        _fail("A", f"{_TEST_MATERIALS['text_004']}{statuses} bodies={bodies}")
        return
    if len(runs) != 1:
        _fail("A", f"{_TEST_MATERIALS['text_005']}{[(str(r.id), r.status) for r in runs]}")
        return
    if crashed:
        _fail("A", _TEST_MATERIALS["text_032"])
        return
    winner = runs[0]
    if winner.status not in {"queued", "running"}:
        _fail("A", f"winner status {winner.status}")
        return
    if anchor_after != anchor_before:
        _fail("A", f"answer_anchor changed {anchor_before} -> {anchor_after}")
        return
    if snap.get("checkpoint_json") is None and winner.status == "queued":
        _fail("A", _TEST_MATERIALS["text_033"])
        return
    if int(snap.get("revision") or 0) != 0:
        _fail("A", f"{_TEST_MATERIALS['text_006']}{snap}")
        return
    if str(ok_bodies[0].get("id")) != str(winner.id):
        _fail("A", _TEST_MATERIALS["text_034"])
        return
    del settings
    _ok("A")


def test_b_closed_skips_model() -> None:
    settings = get_settings()
    calls = {"n": 0}

    def _boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("planner entered")

    import app.agent.service as service

    original = service.next_plan
    service.next_plan = _boom
    try:
        with SessionLocal() as db:
            session = _new_session(db)
            before = int(db.execute(select(LlmCall.id).where(LlmCall.purpose == TOKEN)).scalars().all().__len__())
            for status in ("crashed", "cancelled", "finished", "refused", "error"):
                run = AgentRun(
                    user_id=None,
                    session_id=session.id,
                    goal=f"{TOKEN}{_TEST_MATERIALS['text_029']}{status}",
                    status=status,
                    answer=_TEST_MATERIALS["text_051"],
                    planner="rules",
                    task_kind="factoid",
                    steps=[{"action": "analyze", "thought": "kept", "input": ""}],
                    step_count=1,
                    rolled_back=status in {"crashed", "cancelled"},
                    outline={"answer_anchor": {"run_id": "keep"}},
                )
                db.add(run)
                db.commit()
                steps_before = list(run.steps or [])
                outline_before = dict(run.outline or {})
                execute_agent_run(db, run.id, planner="rules", settings=settings)
                db.refresh(run)
                if calls["n"] != 0:
                    _fail("B", f"{status} entered the planner ({calls['n']} calls)")
                    return
                if list(run.steps or []) != steps_before or dict(run.outline or {}) != outline_before:
                    _fail("B", f"{status} run was rewritten")
                    return
                if run.status != status:
                    _fail("B", f"{status} became {run.status}")
                    return
            snap = session_snapshot(db, session.id)
            if snap.get("checkpoint_json") is not None:
                _fail("B", f"{_TEST_MATERIALS['text_028']}{snap}")
                return
            after = int(db.execute(select(LlmCall.id)).scalars().all().__len__())
            if after != before and calls["n"]:
                _fail("B", "llm_calls grew")
                return
    finally:
        service.next_plan = original
    _ok("B")


def test_b_step_boundary() -> None:
    """Once a run leaves running, the next loop must not plan again."""
    settings = get_settings()
    calls = {"n": 0}
    with SessionLocal() as db:
        session = _new_session(db)
        row = create_agent_run(
            db,
            f"{TOKEN} boundary",
            planner="rules",
            task_kind="factoid",
            session_id=session.id,
            settings=settings,
        )
        run_id = row.id
        # Pretend the worker already claimed it, then another actor crashes it mid-loop.
        row.status = "running"
        row.heartbeat_at = datetime.now(timezone.utc)
        db.commit()

    import app.agent.service as service

    original = service.next_plan

    def _once(*args, **kwargs):
        calls["n"] += 1
        with SessionLocal() as db:
            run = db.get(AgentRun, run_id)
            run.status = "crashed"
            run.rolled_back = True
            run.finished_at = datetime.now(timezone.utc)
            db.commit()
        return Plan(thought="once", action="finish", input="", source="rules")

    from app.agent.tools import ToolResult

    original_execute = service._execute

    def _quiet(plan, ctx, registry):
        return ToolResult({"status": "ok"}, terminal=False)

    service.next_plan = _once
    service._execute = _quiet
    try:
        with SessionLocal() as db:
            execute_agent_run(db, run_id, planner="rules", max_steps=4, settings=settings)
            run = db.get(AgentRun, run_id)
            status = run.status if run is not None else None
    finally:
        service.next_plan = original
        service._execute = original_execute
    if calls["n"] != 1:
        _fail("B2", f"{_TEST_MATERIALS['text_007']}{calls['n']}")
        return
    if status != "crashed":
        _fail("B2", f"{_TEST_MATERIALS['text_008']}{status}")
        return
    _ok("B2")


def test_c_worker_crash_partial() -> None:
    _wipe_queue()
    settings = get_settings()
    with SessionLocal() as db:
        session = _new_session(db)
        row = create_agent_run(
            db,
            f"{TOKEN} partial",
            planner="rules",
            task_kind="factoid",
            session_id=session.id,
            settings=settings,
        )
        run_id = row.id
        session_id = session.id
    pushed = enqueue_agent(
        run_id,
        planner="rules",
        intent=None,
        owner_id=None,
    )
    if pushed is False:
        _fail("C", _TEST_MATERIALS["text_035"])
        return
    job = pop_job(timeout=2)
    if not job or str(job.get("run_id")) != str(run_id):
        _fail("C", f"{_TEST_MATERIALS['text_009']}{job}")
        return
    with SessionLocal() as db:
        run = db.get(AgentRun, run_id)
        run.status = "running"
        run.heartbeat_at = datetime.now(timezone.utc)
        run.steps = [{"action": "analyze", "thought": "kept-step", "input": "", "observation": {}}]
        run.step_count = 1
        db.commit()
        left = pop_job(timeout=1)
    if left is not None:
        _fail("C", f"{_TEST_MATERIALS['text_010']}{left}")
        return
    dead_worker = f"dead-{TOKEN}-partial"
    client = _redis()
    if client is None:
        _fail("C", "redis unavailable")
        return
    client.set(f"provident:run-owner:{run_id}", dead_worker, ex=120)
    try:
        from app.queue.worker import recover_abandoned_jobs
    except ImportError:
        _fail("C", _TEST_MATERIALS["text_042"])
        return
    with SessionLocal() as db:
        recover_abandoned_jobs(db, startup=True)
        run = db.get(AgentRun, run_id)
        snap = session_snapshot(db, session_id)
        status = run.status
        steps = list(run.steps or [])
    if status != "crashed":
        _fail("C", f"{_TEST_MATERIALS['text_011']}{status}, expected crashed")
        return
    if not steps or steps[0].get("thought") != "kept-step":
        _fail("C", f"{_TEST_MATERIALS['text_012']}{steps}")
        return
    if snap.get("checkpoint_json") is not None:
        _fail("C", f"checkpoint still held: {snap}")
        return
    calls = {"n": 0}

    def _boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("replayed")

    import app.agent.service as service

    original = service.next_plan
    service.next_plan = _boom
    try:
        from app.queue.worker import dispatch

        dispatch({"kind": "agent", "run_id": str(run_id), "planner": "rules", "owner_id": None})
    finally:
        service.next_plan = original
    with SessionLocal() as db:
        run = db.get(AgentRun, run_id)
        replayed = run.status
    if calls["n"] != 0 or replayed != "crashed":
        _fail("C", f"{_TEST_MATERIALS['text_013']}{calls['n']} status={replayed}")
        return
    _ok("C")


def test_c_unstarted_requeue() -> None:
    _wipe_queue()
    settings = get_settings()
    with SessionLocal() as db:
        session = _new_session(db)
        row = create_agent_run(
            db,
            f"{TOKEN} unstarted",
            planner="rules",
            task_kind="factoid",
            session_id=session.id,
            settings=settings,
        )
        run_id = row.id
        created_status = row.status
    if created_status != "queued":
        _fail("C-unstarted", f"new run is {created_status}{_TEST_MATERIALS['text_014']}")
        return
    enqueue_agent(run_id, planner="rules", owner_id=None)
    job = pop_job(timeout=2)
    if not job:
        _fail("C-unstarted", _TEST_MATERIALS["text_036"])
        return
    try:
        from app.queue.worker import recover_abandoned_jobs
    except ImportError:
        _fail("C-unstarted", _TEST_MATERIALS["text_043"])
        return
    with SessionLocal() as db:
        recover_abandoned_jobs(db, startup=True)
        run = db.get(AgentRun, run_id)
        status = run.status
        steps = run.step_count
    if status != "queued" or steps:
        _fail("C-unstarted", f"unstarted run became {status} steps={steps}")
        return
    restored = pop_job(timeout=2)
    if not restored or str(restored.get("run_id")) != str(run_id):
        _fail("C-unstarted", f"{_TEST_MATERIALS['text_015']}{restored}")
        return
    _ok("C-unstarted")


def test_d_commit_without_push() -> None:
    _wipe_queue()
    settings = get_settings()
    with SessionLocal() as db:
        before = len(list(db.execute(select(AgentRun.id).where(AgentRun.goal.contains(TOKEN))).scalars()))
        session = _new_session(db)
        row = create_agent_run(
            db,
            f"{TOKEN} nopush",
            planner="rules",
            task_kind="factoid",
            session_id=session.id,
            settings=settings,
        )
        run_id = row.id
        status = row.status
        after = len(list(db.execute(select(AgentRun.id).where(AgentRun.goal.contains(TOKEN))).scalars()))
    if status != "queued":
        _fail("D", f"committed run is {status}{_TEST_MATERIALS['text_016']}")
        return
    if after != before + 1:
        _fail("D", _TEST_MATERIALS["text_037"])
        return
    try:
        from app.queue.worker import deliver_queued_runs
    except ImportError:
        _fail("D", _TEST_MATERIALS["text_044"])
        return
    with SessionLocal() as db:
        deliver_queued_runs(db, only_run_id=run_id)
        deliver_queued_runs(db, only_run_id=run_id)
    first = pop_job(timeout=2)
    second = pop_job(timeout=1)
    if not first or str(first.get("run_id")) != str(run_id):
        _fail("D", f"{_TEST_MATERIALS['text_017']}{first}")
        return
    if second and str(second.get("run_id")) == str(run_id):
        # A duplicate message is allowed only if claim is single-winner.
        pass
    try:
        from app.agent.lifecycle import claim_queued_run
    except ImportError:
        _fail("D", _TEST_MATERIALS["text_045"])
        return
    won = []

    def _claim() -> None:
        won.append(claim_queued_run(run_id))

    threads = [threading.Thread(target=_claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if won.count(True) != 1:
        _fail("D", f"claim winners={won}")
        return
    with SessionLocal() as db:
        count = len(list(db.execute(select(AgentRun.id).where(AgentRun.goal == f"{TOKEN} nopush")).scalars()))
        run = db.get(AgentRun, run_id)
    if count != 1 or run.status != "running":
        _fail("D", f"runs={count} status={run.status}")
        return
    _ok("D")


def test_e_cancel_before_start() -> None:
    settings = get_settings()
    with SessionLocal() as db:
        session = _new_session(db)
        row = create_agent_run(
            db,
            f"{TOKEN} cancel",
            planner="rules",
            task_kind="factoid",
            session_id=session.id,
            settings=settings,
        )
        run_id = row.id
        session_id = session.id
        if session_snapshot(db, session_id).get("checkpoint_json") is None:
            _fail("E", _TEST_MATERIALS["text_046"])
            return
    try:
        from app.agent.session_runtime import cancel_queued_run
    except ImportError:
        with SessionLocal() as db:
            snap = session_snapshot(db, session_id)
            run = db.get(AgentRun, run_id)
        _fail(
            "E",
            f"{_TEST_MATERIALS['text_026']}{run.status} checkpoint={snap.get('checkpoint_json') is not None}",
        )
        return
    with SessionLocal() as db:
        run = db.get(AgentRun, run_id)
        ok = cancel_queued_run(db, run)
        db.refresh(run)
        status = run.status
        snap = session_snapshot(db, session_id)
    if not ok or status != "cancelled":
        _fail("E", f"cancel returned {ok} status={status}")
        return
    if snap.get("checkpoint_json") is not None:
        _fail("E", "checkpoint survived cancel")
        return
    with SessionLocal() as db:
        follow = create_agent_run(
            db,
            f"{TOKEN} follow",
            planner="rules",
            task_kind="factoid",
            session_id=session_id,
            settings=settings,
        )
        follow_status = follow.status
    if follow_status != "queued":
        _fail("E", f"{_TEST_MATERIALS['text_018']}{follow_status}")
        return
    _ok("E")


def test_f_no_inline_thread() -> None:
    before = {thread.name for thread in threading.enumerate()}
    original = queue_client._redis
    queue_client._force_memory = False
    queue_client._redis = lambda **kwargs: None
    try:
        result = enqueue_agent(uuid.uuid4(), planner="rules", owner_id=None)
    finally:
        queue_client._redis = original
    spawned = [
        thread.name for thread in threading.enumerate() if thread.name not in before and thread.name.startswith("job-")
    ]
    if spawned:
        _fail("F", f"{_TEST_MATERIALS['text_019']}{spawned}")
        return
    if result is not False:
        _fail("F", f"{_TEST_MATERIALS['text_020']}{result!r}")
        return
    _ok("F")


def test_g_single_claim() -> None:
    _wipe_queue()
    settings = get_settings()
    calls = {"n": 0}
    with SessionLocal() as db:
        session = _new_session(db)
        row = create_agent_run(
            db,
            f"{TOKEN} claim",
            planner="rules",
            task_kind="factoid",
            session_id=session.id,
            settings=settings,
        )
        run_id = row.id
    enqueue_agent(run_id, planner="rules", owner_id=None)
    job = pop_job(timeout=2)
    if not job:
        _fail("G", "nothing to deliver")
        return
    import app.agent.service as service

    original = service.execute_agent_run

    def _count(*args, **kwargs):
        calls["n"] += 1
        return kwargs.get("run_id")

    service.execute_agent_run = _count
    try:
        from app.queue.worker import dispatch

        threads = [threading.Thread(target=dispatch, args=(dict(job),), name=f"claim-{i}") for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
    finally:
        service.execute_agent_run = original
    if calls["n"] != 1:
        _fail("G", f"duplicate delivery executed {calls['n']} times")
        return
    _ok("G")


def test_h_live_executor_survives_other_startup() -> None:
    """Startup recovery must not crash a run whose worker lease is still alive.

    A second run with a lease and no living worker is the dead-executor case.
    """
    _wipe_queue()
    settings = get_settings()
    from app.models import User

    with SessionLocal() as db:
        owners = list(db.execute(select(User.id).limit(2)).scalars())
    if len(owners) < 2:
        _fail("H", _TEST_MATERIALS["text_038"])
        return
    created: list[tuple[str, uuid.UUID, uuid.UUID, uuid.UUID]] = []
    with SessionLocal() as db:
        for label, owner_id in zip(("live", "dead"), owners):
            session = _new_session(db, owner_id)
            row = create_agent_run(
                db,
                f"{TOKEN} {label}-owner",
                planner="rules",
                task_kind="factoid",
                owner_id=owner_id,
                session_id=session.id,
                settings=settings,
            )
            created.append((label, row.id, session.id, owner_id))
    client = _redis()
    if client is None:
        _fail("H", "redis unavailable")
        return
    live_worker = f"live-{TOKEN}"
    dead_worker = f"dead-{TOKEN}"
    client.set(f"provident:worker:{live_worker}", "1", ex=60)
    for label, run_id, _session_id, owner_id in created:
        enqueue_agent(run_id, planner="rules", max_steps=7, intent="keep-intent", owner_id=owner_id)
        job = pop_job(timeout=2)
        if not job or str(job.get("run_id")) != str(run_id):
            _fail("H", f"{label}{_TEST_MATERIALS['text_027']}{job}")
            return
        owner = live_worker if label == "live" else dead_worker
        client.set(f"provident:run-owner:{run_id}", owner, ex=120)
        with SessionLocal() as db:
            run = db.get(AgentRun, run_id)
            run.status = "running"
            run.heartbeat_at = datetime.now(timezone.utc)
            run.steps = [{"action": "analyze", "thought": f"kept-{label}", "input": "", "observation": {}}]
            run.step_count = 1
            db.commit()
    try:
        from app.queue.worker import abandon_running_on_startup, recover_abandoned_jobs
    except ImportError:
        _fail("H", _TEST_MATERIALS["text_047"])
        return
    with SessionLocal() as db:
        recover_abandoned_jobs(db, startup=True)
        abandon_running_on_startup(db)
        live_run = db.get(AgentRun, created[0][1])
        dead_run = db.get(AgentRun, created[1][1])
        live_snap = session_snapshot(db, created[0][2])
        dead_snap = session_snapshot(db, created[1][2])
        live_status = live_run.status
        dead_status = dead_run.status
        live_step = list(live_run.steps or [])
        dead_step = list(dead_run.steps or [])
    if live_status != "running" or not live_step or live_step[0].get("thought") != "kept-live":
        _fail("H", f"live executor became {live_status} steps={live_step}")
        return
    if live_snap.get("checkpoint_json") is None:
        _fail("H", _TEST_MATERIALS["text_039"])
        return
    if dead_status != "crashed" or not dead_step or dead_step[0].get("thought") != "kept-dead":
        _fail("H", f"dead executor status={dead_status} steps={dead_step}")
        return
    if dead_snap.get("checkpoint_json") is not None:
        _fail("H", _TEST_MATERIALS["text_040"])
        return
    client.delete(
        f"provident:worker:{live_worker}",
        f"provident:run-owner:{created[0][1]}",
        f"provident:run-owner:{created[1][1]}",
    )
    _ok("H")


def test_i_stale_progress_rejected() -> None:
    """A writer that lost the running row must not replace the crashed steps."""
    settings = get_settings()
    with SessionLocal() as db:
        session = _new_session(db)
        row = create_agent_run(
            db,
            f"{TOKEN} stale-progress",
            planner="rules",
            task_kind="factoid",
            session_id=session.id,
            settings=settings,
        )
        run_id = row.id
    from app.agent.lifecycle import claim_queued_run
    from app.agent.service import _persist_progress

    if not claim_queued_run(run_id):
        _fail("I", "claim failed")
        return
    with SessionLocal() as db:
        run = db.get(AgentRun, run_id)
        if run is None:
            _fail("I", _TEST_MATERIALS["text_048"])
            return
        kept = [{"action": "analyze", "thought": "kept-before-crash", "input": "", "observation": {}}]
        run.status = "crashed"
        run.steps = kept
        run.step_count = 1
        db.commit()
    with SessionLocal() as db:
        stale = db.get(AgentRun, run_id)
        if stale is None:
            _fail("I", "run missing")
            return
        wrote = _persist_progress(
            stale,
            [{"action": "retrieve", "thought": "should-not-land", "input": "", "observation": {}}],
            None,
        )
    with SessionLocal() as db:
        run = db.get(AgentRun, run_id)
        thoughts = [item.get("thought") for item in list(run.steps or [])] if run is not None else []
        status = None if run is None else run.status
    if wrote or status != "crashed" or "should-not-land" in thoughts or "kept-before-crash" not in thoughts:
        _fail("I", f"wrote={wrote} status={status} steps={thoughts}")
        return
    _ok("I")


def test_j_expired_presence_cannot_write() -> None:
    """The owner key outlives the worker lease. A missing lease must not write."""
    settings = get_settings()
    with SessionLocal() as db:
        session = _new_session(db)
        row = create_agent_run(
            db,
            f"{TOKEN} expired-lease",
            planner="rules",
            task_kind="factoid",
            session_id=session.id,
            settings=settings,
        )
        run_id = row.id
    from app.agent.lifecycle import WORKER_ID, claim_queued_run
    from app.agent.service import _persist_progress

    if not claim_queued_run(run_id):
        _fail("J", "claim failed")
        return
    kept = [{"action": "analyze", "thought": "kept-while-leased", "input": "", "observation": {}}]
    with SessionLocal() as db:
        run = db.get(AgentRun, run_id)
        if run is None:
            _fail("J", _TEST_MATERIALS["text_049"])
            return
        wrote = _persist_progress(run, kept, None)
    client = _redis()
    if client is None:
        _fail("J", "redis unavailable")
        return
    client.delete(f"provident:worker:{WORKER_ID}")
    try:
        late = [{"action": "retrieve", "thought": "after-lease", "input": "", "observation": {}}]
        with SessionLocal() as db:
            stale = db.get(AgentRun, run_id)
            if stale is None:
                _fail("J", "run missing")
                return
            second = _persist_progress(stale, late, None)
        with SessionLocal() as db:
            run = db.get(AgentRun, run_id)
            thoughts = [item.get("thought") for item in list(run.steps or [])] if run is not None else []
            status = None if run is None else run.status
        if not wrote or second or status != "running" or thoughts != ["kept-while-leased"]:
            _fail("J", f"wrote={wrote} second={second} status={status} steps={thoughts}")
            return
        _ok("J")
    finally:
        client.delete(f"provident:run-owner:{run_id}")


def test_k_other_owner_cannot_write() -> None:
    """A process that does not hold the owner key must not write or take the key."""
    settings = get_settings()
    with SessionLocal() as db:
        session = _new_session(db)
        row = create_agent_run(
            db,
            f"{TOKEN} other-owner",
            planner="rules",
            task_kind="factoid",
            session_id=session.id,
            settings=settings,
        )
        run_id = row.id
    from app.agent import lifecycle
    from app.agent.service import _persist_progress
    from app.queue.client import bind_run_executor

    if not lifecycle.claim_queued_run(run_id):
        _fail("K", "claim failed")
        return
    client = _redis()
    if client is None:
        _fail("K", "redis unavailable")
        return
    other = f"other-{TOKEN}"
    original = lifecycle.WORKER_ID
    client.set(f"provident:worker:{other}", "1", ex=60)
    client.set(f"provident:run-owner:{run_id}", other, ex=120)
    try:
        stolen = bind_run_executor(run_id, original, 120)
        owner = client.get(f"provident:run-owner:{run_id}")
        if isinstance(owner, bytes):
            owner = owner.decode("utf-8")
        denied_steps = [{"action": "retrieve", "thought": "should-not-land", "input": "", "observation": {}}]
        with SessionLocal() as db:
            stale = db.get(AgentRun, run_id)
            if stale is None:
                _fail("K", "run missing")
                return
            denied = _persist_progress(stale, denied_steps, None)
        lifecycle.WORKER_ID = other
        allowed_steps = [{"action": "analyze", "thought": "owner-wrote", "input": "", "observation": {}}]
        with SessionLocal() as db:
            holder = db.get(AgentRun, run_id)
            if holder is None:
                _fail("K", _TEST_MATERIALS["text_050"])
                return
            allowed = _persist_progress(holder, allowed_steps, None)
        with SessionLocal() as db:
            run = db.get(AgentRun, run_id)
            thoughts = [item.get("thought") for item in list(run.steps or [])] if run is not None else []
            status = None if run is None else run.status
        if stolen or owner != other or denied or not allowed or status != "running" or thoughts != ["owner-wrote"]:
            _fail(
                "K",
                f"stolen={stolen} owner={owner} denied={denied} allowed={allowed} status={status} steps={thoughts}",
            )
            return
        _ok("K")
    finally:
        lifecycle.WORKER_ID = original
        client.delete(f"provident:worker:{other}", f"provident:run-owner:{run_id}")


def test_timeouts_distinct() -> None:
    settings = get_settings()
    missing = [
        name
        for name in (
            "llm_connect_timeout_seconds",
            "llm_read_timeout_seconds",
            "llm_write_timeout_seconds",
            "agent_heartbeat_interval_seconds",
            "agent_running_stale_seconds",
        )
        if not hasattr(settings, name)
    ]
    if missing:
        _fail("timeouts", f"missing settings {missing}")
        return
    connect = float(settings.llm_connect_timeout_seconds)
    read = float(settings.llm_read_timeout_seconds)
    interval = int(settings.agent_heartbeat_interval_seconds)
    stale = int(settings.agent_running_stale_seconds)
    if len({connect, read, float(interval), float(stale)}) < 4:
        _fail("timeouts", f"{_TEST_MATERIALS['text_021']}{connect} read={read} beat={interval} stale={stale}")
        return
    if not (connect < read and interval < stale and stale != int(settings.rate_limit_agent_stale_seconds)):
        _fail(
            "timeouts",
            f"{_TEST_MATERIALS['text_022']}{settings.rate_limit_agent_stale_seconds}",
        )
        return
    from app.llm.base import llm_timeout

    timeout = llm_timeout(settings)
    if float(timeout.connect) != connect or float(timeout.read) != read:
        _fail("timeouts", f"client timeout {timeout!r}{_TEST_MATERIALS['text_023']}")
        return
    _ok("timeouts")


def main() -> None:
    _require_postgres()
    _require_redis()
    _isolate_queue()
    _wipe_queue()
    _cleanup_db()
    try:
        for test in (
            test_a_concurrent_submit,
            test_b_closed_skips_model,
            test_b_step_boundary,
            test_c_worker_crash_partial,
            test_c_unstarted_requeue,
            test_d_commit_without_push,
            test_e_cancel_before_start,
            test_f_no_inline_thread,
            test_g_single_claim,
            test_h_live_executor_survives_other_startup,
            test_i_stale_progress_rejected,
            test_j_expired_presence_cannot_write,
            test_k_other_owner_cannot_write,
            test_timeouts_distinct,
        ):
            _cleanup_db()
            _wipe_queue()
            test()
    finally:
        _wipe_queue()
        _cleanup_db()
    if FAILURES:
        print(f"{len(FAILURES)} failed")
        raise SystemExit(1)
    print("lifecycle checks passed")


if __name__ == "__main__":
    main()
