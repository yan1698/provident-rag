"""Same idempotency key must not create a second run. Rediscovery keeps dispatch params."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import delete, select

from app.agent.service import create_agent_run
from app.auth.deps import get_current_user
from app.core.config import get_settings
from app.core.ratelimit import rate_limit_agent_run
from app.db.session import SessionLocal
from app.main import app
from app.models import AgentRun, AgentSession, AgentSessionEvent, AgentSessionState, User
from app.queue import client as queue_client
from app.queue.client import pop_job

TOKEN = uuid.uuid4().hex[:12]
FAILURES: list[str] = []


def _fail(name: str, message: str) -> None:
    FAILURES.append(f"{name}: {message}")
    print(f"FAIL {name}: {message}")


def _ok(name: str) -> None:
    print(f"PASS {name}")


def _cleanup() -> None:
    with SessionLocal() as db:
        run_ids = list(db.execute(select(AgentRun.id).where(AgentRun.goal.contains(TOKEN))).scalars())
        session_ids = list(db.execute(select(AgentSession.id).where(AgentSession.title.contains(TOKEN))).scalars())
        if run_ids:
            db.execute(delete(AgentSessionEvent).where(AgentSessionEvent.run_id.in_(run_ids)))
            db.execute(delete(AgentRun).where(AgentRun.id.in_(run_ids)))
        if session_ids:
            db.execute(delete(AgentSessionEvent).where(AgentSessionEvent.session_id.in_(session_ids)))
            db.execute(delete(AgentSessionState).where(AgentSessionState.session_id.in_(session_ids)))
            db.execute(delete(AgentSession).where(AgentSession.id.in_(session_ids)))
        db.commit()


def _session(db, owner_id) -> AgentSession:
    now = datetime.now(timezone.utc)
    row = AgentSession(user_id=owner_id, title=f"{TOKEN}{_TEST_MATERIALS['text_001']}", created_at=now, updated_at=now)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def test_replay_and_isolation() -> None:
    from fastapi.testclient import TestClient

    with SessionLocal() as db:
        owner = db.execute(select(User.id).limit(1)).scalar_one()
        first = _session(db, owner)
        second = _session(db, owner)
        owner_id = owner
        first_id = str(first.id)
        second_id = str(second.id)

    class _User:
        id = owner_id

    app.dependency_overrides[get_current_user] = lambda: _User()
    app.dependency_overrides[rate_limit_agent_run] = lambda: None
    key = f"idem-{TOKEN}"
    try:
        with TestClient(app) as client:
            payload = {
                "goal": f"{TOKEN}{_TEST_MATERIALS['text_002']}",
                "planner": "rules",
                "task_kind": "factoid",
                "session_id": first_id,
                "idempotency_key": key,
                "max_steps": 4,
                "intent": "explain",
            }
            headers = {"Idempotency-Key": key}
            one = client.post("/agent/run", json=payload, headers=headers)
            two = client.post("/agent/run", json=payload, headers=headers)
            if one.status_code != 200 or two.status_code != 200:
                _fail("replay", f"statuses {one.status_code} {two.status_code} {one.text} {two.text}")
                return
            if one.json()["id"] != two.json()["id"]:
                _fail("replay", f"same key created {one.json()['id']} and {two.json()['id']}")
                return
            cancelled = client.post(f"/agent/{one.json()['id']}/cancel", json={})
            if cancelled.status_code != 200:
                _fail("replay", f"cancel {cancelled.status_code} {cancelled.text}")
                return
            other = client.post(
                "/agent/run",
                headers=headers,
                json={
                    "goal": f"{TOKEN}{_TEST_MATERIALS['text_004']}",
                    "planner": "rules",
                    "task_kind": "factoid",
                    "session_id": second_id,
                    "idempotency_key": key,
                },
            )
            if other.status_code != 200 or other.json()["id"] == one.json()["id"]:
                _fail("isolation", f"other session {other.status_code} {other.text}")
                return
            plain = client.post(
                "/agent/run",
                json={
                    "goal": f"{TOKEN}{_TEST_MATERIALS['text_005']}",
                    "planner": "rules",
                    "task_kind": "factoid",
                    "session_id": second_id,
                },
            )
            if plain.status_code == 200:
                _fail("compat", _TEST_MATERIALS["text_006"])
                return
            if plain.status_code != 409:
                _fail("compat", f"missing key returned {plain.status_code}")
                return
        with SessionLocal() as db:
            count = len(list(db.execute(select(AgentRun.id).where(AgentRun.goal == f"{TOKEN} same")).scalars()))
        if count != 1:
            _fail("replay", f"stored {count}{_TEST_MATERIALS['text_003']}")
            return
        _ok("replay")
        _ok("isolation")
        _ok("compat")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(rate_limit_agent_run, None)


def test_dispatch_survives_redelivery() -> None:
    _cleanup()
    queue_client.QUEUE_KEY = f"provident:testidem:{TOKEN}:jobs"
    queue_client.PROCESSING_KEY = f"provident:testidem:{TOKEN}:processing"
    client = queue_client._redis()
    if client is not None:
        client.delete(queue_client.QUEUE_KEY, queue_client.PROCESSING_KEY)
    settings = get_settings()
    with SessionLocal() as db:
        owner = db.execute(select(User.id).limit(1)).scalar_one()
        session = _session(db, owner)
        try:
            row = create_agent_run(
                db,
                f"{TOKEN} dispatch",
                planner="rules",
                task_kind="factoid",
                owner_id=owner,
                session_id=session.id,
                settings=settings,
                max_steps=4,
                intent="explain",
            )
        except TypeError as exc:
            _fail("dispatch", f"create_agent_run dropped max_steps/intent: {exc}")
            return
        run_id = row.id
    from app.queue.worker import deliver_queued_runs

    with SessionLocal() as db:
        deliver_queued_runs(db, only_run_id=run_id)
    job = pop_job(timeout=2)
    if not job or str(job.get("run_id")) != str(run_id):
        _fail("dispatch", f"redelivery missing: {job}")
        return
    if job.get("max_steps") != 4 or job.get("intent") != "explain" or job.get("planner") != "rules":
        _fail("dispatch", f"params changed: { {k: job.get(k) for k in ('max_steps', 'intent', 'planner')} }")
        return
    _ok("dispatch")


def main() -> None:
    _cleanup()
    try:
        test_replay_and_isolation()
        test_dispatch_survives_redelivery()
    finally:
        _cleanup()
        client = queue_client._redis()
        if client is not None:
            client.delete(queue_client.QUEUE_KEY, queue_client.PROCESSING_KEY)
    if FAILURES:
        print(f"{len(FAILURES)} failed")
        raise SystemExit(1)
    print("idempotency checks passed")


if __name__ == "__main__":
    main()
