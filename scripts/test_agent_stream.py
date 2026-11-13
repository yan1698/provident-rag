"""SSE polls must not keep a database connection between events."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import asyncio
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import delete, select

from app.agent.service import create_agent_run
from app.api.agent import agent_stream
from app.core.config import get_settings
from app.db.session import SessionLocal, engine
from app.main import health
from app.models import AgentRun, AgentSession, AgentSessionEvent, AgentSessionState

TOKEN = uuid.uuid4().hex[:12]


def _fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


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


class _Request:
    def __init__(self) -> None:
        self.gone = False

    async def is_disconnected(self) -> bool:
        return self.gone


async def _open(run_id):
    request = _Request()
    response = await agent_stream(run_id, request, user=None)
    agen = response.body_iterator
    chunk = await agen.__anext__()
    text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
    if "event: snapshot" not in text:
        _fail(f"unexpected first frame: {text[:180]!r}")
    return request, agen


async def _scenario(run_id) -> None:
    request, agen = await _open(run_id)
    held = int(engine.pool.checkedout())
    if held != 0:
        _fail(f"{_TEST_MATERIALS['text_001']}{held} connections")
    streams = [await _open(run_id) for _ in range(8)]
    held_many = int(engine.pool.checkedout())
    if held_many > 2:
        _fail(f"{8 + 1} idle streams hold {held_many} connections")
    started = time.perf_counter()
    body = health()
    health_ms = int((time.perf_counter() - started) * 1000)
    if body.get("status") != "ok":
        _fail(f"{_TEST_MATERIALS['text_002']}{body}")
    pool = body.get("db_pool") or {}
    request.gone = True
    await agen.aclose()
    for _req, stream in streams:
        _req.gone = True
        await stream.aclose()
    if int(engine.pool.checkedout()) != 0:
        _fail(f"{_TEST_MATERIALS['text_003']}{engine.pool.checkedout()}")
    print(
        f"sse idle connections={held_many} health={health_ms}ms "
        f"pool size={pool.get('size')} checked_out={pool.get('checked_out')} "
        f"overflow={pool.get('overflow')} streams=9"
    )


def main() -> None:
    _cleanup()
    settings = get_settings()
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        session = AgentSession(
            user_id=None, title=f"{TOKEN}{_TEST_MATERIALS['text_004']}", created_at=now, updated_at=now
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        row = create_agent_run(
            db,
            f"{TOKEN} stream",
            planner="rules",
            task_kind="factoid",
            session_id=session.id,
            settings=settings,
        )
        run_id = row.id
    try:
        asyncio.run(_scenario(run_id))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
