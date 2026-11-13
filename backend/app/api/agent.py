from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent.jobs import start_agent_job
from app.agent.service import (
    create_agent_run,
    delete_agent_run,
    delete_agent_session,
    get_agent_run,
    get_session,
    list_agent_runs,
    list_agent_sessions,
    list_session_runs,
)
from app.db.session import SessionLocal
from app.auth.deps import get_current_user, owner_id as current_owner
from app.core.config import Settings, get_settings
from app.core.ratelimit import rate_limit_agent_run
from app.db.session import get_db
from app.llm import LLMError, public_llm_failure
from app.models.user import User
from app.generate.general import AnswerSource, source_mode

router = APIRouter(tags=["agent"])


class AgentRunRequest(BaseModel):
    analysis_mode: str | None = Field(default=None, pattern=r"^(auto|quick|synthesis|digest)$")
    target_document_id: uuid.UUID | None = None
    answer_source: AnswerSource = "document_only"
    source_run_id: uuid.UUID | None = None
    selected_context: str = Field(default="", max_length=2000)
    goal: str = Field(..., min_length=1, max_length=4000)
    max_steps: int | None = Field(default=None, ge=1, le=22)
    planner: str | None = Field(default=None, pattern=r"^(auto|rules|llm)$")
    task_kind: str | None = Field(default=None, pattern=r"^(auto|factoid|synthesis|digest)$")
    intent: str | None = Field(default=None, pattern=r"^(explain|compare|plan|inventory)$")
    session_id: uuid.UUID | None = None
    scope_mode: str | None = Field(default=None, pattern=r"^(auto|all|soft|hard)$")
    collection_ids: list[uuid.UUID] = Field(default_factory=list)
    pin_scope: bool = False
    idempotency_key: str | None = Field(default=None, max_length=80)
    clear_pin: bool = False


class AgentStepOut(BaseModel):
    thought: str = ""
    action: str
    input: str = ""
    source: str = ""
    phase: str = ""
    observation: dict[str, Any] | None = None
    elapsed_ms: int = 0
    node_id: str = ""
    usage: list[dict[str, Any]] = Field(default_factory=list)


class OutlineOut(BaseModel):
    mode: str = "generic"
    modules: list[dict[str, Any]] = Field(default_factory=list)
    order_note: str = ""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    task_spec: dict[str, Any] | None = None
    compose_mode: str = ""
    fallback_reason: str = ""
    memory: str = ""
    paragraphs: list[dict[str, Any]] = Field(default_factory=list)
    pending_paragraphs: list[dict[str, Any]] = Field(default_factory=list)
    pending_action: str = ""
    answer_anchor: dict[str, Any] | None = None
    reference: dict[str, Any] | None = None
    anchor_run_id: str = ""
    steps: list[dict[str, Any]] = Field(default_factory=list)
    operation_steps: list[dict[str, Any]] = Field(default_factory=list)
    operation_anchor: dict[str, Any] | None = None
    pending_reference_kind: str = "paragraph"
    task_update: str = "new"
    reading: dict[str, Any] = Field(default_factory=dict)
    run_budget: dict[str, Any] = Field(default_factory=dict)
    evidence_selection: dict[str, Any] = Field(default_factory=dict)


class GapOut(BaseModel):
    topic: str
    reason: str = ""


class CatalogDigestOut(BaseModel):
    filename: str
    sections: list[str] = Field(default_factory=list)


class AgentRunOut(BaseModel):
    routing: dict[str, Any] | None = None
    target_document_id: str = ""
    answer_source: AnswerSource = "document_only"
    source_run_id: str = ""
    id: uuid.UUID
    goal: str
    status: str
    answer: str
    provider: str
    llm_model: str
    planner: str
    task_kind: str = "factoid"
    session_id: uuid.UUID | None = None
    memory: str = ""
    step_count: int
    steps: list[AgentStepOut]
    verification: dict[str, Any] | None = None
    outline: OutlineOut | None = None
    gaps: list[GapOut] = Field(default_factory=list)
    catalog: list[CatalogDigestOut] = Field(default_factory=list)
    task_spec: dict[str, Any] | None = None
    compose_mode: str = ""
    fallback_reason: str = ""
    task_coverage: float | None = None
    rolled_back: bool = False
    session_revision: int = 0
    scope: dict[str, Any] | None = None
    delivery: str = ""
    created_at: datetime
    finished_at: datetime | None = None


class AgentRunSummaryOut(BaseModel):
    answer_source: AnswerSource = "document_only"
    id: uuid.UUID
    goal: str
    status: str
    provider: str
    planner: str
    task_kind: str = "factoid"
    session_id: uuid.UUID | None = None
    gap_count: int = 0
    step_count: int
    created_at: datetime
    finished_at: datetime | None = None


class SessionTurnOut(BaseModel):
    answer_source: AnswerSource = "document_only"
    id: uuid.UUID
    goal: str
    status: str
    answer: str
    task_kind: str = "factoid"
    created_at: datetime
    finished_at: datetime | None = None


class AgentSessionOut(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    runs: list[SessionTurnOut] = Field(default_factory=list)


class AgentSessionSummaryOut(BaseModel):
    answer_source: AnswerSource = "document_only"
    id: uuid.UUID
    title: str
    latest_goal: str = ""
    turn_count: int = 1
    status: str = ""
    task_kind: str = "factoid"
    gap_count: int = 0
    created_at: datetime
    updated_at: datetime


class AgentSessionListOut(BaseModel):
    count: int
    limit: int
    offset: int
    results: list[AgentSessionSummaryOut]


class AgentSessionDeleteOut(BaseModel):
    id: uuid.UUID
    title: str
    deleted: bool = True
    turn_count: int = 0


class AgentRunListOut(BaseModel):
    count: int
    limit: int
    offset: int
    results: list[AgentRunSummaryOut]


class AgentRunDeleteOut(BaseModel):
    id: uuid.UUID
    goal: str
    deleted: bool = True


def _session_revision(row) -> int:
    session_id = getattr(row, "session_id", None)
    if session_id is None:
        return 0
    try:
        from app.agent.session_runtime import session_snapshot
        from app.db.session import SessionLocal

        db = Session.object_session(row)
        if db is None:
            with SessionLocal() as session:
                return int(session_snapshot(session, session_id).get("revision") or 0)
        return int(session_snapshot(db, session_id).get("revision") or 0)
    except Exception:  # noqa: BLE001 — older schema
        return 0


def _to_out(row) -> AgentRunOut:
    steps = []
    for item in row.steps or []:
        if not isinstance(item, dict):
            continue
        steps.append(
            AgentStepOut(
                thought=str(item.get("thought") or ""),
                action=str(item.get("action") or ""),
                input=str(item.get("input") or ""),
                source=str(item.get("source") or ""),
                phase=str(item.get("phase") or ""),
                observation=item.get("observation") if isinstance(item.get("observation"), dict) else None,
                elapsed_ms=int(item.get("elapsed_ms") or 0),
                node_id=str(item.get("node_id") or ""),
                usage=list(item.get("usage") or []) if isinstance(item.get("usage"), list) else [],
            )
        )
    outline = None
    raw_outline = getattr(row, "outline", None)
    task_spec = None
    compose_mode = ""
    fallback_reason = ""
    memory = ""
    if isinstance(raw_outline, dict):
        raw_spec = raw_outline.get("task_spec")
        task_spec = raw_spec if isinstance(raw_spec, dict) else None
        compose_mode = str(raw_outline.get("compose_mode") or "")
        fallback_reason = str(raw_outline.get("fallback_reason") or "")
        memory = str(raw_outline.get("memory") or "")
        raw_anchor = raw_outline.get("answer_anchor")
        raw_reference = raw_outline.get("reference")
        outline = OutlineOut(
            mode=str(raw_outline.get("mode") or "generic"),
            modules=list(raw_outline.get("modules") or []),
            order_note=str(raw_outline.get("order_note") or ""),
            evidence=list(raw_outline.get("evidence") or []),
            task_spec=task_spec,
            compose_mode=compose_mode,
            fallback_reason=fallback_reason,
            memory=memory,
            paragraphs=list(raw_outline.get("paragraphs") or [])
            if isinstance(raw_outline.get("paragraphs"), list)
            else [],
            pending_paragraphs=list(raw_outline.get("pending_paragraphs") or [])
            if isinstance(raw_outline.get("pending_paragraphs"), list)
            else [],
            pending_action=str(raw_outline.get("pending_action") or ""),
            answer_anchor=raw_anchor if isinstance(raw_anchor, dict) else None,
            reference=raw_reference if isinstance(raw_reference, dict) else None,
            anchor_run_id=str(raw_outline.get("anchor_run_id") or ""),
            steps=list(raw_outline.get("steps") or []),
            operation_steps=list(raw_outline.get("operation_steps") or []),
            operation_anchor=(
                raw_outline["operation_anchor"]
                if isinstance(raw_outline.get("operation_anchor"), dict)
                else {"run_id": raw_outline["operation_anchor"]}
                if raw_outline.get("operation_anchor")
                else None
            ),
            pending_reference_kind=str(raw_outline.get("pending_reference_kind") or "paragraph"),
            task_update=str(raw_outline.get("task_update") or "new"),
            reading=dict(raw_outline.get("reading") or {}),
            run_budget=dict(raw_outline.get("run_budget") or {}),
            evidence_selection=dict(raw_outline.get("evidence_selection") or {}),
        )
    if task_spec is None:
        for item in reversed(row.steps or []):
            if not isinstance(item, dict) or item.get("action") != "analyze":
                continue
            obs = item.get("observation")
            if isinstance(obs, dict) and obs.get("intent"):
                task_spec = obs
                break
    task_coverage = None
    verification = row.verification if isinstance(row.verification, dict) else None
    if verification is not None and verification.get("task_coverage") is not None:
        try:
            task_coverage = float(verification.get("task_coverage"))
        except (TypeError, ValueError):
            task_coverage = None
    gaps = []
    for item in getattr(row, "gaps", None) or []:
        if isinstance(item, dict) and item.get("topic"):
            gaps.append(GapOut(topic=str(item.get("topic") or ""), reason=str(item.get("reason") or "")))
    catalog = []
    for item in getattr(row, "catalog_digest", None) or []:
        if isinstance(item, dict) and item.get("filename"):
            catalog.append(
                CatalogDigestOut(
                    filename=str(item.get("filename") or ""),
                    sections=list(item.get("sections") or []),
                )
            )
    return AgentRunOut(
        routing=(row.outline or {}).get("routing"),
        target_document_id=str((row.outline or {}).get("target_document_id") or ""),
        answer_source=source_mode(row),
        source_run_id=str((raw_outline or {}).get("source_run_id") or ""),
        id=row.id,
        goal=row.goal,
        status=row.status,
        answer=row.answer or "",
        provider=row.provider,
        llm_model=row.llm_model,
        planner=row.planner,
        task_kind=str(getattr(row, "task_kind", None) or "factoid"),
        session_id=getattr(row, "session_id", None),
        memory=memory,
        step_count=row.step_count,
        steps=steps,
        verification=row.verification,
        outline=outline,
        gaps=gaps,
        catalog=catalog,
        task_spec=task_spec,
        compose_mode=compose_mode,
        fallback_reason=fallback_reason,
        task_coverage=task_coverage,
        rolled_back=bool(getattr(row, "rolled_back", False)),
        session_revision=_session_revision(row),
        scope=row.scope if isinstance(getattr(row, "scope", None), dict) else None,
        created_at=row.created_at,
        finished_at=row.finished_at,
    )


@router.post("/agent/run", response_model=AgentRunOut)
def agent_run(
    body: AgentRunRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User | None = Depends(get_current_user),
    _: None = Depends(rate_limit_agent_run),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> AgentRunOut:
    """Enqueue an agent run and return immediately (poll GET /agent/{id} for progress)."""
    goal = body.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="goal must not be empty")
    uid = current_owner(user)
    try:
        row = create_agent_run(
            db,
            goal,
            planner=body.planner,
            task_kind=body.task_kind,
            owner_id=uid,
            session_id=body.session_id,
            settings=settings,
            scope_mode=body.scope_mode,
            collection_ids=[str(item) for item in body.collection_ids],
            pin_scope=body.pin_scope,
            clear_pin=body.clear_pin,
            max_steps=body.max_steps,
            intent=body.intent,
            idempotency_key=body.idempotency_key or idempotency_key,
            answer_source=body.answer_source,
            source_run_id=body.source_run_id,
            selected_context=body.selected_context,
            analysis_mode=body.analysis_mode,
            target_document_id=body.target_document_id,
        )
    except ValueError as exc:
        detail = str(exc)
        if "幂等" in detail:
            code = 409
        elif "not found" in detail:
            code = 404
        else:
            code = 400
        raise HTTPException(status_code=code, detail=detail) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"模型服务未就绪：{public_llm_failure(exc)}。"
                "请配置 DASHSCOPE_API_KEY；离线测试需 ANSWER_LLM_PROVIDER=stub 且 ALLOW_LLM_STUB=true。"
            ),
        ) from exc
    except Exception as exc:
        from app.agent.lifecycle import SessionBusy

        if not isinstance(exc, SessionBusy):
            raise
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    replay = bool(getattr(row, "_idempotent_replay", False))
    enqueued = False
    if not replay:
        enqueued = start_agent_job(
            row.id,
            max_steps=body.max_steps,
            planner=body.planner,
            intent=body.intent,
            owner_id=uid,
        )
    out = _to_out(row)
    if replay:
        out.delivery = "replayed"
    else:
        out.delivery = "accepted" if enqueued else "deferred"
    return out


@router.get("/agent", response_model=AgentRunListOut)
def agent_list(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> AgentRunListOut:
    rows = list_agent_runs(db, limit=limit, offset=offset, owner_id=current_owner(user))
    return AgentRunListOut(
        count=len(rows),
        limit=limit,
        offset=offset,
        results=[
            AgentRunSummaryOut(
                answer_source=source_mode(row),
                id=row.id,
                goal=row.goal,
                status=row.status,
                provider=row.provider,
                planner=row.planner,
                task_kind=str(getattr(row, "task_kind", None) or "factoid"),
                session_id=getattr(row, "session_id", None),
                gap_count=len(getattr(row, "gaps", None) or []),
                step_count=row.step_count,
                created_at=row.created_at,
                finished_at=row.finished_at,
            )
            for row in rows
        ],
    )


@router.get("/agent/sessions", response_model=AgentSessionListOut)
def agent_session_list(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> AgentSessionListOut:
    rows = list_agent_sessions(db, limit=limit, offset=offset, owner_id=current_owner(user))
    return AgentSessionListOut(
        count=len(rows),
        limit=limit,
        offset=offset,
        results=[
            AgentSessionSummaryOut(
                id=item["id"],
                title=item["title"],
                latest_goal=item["latest_goal"],
                answer_source=item.get("answer_source", "document_only"),
                turn_count=item["turn_count"],
                status=item["status"],
                task_kind=item["task_kind"],
                gap_count=item["gap_count"],
                created_at=item["created_at"],
                updated_at=item["updated_at"],
            )
            for item in rows
        ],
    )


def _snip_answer(text: str, limit: int = 480) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


@router.get("/agent/session/{session_id}/state")
def agent_session_state(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
    since: int | None = Query(default=None),
) -> dict[str, Any]:
    """Reconnect always returns a full snapshot. `since` is ignored (never resume)."""
    uid = current_owner(user)
    session = get_session(db, session_id, owner_id=uid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"agent session {session_id} not found")
    from app.agent.runtime import sse_snapshot_frame
    from app.agent.session_runtime import session_snapshot

    snap = session_snapshot(db, session_id)
    return sse_snapshot_frame(
        current_revision=int(snap.get("revision") or 0),
        snapshot=snap,
        client_since=since,
    )


@router.get("/agent/session/{session_id}", response_model=AgentSessionOut)
def agent_session(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> AgentSessionOut:
    uid = current_owner(user)
    session = get_session(db, session_id, owner_id=uid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"agent session {session_id} not found")
    rows = list_session_runs(db, session_id, owner_id=uid)
    return AgentSessionOut(
        id=session.id,
        title=session.title or "",
        created_at=session.created_at,
        updated_at=session.updated_at,
        runs=[
            SessionTurnOut(
                answer_source=source_mode(row),
                id=row.id,
                goal=row.goal,
                status=row.status,
                answer=_snip_answer(row.answer or ""),
                task_kind=str(getattr(row, "task_kind", None) or "factoid"),
                created_at=row.created_at,
                finished_at=row.finished_at,
            )
            for row in rows
        ],
    )


@router.delete("/agent/session/{session_id}", response_model=AgentSessionDeleteOut)
def agent_session_delete(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> AgentSessionDeleteOut:
    row = delete_agent_session(db, session_id, owner_id=current_owner(user))
    if row is None:
        raise HTTPException(status_code=404, detail=f"agent session {session_id} not found")
    return AgentSessionDeleteOut(
        id=session_id,
        title=row.title,
        deleted=True,
        turn_count=row.turn_count,
    )


@router.post("/agent/{run_id}/cancel", response_model=AgentRunOut)
def agent_cancel(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> AgentRunOut:
    row = get_agent_run(db, run_id, owner_id=current_owner(user))
    if row is None:
        raise HTTPException(status_code=404, detail=f"agent run {run_id} not found")
    from app.agent.lifecycle import QUEUED, RUNNING
    from app.agent.session_runtime import cancel_queued_run
    from app.queue.client import cancel_agent

    if row.status == QUEUED:
        if cancel_queued_run(db, row):
            db.refresh(row)
            return _to_out(row)
        db.refresh(row)
    if row.status == RUNNING:
        if not cancel_agent(run_id):
            raise HTTPException(status_code=503, detail="队列不可用，取消标志没有写入，任务仍在执行")
    return _to_out(row)


@router.get("/agent/{run_id}/stream")
async def agent_stream(
    run_id: uuid.UUID,
    request: Request,
    user: User | None = Depends(get_current_user),
):
    uid = current_owner(user)
    with SessionLocal() as db:
        row = get_agent_run(db, run_id, owner_id=uid)
        if row is None:
            raise HTTPException(status_code=404, detail=f"agent run {run_id} not found")

    async def events():
        last: tuple[int, str] = (-1, "")
        while True:
            if await request.is_disconnected():
                return
            snapshot = ""
            missing = False
            done = False
            with SessionLocal() as session:
                current = get_agent_run(session, run_id, owner_id=uid)
                if current is None:
                    missing = True
                else:
                    marker = (int(current.step_count or 0), str(current.status or ""))
                    if marker != last:
                        payload = _to_out(current).model_dump(mode="json")
                        snapshot = f"event: snapshot\ndata: {json.dumps(payload, default=str)}\n\n"
                        last = marker
                    done = str(current.status) in {"finished", "refused", "error", "cancelled", "crashed"}
            if missing:
                yield 'event: error\ndata: {"detail":"not found"}\n\n'
                return
            if snapshot:
                yield snapshot
            if done:
                yield "event: done\ndata: {}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/agent/{run_id}", response_model=AgentRunOut)
def agent_get(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> AgentRunOut:
    row = get_agent_run(db, run_id, owner_id=current_owner(user))
    if row is None:
        raise HTTPException(status_code=404, detail=f"agent run {run_id} not found")
    return _to_out(row)


@router.delete("/agent/{run_id}", response_model=AgentRunDeleteOut)
def agent_delete(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> AgentRunDeleteOut:
    row = delete_agent_run(db, run_id, owner_id=current_owner(user))
    if row is None:
        raise HTTPException(status_code=404, detail=f"agent run {run_id} not found")
    return AgentRunDeleteOut(id=run_id, goal=row.goal, deleted=True)
