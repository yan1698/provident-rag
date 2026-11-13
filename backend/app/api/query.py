from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.verify import VerificationOut, to_verification_out
from app.auth.deps import get_current_user, owner_id as current_owner
from app.core.config import Settings, get_settings
from app.core.ratelimit import rate_limit_query
from app.db.session import get_db
from app.generate.service import answer_query
from app.llm import LLMError, public_llm_failure
from app.models.user import User
from app.trace.service import llm_model_name, save_trace
from app.verify.service import verify_answer
from app.generate.general import AnswerSource, explain

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    session_id: uuid.UUID | None = None
    source_run_id: uuid.UUID | None = None
    analysis_mode: str | None = Field(default=None, pattern=r"^(auto|quick|synthesis|digest)$")
    target_document_id: uuid.UUID | None = None
    answer_source: AnswerSource = "document_only"
    source_trace_id: uuid.UUID | None = None
    selected_context: str = Field(default="", max_length=2000)
    query: str = Field(..., min_length=1, max_length=4000, description="Natural language question")
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Number of reranked evidence chunks fed to the LLM"
    )
    candidates: int | None = Field(
        default=None, ge=1, le=200, description="Hybrid (RRF) candidate pool size before reranking"
    )
    verify: bool | None = Field(
        default=None,
        description="Run Day 7 Verifier on the generated answer (default: QA_VERIFY)",
    )
    persist: bool | None = Field(
        default=None,
        description="Persist this run as a query_traces row (default: QA_PERSIST_TRACES)",
    )
    eval_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Optional golden-query id stored on the trace for Day 9 eval",
    )
    scope_mode: str | None = Field(default=None, pattern=r"^(auto|all|soft|hard)$")
    collection_ids: list[str] = Field(default_factory=list)


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    page: int


class EvidenceOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    page: int
    text: str
    chunk_hash: str = ""
    section_path: str = ""
    start_offset: int = 0


class QueryResponse(BaseModel):
    answer_source: AnswerSource = "document_only"
    source_trace_id: str = ""
    query: str
    answer: str
    provider: str
    citations: list[CitationOut]
    evidence: list[EvidenceOut]
    verification: VerificationOut | None = None
    usage: list[dict[str, Any]] = Field(default_factory=list)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    timing_meta: dict[str, str | int] = Field(default_factory=dict)
    citation_integrity: list[dict[str, Any]] = Field(default_factory=list)
    evidence_selection: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    eval_id: str | None = None
    scope: dict | None = None


@router.post("/query/route")
def route_question(body: QueryRequest, db: Session = Depends(get_db), user: User | None = Depends(get_current_user)):
    from app.agent.routing import decide, validate_scope, validate_target
    from app.corpus.router import resolve_scope

    try:
        if body.answer_source == "document_only":
            validate_scope(db, current_owner(user), body.scope_mode, body.collection_ids)
        context = None
        if body.source_run_id:
            from app.agent.service import get_agent_run

            context = get_agent_run(db, body.source_run_id, owner_id=current_owner(user))
            if context is None or context.user_id != current_owner(user) or context.session_id != body.session_id:
                raise HTTPException(status_code=404, detail="source run not found")
        decision = decide(
            body.query,
            body.analysis_mode or "auto",
            answer_source=body.answer_source,
            target_document_id=body.target_document_id,
            has_answer_context=bool(context),
        )
        if body.answer_source == "document_only" and body.target_document_id:
            scope = resolve_scope(
                db, current_owner(user), body.query, mode=body.scope_mode or "auto", collection_ids=body.collection_ids
            )
            validate_target(db, current_owner(user), body.target_document_id, scope)
        return decision
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User | None = Depends(get_current_user),
    _: None = Depends(rate_limit_query),
) -> QueryResponse:
    query_text = body.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="query must not be empty")

    uid = current_owner(user)
    route_decision = None
    if body.analysis_mode is not None:
        route_decision = route_question(body, db=db, user=user)
        if route_decision["actual"] not in {"quick", "general_explanation"}:
            raise HTTPException(status_code=409, detail="此任务需要 Agent 执行，请重新提交")
    if body.answer_source == "general_explanation":
        return _general_query(body, db=db, settings=settings, uid=uid)
    from app.corpus.router import resolve_scope
    from app.corpus.scope import Scope
    from app.llm.ledger import attach_call_refs, bind_call_context
    from app.search.stages import add_stage, note_meta, reset_stages

    reset_stages(capture_candidates=settings.search_trace_candidates)
    scope_started = time.perf_counter()
    try:
        scope = resolve_scope(
            db,
            uid,
            query_text,
            mode=body.scope_mode or "auto",
            collection_ids=body.collection_ids,
        )
    except Exception as exc:  # noqa: BLE001 — collections table may be absent
        if "collections" not in str(exc) and "does not exist" not in str(exc).lower():
            raise
        scope = Scope()
    add_stage("scope_ms", scope_started)
    from app.core.request_id import get_request_id

    request_id = get_request_id()
    if request_id:
        note_meta("request_id", request_id)
    try:
        with bind_call_context(user_id=uid):
            result = answer_query(
                db,
                query_text,
                top_k=body.top_k,
                candidates=body.candidates,
                owner_id=uid,
                scope=scope,
                settings=settings,
            )
            if scope.mode == "soft" and not result.evidence:
                result = answer_query(
                    db,
                    query_text,
                    top_k=body.top_k,
                    candidates=body.candidates,
                    owner_id=uid,
                    scope=Scope(),
                    settings=settings,
                )
                if result.evidence:
                    scope = Scope(
                        mode=scope.mode,
                        collection_ids=scope.collection_ids,
                        labels=scope.labels,
                        widened=True,
                    )
    except LLMError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"模型服务未就绪：{public_llm_failure(exc)}。"
                "请配置 DASHSCOPE_API_KEY；离线测试需 ANSWER_LLM_PROVIDER=stub 且 ALLOW_LLM_STUB=true。"
            ),
        ) from exc

    should_verify = settings.qa_verify if body.verify is None else body.verify
    verification = None
    if should_verify:
        from app.search.stages import add_stage

        started = time.perf_counter()
        verification = to_verification_out(verify_answer(result.answer, result.evidence, settings=settings))
        add_stage("verify_ms", started)
    from app.search.stages import snapshot_candidates, snapshot_meta, snapshot_stages
    from app.verify.structure import assess_citations, load_citation_rows

    citations = [CitationOut(**vars(c)) for c in result.citations]
    evidence = [
        EvidenceOut(
            index=item.index,
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            page=item.page,
            text=item.text,
            chunk_hash=item.chunk_hash,
            section_path=item.section_path,
            start_offset=item.start_offset,
        )
        for item in result.evidence
    ]
    citation_integrity = assess_citations(
        result.evidence,
        load_citation_rows(db, [item.chunk_id for item in result.evidence]),
        owner_id=uid,
    )

    trace_id = None
    should_persist = settings.qa_persist_traces if body.persist is None else body.persist
    if should_persist:
        row = save_trace(
            db,
            query=result.query,
            answer=result.answer,
            provider=result.provider,
            llm_model=llm_model_name(settings),
            top_k=body.top_k if body.top_k is not None else settings.qa_top_k,
            citations=[item.model_dump() for item in citations],
            evidence=[item.model_dump() for item in evidence],
            verification=None if verification is None else verification.model_dump(),
            usage=list(result.usage or []),
            eval_id=body.eval_id,
            user_id=uid,
            diagnostics={
                "answer_source": "document_only",
                "routing": route_decision,
                "scope": scope.to_dict(),
                "stages_ms": snapshot_stages(),
                "meta": snapshot_meta(),
                "candidate_stages": snapshot_candidates(),
                "citation_integrity": citation_integrity,
                "prompt_version": str(snapshot_meta().get("prompt_version") or ""),
                "evidence_selection": result.evidence_selection,
            },
        )
        attach_call_refs(trace_id=row.id)
        trace_id = str(row.id)

    if scope.mode == "soft":
        from sqlalchemy import select

        from app.corpus.library import log_scope_search
        from app.models import Document

        outside = []
        if scope.widened:
            ids = []
            for item in result.evidence:
                try:
                    ids.append(uuid.UUID(str(item.doc_id)))
                except ValueError:
                    continue
            if ids:
                for doc_id, collection_id in db.execute(
                    select(Document.id, Document.collection_id).where(Document.id.in_(ids))
                ).all():
                    outside.append((doc_id, collection_id))
        try:
            log_scope_search(db, uid, scope, query=query_text, widened_docs=outside)
            db.commit()
        except Exception:  # noqa: BLE001
            db.rollback()

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        provider=result.provider,
        citations=citations,
        evidence=evidence,
        verification=verification,
        usage=list(result.usage or []),
        timings_ms=snapshot_stages(),
        timing_meta=snapshot_meta(),
        citation_integrity=citation_integrity,
        evidence_selection=result.evidence_selection,
        trace_id=trace_id,
        eval_id=body.eval_id,
        scope=scope.to_dict(),
    )


def _general_query(body, *, db, settings, uid):
    from app.trace.service import get_trace
    from app.generate.general import source_mode
    from app.llm.ledger import attach_call_refs, bind_call_context

    source = get_trace(db, body.source_trace_id, owner_id=uid) if body.source_trace_id else None
    if source is None or source.user_id != uid or source_mode(source) != "document_only":
        raise HTTPException(status_code=404, detail="document source trace not found")
    if body.selected_context and body.selected_context not in source.answer:
        raise HTTPException(status_code=400, detail="selected_context must be selected from source answer")
    try:
        with bind_call_context(user_id=uid):
            result = explain(
                body.query.strip(), topic=source.query, selected_context=body.selected_context, settings=settings
            )
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=public_llm_failure(exc)) from exc
    trace_id = None
    if settings.qa_persist_traces if body.persist is None else body.persist:
        row = save_trace(
            db,
            query=result.query,
            answer=result.answer,
            provider=result.provider,
            llm_model=llm_model_name(settings),
            top_k=0,
            citations=[],
            evidence=[],
            verification=None,
            usage=result.usage,
            user_id=uid,
            diagnostics={"answer_source": "general_explanation", "source_trace_id": str(source.id)},
        )
        attach_call_refs(trace_id=row.id)
        trace_id = str(row.id)
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        provider=result.provider,
        citations=[],
        evidence=[],
        usage=result.usage,
        trace_id=trace_id,
        answer_source="general_explanation",
        source_trace_id=str(source.id),
    )
