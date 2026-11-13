from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.generate.prompt import PROMPT_VERSION, SYSTEM_PROMPT, EvidenceItem, prepare_prompt
from app.llm import answer_provider_name, get_llm_client
from app.models import Chunk

_CITATION_RE = re.compile(r"\[S(\d+)\]")


@dataclass
class Citation:
    index: int
    chunk_id: str
    doc_id: str
    page: int


@dataclass
class AnswerResult:
    query: str
    answer: str
    provider: str
    citations: list[Citation]
    evidence: list[EvidenceItem]
    usage: list[dict] = field(default_factory=list)
    evidence_selection: dict = field(default_factory=dict)


_NO_EVIDENCE_ANSWER = "现有证据不足以回答该问题（检索未命中任何相关片段）。"


def answer_query(
    db: Session,
    query: str,
    *,
    top_k: int | None = None,
    candidates: int | None = None,
    owner_id: UUID | None = None,
    scope=None,
    settings: Settings | None = None,
) -> AnswerResult:
    settings = settings or get_settings()
    q = query.strip()
    if not q:
        raise ValueError("query must not be empty")

    k = top_k if top_k is not None else settings.qa_top_k
    from app.search.rerank import rerank_search

    hits = rerank_search(db, q, top_k=k, candidates=candidates, owner_id=owner_id, scope=scope, settings=settings)
    if not hits:
        return AnswerResult(
            query=q,
            answer=_NO_EVIDENCE_ANSWER,
            provider=answer_provider_name(settings),
            citations=[],
            evidence=[],
        )

    rows = _fetch_chunk_rows(db, [hit.chunk_id for hit in hits])
    evidence = []
    for i, hit in enumerate(hits):
        text, chunk_hash, section_path, start_offset = rows.get(
            hit.chunk_id,
            (hit.snippet, "", "", hit.start),
        )
        evidence.append(
            EvidenceItem(
                index=i + 1,
                chunk_id=str(hit.chunk_id),
                doc_id=str(hit.doc_id),
                page=hit.page,
                text=text,
                chunk_hash=chunk_hash,
                section_path=section_path,
                start_offset=int(start_offset or 0),
            )
        )
    return answer_from_evidence(q, evidence, settings=settings)


def answer_from_evidence(
    query: str,
    evidence: list[EvidenceItem],
    *,
    settings: Settings | None = None,
) -> AnswerResult:
    """Generate a cited answer from already-retrieved evidence (Day 13 agent reuse)."""
    settings = settings or get_settings()
    q = query.strip()
    if not evidence:
        return AnswerResult(
            query=q,
            answer=_NO_EVIDENCE_ANSWER,
            provider=answer_provider_name(settings),
            citations=[],
            evidence=[],
        )

    prepared = prepare_prompt(q, evidence, max_context_chars=settings.qa_max_context_chars)
    prompt = prepared.text
    evidence = prepared.evidence
    client = get_llm_client(settings)
    from app.search.stages import add_stage, note_meta

    note_meta("generate_prompt_chars", len(prompt))
    note_meta("llm_provider", answer_provider_name(settings))
    note_meta("prompt_version", PROMPT_VERSION)
    note_meta("evidence_candidate_count", len(prepared.selection["candidate_evidence"]))
    note_meta("evidence_sent_count", len(evidence))
    note_meta("evidence_omitted_count", len(prepared.selection["omitted_indices"]))
    started = time.perf_counter()
    response = client.generate(
        system=SYSTEM_PROMPT,
        prompt=prompt,
        max_output_tokens=settings.llm_max_output_tokens,
        temperature=settings.llm_temperature,
        purpose="answer",
    )
    add_stage("generate_ms", started)
    note_meta("llm_model", response.usage.model or "")
    note_meta("prompt_tokens", int(response.usage.prompt_tokens or 0))
    note_meta("completion_tokens", int(response.usage.completion_tokens or 0))
    from app.generate.synthesis import plain_answer

    answer_text = plain_answer(response.text)
    valid_indices = {item.index for item in evidence}
    invalid_indices = sorted({int(match.group(1)) for match in _CITATION_RE.finditer(answer_text)} - valid_indices)
    prepared.selection["invalid_citation_indices"] = invalid_indices
    if invalid_indices:
        # Fail closed for the whole draft: merely dropping metadata would leave false source badges.
        answer_text = "现有证据不足以回答该问题（模型引用了未提供的来源，本次回答无法核验，请重试）。"
    citations = _extract_citations(answer_text, evidence)
    return AnswerResult(
        query=q,
        answer=answer_text,
        provider=answer_provider_name(settings),
        citations=citations,
        evidence=evidence,
        usage=[response.usage.to_dict()],
        evidence_selection=prepared.selection,
    )


def _fetch_chunk_rows(db: Session, chunk_ids: list[UUID]) -> dict[UUID, tuple[str, str, str, int]]:
    """chunk_id -> (全文, chunk_hash, section_path, start_offset)."""
    if not chunk_ids:
        return {}
    rows = db.execute(
        select(Chunk.id, Chunk.text, Chunk.chunk_hash, Chunk.section_path, Chunk.start_offset).where(
            Chunk.id.in_(chunk_ids)
        )
    ).all()
    return {
        row.id: (row.text or "", row.chunk_hash or "", row.section_path or "", int(row.start_offset or 0))
        for row in rows
    }


def _extract_citations(answer_text: str, evidence: list[EvidenceItem]) -> list[Citation]:
    by_index = {item.index: item for item in evidence}
    seen: set[int] = set()
    citations: list[Citation] = []
    for match in _CITATION_RE.finditer(answer_text):
        idx = int(match.group(1))
        if idx in seen or idx not in by_index:
            continue
        seen.add(idx)
        item = by_index[idx]
        citations.append(Citation(index=idx, chunk_id=item.chunk_id, doc_id=item.doc_id, page=item.page))
    return citations
